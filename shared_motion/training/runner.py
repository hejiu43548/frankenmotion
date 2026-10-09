"""Three staged optimizers with fixed scans, provenance and exact RNG resume."""

import fcntl
import json
import os
from pathlib import Path
import random
import traceback

from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import torch

from .catalog import COMMAND_RANGES, TASK_NAMES, command_error, measure
from .data import MotionDataset, StatefulSampler, assert_disjoint
from .geometry import Skeleton
from .model import build_model, file_sha256
from .turn import TURN_POLICY, TURN_NATIVE_SPEED, scan_indices, sample_commands


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes():
    repository = Path(__file__).resolve().parents[2]
    paths = (
        list((repository / "shared_motion/training").glob("*.py"))
        + list((repository / "shared_motion/adapter").glob("*.py"))
        + list((repository / "scripts").glob("train*.py"))
        + list((repository / "src/model/backbones").rglob("*.py"))
        + list((repository / "src/model/schedule").rglob("*.py"))
        + [repository / "src/tools/geometry.py"]
    )
    return {
        str(path.relative_to(repository)): file_sha256(path)
        for path in paths
        if not path.name.startswith("._")
    }


def frozen_state(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }


def verify_frozen(model, snapshot):
    parameters = dict(model.named_parameters())
    if any(
        parameters[name].grad is not None
        or not torch.equal(parameters[name].detach().cpu(), value)
        for name, value in snapshot.items()
    ):
        raise RuntimeError("Frozen backbone/root parameters changed")


def checkpoint_compatible(checkpoint, model, tasks, stage=None):
    if checkpoint.get("format") == "verified_external_root_v1":
        checks = checkpoint.get("import_checks", {})
        if (
            stage != 1
            or model.kind != "charlie_root"
            or checkpoint["controller_kind"] != model.kind
            or checkpoint["backbone_sha256"] != model.backbone_sha256
            or checkpoint["backbone_config"] != model.backbone_configuration
            or not checks.get("parameters_exact")
            or not checks.get("forward_bitwise_equal")
            or not checkpoint.get("source", {}).get("checkpoint_sha256")
        ):
            raise ValueError(
                "External root import is valid only for verified stage2 root initialization"
            )
        return
    if (
        checkpoint["controller_kind"] != model.kind
        or checkpoint["backbone_sha256"] != model.backbone_sha256
        or checkpoint["tasks"] != tasks
    ):
        raise ValueError(
            "Checkpoint controller/backbone/task catalog does not match config"
        )
    if checkpoint.get("task_policy") != model.task_policy:
        raise ValueError("Checkpoint task semantics differ: walking_turn_v2 required")
    if checkpoint["backbone_config"] != model.backbone_configuration:
        raise ValueError(
            "Checkpoint backbone architecture/schedule config does not match"
        )
    if stage is not None and checkpoint["stage"] != stage:
        raise ValueError(
            f"Expected stage {stage} checkpoint, got {checkpoint['stage']}"
        )


def rng_state():
    return dict(
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        python=random.getstate(),
        numpy=np.random.get_state(),
    )


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


@torch.no_grad()
def validate(model, skeleton, dataset, recipe, config):
    device = torch.device(config.runtime.device)
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    model.eval()
    total = 0.0
    count = 0
    validation_recipe = (
        recipe
        if recipe.mode != "free"
        else instantiate(config.validation.supervised_loss)
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.validation.seed)
        for start in range(0, len(dataset.rows), config.validation.batch_size):
            indices = list(
                range(
                    start, min(start + config.validation.batch_size, len(dataset.rows))
                )
            )
            batch = dataset.batch(indices, device)
            loss, _ = validation_recipe.supervised(
                model, skeleton, batch, training=False
            )
            total += float(loss) * len(indices)
            count += len(indices)
    return total / count


@torch.no_grad()
def scan(model, skeleton, dataset, config, artifact_path, step):
    model.eval()
    tasks = []
    motions = []
    lengths = []
    task_indices = []
    requested_values = []
    measured_values = []
    for task_name in config.data.tasks:
        task_index = TASK_NAMES.index(task_name)
        commands = torch.linspace(
            *COMMAND_RANGES[task_index],
            config.validation.points,
            device=config.runtime.device,
        )
        selected = scan_indices(dataset, task_index, commands, TASK_NAMES.index("turn"))
        batch = dataset.batch(selected, config.runtime.device)
        motion = model.sample(
            batch,
            commands,
            skeleton,
            [config.validation.noise_seed + task_index] * len(commands),
            config.validation.ddim_steps,
        )
        measured = measure(skeleton, motion, batch["task"], batch["lengths"])
        errors = command_error(measured, commands, batch["task"]).abs()
        if not torch.isfinite(motion).all() or not torch.isfinite(errors).all():
            raise RuntimeError("Nonfinite generation scan")
        bounds = COMMAND_RANGES[task_index]
        tasks.append(
            dict(
                task=task_name,
                requested=commands.tolist(),
                measured=measured.tolist(),
                mae=float(errors.mean()),
                normalized_mae=float(errors.mean()) / (bounds[1] - bounds[0]),
                keys=[dataset.rows[index]["key"] for index in selected],
                families=[dataset.rows[index]["family"] for index in selected],
                noise_seed=config.validation.noise_seed + task_index,
            )
        )
        if task_name == "turn":
            speeds = torch.stack(
                [
                    motion[index, : int(batch["lengths"][index]) - 1, 1:3]
                    .norm(dim=-1)
                    .mean()
                    * 20
                    for index in range(len(motion))
                ]
            )
            tasks[-1].update(
                requested_speed_m_s=TURN_NATIVE_SPEED,
                measured_speed_m_s=speeds.tolist(),
                speed_mae_m_s=float((speeds - TURN_NATIVE_SPEED).abs().mean()),
            )
        motions.append(motion.cpu().numpy())
        lengths.append(batch["lengths"].cpu().numpy())
        task_indices.append(batch["task"].cpu().numpy())
        requested_values.append(commands.cpu().numpy())
        measured_values.append(measured.cpu().numpy())
    artifact_path.mkdir(parents=True, exist_ok=True)
    archive = artifact_path / f"scan_{step:06d}.npz"
    np.savez_compressed(
        archive,
        motion=np.concatenate(motions),
        lengths=np.concatenate(lengths),
        task=np.concatenate(task_indices),
        requested=np.concatenate(requested_values),
        measured=np.concatenate(measured_values),
    )
    return dict(
        step=step,
        tasks=tasks,
        macro_normalized_mae=float(
            np.mean([entry["normalized_mae"] for entry in tasks])
        ),
        motion_archive=str(archive),
        motion_sha256=file_sha256(archive),
        protocol="fixed validation text per task, fixed seeds, pure-noise DDIM, requested root controls only",
    )


def audit_scans(report_path, skeleton, device):
    maximum_error = 0.0
    count = 0
    for path in sorted((report_path / "scans").glob("scan_*.json")):
        result = json.loads(path.read_text())
        if file_sha256(result["motion_archive"]) != result["motion_sha256"]:
            raise RuntimeError("Saved rollout archive hash mismatch")
        with np.load(result["motion_archive"]) as archive:
            motion = torch.tensor(archive["motion"], device=device)
            task_indices = torch.tensor(archive["task"], device=device)
            lengths = torch.tensor(archive["lengths"], device=device)
            with torch.no_grad():
                recomputed = measure(skeleton, motion, task_indices, lengths)
            expected = torch.tensor(archive["measured"], device=device)
            error = float(command_error(recomputed, expected, task_indices).abs().max())
            if error > 1e-5:
                raise RuntimeError("Saved rollout measurements failed FK recomputation")
            maximum_error = max(maximum_error, error)
            count += len(motion)
    return dict(rollouts=count, max_quantity_error=maximum_error)


def _run_locked(config):
    resolved = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    phase = "root" if config.stage.index == 1 else "task"
    expected_mode = {1: "root", 2: "supervised", 3: "free"}[config.stage.index]
    recipe = instantiate(config.loss)
    if recipe.mode != expected_mode:
        raise ValueError(
            f"Stage {config.stage.index} requires a {expected_mode} loss; got {recipe.mode}"
        )
    if len(set(config.data.tasks)) != len(config.data.tasks) or set(
        config.data.tasks
    ) - set(TASK_NAMES):
        raise ValueError("Unsupported or duplicate task catalog")
    if config.validation.points < 2 or not 2 <= config.validation.ddim_steps <= 100:
        raise ValueError("Scans need >=2 command points and 2..100 DDIM steps")
    if (
        config.stage.batch_size < 1
        or config.stage.accumulate < 1
        or (config.stage.index != 1 and config.stage.accumulate != 1)
    ):
        raise ValueError(
            "Positive batch size required; accumulation is supported by root epochs only"
        )
    report_path = Path(config.paths.report).resolve()
    artifact_path = Path(config.paths.artifacts).resolve()
    report_path.mkdir(parents=True, exist_ok=True)
    artifact_path.mkdir(parents=True, exist_ok=True)
    if (report_path / "protocol.json").exists() and not config.runtime.resume:
        raise FileExistsError(
            "Run already initialized; choose a new experiment or explicit runtime.resume checkpoint"
        )
    torch.set_num_threads(config.runtime.threads)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    training = MotionDataset(
        config.data.train_manifest,
        "train",
        config.data.tasks,
        config.data.path_root,
        config.stage.index == 1,
    )
    validation = MotionDataset(
        config.data.val_manifest,
        "val",
        config.data.tasks,
        config.data.path_root,
        config.stage.index == 1,
    )
    assert_disjoint(training, validation)
    skeleton = Skeleton(config.data.skeleton).to(config.runtime.device)
    model = build_model(config, phase).to(config.runtime.device)
    previous = None
    if config.stage.index > 1:
        if not config.initial:
            raise ValueError("Stages2/3 require initial=<previous-stage checkpoint>")
        previous = torch.load(config.initial, map_location="cpu", weights_only=False)
        checkpoint_compatible(
            previous, model, list(config.data.tasks), config.stage.index - 1
        )
        model.load_adapter(previous["adapter"], root_only=config.stage.index == 2)
    snapshot = frozen_state(model)
    reference = None
    if config.stage.index == 3:
        reference = build_model(config, "task").to(config.runtime.device)
        reference.load_adapter(previous["adapter"])
        reference.requires_grad_(False).eval()
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.stage.learning_rate,
        weight_decay=config.stage.weight_decay,
    )
    sampler = StatefulSampler(training, config.seed, config.stage.index == 1)
    protocol = dict(
        controller_kind=model.kind,
        task_policy=TURN_POLICY,
        stage=config.stage.index,
        backbone_sha256=model.backbone_sha256,
        backbone_config=OmegaConf.to_container(config.backbone, resolve=True),
        tasks=list(config.data.tasks),
        loss=OmegaConf.to_container(config.loss, resolve=True),
        controller=OmegaConf.to_container(config.controller, resolve=True),
        schedule=OmegaConf.to_container(config.stage, resolve=True),
        validation=OmegaConf.to_container(config.validation, resolve=True),
        seed=config.seed,
        initial_sha256=file_sha256(config.initial) if config.initial else None,
        initial_source=previous.get("source") if previous else None,
        data_sha256={
            "train": file_sha256(config.data.train_manifest),
            "val": file_sha256(config.data.val_manifest),
        },
        cache_sha256=dict(train=training.fingerprint, val=validation.fingerprint),
        skeleton_sha256=file_sha256(config.data.skeleton),
        sources=source_hashes(),
        trainable_parameters=sum(parameter.numel() for parameter in parameters),
    )
    best = float("inf")
    stale = 0
    step = 0
    best_step = None
    if config.runtime.resume:
        checkpoint = torch.load(
            config.runtime.resume, map_location="cpu", weights_only=False
        )
        checkpoint_compatible(
            checkpoint, model, list(config.data.tasks), config.stage.index
        )
        if checkpoint["protocol"] != protocol:
            raise ValueError(
                "Resume protocol differs: data, sources, controller, loss and stage settings must match"
            )
        model.load_adapter(checkpoint["adapter"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        sampler.load_state_dict(checkpoint["sampler"])
        step, best, stale, best_step = (
            checkpoint["step"],
            checkpoint["best"],
            checkpoint["stale"],
            checkpoint["best_step"],
        )
        restore_rng(checkpoint["rng"])
        verify_frozen(model, snapshot)
    else:
        save_json(report_path / "protocol.json", protocol)
        OmegaConf.save(config, report_path / "config.yaml", resolve=True)
        save_json(report_path / "data_index/train.json", training.rows)
        save_json(report_path / "data_index/val.json", validation.rows)
        save_json(
            report_path / "data_index/cache_sha256.json",
            dict(train=training.cache_hashes, val=validation.cache_hashes),
        )
        save_json(report_path / "configuration.json", resolved)

    def save_checkpoint():
        state = dict(
            adapter=model.adapter_state(),
            optimizer=optimizer.state_dict(),
            sampler=sampler.state_dict(),
            rng=rng_state(),
            step=step,
            best=best,
            stale=stale,
            best_step=best_step,
            protocol=protocol,
            controller_kind=model.kind,
            task_policy=TURN_POLICY,
            stage=config.stage.index,
            backbone_sha256=model.backbone_sha256,
            backbone_config=OmegaConf.to_container(config.backbone, resolve=True),
            tasks=list(config.data.tasks),
        )
        temporary = artifact_path / "latest.tmp"
        torch.save(state, temporary)
        temporary.replace(artifact_path / "latest.pt")

    def evaluate():
        nonlocal best, stale, best_step
        validation_loss = validate(model, skeleton, validation, recipe, config)
        result = dict(step=step, epoch=sampler.epoch, validation_loss=validation_loss)
        metric = validation_loss
        if config.stage.index > 1:
            result["scan"] = scan(
                model, skeleton, validation, config, artifact_path / "motions", step
            )
            save_json(report_path / "scans" / f"scan_{step:06d}.json", result["scan"])
            metric = result["scan"]["macro_normalized_mae"]
        improved = metric < best - (
            config.stage.min_delta if config.stage.index == 1 else 0
        )
        if improved:
            best, stale, best_step = metric, 0, step
        else:
            stale += 1
        verify_frozen(model, snapshot)
        save_checkpoint()
        archive = artifact_path / f"checkpoint_{step:06d}.pt"
        if archive.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint archive {archive}")
        os.link(artifact_path / "latest.pt", archive)
        if improved:
            temporary = artifact_path / "best.tmp"
            if temporary.exists():
                temporary.unlink()
            os.link(archive, temporary)
            temporary.replace(artifact_path / "best.pt")
            save_json(
                report_path / "best.json",
                dict(
                    step=step,
                    metric=best,
                    checkpoint=str(archive),
                    sha256=file_sha256(archive),
                ),
            )
        save_json(report_path / f"validation_{step:06d}.json", result)
        print(
            json.dumps(dict(event="evaluation", step=step, metric=metric)), flush=True
        )

    try:
        if not config.runtime.resume:
            evaluate()
        while True:
            if config.stage.index == 1:
                if (
                    sampler.epoch >= config.stage.epochs
                    or stale >= config.stage.patience
                ):
                    break
            elif step >= config.stage.steps:
                break
            previous_epoch = sampler.epoch
            batches = sampler.batches(config.stage.batch_size, config.stage.accumulate)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for indices in batches:
                batch = training.batch(indices, config.runtime.device)
                if recipe.mode == "free":
                    commands = sample_commands(
                        batch, COMMAND_RANGES, TASK_NAMES.index("turn")
                    )
                    seeds = [
                        config.seed
                        + 510000000
                        + (step + 1) * config.stage.batch_size
                        + sample_index
                        for sample_index in range(len(indices))
                    ]
                    loss, metrics = recipe.free(
                        model, reference, skeleton, batch, commands, seeds
                    )
                else:
                    loss, metrics = recipe.supervised(
                        model, skeleton, batch, dropout=config.stage.root_dropout
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                (loss / len(batches)).backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, config.stage.clip, error_if_nonfinite=True
            )
            optimizer.step()
            step += 1
            if step % config.runtime.log_every == 0:
                save_json(
                    report_path / "status.json",
                    dict(
                        state="training",
                        stage=config.stage.index,
                        step=step,
                        epoch=sampler.epoch,
                        pid=os.getpid(),
                        loss=float(loss.detach()),
                        metrics={name: float(value) for name, value in metrics.items()},
                    ),
                )
            evaluated = (
                sampler.epoch > previous_epoch
                if config.stage.index == 1
                else step % config.stage.eval_every == 0 or step == config.stage.steps
            )
            if evaluated:
                evaluate()
            elif step % config.runtime.checkpoint_every == 0:
                save_checkpoint()
            if (
                config.runtime.stop_after is not None
                and step >= config.runtime.stop_after
            ):
                save_checkpoint()
                save_json(
                    report_path / "status.json",
                    dict(
                        state="interrupted",
                        stage=config.stage.index,
                        step=step,
                        epoch=sampler.epoch,
                    ),
                )
                return
        verify_frozen(model, snapshot)
        save_checkpoint()
        if config.stage.index < 3 and len(sampler.seen) != len(training.rows):
            raise RuntimeError(
                "Not every training record was sampled; increase the run length before marking complete"
            )
        audit = audit_scans(report_path, skeleton, config.runtime.device)
        audit.update(
            frozen_parameters_verified=True,
            unique_training_records_seen=len(sampler.seen),
            training_records=len(training.rows),
            best_step=best_step,
        )
        save_json(report_path / "completion_audit.json", audit)
        save_json(
            report_path / "status.json",
            dict(
                state="complete",
                stage=config.stage.index,
                step=step,
                epoch=sampler.epoch,
                best_metric=best,
            ),
        )
        (report_path / "report.txt").write_text(
            f"Stage {config.stage.index} completed at optimizer step {step}.\nController: {model.kind}\nBest evaluation step: {best_step}; metric: {best}\nFrozen parameters and saved rollout quantities verified.\nCommand error alone does not certify natural motion.\nSee config.yaml, protocol.json, data_index/, completion_audit.json and best.json.\nLarge artifacts: {artifact_path}\n"
        )
    except BaseException as error:
        save_json(
            report_path / "failure.json",
            dict(error=repr(error), traceback=traceback.format_exc(), step=step),
        )
        raise


def run(config):
    report_path = Path(config.paths.report).resolve()
    report_path.mkdir(parents=True, exist_ok=True)
    with (report_path / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(config)
