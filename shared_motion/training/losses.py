"""Configurable loss recipes, including the two experimentally used free losses."""

import torch
from torch.nn import functional as functional

from .catalog import COMMAND_RANGES, TASK_NAMES, command_error, measure
from .model import root_signals, smooth_profile, training_root


class LossRecipe:
    def __init__(
        self,
        mode,
        command_kind="huber",
        circular_turn=False,
        reconstruction=1.0,
        root_feature_weight=5.0,
        command_weight=0.2,
        speed_weight=0.1,
        yaw_weight=0.1,
        pose_weight=0.0,
        root_position_weight=0.0,
        velocity_excess_weight=0.0,
        velocity_match_weight=0.0,
        support_weight=0.0,
        retain_weight=0.0,
        stationary_priors=False,
        prototype_weight=0.0,
        reference_mode="root_only",
        ddim_steps=10,
    ):
        if mode not in ["root", "supervised", "free"] or command_kind not in [
            "huber",
            "mse",
        ]:
            raise ValueError("Unsupported loss mode or command error")
        if (
            reference_mode not in ["task", "root_only", "backbone"]
            or not 2 <= ddim_steps <= 100
        ):
            raise ValueError("Invalid reference mode or DDIM steps")
        if circular_turn:
            raise ValueError("walking_turn_v2 requires signed unwrapped turn error")
        self.options = dict(locals())
        self.options.pop("self")
        for name, value in self.options.items():
            setattr(self, name, value)

    def difference(self, predicted, requested, task_indices):
        return (
            command_error(predicted, requested, task_indices)
            if self.circular_turn
            else predicted - requested
        )

    def command_loss(self, error):
        if self.command_kind == "mse":
            return error.square().mean()
        return functional.smooth_l1_loss(error, torch.zeros_like(error))

    def supervised(self, model, skeleton, batch, training=True, dropout=0.15):
        clean, conditioning = model.inputs(batch)
        target, valid, available, control = training_root(
            batch, dropout if training else 0, only_supported=self.mode != "root"
        )
        timesteps = torch.randint(len(model.alphas), (len(clean),), device=clean.device)
        noise = torch.randn_like(clean[..., :205])
        alpha = model.alphas[timesteps, None, None]
        noisy = torch.cat(
            [
                alpha.sqrt() * clean[..., :205] + (1 - alpha).sqrt() * noise,
                clean[..., 205:],
            ],
            -1,
        )
        prediction = model.predict(
            noisy,
            conditioning,
            timesteps,
            batch,
            batch["quantity"],
            control,
            "root_only" if self.mode == "root" else "task",
        )
        feature_weights = prediction.new_ones(205)
        feature_weights[:4] = self.root_feature_weight
        per_frame = (
            (prediction[..., :205] - clean[..., :205]).square() * feature_weights
        ).mean(-1)
        if self.mode == "root":
            reconstruction = (per_frame * batch["mask"]).sum() / batch["mask"].sum()
        else:
            reconstruction = (
                (per_frame * batch["mask"]).sum(1) / batch["lengths"]
            ).mean()
        raw = prediction[..., :205] * (model.std[:205] + 1e-12) + model.mean[:205]
        predicted_profile = smooth_profile(root_signals(raw), valid)
        profile_error = predicted_profile - target
        if self.mode == "root":
            speed = (
                profile_error[..., 0].square() * available[..., 0]
            ).sum() / available[..., 0].sum().clamp_min(1)
            yaw = (
                profile_error[..., 1].square() * available[..., 1]
            ).sum() / available[..., 1].sum().clamp_min(1)
        else:
            speed = (
                (profile_error[..., 0].square() * valid).sum(1)
                / valid.sum(1).clamp_min(1)
            ).mean()
            yaw = (
                (profile_error[..., 1].square() * valid).sum(1)
                / valid.sum(1).clamp_min(1)
            ).mean()
        command_term = raw.new_zeros(())
        if self.command_weight and self.mode != "root":
            ranges = raw.new_tensor(COMMAND_RANGES)[batch["task"]]
            measured = measure(skeleton, raw, batch["task"], batch["lengths"])
            error = self.difference(measured, batch["quantity"], batch["task"]) / (
                ranges[:, 1] - ranges[:, 0]
            )
            command_term = self.command_loss(error)
        loss = (
            self.reconstruction * reconstruction
            + self.command_weight * command_term
            + self.speed_weight * speed
            + self.yaw_weight * yaw
        )
        return loss, dict(
            reconstruction=reconstruction.detach(),
            command=command_term.detach(),
            speed=speed.detach(),
            yaw=yaw.detach(),
        )

    def free(self, model, reference, skeleton, batch, commands, seeds):
        with torch.no_grad():
            anchor = reference.sample(
                batch, commands, skeleton, seeds, self.ddim_steps, self.reference_mode
            )
        raw = model.sample(batch, commands, skeleton, seeds, self.ddim_steps)
        positions = skeleton(raw)
        target_positions = skeleton(anchor)
        mask = batch["mask"]
        transition = mask[:, 1:]
        ranges = raw.new_tensor(COMMAND_RANGES)[batch["task"]]
        error = self.difference(
            measure(skeleton, raw, batch["task"], batch["lengths"]),
            commands,
            batch["task"],
        ) / (ranges[:, 1] - ranges[:, 0])
        command_term = self.command_loss(error)
        relative = (positions - positions[:, :, :1]) - (
            target_positions - target_positions[:, :, :1]
        )
        weights = raw.new_ones(len(raw), 24)
        # The Charlie recipe downweights the controlled limb in its pose anchor.
        if self.reference_mode == "root_only":
            for joint_index in [17, 19, 21, 23]:
                weights[:, joint_index] = torch.where(
                    batch["task"] <= 3, 0.15, weights[:, joint_index]
                )
            for joint_index in [2, 5, 8, 11]:
                weights[:, joint_index] = torch.where(
                    batch["task"] == 7, 0.15, weights[:, joint_index]
                )
        pose = (
            relative.square() * weights[:, None, :, None] * mask[:, :, None, None]
        ).sum() / (mask.sum() * 24 * 3)
        root_position = (
            (positions[:, :, 0] - target_positions[:, :, 0]).square() * mask[..., None]
        ).sum() / (mask.sum() * 3)
        velocity = (positions[:, 1:] - positions[:, :-1]) * 20
        target_velocity = (target_positions[:, 1:] - target_positions[:, :-1]) * 20
        excess = (
            (velocity.norm(dim=-1) - 1.5 * target_velocity.norm(dim=-1) - 0.5)
            .clamp_min(0)
            .square()
        )
        velocity_excess = (excess * transition[..., None]).sum() / (
            transition.sum() * 24
        )
        velocity_match = (
            (velocity - target_velocity).square() * transition[:, :, None, None]
        ).sum() / (transition.sum() * 24 * 3)
        target_feet = target_positions[:, :, [7, 8]]
        lowest = (
            target_feet[..., 2]
            .masked_fill(~mask[..., None], float("inf"))
            .amin(1, keepdim=True)
        )
        support = (
            (target_feet[:, 1:, :, 2] < lowest + 0.05)
            & (target_velocity[:, :, [7, 8]].norm(dim=-1) < 0.15)
            & transition[..., None]
        )
        foot_speed = (
            velocity[:, :, [7, 8], :2].square().sum(-1) * support
        ).sum() / support.sum().clamp_min(1)
        retain = raw.new_zeros(())
        if self.retain_weight:
            # Replay all20 command types, independent of the sampled minibatch class.
            replay = {
                name: value[:1].expand(len(TASK_NAMES), *value.shape[1:])
                for name, value in batch.items()
            }
            replay["task"] = torch.arange(len(TASK_NAMES), device=raw.device)
            bounds = raw.new_tensor(COMMAND_RANGES)
            replay_commands = bounds[:, 0] + torch.rand(
                len(TASK_NAMES), device=raw.device
            ) * (bounds[:, 1] - bounds[:, 0])
            with torch.no_grad():
                targets = reference.controller_outputs(
                    replay, replay_commands, skeleton
                )
            predictions = model.controller_outputs(replay, replay_commands, skeleton)
            retain = sum(
                (predicted - target).square().mean()
                for predicted, target in zip(predictions, targets)
            )
        prior = raw.new_zeros(())
        if self.stationary_priors:
            for sample_index, task_index in enumerate(batch["task"].tolist()):
                name = TASK_NAMES[task_index]
                if task_index < 11 or name in ["jog", "march"]:
                    continue
                length = int(batch["lengths"][sample_index])
                sample_positions = positions[sample_index, :length]
                sample_velocity = velocity[sample_index, : length - 1]
                prior = (
                    prior
                    + 0.4 * sample_velocity[:, 0, :2].square().mean()
                    + 0.1 * sample_velocity[:, [7, 8]].square().mean()
                )
                if name != "squat":
                    ankles = sample_positions[:, [7, 8]].mean(1)
                    height = sample_positions[:, 0, 2] - ankles[:, 2]
                    offset = (sample_positions[:, 0, :2] - ankles[:, :2]).norm(dim=-1)
                    prior = (
                        prior
                        + 20 * (0.82 - height).relu().square().mean()
                        + 5 * (offset - 0.12).relu().square().mean()
                    )
            prior = prior / len(raw)
        prototype = raw.new_zeros(())
        if self.prototype_weight:
            # Available cached GT motion provides the same two task-specific cues as main.
            with torch.no_grad():
                source_positions = skeleton(batch["motion"])
            for sample_index, task_index in enumerate(batch["task"].tolist()):
                name = TASK_NAMES[task_index]
                length = int(batch["lengths"][sample_index])
                predicted = positions[sample_index, :length]
                source = source_positions[sample_index, :length]
                if name == "arm_circle":
                    predicted_direction = (
                        predicted[:, [20, 21]] - predicted[:, [16, 17]]
                    )
                    source_direction = source[:, [20, 21]] - source[:, [16, 17]]
                    prototype = (
                        prototype
                        + 5
                        * (
                            functional.normalize(predicted_direction, dim=-1)
                            - functional.normalize(source_direction, dim=-1)
                        )
                        .square()
                        .mean()
                    )
                elif name == "clap":
                    gap = (predicted[:, 20] - predicted[:, 21]).norm(dim=-1)
                    source_gap = (source[:, 20] - source[:, 21]).norm(dim=-1)
                    pattern = (source_gap - source_gap.min()) / (
                        source_gap.max() - source_gap.min()
                    ).clamp_min(0.05)
                    prototype = (
                        prototype
                        + 25
                        * (gap - (0.12 + commands[sample_index] * pattern))
                        .square()
                        .mean()
                    )
            prototype = prototype / len(raw)
        loss = (
            self.command_weight * command_term
            + self.pose_weight * pose
            + self.root_position_weight * root_position
            + self.velocity_excess_weight * velocity_excess
            + self.velocity_match_weight * velocity_match
            + self.support_weight * foot_speed
            + self.retain_weight * retain
            + prior
            + self.prototype_weight * prototype
        )
        metrics = dict(
            command=command_term,
            pose=pose,
            root_position=root_position,
            velocity_excess=velocity_excess,
            velocity_match=velocity_match,
            support=foot_speed,
            retain=retain,
            stationary_prior=prior,
            prototype=prototype,
        )
        return loss, {name: value.detach() for name, value in metrics.items()}
