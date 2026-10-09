"""Freeze a complete validation-only checkpoint ranking before test evaluation."""

import hashlib
import json
import math
from pathlib import Path

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_checkpoint(configuration):
    expected = list(configuration["expected_iterations"])
    candidates = list(configuration["candidates"])
    observed = [entry["iteration"] for entry in candidates]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("Expected iterations must be nonempty and unique")
    if sorted(observed) != sorted(expected):
        raise ValueError("Candidate set does not exactly match the declared iterations")
    reference_conditions = None
    ranking = []
    for entry in candidates:
        metrics_path = Path(entry["metrics"])
        metrics = json.loads(metrics_path.read_text())
        evaluation = metrics["configuration"]
        if evaluation["split"] != "val" or evaluation["perturbation"] != 0:
            raise ValueError("Selection requires unperturbed validation metrics")
        if not metrics["backend"].startswith("native MuJoCo"):
            raise ValueError("Selection requires native MuJoCo metrics")
        conditions = {
            name: metrics[name] for name in ["backend", "scene_sha256", "references"]
        }
        conditions["seed"] = evaluation["seed"]
        identities = [
            f"{episode['task']}:{episode['motion_seed']}"
            for episode in metrics["episodes"]
        ]
        if (
            len(identities) != configuration["expected_episodes"]
            or len(set(identities)) != len(identities)
            or set(identities) != set(metrics["references"])
        ):
            raise ValueError("Missing, duplicated, or mismatched validation episodes")
        conditions["horizons"] = {
            identity: episode["expected_frames"]
            for identity, episode in zip(identities, metrics["episodes"], strict=True)
        }
        if reference_conditions is None:
            reference_conditions = conditions
        elif reference_conditions != conditions:
            raise ValueError("Candidate evaluation conditions differ")
        policy = Path(evaluation["policy"])
        contract_path = Path(evaluation["contract"])
        contract = json.loads(contract_path.read_text())
        if sha256(policy) != metrics["policy_sha256"]:
            raise ValueError("Policy changed after evaluation")
        # Original screening exports predate schema 2 and lack an embedded policy
        # hash. Their actual policy bytes are still checked against evaluation.
        contract_policy_hash = contract.get("policy_sha256")
        if contract.get("schema_version") == 2 and contract_policy_hash is None:
            raise ValueError("Schema 2 export is missing its policy hash")
        if (
            contract_policy_hash is not None
            and contract_policy_hash != metrics["policy_sha256"]
        ):
            raise ValueError("Export contract does not match evaluated policy")
        if sha256(evaluation["scene"]) != metrics["scene_sha256"]:
            raise ValueError("Scene changed after evaluation")
        selected_metrics = metrics["macro"]
        for name in ["tracking_success", "complete", "root_m_failure_penalized"]:
            if not math.isfinite(selected_metrics[name]):
                raise ValueError("Nonfinite selection metric")
        ranking.append(
            {
                "iteration": entry["iteration"],
                "metrics": selected_metrics,
                "metrics_path": str(metrics_path),
                "metrics_sha256": sha256(metrics_path),
                "policy": str(policy),
                "policy_sha256": metrics["policy_sha256"],
                "contract": str(contract_path),
                "contract_sha256": sha256(contract_path),
                "checkpoint_sha256": contract["checkpoint_sha256"],
                "legacy_contract_without_policy_hash": contract_policy_hash is None,
            }
        )
    ranking.sort(
        key=lambda entry: (
            -entry["metrics"]["tracking_success"],
            -entry["metrics"]["complete"],
            entry["metrics"]["root_m_failure_penalized"],
            entry["iteration"],
        )
    )
    return {
        "experiment": configuration["experiment"],
        "test_used": False,
        "rule": "strict success, completion, lower failure-penalized torso-anchor error, earlier iteration",
        "configuration": configuration,
        "evaluation_conditions": reference_conditions,
        "selected": ranking[0],
        "ranking": ranking,
    }


@hydra.main(
    config_path="../config/tracker_rl", config_name="select", version_base="1.3"
)
def main(configuration: DictConfig):
    result = select_checkpoint(OmegaConf.to_container(configuration, resolve=True))
    output = Path(configuration.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as destination:
        json.dump(result, destination, indent=2)
    print(json.dumps(result["selected"], indent=2))


if __name__ == "__main__":
    main()
