"""Verify official ONNX parity before exporting clean batched SONIC networks."""

import hashlib
import json
from pathlib import Path
import sys

import hydra
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
import onnxruntime
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_motion.rl.sonic import RELEASE_SHA256
from shared_motion.rl.sonic import SonicModeZeroPolicy
from shared_motion.rl.sonic import SonicReleaseDecoder
from shared_motion.rl.sonic import SonicReleaseEncoder


@hydra.main(
    config_path="../config/tracker_rl", config_name="sonic_import", version_base="1.3"
)
def main(configuration: DictConfig):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(configuration.output)
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(configuration, output / "config.yaml")
    random = np.random.default_rng(configuration.seed)
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 2
    results = {}
    for kind, mode in [("encoder", 0), ("encoder", 2), ("decoder", 0)]:
        module = (
            SonicReleaseEncoder(configuration.directory, mode)
            if kind == "encoder"
            else SonicReleaseDecoder(configuration.directory)
        ).eval()
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise AssertionError("Official base parameters must be frozen")
        session = onnxruntime.InferenceSession(
            str(Path(configuration.directory) / f"model_{kind}.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )
        samples = random.normal(
            0, 0.4, (configuration.samples, 1762 if kind == "encoder" else 994)
        ).astype(np.float32)
        if kind == "encoder":
            samples[:, :4] = 0
            samples[:, 0] = mode
        expected = np.concatenate(
            [
                session.run(None, {"obs_dict": sample[None]})[0].reshape(1, -1)
                for sample in samples
            ]
        )
        device_errors = {}
        for device in ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]:
            module.to(device)
            with torch.no_grad():
                actual = module(torch.from_numpy(samples).to(device)).cpu().numpy()
            error = float(np.max(np.abs(actual - expected)))
            device_errors[device] = error
            if error >= 2e-4:
                raise AssertionError(
                    f"{kind} mode {mode} {device} parity failed: {error}"
                )
        name = f"{kind}_mode{mode}"
        torch.jit.script(module.cpu()).save(str(output / f"{name}.pt"))
        results[name] = {"samples": configuration.samples, "max_error": device_errors}
        print(name, device_errors, flush=True)
    contract = json.loads(Path(configuration.native_contract).read_text())
    policy = SonicModeZeroPolicy(configuration.directory, contract).eval()
    torch.jit.script(policy).save(str(output / "policy.pt"))
    (output / "policy.json").write_text(
        json.dumps(
            {
                "preview_offsets": [5, 10, 20, 35, 50],
                "input_dimensions": 2350,
                "stateful": True,
                "reset_required": True,
                "source": "official SONIC release ONNX, no residual head",
            },
            indent=2,
        )
    )
    report = {
        "source_sha256": RELEASE_SHA256,
        "policy_sha256": hashlib.sha256(
            (output / "policy.pt").read_bytes()
        ).hexdigest(),
        "parity": results,
        "limitations": "Random ABI parity is not a trajectory or deployment quality test. Native dynamics differ from official deployment gains.",
        "source": "https://huggingface.co/nvidia/GEAR-SONIC/tree/6733128a3d8a523b1418b06bca3cdf61c8b0987f",
        "provenance_evidence": "Local HF snapshot symlinks resolve to matching content-addressed SHA256 blobs; public network revalidation unavailable at import time.",
        "historical_tracker_weights_loaded": False,
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
