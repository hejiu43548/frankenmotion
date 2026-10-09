"""Frozen batched SONIC release networks loaded directly from official ONNX.

The SHA allowlist pins graph semantics as well as weights. No historical tracker
checkpoint, residual head, normalizer, expert actions, or rollouts are loaded.
ONNX is needed only when importing weights; exported modules need PyTorch alone.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch


RELEASE_SHA256 = {
    "encoder": "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3",
    "decoder": "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed",
}


def read_release_graph(directory, kind):
    import onnx

    path = Path(directory) / f"model_{kind}.onnx"
    if hashlib.sha256(path.read_bytes()).hexdigest() != RELEASE_SHA256[kind]:
        raise ValueError(f"Unsupported SONIC {kind} graph: release checksum differs")
    graph = onnx.load(str(path)).graph
    tensors = {
        tensor.name: np.array(onnx.numpy_helper.to_array(tensor), copy=True)
        for tensor in graph.initializer
    }
    for node in graph.node:
        if node.op_type == "Constant":
            attributes = {
                attribute.name: onnx.helper.get_attribute_value(attribute)
                for attribute in node.attribute
            }
            tensors[node.output[0]] = np.array(
                onnx.numpy_helper.to_array(attributes["value"]), copy=True
            )
    return graph, tensors


def frozen_linear(weight, bias):
    layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(weight))
        layer.bias.copy_(torch.from_numpy(bias))
    layer.requires_grad_(False)
    return layer


class SonicReleaseEncoder(torch.nn.Module):
    """Official mode 0 (G1) or mode 2 (SMPL), accepting the 1762D release ABI."""

    def __init__(self, directory, mode=0):
        super().__init__()
        if mode not in [0, 2]:
            raise ValueError("Only SONIC release modes 0 and 2 are implemented")
        self.mode = mode
        _, tensors = read_release_graph(directory, "encoder")
        name = "g1" if mode == 0 else "smpl"
        layers = []
        for index in [0, 2, 4, 6, 8]:
            prefix = f"module.encoders.{name}.module.{index}"
            layers.append(
                frozen_linear(tensors[prefix + ".weight"], tensors[prefix + ".bias"])
            )
            if index != 8:
                layers.append(torch.nn.SiLU())
        self.network = torch.nn.Sequential(*layers)
        prefix = "/quantizer" if mode == 0 else "/quantizer_2"
        for name, index in [("shift", 1), ("bound", 2), ("offset", 3), ("divisor", 4)]:
            self.register_buffer(
                name, torch.from_numpy(tensors[f"{prefix}/Constant_{index}_output_0"])
            )

    def forward(self, observation):
        if self.mode == 0:
            features = torch.cat(
                [
                    observation[:, 4:584].reshape(-1, 10, 58),
                    observation[:, 601:661].reshape(-1, 10, 6),
                ],
                dim=-1,
            ).flatten(1)
        else:
            features = torch.cat(
                [
                    observation[:, 922:1642].reshape(-1, 10, 72),
                    observation[:, 1642:1702].reshape(-1, 10, 6),
                    observation[:, 1702:1762].reshape(-1, 10, 6),
                ],
                dim=-1,
            ).flatten(1)
        latent = self.network(features).reshape(-1, 2, 32)
        bounded = torch.tanh(latent + self.shift) * self.bound - self.offset
        quantized = torch.round(bounded) / self.divisor
        return quantized.reshape(-1, 64)


class SonicReleaseDecoder(torch.nn.Module):
    """Shared frozen 994D token/history to 29D action decoder, arbitrary batch."""

    def __init__(self, directory):
        super().__init__()
        graph, tensors = read_release_graph(directory, "decoder")
        weights = [node.input[1] for node in graph.node if node.op_type == "MatMul"]
        layers = []
        for index, weight_name in enumerate(weights):
            bias_name = f"module.decoders.g1_dyn.module.{2 * index}.bias"
            layers.append(frozen_linear(tensors[weight_name].T, tensors[bias_name]))
            if index != len(weights) - 1:
                layers.append(torch.nn.SiLU())
        self.network = torch.nn.Sequential(*layers)

    def forward(self, observation):
        return self.network(observation)


class SonicModeZeroPolicy(torch.nn.Module):
    """Standalone official policy with causal history and shared native ABI.

    The middle 495 features of the historical 2350D ABI are ignored. They were
    used by the old residual head; no such head or learned normalization exists
    here. Conversion constants are joint orders and physical action scales.
    """

    def __init__(self, directory, contract):
        super().__init__()
        self.encoder = SonicReleaseEncoder(directory, mode=0)
        self.decoder = SonicReleaseDecoder(directory)
        mujoco_to_isaac = np.array(
            [
                0,
                6,
                12,
                1,
                7,
                13,
                2,
                8,
                14,
                3,
                9,
                15,
                22,
                4,
                10,
                16,
                23,
                5,
                11,
                17,
                24,
                18,
                25,
                19,
                26,
                20,
                27,
                21,
                28,
            ]
        )
        self.register_buffer("il_to_mj", torch.from_numpy(np.argsort(mujoco_to_isaac)))
        self.register_buffer("history", torch.zeros(10, 93))
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("q0", torch.tensor(contract["sonic_default_positions"]))
        self.register_buffer(
            "sonic_scale", torch.tensor(contract["sonic_action_scale"])
        )
        self.register_buffer("offset", torch.tensor(contract["action_offset"]))
        self.register_buffer("scale", torch.tensor(contract["action_scale"]))
        self.register_buffer(
            "action_order",
            torch.tensor(
                [
                    contract["sonic_source_joint_names"].index(name)
                    for name in contract["action_target_names"]
                ]
            ),
        )

    def forward(self, observation):
        state = observation[:, 2257:2350]
        if not bool(self.initialized):
            self.history.copy_(state.expand(10, 93))
            self.initialized.fill_(True)
        else:
            self.history.copy_(torch.cat([self.history[1:], state], dim=0))
        token = self.encoder(observation[:, :1762])
        decoder_input = torch.cat(
            [
                token,
                self.history[:, :3].reshape(1, -1),
                self.history[:, 3:32].reshape(1, -1),
                self.history[:, 32:61].reshape(1, -1),
                self.history[:, 61:90].reshape(1, -1),
                self.history[:, 90:93].reshape(1, -1),
            ],
            dim=1,
        )
        raw = self.decoder(decoder_input).clamp(-20, 20)
        target = self.q0 + raw[:, self.il_to_mj] * self.sonic_scale
        return (target[:, self.action_order] - self.offset) / self.scale

    @torch.jit.export
    def reset(self):
        self.initialized.fill_(False)
        self.history.zero_()


class SonicResidualPolicy(torch.nn.Module):
    """One deployable module containing frozen SONIC and the online PPO residual."""

    def __init__(self, base, residual, clip_actions=None):
        super().__init__()
        self.base = base
        self.residual = residual
        self.clip_limit = float("inf") if clip_actions is None else float(clip_actions)
        self.register_buffer("il_to_mj", base.il_to_mj.clone())

    def forward(self, observation):
        nominal = self.base(observation)
        token = self.base.encoder(observation[:, :1762])
        # Native legacy ABI has five previews, while this residual uses the
        # first three (5,10,20). Remaining previews are deliberately ignored.
        features = torch.cat([observation[:, 1762:2123], nominal, token], dim=1)
        return (nominal + self.residual(features)).clamp(
            -self.clip_limit, self.clip_limit
        )

    @torch.jit.export
    def reset(self):
        self.base.reset()
