"""Selected architecture; reconstructable from its one standalone export."""

import copy
import torch


class SharedTracker(torch.nn.Module):
    def __init__(self, export):
        super().__init__()
        self.encoder = copy.deepcopy(export.encoder)
        self.decoder = copy.deepcopy(export.decoder)
        self.head = copy.deepcopy(export.head)
        for name in [
            "mean",
            "std",
            "history",
            "initialized",
            "il_to_mj",
            "action_order",
            "q0",
            "sonic_scale",
            "offset",
            "scale",
        ]:
            self.register_buffer(name, getattr(export, name).detach().clone())
        self.reset()

    def correct(self, correction_features):
        return correction_features[:, 495:524] + 3.0 * torch.tanh(
            self.head(
                torch.clamp((correction_features - self.mean) / self.std, -10, 10)
            )
        )

    def forward(self, observation):
        state = observation[:, 2257:2350]
        if not bool(self.initialized):
            self.history.copy_(state.expand(10, 93))
            self.initialized.fill_(True)
        else:
            self.history.copy_(torch.cat([self.history[1:], state], dim=0))
        token = self.encoder(observation[:, :1762]).reshape(1, 64)
        state_history = self.history
        decoder_input = torch.cat(
            [
                token,
                state_history[:, :3].reshape(1, -1),
                state_history[:, 3:32].reshape(1, -1),
                state_history[:, 32:61].reshape(1, -1),
                state_history[:, 61:90].reshape(1, -1),
                state_history[:, 90:93].reshape(1, -1),
            ],
            dim=1,
        )
        decoder_action = torch.clamp(
            self.decoder(decoder_input).reshape(1, 29), -20, 20
        )
        target = self.q0 + decoder_action[:, self.il_to_mj] * self.sonic_scale
        action = (target[:, self.action_order] - self.offset) / self.scale
        return self.correct(
            torch.cat([observation[:, 1762:2257], action, token], dim=1)
        )

    @torch.jit.export
    def reset(self):
        self.initialized.fill_(False)
        self.history.zero_()
