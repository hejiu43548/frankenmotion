"""Selected architecture; reconstructable from its one standalone export."""

import copy, torch


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

    def correct(self, x):
        return x[:, 495:524] + 3.0 * torch.tanh(
            self.head(torch.clamp((x - self.mean) / self.std, -10, 10))
        )

    def forward(self, x):
        state = x[:, 2257:2350]
        if not bool(self.initialized):
            self.history.copy_(state.expand(10, 93))
            self.initialized.fill_(True)
        else:
            self.history.copy_(torch.cat([self.history[1:], state], dim=0))
        token = self.encoder(x[:, :1762]).reshape(1, 64)
        h = self.history
        dec = torch.cat(
            [
                token,
                h[:, :3].reshape(1, -1),
                h[:, 3:32].reshape(1, -1),
                h[:, 32:61].reshape(1, -1),
                h[:, 61:90].reshape(1, -1),
                h[:, 90:93].reshape(1, -1),
            ],
            dim=1,
        )
        raw = torch.clamp(self.decoder(dec).reshape(1, 29), -20, 20)
        target = self.q0 + raw[:, self.il_to_mj] * self.sonic_scale
        action = (target[:, self.action_order] - self.offset) / self.scale
        return self.correct(torch.cat([x[:, 1762:2257], action, token], dim=1))

    @torch.jit.export
    def reset(self):
        self.initialized.fill_(False)
        self.history.zero_()
