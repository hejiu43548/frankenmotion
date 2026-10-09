import numpy as np, torch
from src.tools.geometry import (
    rotation_6d_to_matrix,
    matrix_to_euler_angles,
    axis_angle_rotation,
    matrix_to_axis_angle,
)
from .schema import TASKS


class FK:
    def __init__(self, skeleton, device="cpu"):
        z = np.load(skeleton)
        self.J = torch.tensor(z["J"], device=device, dtype=torch.float32)
        self.parents = z["parents"][:22].tolist()
        self.height = float(z["height"])

    def __call__(self, raw, canonical=True, return_pose=False):
        # Same SMPL-RIFKE root reconstruction as upstream; no redundant xyz prediction.
        b, t, _ = raw.shape
        mat = rotation_6d_to_matrix(raw[..., 4:136].reshape(b, t, 22, 6))
        e = matrix_to_euler_angles(mat[:, :, 0], "ZYX")
        yaw = torch.cat(
            [torch.zeros_like(raw[:, :1, 3]), torch.cumsum(raw[:, :-1, 3], 1)], 1
        )
        rz = axis_angle_rotation("Z", yaw)
        r0 = (
            rz
            @ axis_angle_rotation("Y", e[..., 1])
            @ axis_angle_rotation("X", e[..., 2])
        )
        mats = torch.cat([r0[:, :, None], mat[:, :, 1:]], 2)
        v = (rz[..., :2, :2] @ raw[..., 1:3, None]).squeeze(-1)
        xy = torch.cat([torch.zeros_like(v[:, :1]), torch.cumsum(v[:, :-1], 1)], 1)
        trans = torch.cat([xy, raw[..., :1]], -1)
        g = []
        p = []
        for j in range(22):
            if j == 0:
                g.append(mats[:, :, 0])
                p.append(self.J[0].expand(b, t, 3) + trans)
            else:
                par = self.parents[j]
                g.append(g[par] @ mats[:, :, j])
                p.append(
                    p[par] + (g[par] @ (self.J[j] - self.J[par])[:, None]).squeeze(-1)
                )
        p.extend(
            [
                p[20] + (g[20] @ (self.J[22] - self.J[20])[:, None]).squeeze(-1),
                p[21] + (g[21] @ (self.J[37] - self.J[21])[:, None]).squeeze(-1),
            ]
        )
        p = torch.stack(p, 2)
        if canonical:
            side = p[:, 0, 1, :2] - p[:, 0, 2, :2]
            angle = torch.atan2(side[:, 1], side[:, 0]) - np.pi / 2
            rot = axis_angle_rotation("Z", -angle)
            p = (rot[:, None, None] @ p[..., None]).squeeze(-1)
        if return_pose:
            return p, matrix_to_axis_angle(mats).reshape(b, t, 66), trans
        return p


def quantity(p, task, scale=1.0, net_walk=False):
    # Frozen screenshot quantities, including legacy path-speed walk; net-speed is separately reported.
    task = TASKS[task] if isinstance(task, int) else task
    root = p[:, :, 0]
    span = (p.shape[1] - 1) / 20
    if task == "raise_hand":
        q = torch.quantile((p[:, :, 21] - root)[..., 2], 0.95, dim=1)
    elif task == "reach":
        q = torch.quantile((p[:, :, 21] - root)[..., 0], 0.95, dim=1)
    elif task == "strike":
        w = p[:, :, 21]
        s = 0.25 * w[:, :-2] + 0.5 * w[:, 1:-1] + 0.25 * w[:, 2:]
        v = (s[:, 2:] - s[:, :-2]).norm(dim=-1) * 10
        q = v[:, 14:32].amax(1)
    elif task == "wave":
        a = p[:, 16:101]
        lat = a[:, :, 16] - a[:, :, 17]
        lat = lat / lat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        s = ((a[:, :, 21] - (a[:, :, 16] + a[:, :, 17]) / 2) * lat).sum(-1)
        q = (torch.quantile(s, 0.95, dim=1) - torch.quantile(s, 0.05, dim=1)) / 2
    elif task == "turn":
        side = p[:, :, 1, :2] - p[:, :, 2, :2]
        yaw = torch.atan2(side[..., 1], side[..., 0])
        diff = yaw[:, -1] - yaw[:, 0]
        return -torch.atan2(torch.sin(diff), torch.cos(diff))
    elif task == "sidestep":
        q = -(root[:, :, 1] - root[:, :1, 1]).amin(1)
    elif task == "back_walk":
        q = -(root[:, -1, 0] - root[:, 0, 0]) / span
    elif task == "kick":
        s = (p[:, :, 8] - root)[..., 0]
        q = s[:, 10:51].amax(1) - s[:, 0]
    elif task == "jump":
        q = root[:, :, 2].amax(1) - root[:, 0, 2]
    elif task == "lean":
        v = (p[:, :, 16] + p[:, :, 17]) / 2 - root
        pitch = torch.atan2(v[..., 0], v[..., 2])
        return torch.quantile(pitch[:, 20:59], 0.9, dim=1)
    elif task == "walk":
        q = (
            (root[:, -1, 0] - root[:, 0, 0]) / span
            if net_walk
            else (root[:, 1:, :2] - root[:, :-1, :2]).norm(dim=-1).sum(1) / span
        )
    else:
        raise ValueError(task)
    return q * scale
