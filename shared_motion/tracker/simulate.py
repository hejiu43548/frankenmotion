"""Run a prepared 50 Hz G1 reference using one exported tracker and fixed dynamics."""

import argparse, json
from pathlib import Path
import numpy as np, torch, mujoco
from scipy.spatial.transform import Rotation
from .runtime import Tracker


def run(actor, scene, contract, reference, initial_state, output):
    torch.set_num_threads(1)
    tracker = Tracker(actor)
    net = tracker.actor
    c = json.loads(Path(contract).read_text())
    c["preview_offsets"] = json.loads(Path(actor).with_suffix(".json").read_text())[
        "preview_offsets"
    ]
    m = mujoco.MjModel.from_binary_path(str(scene))
    d = mujoco.MjData(m)
    ref = dict(np.load(reference))
    q = np.load(initial_state)["qpos"]
    d.qpos[:] = q[0] if q.ndim == 2 else q
    mujoco.mj_forward(m, d)
    qa = [m.joint("robot/" + n).qposadr[0] for n in c["joint_names"]]
    va = [m.joint("robot/" + n).dofadr[0] for n in c["joint_names"]]
    aids = [
        int(np.where(m.actuator_trnid[:, 0] == m.joint("robot/" + n).id)[0][0])
        for n in c["action_target_names"]
    ]
    anchor = m.body(c["anchor_body_name"]).id
    pelvis = m.body("robot/pelvis").id
    names = c["sonic_source_joint_names"]
    order = [c["joint_names"].index(n) for n in names]
    action_order = [names.index(n) for n in c["action_target_names"]]
    mj_to_il = np.argsort(net.il_to_mj.numpy())
    last = np.zeros(29)
    n = len(ref["joint_pos"])
    states = [d.qpos.copy()]
    actions = []
    failure = None

    def rot(q):
        return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]])

    def sensor(name):
        s = m.sensor(name)
        return d.sensordata[s.adr[0] : s.adr[0] + s.dim[0]].copy()

    for i in range(n - 1):
        inv = rot(d.xquat[anchor]).inv()

        def relative(k):
            return inv.apply(
                ref["body_pos_w"][k, c["reference_anchor_index"]] - d.xpos[anchor]
            ), (
                inv * rot(ref["body_quat_w"][k, c["reference_anchor_index"]])
            ).as_matrix()[
                :, :2
            ].reshape(
                -1
            )

        pos, r = relative(i)
        obs = [
            ref["joint_pos"][i],
            ref["joint_vel"][i],
            pos,
            r,
            sensor(c["linear_velocity_sensor"]),
            sensor(c["angular_velocity_sensor"]),
            d.qpos[qa] - np.array(c["default_joint_pos"]),
            d.qvel[va],
            last,
        ]
        for offset in c["preview_offsets"]:
            k = min(i + offset, n - 1)
            pos, r = relative(k)
            obs.extend([ref["joint_pos"][k], ref["joint_vel"][k], pos, r])
        ids = np.minimum(i + np.arange(10) * 5, n - 1)
        enc = np.zeros(1762, np.float32)
        enc[4:294] = ref["joint_pos"][ids][:, order][:, mj_to_il].ravel()
        enc[294:584] = ref["joint_vel"][ids][:, order][:, mj_to_il].ravel()
        robot = rot(d.qpos[3:7])
        rr = Rotation.from_quat(ref["body_quat_w"][ids, 0][:, [1, 2, 3, 0]])
        enc[601:661] = (robot.inv() * rr).as_matrix()[:, :, :2].ravel()
        raw = (
            np.zeros(29)
            if i == 0
            else (
                (
                    (last * np.array(c["action_scale"]) + np.array(c["action_offset"]))[
                        np.argsort(action_order)
                    ]
                    - np.array(c["sonic_default_positions"])
                )
                / np.array(c["sonic_action_scale"])
            )[mj_to_il]
        )
        state = np.r_[
            d.qvel[3:6],
            (d.qpos[qa][order] - np.array(c["sonic_default_positions"]))[mj_to_il],
            d.qvel[va][order][mj_to_il],
            raw,
            robot.inv().apply([0, 0, -1]),
        ]
        x = np.r_[enc, np.concatenate(obs), state].astype(np.float32)
        last = tracker(x[None])[0].numpy()
        actions.append(last.copy())
        d.ctrl[aids] = last * np.array(c["action_scale"]) + np.array(c["action_offset"])
        for _ in range(round(c["control_timestep"] / m.opt.timestep)):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        states.append(d.qpos.copy())
        tilt = np.arccos(np.clip(d.xmat[pelvis].reshape(3, 3)[2, 2], -1, 1))
        if (
            not np.isfinite(d.qpos).all()
            or d.xpos[pelvis, 2] < 0.35
            or tilt > np.pi / 3
        ):
            failure = float(d.time)
            break
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, qpos=states, actions=actions, fps=50.0)
    output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                complete=failure is None,
                termination_time=failure,
                frames=len(states),
                tracker=str(actor),
                reference=str(reference),
                state_reset_after_initialization=False,
            ),
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser()
    for name in ["actor", "scene", "contract", "reference", "initial-state", "output"]:
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    run(a.actor, a.scene, a.contract, a.reference, a.initial_state, a.output)


if __name__ == "__main__":
    main()
