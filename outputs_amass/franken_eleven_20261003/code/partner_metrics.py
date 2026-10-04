"""NumPy parity measurements for frozen 20 Hz PHC24 and G1 joint centres.

The task quantity follows ``physics_refine.experiments.tasks``. Distances in
G1 event proxies use the frozen shoulder-to-ankle height ratio; angles do not.
This module measures saved trajectories and does not edit reference motions.
"""

from __future__ import annotations

import numpy as np


TASK_FRAMES = {"reach": 60, "strike": 60, "wave": 120, "turn": 120,
               "sidestep": 120, "kick": 60, "jump": 60, "walk": 120}


def _index(names, key):
    return names.index(key)


def _pitch(y):
    return np.arctan2(np.sin(y), np.cos(y))


def _hip_yaw(p, names):
    side = p[:, _index(names, "L_Hip"), :2] - p[:, _index(names, "R_Hip"), :2]
    return np.arctan2(side[:, 1], side[:, 0])


def _event(p, start, end):
    return p[round(start * 20):round(end * 20) + 1]


def measure(task: str, positions: np.ndarray, names: list[str] | tuple[str, ...],
            *, distance_ratio: float = 1.) -> dict:
    p = np.asarray(positions, dtype=np.float64)
    names = list(names)
    if task not in TASK_FRAMES or p.shape != (TASK_FRAMES[task], len(names), 3):
        raise ValueError("Wrong task, frame count, or joint shape")
    if len(set(names)) != len(names) or not np.isfinite(p).all():
        raise ValueError("Invalid task trajectory")
    if not np.isfinite(distance_ratio) or distance_ratio <= 0:
        raise ValueError("Invalid morphology scale")
    at = lambda key: p[:, _index(names, key)]
    pelvis = at("Pelvis")
    span = (len(p) - 1) / 20
    ratio = distance_ratio

    if task == "reach":
        signal = (at("R_Wrist") - pelvis)[:, 0]
        rise = float(np.max(signal[1:] - np.minimum.accumulate(signal[:-1])))
        return {"quantity": float(np.quantile(signal, .95)), "ordered_rise_m": rise,
                "event_pass": rise >= .08 * ratio}

    if task == "strike":
        wrist = at("R_Wrist")
        smooth = .25 * wrist[:-2] + .5 * wrist[1:-1] + .25 * wrist[2:]
        speed = np.linalg.norm(smooth[2:] - smooth[:-2], axis=-1) * 10
        quantity = float(speed[round(.8 * 20)-2:round(1.6 * 20)-1].max())
        rel = (wrist - pelvis)[:, 0]
        event = rel[round(.8 * 20):round(1.6 * 20)+1]
        excursion = float(event.max() - rel[0])
        retraction = float(event.max() - rel[-1])
        return {"quantity": quantity, "forward_excursion_m": excursion,
                "retraction_m": retraction,
                "event_pass": excursion >= .08 * ratio and retraction >= .08 * ratio}

    if task == "wave":
        x = _event(p, .8, 5.)
        left = x[:, _index(names, "L_Shoulder")]
        right = x[:, _index(names, "R_Shoulder")]
        lateral = left - right
        lateral /= np.maximum(np.linalg.norm(lateral, axis=-1, keepdims=True), 1e-8)
        center = (left + right) / 2
        signal = np.sum((x[:, _index(names, "R_Wrist")] - center) * lateral, axis=-1)
        low, high = np.quantile(signal, [.05, .95])
        state, traversals = None, 0
        for value in signal:
            side = -1 if value <= low + .2 * (high-low) else (1 if value >= high - .2 * (high-low) else None)
            if side is not None:
                traversals += int(state is not None and side != state)
                state = side
        return {"quantity": float((high-low)/2), "full_cycles": traversals // 2,
                "event_pass": traversals >= 4 and high-low >= .04 * ratio}

    if task == "turn":
        yaw = _hip_yaw(p, names)
        right_angle = float(-_pitch(yaw[-1] - yaw[0]))
        increments = _pitch(np.diff(yaw))
        cumulative = np.r_[0., np.cumsum(increments)]
        right_peak = float(-cumulative.min())
        return {"quantity": right_angle, "right_peak_rad": right_peak,
                "event_pass": right_angle >= .3 and right_peak >= .3}

    if task == "sidestep":
        lateral = -(pelvis[:, 1] - pelvis[0, 1])
        forward = pelvis[:, 0] - pelvis[0, 0]
        yaw = _hip_yaw(p, names)
        heading = float(np.max(np.abs(_pitch(yaw-yaw[0]))))
        quantity = float(lateral.max())
        return {"quantity": quantity, "forward_drift_m": float(np.max(np.abs(forward))),
                "heading_change_rad": heading,
                "event_pass": quantity >= .3 * ratio and heading <= .55}

    if task == "kick":
        ankle = at("R_Ankle")
        signal = (ankle - pelvis)[:, 0]
        event = signal[round(.5 * 20):round(2.5 * 20)+1]
        quantity = float(event.max() - signal[0])
        lift = float(ankle[:, 2].max() - ankle[0, 2])
        retraction = float(event.max() - signal[-1])
        return {"quantity": quantity, "ankle_lift_m": lift, "retraction_m": retraction,
                "event_pass": quantity >= .2 * ratio and lift >= .12 * ratio and
                              retraction >= .12 * ratio}

    if task == "jump":
        z = pelvis[:, 2]
        quantity = float(z.max() - z[0])
        clearance = np.minimum(at("L_Ankle")[:, 2] - at("L_Ankle")[0, 2],
                               at("R_Ankle")[:, 2] - at("R_Ankle")[0, 2])
        peak_index = int(np.argmax(clearance))
        peak = float(clearance[peak_index])
        landed = bool(np.any(clearance[peak_index:] <= .08 * ratio))
        return {"quantity": quantity, "both_foot_clearance_m": peak, "landed": landed,
                "event_pass": quantity >= .18 * ratio and peak >= .12 * ratio and landed}

    # walk: original task is XY path length divided by the 5.95 s task span.
    xy = pelvis[:, :2]
    length = float(np.linalg.norm(np.diff(xy, axis=0), axis=-1).sum())
    progress = float(xy[-1, 0] - xy[0, 0])
    lateral = float(np.max(np.abs(xy[:, 1] - xy[0, 1])))
    forward_ratio = progress / max(length, 1e-8)
    lateral_ratio = lateral / max(progress, 1e-8)
    return {"quantity": length / span, "forward_ratio": forward_ratio,
            "lateral_ratio": lateral_ratio,
            "event_pass": length > 1e-8 and progress > 1e-8 and
                          forward_ratio >= .8 and lateral_ratio <= .2}
