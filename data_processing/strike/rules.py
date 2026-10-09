"""Semantic and physical admission for the user's right straight-punch scope."""

import re
import numpy as np
from scipy.signal import find_peaks

POLICY = dict(
    revision="right_forward_straight_punch_v3",
    fps=20,
    frames=60,
    peak_frame=28,
    crop_frame_options=[60, 50, 40],
    peak_frame_options=[28, 24, 20, 16, 32],
    minimum_forward_speed_m_s=0.8,
    minimum_forward_extension_m=0.16,
    minimum_forward_reach_m=0.28,
    minimum_extended_elbow_deg=125,
    maximum_root_excursion_m=0.40,
    maximum_foot_excursion_m=0.35,
    maximum_foot_height_range_m=0.18,
    maximum_root_height_range_m=0.20,
    maximum_torso_tilt_rad=0.65,
    minimum_forward_speed_fraction=0.60,
    minimum_right_left_speed_ratio=1.1,
    context_left_punch_min_speed_m_s=1.0,
    context_left_punch_min_extension_m=0.16,
    context_left_punch_min_elbow_deg=135,
    semantics="Right-arm forward straight punch/jab only, user confirmed. Preserve original motion; no time warping, padding, mirroring or synthetic prompt.",
    metric="Existing right-wrist smoothed world-speed maximum, indices14:32 (original frames16..33). Outbound event peak placed at frame28; check metric peak is outbound.",
    source_priority="HDM05 scene03-02 with BABEL punch frame boundaries first, then other BABEL act_cat=punch frames. Official native cut table unavailable; do not claim official cut boundaries.",
)
ALLOWED_CATS = {
    "punch",
    "hand movements",
    "arm movements",
    "forward movement",
    "martial art",
    "sports move",
    "exercise/training",
}
CONTEXT_CATS = ALLOWED_CATS | {
    "transition",
    "stand",
    "body part movement",
    "move body part",
    "look",
    "head movements",
    "raising body part",
    "lowering body part",
}
BAD_STYLE = re.compile(
    r"\b(?:hook|uppercut|upper cut|lob|wide|side|sideways|backward\w*|circular|circle|down|low|top|up|squat|crouch|kick|throw)\b|to (?:the )?(?:left|right)",
    re.I,
)
LEFT_ONLY = re.compile(r"\bleft\b", re.I)


def semantic_reasons(label):
    reasons = []
    if "punch" not in (label.get("act_cat") or []):
        reasons.append("no_punch_act_cat")
    if set(label.get("act_cat") or []) - ALLOWED_CATS:
        reasons.append("conflicting_action_category")
    text = label.get("proc_label") or label.get("raw_label") or ""
    if BAD_STYLE.search(text):
        reasons.append("not_forward_straight_punch_style")
    if LEFT_ONLY.search(text):
        reasons.append("left_hand_or_left_direction")
    if "both" in text:
        reasons.append("bilateral_punch_label")
    return reasons


def signals(j):
    side = j[:, 1, :2] - j[:, 2, :2]
    side /= np.maximum(np.linalg.norm(side, axis=-1, keepdims=True), 1e-8)
    forward = np.c_[side[:, 1], -side[:, 0], np.zeros(len(j))]
    rel = j[:, 21] - j[:, 17]
    smooth = lambda p: 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]
    rv = (smooth(rel)[2:] - smooth(rel)[:-2]) * 10
    world = (smooth(j[:, 21])[2:] - smooth(j[:, 21])[:-2]) * 10
    left = (smooth(j[:, 20] - j[:, 16])[2:] - smooth(j[:, 20] - j[:, 16])[:-2]) * 10
    fs = (rv * forward[2:-2]).sum(-1)
    upper = j[:, 17] - j[:, 19]
    lower = j[:, 21] - j[:, 19]
    cosine = (upper * lower).sum(-1) / np.maximum(
        np.linalg.norm(upper, axis=-1) * np.linalg.norm(lower, axis=-1), 1e-8
    )
    elbow = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
    return dict(
        forward_speed=fs,
        relative_speed=np.linalg.norm(rv, axis=-1),
        world_speed=np.linalg.norm(world, axis=-1),
        left_speed=np.linalg.norm(left, axis=-1),
        forward_reach=(rel * forward).sum(-1),
        elbow=elbow,
    )


def peaks(j, start, stop):
    signal = signals(j)
    indices = (
        find_peaks(
            signal["forward_speed"],
            height=POLICY["minimum_forward_speed_m_s"],
            prominence=0.5,
            distance=10,
        )[0]
        + 2
    )
    return [int(p) for p in indices if start <= p < stop]


def left_punch_events(j):
    """Evaluate left-arm anatomy without mirroring or changing the source motion."""
    swapped = j.copy()
    swapped[:, [17, 19, 21]] = j[:, [16, 18, 20]]
    swapped[:, [16, 18, 20]] = j[:, [17, 19, 21]]
    s = signals(swapped)
    events = []
    for i in find_peaks(
        s["forward_speed"],
        height=POLICY["context_left_punch_min_speed_m_s"],
        prominence=0.5,
        distance=10,
    )[0]:
        p = int(i + 2)
        stop = min(len(j), p + 9)
        reach = s["forward_reach"]
        target = p + int(np.argmax(reach[p:stop]))
        extension = float(reach[p:stop].max() - reach[max(0, p - 10) : p + 1].min())
        if (
            extension > POLICY["context_left_punch_min_extension_m"]
            and s["elbow"][target] > POLICY["context_left_punch_min_elbow_deg"]
        ):
            events.append(
                dict(
                    frame=p,
                    speed_m_s=float(s["forward_speed"][i]),
                    extension_m=extension,
                    elbow_deg=float(s["elbow"][target]),
                )
            )
    return events


def admission(j, peak_frame=None):
    s = signals(j)
    p = POLICY["peak_frame"] if peak_frame is None else peak_frame
    pi = p - 2
    root = j[:, 0]
    feet = j[:, [7, 8, 10, 11]]
    reach = s["forward_reach"]
    future = slice(p, min(len(j), p + 9))
    past = slice(max(0, p - 12), p + 1)
    target = int(p + np.argmax(reach[future]))
    torso = (j[:, 16] + j[:, 17]) / 2 - root
    metric_peak = int(14 + np.argmax(s["world_speed"][14:32]))
    m = dict(
        forward_speed_m_s=float(s["forward_speed"][pi]),
        forward_speed_fraction=float(
            s["forward_speed"][pi] / max(s["relative_speed"][pi], 1e-8)
        ),
        forward_extension_m=float(reach[future].max() - reach[past].min()),
        extended_forward_reach_m=float(reach[target]),
        extended_elbow_deg=float(s["elbow"][target]),
        right_left_speed_ratio=float(
            s["relative_speed"][pi]
            / max(s["left_speed"][max(0, pi - 2) : pi + 3].max(), 0.05)
        ),
        root_excursion_m=float(
            np.linalg.norm(root[:, :2] - root[:1, :2], axis=-1).max()
        ),
        foot_excursion_m=float(
            np.linalg.norm(feet[:, :, :2] - feet[:1, :, :2], axis=-1).max()
        ),
        foot_height_range_m=float(np.ptp(feet[:, :, 2], axis=0).max()),
        root_height_range_m=float(np.ptp(root[:, 2])),
        max_torso_tilt_rad=float(
            np.arctan2(np.linalg.norm(torso[:, :2], axis=-1), torso[:, 2]).max()
        ),
        metric_peak_frame=metric_peak + 2,
        metric_peak_outbound_speed_m_s=float(s["forward_speed"][metric_peak]),
        selected_to_metric_peak_ratio=float(
            s["world_speed"][pi] / max(s["world_speed"][metric_peak], 1e-8)
        ),
        elbow_extension_deg=float(s["elbow"][target] - s["elbow"][past].min()),
    )
    reasons = []
    m["left_punches_in_context"] = left_punch_events(j)
    if m["left_punches_in_context"]:
        reasons.append("left_straight_punch_inside_training_crop")
    for metric, bound in [
        ("forward_speed_m_s", "minimum_forward_speed_m_s"),
        ("forward_speed_fraction", "minimum_forward_speed_fraction"),
        ("forward_extension_m", "minimum_forward_extension_m"),
        ("extended_forward_reach_m", "minimum_forward_reach_m"),
        ("extended_elbow_deg", "minimum_extended_elbow_deg"),
        ("right_left_speed_ratio", "minimum_right_left_speed_ratio"),
    ]:
        if m[metric] < POLICY[bound]:
            reasons.append(metric + "_below_limit")
    for metric, bound in [
        ("root_excursion_m", "maximum_root_excursion_m"),
        ("foot_excursion_m", "maximum_foot_excursion_m"),
        ("foot_height_range_m", "maximum_foot_height_range_m"),
        ("root_height_range_m", "maximum_root_height_range_m"),
        ("max_torso_tilt_rad", "maximum_torso_tilt_rad"),
    ]:
        if m[metric] > POLICY[bound]:
            reasons.append(metric + "_above_limit")
    if (
        m["metric_peak_outbound_speed_m_s"] < 0.5 * m["forward_speed_m_s"]
        or m["selected_to_metric_peak_ratio"] < 0.8
    ):
        reasons.append("measurement_window_peak_is_not_selected_outbound_punch")
    if m["elbow_extension_deg"] < 15:
        reasons.append("insufficient_elbow_extension")
    return m, reasons
