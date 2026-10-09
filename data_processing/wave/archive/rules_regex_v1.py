"""Conservative standing, right-compatible wave admission in source time."""

import re
import numpy as np

WAVE = re.compile(r"\bwav(?:e|es|ed|ing)\b", re.I)
CONFLICT = re.compile(
    r"\b(?:walk\w*|step\w*|run|runs|running|ran|jog\w*|march\w*|"
    r"jump\w*|hop|hops|hopping|dance|dances|dancing|danced|kick\w*|"
    r"stumb\w*|crawl\w*|swim\w*|punch\w*|box\w*|knock\w*|scratch\w*|"
    r"clap\w*|throw\w*|cartwheel\w*|sit|sits|sitting|kneel\w*|"
    r"wipe\w*|wiping|flap\w*|flutt\w*|elephant|swat\w*)\b", re.I
)
POLICY = dict(
    name="standing_right_compatible_wave_v1", fps=20,
    minimum_frames=40, maximum_frames=120, confidence=3,
    root_excursion_m=.18, root_mean_speed_m_s=.12,
    foot_excursion_m=.25, root_height_range_m=.15,
    right_wrist_relative_excursion_m=.12,
    semantics="Right or both hands waving while standing; no locomotion or mixed conflicting caption. Left-only is incompatible with the existing right-wrist command metric.",
    wave_regex=WAVE.pattern, conflict_regex=CONFLICT.pattern,
    thresholds="Conservative admission choices, not ground-truth contact labels.",
)


def evidence(annotation, frames):
    """No caption fallback; return a high-confidence timed right-arm mask."""
    times = np.arange(frames) / 20
    bounds = (times >= annotation["start"]) & (times < annotation["end"])
    right = np.zeros(frames, bool)
    conflict = np.zeros(frames, bool)
    for segment in annotation["annotations"]:
        if segment.get("confidence", 0) < POLICY["confidence"]:
            continue
        active = (times >= segment["start"]) & (times < segment["end"])
        if segment["bodypart"] == "right_arm" and WAVE.search(segment["text"]):
            right |= active
        if segment["bodypart"] != "sequence_caption" and CONFLICT.search(segment["text"]):
            conflict |= active
    return bounds & right & ~conflict


def caption_reasons(annotation):
    caption = annotation.get("caption_label", "")
    reasons = []
    if not WAVE.search(caption):
        reasons.append("no_exact_wave_word_in_caption")
    if CONFLICT.search(caption):
        reasons.append("conflicting_global_caption")
    return reasons


def windows(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    result = []
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start < POLICY["minimum_frames"]:
            continue
        boundaries = np.linspace(start, stop, int(np.ceil((stop-start)/POLICY["maximum_frames"]))+1, dtype=int)
        result.extend((int(s), int(e)) for s, e in zip(boundaries[:-1], boundaries[1:]))
    return result


def metrics(positions):
    root = positions[:, 0]
    relative_wrist = positions[:, 21] - root
    feet = positions[:, [7, 8, 10, 11], :2]
    return dict(
        root_excursion_m=float(np.linalg.norm(root[:, :2]-root[:1, :2], axis=-1).max()),
        root_mean_speed_m_s=float(np.linalg.norm(np.diff(root[:, :2], axis=0), axis=-1).mean()*20),
        foot_excursion_m=float(np.linalg.norm(feet-feet[:1], axis=-1).max()),
        root_height_range_m=float(np.ptp(root[:, 2])),
        right_wrist_relative_excursion_m=float(np.linalg.norm(relative_wrist-relative_wrist[:1], axis=-1).max()),
    )


def motion_reasons(values):
    result = []
    for name in ["root_excursion_m", "root_mean_speed_m_s", "foot_excursion_m", "root_height_range_m"]:
        if not np.isfinite(values[name]) or values[name] > POLICY[name]:
            result.append(name + "_above_limit")
    name = "right_wrist_relative_excursion_m"
    if not np.isfinite(values[name]) or values[name] < POLICY[name]:
        result.append(name + "_below_limit")
    return result
