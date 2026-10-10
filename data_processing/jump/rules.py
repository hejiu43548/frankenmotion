"""Conservative event and kinematic checks; feet are proxies, not contact truth."""

import re

import numpy as np

ALLOWED_EVENT_CATEGORIES = {
    "jump",
    "hop",
    "foot movements",
    "leg movements",
    "exercise/training",
    "raising body part",
    "lowering body part",
    "arm movements",
    "hand movements",
    "swing body part",
}
ALLOWED_CONTEXT_CATEGORIES = ALLOWED_EVENT_CATEGORIES | {
    "stand",
    "transition",
    "prepare",
    "crouch",
    "bend",
    "squat",
    "move body part",
    "body part movement",
}
# Text only rejects variants within category-backed candidates. It never admits a source.
WRONG_STYLE = re.compile(
    r"rope|jack|obstacle|barrier|hurdle|bench|stair|platform|table|box|vault|"
    r"jump (?:down|off|over)|hop over|leap over|single|one[- ](?:leg|foot)|"
    r"(?:left|right) (?:leg|foot)|(?:leg|foot) (?:left|right)|alternat|"
    r"crane|kick|spin|twist|somersault|flip|skip|run|jog|basket|ball|"
    r"prepare to|t[- ]?pose",
    re.I,
)
DIRECTIONAL_JUMP = re.compile(
    r"\b(?:left|right|forward\w*|backward\w*|back|sideways|sideward\w*|lateral\w*|"
    r"across|away|ahead|travel\w*|diagonal\w*)\b|side[- ]to[- ]side",
    re.I,
)


def semantic_failures(label):
    categories = set(label.get("act_cat") or [])
    failures = []
    if not categories & {"jump", "hop"}:
        failures.append("missing_jump_hop_category")
    if categories - ALLOWED_EVENT_CATEGORIES:
        failures.append("conflicting_event_category")
    text = " ".join(label.get(name) or "" for name in ["proc_label", "raw_label"])
    if WRONG_STYLE.search(text):
        failures.append("non_flat_two_foot_jump_description")
    if DIRECTIONAL_JUMP.search(text):
        failures.append("directional_jump_description")
    return failures


def runs(mask):
    changes = np.diff(np.pad(np.asarray(mask, dtype=int), (1, 1)))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def foot_heights(joints):
    return np.minimum(joints[:, [7, 8], 2], joints[:, [10, 11], 2])


def propose_windows(joints, event_start, event_stop, rules):
    """Locate shared flight, then bound it by actual bilateral support."""
    lower = max(0, event_start - rules.context_frames)
    upper = min(len(joints), event_stop + rules.context_frames)
    feet = foot_heights(joints[lower:upper])
    floor = np.quantile(feet, 0.1, axis=0)
    clearance = feet - floor
    both_air = (clearance > rules.flight_clearance_m).all(axis=1)
    both_support = (clearance < rules.contact_clearance_m).all(axis=1)
    windows = []
    for flight_start, flight_stop in runs(both_air):
        absolute_start = lower + int(flight_start)
        absolute_stop = lower + int(flight_stop)
        if not event_start <= (absolute_start + absolute_stop) / 2 < event_stop:
            continue
        if (
            not rules.minimum_flight_frames
            <= flight_stop - flight_start
            <= rules.maximum_flight_frames
        ):
            continue
        previous_flight_end = max(
            [0] + [int(stop) for start, stop in runs(both_air[:flight_start])]
        )
        preceding = [
            frame
            for frame in range(
                max(previous_flight_end, flight_start - rules.context_frames),
                flight_start - rules.support_frames + 1,
            )
            if both_support[frame : frame + rules.support_frames].all()
        ]
        following = [
            frame
            for frame in range(
                flight_stop,
                min(
                    len(feet) - rules.support_frames + 1,
                    flight_stop + rules.context_frames,
                ),
            )
            if both_support[frame : frame + rules.support_frames].all()
        ]
        if not preceding or not following:
            continue
        # Start near upright before the countermovement; the original metric uses frame0.
        start = max(preceding, key=lambda frame: float(joints[lower + frame, 0, 2]))
        stop = following[-1] + rules.support_frames
        # Do not include the next flight in a repeated hopping sequence.
        later_flights = np.flatnonzero(both_air[flight_stop:stop])
        if len(later_flights):
            stop = flight_stop + int(later_flights[0])
        if not rules.minimum_clip_frames <= stop - start <= rules.maximum_clip_frames:
            continue
        windows.append(
            tuple(
                map(int, (lower + start, lower + stop, absolute_start, absolute_stop))
            )
        )
    return windows


def check_clip(joints, rules):
    feet = foot_heights(joints)
    support = rules.support_frames
    start_floor = np.median(feet[:support], axis=0)
    end_floor = np.median(feet[-support:], axis=0)
    floor = (start_floor + end_floor) / 2
    clearance = feet - floor
    flight_runs = runs((clearance > rules.flight_clearance_m).all(axis=1))
    sustained = [
        (start, stop)
        for start, stop in flight_runs
        if stop - start >= rules.minimum_flight_frames
    ]
    failures = []
    if not rules.minimum_clip_frames <= len(joints) <= rules.maximum_clip_frames:
        failures.append("invalid_unpadded_clip_length")
    root_height = joints[:, 0, 2]
    side = joints[:, 1, :2] - joints[:, 2, :2]
    side /= np.linalg.norm(side, axis=1, keepdims=True).clip(1e-8)
    forward = np.stack([side[:, 1], -side[:, 0]], axis=1)
    feet_difference = joints[:, 7, :2] - joints[:, 8, :2]
    separation = np.linalg.norm(feet_difference, axis=1)
    torso = (joints[:, 16] + joints[:, 17]) / 2 - joints[:, 0]
    upper_legs = joints[:, [1, 2]] - joints[:, [4, 5]]
    lower_legs = joints[:, [7, 8]] - joints[:, [4, 5]]
    cosine = (upper_legs * lower_legs).sum(axis=-1) / (
        np.linalg.norm(upper_legs, axis=-1) * np.linalg.norm(lower_legs, axis=-1)
    ).clip(1e-8)
    knees = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
    metrics = dict(
        root_rise_m=float(root_height.max() - root_height[0]),
        floor_height_difference_m=float(np.abs(end_floor - start_floor).max()),
        support_foot_height_difference_m=float(
            max(abs(start_floor[0] - start_floor[1]), abs(end_floor[0] - end_floor[1]))
        ),
        endpoint_root_height_difference_m=float(
            abs(root_height[-support:].mean() - root_height[:support].mean())
        ),
        heading_change_rad=float(np.ptp(np.unwrap(np.arctan2(side[:, 1], side[:, 0])))),
        maximum_torso_tilt_rad=float(
            np.arctan2(np.linalg.norm(torso[:, :2], axis=1), torso[:, 2]).max()
        ),
        foot_separation_change_m=float(np.ptp(separation)),
        maximum_foot_forward_difference_m=float(
            np.abs((feet_difference * forward).sum(axis=1)).max()
        ),
        horizontal_displacement_m=float(
            np.linalg.norm(joints[-1, 0, :2] - joints[0, 0, :2])
        ),
        root_endpoint_offset_m=float(
            np.linalg.norm(
                joints[-support:, 0, :2].mean(0) - joints[:support, 0, :2].mean(0)
            )
        ),
        root_horizontal_excursion_m=float(
            np.linalg.norm(
                joints[:, 0, :2] - joints[:support, 0, :2].mean(0), axis=1
            ).max()
        ),
        flight_count=len(sustained),
        takeoff_difference_frames=None,
        landing_difference_frames=None,
        start_knee_angle_deg=float(knees[:support].mean(axis=0).min()),
        end_knee_angle_deg=float(knees[-support:].mean(axis=0).min()),
    )
    if len(sustained) != 1:
        failures.append("not_exactly_one_bilateral_flight")
    else:
        flight_start, flight_stop = sustained[0]
        onsets = []
        landings = []
        for foot in range(2):
            candidates = [
                (start, stop)
                for start, stop in runs(clearance[:, foot] > rules.contact_clearance_m)
                if start <= flight_start and stop >= flight_stop
            ]
            if not candidates:
                failures.append("missing_continuous_foot_flight")
                break
            onsets.append(candidates[0][0])
            landings.append(candidates[0][1])
        if len(onsets) == 2:
            metrics["takeoff_difference_frames"] = int(abs(onsets[0] - onsets[1]))
            metrics["landing_difference_frames"] = int(abs(landings[0] - landings[1]))
            if (
                max(
                    metrics["takeoff_difference_frames"],
                    metrics["landing_difference_frames"],
                )
                > rules.maximum_foot_timing_difference_frames
            ):
                failures.append("asynchronous_takeoff_or_landing")
            # Compare first landing, not just the last frame: a jump out and step
            # back must not pass. Also constrain final recovery against takeoff.
            foot_xy = (joints[:, [7, 8], :2] + joints[:, [10, 11], :2]) / 2
            start_xy = foot_xy[:support].mean(0)
            takeoff = min(onsets)
            landed = max(landings)
            before_xy = (
                foot_xy[max(0, takeoff - support) : takeoff].mean(0)
                if takeoff
                else start_xy
            )
            landing_xy = (
                foot_xy[landed : landed + support].mean(0)
                if landed < len(joints)
                else foot_xy[-support:].mean(0)
            )
            recovery_xy = foot_xy[-support:].mean(0)
            positions = [before_xy, landing_xy, recovery_xy]
            metrics["landing_center_offset_m"] = float(
                max(
                    np.linalg.norm(position.mean(0) - start_xy.mean(0))
                    for position in positions
                )
            )
            metrics["each_foot_landing_offset_m"] = float(
                max(
                    np.linalg.norm(position - start_xy, axis=1).max()
                    for position in positions
                )
            )
            metrics["flight_landing_center_offset_m"] = float(
                np.linalg.norm(landing_xy.mean(0) - before_xy.mean(0))
            )
            metrics["flight_each_foot_landing_offset_m"] = float(
                np.linalg.norm(landing_xy - before_xy, axis=1).max()
            )
            for metric, maximum in [
                ("landing_center_offset_m", rules.maximum_landing_center_offset_m),
                (
                    "flight_landing_center_offset_m",
                    rules.maximum_landing_center_offset_m,
                ),
                (
                    "each_foot_landing_offset_m",
                    rules.maximum_each_foot_landing_offset_m,
                ),
                (
                    "flight_each_foot_landing_offset_m",
                    rules.maximum_each_foot_landing_offset_m,
                ),
            ]:
                if metrics[metric] > maximum:
                    failures.append(metric + "_above_limit")
        if not flight_start <= int(root_height.argmax()) < flight_stop:
            failures.append("root_peak_outside_bilateral_flight")
        metrics["flight_frames"] = int(flight_stop - flight_start)
        if metrics["flight_frames"] > rules.maximum_flight_frames:
            failures.append("implausibly_long_flight")
        metrics["flight_start"] = int(flight_start)
        metrics["flight_stop"] = int(flight_stop)
        metrics["airborne_foot_height_difference_m"] = float(
            np.abs(
                feet[flight_start:flight_stop, 0] - feet[flight_start:flight_stop, 1]
            ).max()
        )
        if (
            metrics["airborne_foot_height_difference_m"]
            > rules.maximum_airborne_foot_height_difference_m
        ):
            failures.append("asymmetric_leg_lift_in_flight")
    if metrics["start_knee_angle_deg"] < rules.minimum_start_knee_angle_deg:
        failures.append("crop_starts_crouched")
    if metrics["end_knee_angle_deg"] < rules.minimum_end_knee_angle_deg:
        failures.append("crop_ends_before_landing_recovery")
    if (
        not (np.abs(clearance[:support]) < rules.contact_clearance_m).all()
        or not (np.abs(clearance[-support:]) < rules.contact_clearance_m).all()
    ):
        failures.append("missing_bilateral_support_at_endpoints")
    if metrics["root_rise_m"] < rules.minimum_root_rise_m:
        failures.append("insufficient_actual_root_rise")
    for metric, maximum in [
        ("root_endpoint_offset_m", rules.maximum_root_endpoint_offset_m),
        ("root_horizontal_excursion_m", rules.maximum_root_horizontal_excursion_m),
        ("floor_height_difference_m", rules.maximum_floor_height_difference_m),
        (
            "support_foot_height_difference_m",
            rules.maximum_support_foot_height_difference_m,
        ),
        (
            "endpoint_root_height_difference_m",
            rules.maximum_endpoint_root_height_difference_m,
        ),
        ("heading_change_rad", rules.maximum_heading_change_rad),
        ("maximum_torso_tilt_rad", rules.maximum_torso_tilt_rad),
        ("foot_separation_change_m", rules.maximum_foot_separation_change_m),
        ("maximum_foot_forward_difference_m", rules.maximum_foot_forward_difference_m),
    ]:
        if metrics[metric] > maximum:
            failures.append(metric + "_above_limit")
    return metrics, failures
