"""Category-backed running admission and mutually exclusive movement classes."""

import re

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

RUN_CATEGORIES = {"run", "jog"}
COMPATIBLE = RUN_CATEGORIES | {
    "forward movement",
    "circular movement",
    "exercise/training",
}
CONTEXT = COMPATIBLE | {"transition", "turn"}
WRONG_STYLE = re.compile(
    r"backward|\bback\b|sideways|sideway|side[- ]?run|treadmill|pretend|"
    r"stair|slope|uphill|downhill|jump|hop|skip|dance|moonwalk|limp|"
    r"slow[- ]?mo|object|carry|hold|weight|ball|kick|punch|stomp|march",
    re.I,
)


def semantic_failures(label):
    categories = set(label.get("act_cat") or [])
    failures = []
    if not categories & RUN_CATEGORIES:
        failures.append("missing_run_jog_category")
    if categories - COMPATIBLE:
        failures.append("incompatible_running_category")
    if WRONG_STYLE.search(
        " ".join(label.get(name) or "" for name in ["proc_label", "raw_label"])
    ):
        failures.append("nonstandard_or_nonrunning_description")
    return failures


def runs(mask):
    differences = np.diff(np.pad(np.asarray(mask, dtype=int), (1, 1)))
    return [
        (int(start), int(stop))
        for start, stop in zip(
            np.flatnonzero(differences == 1), np.flatnonzero(differences == -1)
        )
    ]


def movement_signals(joints, smooth_frames):
    lateral = joints[:, 1, :2] - joints[:, 2, :2]
    lateral /= np.linalg.norm(lateral, axis=1, keepdims=True).clip(1e-8)
    forward = np.stack([lateral[:, 1], -lateral[:, 0]], axis=1)
    velocity = np.gradient(joints[:, 0, :2], axis=0) * 20
    return dict(
        speed=uniform_filter1d(
            np.linalg.norm(velocity, axis=1), smooth_frames, mode="nearest"
        ),
        forward=uniform_filter1d(
            (velocity * forward).sum(1), smooth_frames, mode="nearest"
        ),
        lateral=uniform_filter1d(
            (velocity * lateral).sum(1), smooth_frames, mode="nearest"
        ),
    )


def propose_windows(joints, start, stop, valid, rules):
    signals = movement_signals(joints, rules.smooth_frames)
    event = np.zeros(len(joints), dtype=bool)
    event[start:stop] = True
    output = []
    for task, condition in [
        ("jog", (signals["speed"] >= 0.60) & (signals["forward"] >= 0.50)),
        ("march", signals["speed"] <= 0.35),
    ]:
        for begin, end in runs(event & valid & condition):
            if end - begin < rules.minimum_frames:
                continue
            pieces = int(np.ceil((end - begin) / rules.maximum_frames))
            boundaries = np.linspace(begin, end, pieces + 1, dtype=int)
            output.extend(
                (task, int(lower), int(upper))
                for lower, upper in zip(boundaries[:-1], boundaries[1:])
                if upper - lower >= rules.minimum_frames
            )
    return output


def classify_clip(joints, rules):
    """Return jog/march/None, real metrics and all failed shared gait checks."""
    root = joints[:, 0]
    velocities = np.diff(root[:, :2], axis=0) * 20
    speed = np.linalg.norm(velocities, axis=1)
    signals = movement_signals(joints, rules.smooth_frames)
    mean_speed = float(speed.mean())
    feet = np.minimum(joints[:, [7, 8], 2], joints[:, [10, 11], 2])
    floor = np.quantile(feet, 0.05, axis=0)
    clearance = feet - floor
    smooth_clearance = uniform_filter1d(clearance, 3, axis=0, mode="nearest")
    peaks = [
        find_peaks(
            smooth_clearance[:, foot],
            height=rules.peak_height_m,
            prominence=rules.peak_prominence_m,
            distance=4,
        )[0]
        for foot in range(2)
    ]
    events = sorted(
        (int(frame), foot)
        for foot, foot_peaks in enumerate(peaks)
        for frame in foot_peaks
    )
    alternation = (
        np.mean(
            [first[1] != second[1] for first, second in zip(events[:-1], events[1:])]
        )
        if len(events) > 1
        else 0.0
    )
    simultaneous = sum(
        any(abs(int(frame) - int(other)) <= 2 for other in peaks[1 - foot])
        for foot in range(2)
        for frame in peaks[foot]
    ) / max(len(events), 1)
    intervals = np.concatenate([np.diff(foot_peaks) / 20 for foot_peaks in peaks])
    step_intervals = np.diff([frame for frame, foot in events]) / 20
    interval_cv = (
        float(intervals.std() / max(intervals.mean(), 1e-8)) if len(intervals) else 1.0
    )
    flight = (clearance > rules.flight_clearance_m).all(axis=1)
    ground_speed = np.minimum(
        np.linalg.norm(np.gradient(joints[:, [7, 8], :2], axis=0) * 20, axis=-1),
        np.linalg.norm(np.gradient(joints[:, [10, 11], :2], axis=0) * 20, axis=-1),
    )
    support = (clearance < rules.support_clearance_m) & (
        np.abs(np.gradient(feet, axis=0) * 20)
        < rules.maximum_support_vertical_speed_m_s
    )
    support_values = ground_speed[support]
    torso = (joints[:, 16] + joints[:, 17]) / 2 - root
    metrics = dict(
        mean_path_speed_m_s=mean_speed,
        forward_path_fraction=float(
            np.maximum(signals["forward"], 0).sum() / max(signals["speed"].sum(), 1e-8)
        ),
        lateral_path_fraction=float(
            np.abs(signals["lateral"]).sum() / max(signals["speed"].sum(), 1e-8)
        ),
        backward_fraction=float((signals["forward"] < -0.2).mean()),
        pause_fraction=float((signals["speed"] < 0.5).mean()),
        root_excursion_m=float(
            np.linalg.norm(root[:, :2] - root[:1, :2], axis=1).max()
        ),
        root_net_displacement_m=float(np.linalg.norm(root[-1, :2] - root[0, :2])),
        peak_counts=[len(foot_peaks) for foot_peaks in peaks],
        alternation_fraction=float(alternation),
        simultaneous_peak_fraction=float(simultaneous),
        step_rate_hz=(
            float(1 / np.median(step_intervals))
            if len(step_intervals) and np.median(step_intervals) > 0
            else 0.0
        ),
        maximum_step_gap_s=(
            float(step_intervals.max()) if len(step_intervals) else 100.0
        ),
        maximum_double_support_frames=max(
            [
                stop - start
                for start, stop in runs(
                    (clearance < rules.flight_clearance_m).all(axis=1)
                )
            ]
            or [0]
        ),
        stride_interval_cv=interval_cv,
        flight_fraction=float(flight.mean()),
        flight_runs=len(runs(flight)),
        support_frames_by_foot=support.sum(axis=0).tolist(),
        median_support_speed_m_s=(
            float(np.median(support_values)) if len(support_values) else 100.0
        ),
        maximum_torso_tilt_rad=float(
            np.arctan2(np.linalg.norm(torso[:, :2], axis=1), torso[:, 2]).max()
        ),
    )
    failures = []
    if min(metrics["peak_counts"]) < rules.minimum_peaks_per_foot:
        failures.append("insufficient_bilateral_stride_cycles")
    if alternation < rules.minimum_alternation_fraction:
        failures.append("nonalternating_leg_cycles")
    if simultaneous > rules.maximum_simultaneous_peak_fraction:
        failures.append("synchronous_hopping_instead_of_running")
    if (
        not rules.minimum_step_rate_hz
        <= metrics["step_rate_hz"]
        <= rules.maximum_step_rate_hz
    ):
        failures.append("step_cadence_outside_running_gate")
    if interval_cv > rules.maximum_stride_interval_cv:
        failures.append("irregular_stride_cycles")
    if metrics["maximum_step_gap_s"] > rules.maximum_step_gap_s:
        failures.append("long_pause_between_steps")
    if metrics["maximum_double_support_frames"] > rules.maximum_double_support_frames:
        failures.append("standing_or_walk_pause_inside_running_crop")
    if (
        metrics["flight_fraction"] < rules.minimum_flight_fraction
        or metrics["flight_runs"] < rules.minimum_flight_runs
    ):
        failures.append("no_repeated_running_flight_evidence")
    if min(metrics["support_frames_by_foot"]) < 3:
        failures.append("insufficient_alternating_support_evidence")
    if metrics["median_support_speed_m_s"] > rules.maximum_median_support_speed_m_s:
        failures.append("support_foot_sliding_or_treadmill_motion")
    if metrics["maximum_torso_tilt_rad"] > rules.maximum_torso_tilt_rad:
        failures.append("bent_or_mixed_upper_body_motion")
    task = None
    if mean_speed >= rules.jog_minimum_mean_speed_m_s:
        if (
            metrics["forward_path_fraction"] < rules.jog_minimum_forward_fraction
            or metrics["lateral_path_fraction"] > rules.jog_maximum_lateral_fraction
            or metrics["backward_fraction"] > rules.maximum_backward_fraction
            or metrics["pause_fraction"] > rules.jog_maximum_pause_fraction
        ):
            failures.append("not_continuous_forward_running")
        else:
            task = "jog"
    elif (
        mean_speed <= rules.march_maximum_mean_speed_m_s
        and metrics["root_excursion_m"] <= rules.march_maximum_excursion_m
        and metrics["root_net_displacement_m"] <= rules.march_maximum_net_displacement_m
    ):
        task = "march"
    else:
        failures.append("ambiguous_travel_or_stationarity")
    return (None if failures else task), metrics, failures


def trim_to_running_cycles(joints, rules):
    """Discard stationary prefix/suffix instead of relabelling the whole sequence."""
    feet = np.minimum(joints[:, [7, 8], 2], joints[:, [10, 11], 2])
    clearance = uniform_filter1d(
        feet - np.quantile(feet, 0.05, axis=0), 3, axis=0, mode="nearest"
    )
    peaks = sorted(
        int(frame)
        for foot in range(2)
        for frame in find_peaks(
            clearance[:, foot],
            height=rules.peak_height_m,
            prominence=rules.peak_prominence_m,
            distance=4,
        )[0]
    )
    if len(peaks) < 4:
        return None
    # A crop can begin in a genuine swing phase before its first complete peak.
    # Trim actual inactivity, not fixed margins around peaks (which discard it).
    raw_clearance = feet - np.quantile(feet, 0.05, axis=0)
    active = np.flatnonzero((raw_clearance > rules.activity_clearance_m).any(axis=1))
    if not len(active):
        return None
    begin = max(0, int(active[0]) - 1)
    end = min(len(joints), int(active[-1]) + 2)
    return (begin, end) if end - begin >= rules.minimum_frames else None
