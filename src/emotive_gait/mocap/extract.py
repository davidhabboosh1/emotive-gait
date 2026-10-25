"""Per-cycle gait measurements from a :class:`SkeletonClip` (NumPy and SciPy only).

Every landmark track is low-pass filtered (zero-lag Butterworth; NaN gaps are bridged for the
filter and restored after): at ``EVENT_FILTER_HZ`` for timing, so marker noise cannot break a
stance run, and at the lighter ``FILTER_HZ`` for the measurements, so swing peaks and arm swing
keep their amplitude. The pelvis gives the path frame and each foot (toe when present, else
ankle) its events (Task 2); the frame is then rebuilt with a heading window of one measured
stride (the median cycle), which cancels the lateral sway. A cycle is two consecutive left
strikes with one right strike between; it is skipped when it touches a ``turning`` frame, and
as ``irregular`` when a foot does not swing exactly once in it or its duration is further than
``MAX_CYCLE_DEVIATION`` from the median. Within a cycle, with ``(forward, left, up)`` the path
frame axes ("detrended" removes the straight line through the cycle's end values):

=====================  ======================================================================
``cadence``            steps/s: 2 / cycle duration.
``speed``              horizontal pelvis displacement over the cycle / cycle duration (m/s).
``step_length``        half the horizontal pelvis displacement over the cycle (m), so
                       ``speed == step_length * cadence``.
``step_height``        foot (toe, else ankle) peak swing height above the median height of
                       the same foot in stance during the cycle (m); both feet averaged.
``double_support_frac`` per step (strike to the next contralateral strike), the fraction of
                       frames with both feet in stance; the cycle's two steps averaged.
``swing_fraction``     fraction of the cycle a foot is not in stance; both feet averaged.
``feet_spacing``       mean ``|ankle_L - ankle_R|`` along the cycle's mean ``left`` axis (m).
``pelvis_height``      mean pelvis z minus the mean stance-ankle z, plus the ankle's
                       5th-percentile height (the same ankle term as :func:`leg_length`).
``com_bounce_amp``     half peak-to-peak of the detrended pelvis z.
``lateral_sway_amp``   half peak-to-peak of the detrended pelvis displacement along the
                       cycle's mean ``left`` axis.
``trunk_pitch``        mean ``atan2(forward, up)`` of ``NECK - PELVIS``; forward is positive.
``neck_pitch``         mean of the same angle for ``HEAD - NECK`` minus the trunk's (head on
                       trunk); positive is gaze down.
``arm_swing_amp``      half peak-to-peak of the upper arm's sagittal angle relative to the trunk:
                       ``atan2(forward, -up)`` of ``ELBOW - SHOULDER`` plus the trunk angle (0
                       when the arm hangs along the trunk line, forward positive); arms averaged.
``arm_abduction``      mean ``asin`` of the unit upper arm's component along ``left``, signed so
                       that away from the body is positive. The side is decided by the sign of
                       ``SHOULDER_L - SHOULDER_R`` along ``left``, never by ``left`` alone, so a
                       mirrored coordinate system cannot flip it; arms averaged.
``elbow_flexion``      mean ``pi - angle(SHOULDER - ELBOW, WRIST - ELBOW)``; 0 is straight.
``elbow_swing_ratio``  slope of ``np.polyfit(arm_dev, elbow_dev, 1)`` over the cycle, the
                       deviations from the cycle means of the arm angle above and the elbow
                       flexion; arms averaged.
=====================  ======================================================================

A field whose landmarks are absent is ``NaN``; nothing else is.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace

import numpy as np
from scipy.signal import butter, filtfilt

from emotive_gait.mocap.events import FootEvents, PathFrame, cycles, foot_events, path_frame
from emotive_gait.mocap.skeleton import Landmark, LandmarkMap, SkeletonClip

EVENT_FILTER_HZ = 6.0  # low-pass cutoff for the tracks that give the path frame and events
FILTER_HZ = 12.0  # low-pass cutoff for the tracks that give the measurements
FILTER_ORDER = 2  # applied forward and backward, so the effective order is 4
MAX_CYCLE_DEVIATION = 0.25  # a cycle further than this from the median duration is irregular
MIN_CYCLES = 2  # a clip with fewer usable cycles summarises to zero cycles

L = Landmark
_TIMING = (L.PELVIS, L.ANKLE_L, L.ANKLE_R, L.TOE_L, L.TOE_R)


@dataclass(frozen=True)
class GaitMeasurement:
    """One gait cycle's measurements in SI units and radians (see the module table)."""

    cadence: float
    step_length: float
    step_height: float
    double_support_frac: float
    feet_spacing: float
    pelvis_height: float
    com_bounce_amp: float
    lateral_sway_amp: float
    trunk_pitch: float
    neck_pitch: float
    arm_swing_amp: float
    arm_abduction: float
    elbow_flexion: float
    elbow_swing_ratio: float
    speed: float
    swing_fraction: float
    start_frame: int
    end_frame: int


_FLOAT_FIELDS = tuple(f.name for f in fields(GaitMeasurement) if f.type == "float")


@dataclass(frozen=True)
class ClipSummary:
    """Cycle-mean measurement of a clip.

    ``n_discarded`` counts cycles that were found but not used; ``discard_reasons`` counts why,
    per cycle, plus one entry for a clip-level reason (no forward travel, a missing pelvis or
    ankle, no complete cycle). ``n_cycles == 0`` means ``mean`` is all ``NaN`` and
    ``discard_reasons`` is non-empty.
    """

    mean: GaitMeasurement
    n_cycles: int
    n_discarded: int
    discard_reasons: Mapping[str, int]
    leg_length: float


def _nan(n: int) -> np.ndarray:
    return np.full(n, np.nan)


def _nanmean(values) -> float:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else math.nan


def _half_ptp(values: np.ndarray) -> float:
    a = values[np.isfinite(values)]
    return float(np.ptp(a) / 2) if a.size >= 2 else math.nan


def _detrended_half_ptp(values: np.ndarray) -> float:
    """Half peak-to-peak after removing the straight line through the cycle's end values.

    The end-to-end line removes drift (a sloped floor, a slow turn) and leaves a periodic
    signal untouched; a least-squares line would tilt a whole-period sinusoid and inflate it.
    """
    if not (np.isfinite(values[0]) and np.isfinite(values[-1])):
        return math.nan
    line = np.linspace(values[0], values[-1], len(values))
    return _half_ptp(values - line)


def _lowpass(track: np.ndarray, fps: float, cutoff_hz: float) -> np.ndarray:
    """Zero-lag low-pass of a (T, 3) track; NaN rows are bridged linearly and kept NaN."""
    track = np.asarray(track, dtype=float)
    nyquist = fps / 2.0
    ok = np.all(np.isfinite(track), axis=1)
    padlen = 3 * (FILTER_ORDER + 1)
    if cutoff_hz >= nyquist or np.count_nonzero(ok) <= padlen:
        return track
    k = np.arange(len(track))
    filled = np.stack([np.interp(k, k[ok], track[ok, i]) for i in range(3)], axis=1)
    b, a = butter(FILTER_ORDER, cutoff_hz / nyquist)
    out = filtfilt(b, a, filled, axis=0, padlen=min(padlen, len(track) - 1))
    out[~ok] = np.nan
    return out


def _sagittal(v: np.ndarray, frame: PathFrame, *, down: bool = False) -> np.ndarray:
    """``atan2(forward, up)`` of ``v`` (or ``atan2(forward, -up)`` for a limb hanging down)."""
    f = np.sum(v * frame.forward, axis=1)
    return np.arctan2(f, -v[:, 2] if down else v[:, 2])


def _unit(v: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def _between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cos = np.sum(_unit(a) * _unit(b), axis=1)
    return np.arccos(np.clip(cos, -1.0, 1.0))


@dataclass(frozen=True)
class _Angles:
    """Per-frame angle tracks (NaN where a landmark is absent)."""

    trunk: np.ndarray
    neck: np.ndarray
    arm: tuple[np.ndarray, np.ndarray]  # sagittal, relative to the trunk; (left, right)
    abduction: tuple[np.ndarray, np.ndarray]
    elbow: tuple[np.ndarray, np.ndarray]


def _angles(tracks: Mapping[Landmark, np.ndarray], frame: PathFrame) -> _Angles:
    n = len(frame.origin)
    trunk = _sagittal(tracks[L.NECK] - tracks[L.PELVIS], frame) if L.NECK in tracks else _nan(n)
    neck = (
        _sagittal(tracks[L.HEAD] - tracks[L.NECK], frame) - trunk
        if L.HEAD in tracks and L.NECK in tracks
        else _nan(n)
    )
    if L.SHOULDER_L in tracks and L.SHOULDER_R in tracks:
        across = np.sum((tracks[L.SHOULDER_L] - tracks[L.SHOULDER_R]) * frame.left, axis=1)
        side = np.sign(across)
    else:
        side = _nan(n)
    arm, abduction, elbow = [], [], []
    for shoulder, elb, wrist, outward in (
        (L.SHOULDER_L, L.ELBOW_L, L.WRIST_L, side),
        (L.SHOULDER_R, L.ELBOW_R, L.WRIST_R, -side),
    ):
        if shoulder in tracks and elb in tracks:
            upper = tracks[elb] - tracks[shoulder]
            arm.append(_sagittal(upper, frame, down=True) + trunk)
            lateral = np.sum(_unit(upper) * frame.left, axis=1) * outward
            abduction.append(np.arcsin(np.clip(lateral, -1.0, 1.0)))
        else:
            arm.append(_nan(n))
            abduction.append(_nan(n))
        if shoulder in tracks and elb in tracks and wrist in tracks:
            e = tracks[elb]
            elbow.append(np.pi - _between(tracks[shoulder] - e, tracks[wrist] - e))
        else:
            elbow.append(_nan(n))
    return _Angles(trunk, neck, tuple(arm), tuple(abduction), tuple(elbow))


def _swing_ratio(arm: np.ndarray, elbow: np.ndarray) -> float:
    ok = np.isfinite(arm) & np.isfinite(elbow)
    if np.count_nonzero(ok) < 3 or np.ptp(arm[ok]) < 1e-6:
        return math.nan
    x, y = arm[ok] - arm[ok].mean(), elbow[ok] - elbow[ok].mean()
    return float(np.polyfit(x, y, 1)[0])


def _step_height(z: np.ndarray, stance: np.ndarray) -> float:
    if not stance.any() or stance.all():
        return math.nan
    swing_peak = np.nanmax(z[~stance]) if np.isfinite(z[~stance]).any() else math.nan
    return float(swing_peak - _nanmedian(z[stance]))


def _nanmedian(values: np.ndarray) -> float:
    a = values[np.isfinite(values)]
    return float(np.median(a)) if a.size else math.nan


def _ankle_floor_height(tracks: Mapping[Landmark, np.ndarray]) -> float:
    """The ankle's 5th-percentile z, left and right averaged (shared with ``leg_length``)."""
    return _nanmean(
        [_nanpercentile(tracks[a][:, 2], 5) for a in (L.ANKLE_L, L.ANKLE_R) if a in tracks]
    )


def _nanpercentile(values: np.ndarray, q: float) -> float:
    a = values[np.isfinite(values)]
    return float(np.percentile(a, q)) if a.size else math.nan


def _measure_cycle(
    s: int,
    e: int,
    tracks: Mapping[Landmark, np.ndarray],
    frame: PathFrame,
    feet: tuple[np.ndarray, np.ndarray],
    events: tuple[FootEvents, FootEvents],
    angles: _Angles,
    fps: float,
) -> GaitMeasurement:
    window = slice(s, e + 1)
    duration = (e - s) / fps
    pelvis = tracks[L.PELVIS]
    travel = pelvis[e, :2] - pelvis[s, :2]
    displacement = float(np.hypot(*travel))
    left_ev, right_ev = events
    stance = (left_ev.stance[window], right_ev.stance[window])

    mid = int(right_ev.strikes[(right_ev.strikes > s) & (right_ev.strikes < e)][0])
    both = left_ev.stance & right_ev.stance
    double_support = 0.5 * (both[s:mid].mean() + both[mid:e].mean())

    ankles = (tracks[L.ANKLE_L], tracks[L.ANKLE_R])
    left_axis = frame.left[window].mean(axis=0)
    left_axis /= np.linalg.norm(left_axis)
    lateral = (ankles[0][window] - ankles[1][window]) @ left_axis
    stance_ankle_z = _nanmean([_nanmedian(a[window, 2][st]) for a, st in zip(ankles, stance)])
    pelvis_height = _nanmean(pelvis[window, 2]) - stance_ankle_z + _ankle_floor_height(tracks)
    sway = (pelvis[window] - pelvis[s]) @ left_axis

    return GaitMeasurement(
        cadence=2.0 / duration,
        step_length=displacement / 2.0,
        step_height=_nanmean([_step_height(f[window, 2], st) for f, st in zip(feet, stance)]),
        double_support_frac=float(double_support),
        feet_spacing=_nanmean(np.abs(lateral)),
        pelvis_height=pelvis_height,
        com_bounce_amp=_detrended_half_ptp(pelvis[window, 2]),
        lateral_sway_amp=_detrended_half_ptp(sway),
        trunk_pitch=_nanmean(angles.trunk[window]),
        neck_pitch=_nanmean(angles.neck[window]),
        arm_swing_amp=_nanmean([_half_ptp(a[window]) for a in angles.arm]),
        arm_abduction=_nanmean([_nanmean(a[window]) for a in angles.abduction]),
        elbow_flexion=_nanmean([_nanmean(a[window]) for a in angles.elbow]),
        elbow_swing_ratio=_nanmean(
            [_swing_ratio(a[window], b[window]) for a, b in zip(angles.arm, angles.elbow)]
        ),
        speed=displacement / duration,
        swing_fraction=_nanmean([1.0 - st.mean() for st in stance]),
        start_frame=s,
        end_frame=e,
    )


def _regular(s: int, e: int, events: tuple[FootEvents, FootEvents], typical: float) -> bool:
    """One toe-off per foot inside the cycle and a duration near the clip's median cycle.

    A stance run split by marker noise adds a strike, which pairs into a short cycle that
    ``cycles`` alone cannot tell from a real one.
    """
    one_swing = all(
        np.count_nonzero(ev.stance[s:e] & ~ev.stance[s + 1 : e + 1]) == 1 for ev in events
    )
    return one_swing and abs((e - s) / typical - 1.0) <= MAX_CYCLE_DEVIATION


def _extract(clip: SkeletonClip, landmarks: LandmarkMap) -> tuple[list[GaitMeasurement], Counter]:
    reasons: Counter = Counter()
    fps = float(clip.fps)
    raw = clip.landmark_positions(landmarks)
    if any(k not in raw for k in (L.PELVIS, L.ANKLE_L, L.ANKLE_R)):
        reasons["missing pelvis or ankle landmark"] += 1
        return [], reasons
    tracks = {k: _lowpass(v, fps, FILTER_HZ) for k, v in raw.items()}
    timing = {k: _lowpass(v, fps, EVENT_FILTER_HZ) for k, v in raw.items() if k in _TIMING}
    try:
        # Events need only the frame's mean speed; the heading and the turning flags are then
        # rebuilt with a window of one measured stride, which cancels the lateral sway exactly.
        frame = path_frame(timing[L.PELVIS], fps)
    except ValueError:
        reasons["no forward travel"] += 1
        return [], reasons
    feet = (tracks.get(L.TOE_L, tracks[L.ANKLE_L]), tracks.get(L.TOE_R, tracks[L.ANKLE_R]))
    timing_feet = (
        timing.get(L.TOE_L, timing[L.ANKLE_L]),
        timing.get(L.TOE_R, timing[L.ANKLE_R]),
    )
    events = (foot_events(timing_feet[0], frame, fps), foot_events(timing_feet[1], frame, fps))
    pairs = cycles(*events)
    if not pairs:
        reasons["no complete cycle"] += 1
        return [], reasons
    typical = float(np.median([e - s for s, e in pairs]))
    frame = path_frame(timing[L.PELVIS], fps, smooth_s=typical / fps)
    frame = replace(frame, origin=tracks[L.PELVIS])
    angles = _angles(tracks, frame)
    out = []
    for s, e in pairs:
        if frame.turning[s : e + 1].any():
            reasons["turning"] += 1
            continue
        if not _regular(s, e, events, typical):
            reasons["irregular"] += 1
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out.append(_measure_cycle(s, e, tracks, frame, feet, events, angles, fps))
    return out, reasons


def extract_cycles(clip: SkeletonClip, landmarks: LandmarkMap) -> list[GaitMeasurement]:
    """One measurement per usable gait cycle; ``[]`` when there is none (never raises)."""
    return _extract(clip, landmarks)[0]


def leg_length(clip: SkeletonClip, landmarks: LandmarkMap) -> float:
    """Median hip-knee plus median knee-ankle distance plus the ankle's 5th-percentile height.

    Per side, then left and right averaged; ``NaN`` when no side has all three landmarks.
    """
    tracks = clip.landmark_positions(landmarks)
    sides = []
    for hip, knee, ankle in ((L.HIP_L, L.KNEE_L, L.ANKLE_L), (L.HIP_R, L.KNEE_R, L.ANKLE_R)):
        if not all(k in tracks for k in (hip, knee, ankle)):
            continue
        thigh = _nanmedian(np.linalg.norm(tracks[hip] - tracks[knee], axis=1))
        shank = _nanmedian(np.linalg.norm(tracks[knee] - tracks[ankle], axis=1))
        sides.append(thigh + shank + _nanpercentile(tracks[ankle][:, 2], 5))
    return _nanmean(sides)


def _mean(measurements: list[GaitMeasurement]) -> GaitMeasurement:
    values = {name: _nanmean([getattr(m, name) for m in measurements]) for name in _FLOAT_FIELDS}
    start = measurements[0].start_frame if measurements else 0
    end = measurements[-1].end_frame if measurements else 0
    return GaitMeasurement(**values, start_frame=start, end_frame=end)


def summarise(clip: SkeletonClip, landmarks: LandmarkMap) -> ClipSummary:
    """Mean over the usable cycles; fewer than ``MIN_CYCLES`` gives a zero-cycle summary."""
    kept, reasons = _extract(clip, landmarks)
    n_discarded = reasons["turning"]
    if len(kept) < MIN_CYCLES:
        if kept:
            reasons[f"fewer than {MIN_CYCLES} cycles"] += len(kept)
            n_discarded += len(kept)
        kept = []
    return ClipSummary(
        mean=_mean(kept),
        n_cycles=len(kept),
        n_discarded=n_discarded,
        discard_reasons=dict(reasons),
        leg_length=leg_length(clip, landmarks),
    )
