"""Path frame and gait events from joint tracks (NumPy and SciPy only).

All positions are ``(T, 3)`` world coordinates in metres, +z up. The thresholds below are
keyword defaults, frozen against the G1 kinematic round trip in Task 3
(``tests/mocap/test_roundtrip.py``: neutral, sad, angry and happy at 120 Hz, plus neutral and
sad at 60 Hz with 3 mm marker noise). ``mocap.extract`` feeds these functions tracks low-passed at
6 Hz (``EVENT_FILTER_HZ``):

``smooth_s`` (1.0 s)
    Width of the moving-average window applied to the pelvis velocity before taking the
    travel heading. About one stride, so lateral sway (one period per stride) averages out;
    0.25 s flags most of a swaying straight walk as turning. A fixed 1.0 s is not enough for a
    slow stride: sad (1.32 steps/s, a 1.52 s stride, 0.025 m sway at 0.34 m/s) keeps a
    0.54 rad/s heading wobble and 26 % of its frames read as turning. ``mocap.extract``
    therefore uses this default only to find the events, then rebuilds the frame with
    ``smooth_s`` set to the clip's median cycle duration; the default itself is unchanged.
``yaw_rate_max`` (0.5 rad/s)
    A frame is ``turning`` where the heading of the path frame rotates faster than this
    (diagnostic; ``mocap.extract`` discards cycles by their total heading change instead).
Foot events (Task 3 fix round)
    The position-extremum rule (Zeni, Richards & Higginson 2008) replaced Task 2's speed-and-height rule in Task 3's fix round. On the real BML, 100STYLE
    and CMU files the speed rule gave stance 0.30-0.37 of the cycle and double support about
    0.00 on every human clip: the ankle marker moves at 0.3-1.0 x pelvis speed from heel strike
    to foot-flat and from heel-off to toe-off, so it is never "slow" then. A speed threshold
    proportional to walking speed also sat inside the filtered velocity-noise floor for slow
    noisy walks (sad at 60 Hz with 3 mm noise: threshold 0.040 m/s, noise median 0.025 m/s,
    90th percentile 0.049 m/s), fragmenting stance. The extremum prototype gave stance
    0.59-0.65 and 0.16-0.22 s of double support per step on the same files.
``MIN_PEAK_S`` (0.4 s)
    Minimum spacing of two strikes (or two toe-offs) of one foot; the fastest preset's
    stride is 0.83 s.
``PROMINENCE_FRAC`` (0.3)
    An extremum must stand out by this fraction of the 5th-95th percentile range of the
    foot's forward coordinate (about the stride length), so noise never makes one.

On a swing that starts and ends at rest (smoothstep) the extremum falls where the foot's
speed equals the pelvis speed, a few frames inside the swing, so the rule reads stance (and
double support) longer than the swing schedule; the tests derive that offset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

MIN_FORWARD_SPEED = 0.05  # m/s; slower mean travel means walking in place
MIN_PEAK_S = 0.4  # s; extrema of one kind closer than this are one event
PROMINENCE_FRAC = 0.3  # of the 5th-95th percentile range of the foot's forward coordinate


@dataclass(frozen=True)
class PathFrame:
    """Per-frame heading frame of the pelvis path (forward and left are horizontal)."""

    origin: np.ndarray  # (T, 3) pelvis
    forward: np.ndarray  # (T, 3) unit, horizontal
    left: np.ndarray  # (T, 3) unit
    turning: np.ndarray  # (T,) bool
    mean_speed: float = 0.0  # m/s, mean horizontal pelvis speed from the smoothed velocity

    def to_path(self, world: np.ndarray) -> np.ndarray:
        """Coordinates ``(forward, left, up)`` of ``world - origin`` for every frame."""
        rel = np.asarray(world, dtype=float) - self.origin
        return np.stack(
            [
                np.sum(rel * self.forward, axis=1),
                np.sum(rel * self.left, axis=1),
                rel[:, 2],
            ],
            axis=1,
        )


def _horizontal_velocity(positions: np.ndarray, fps: float) -> np.ndarray:
    velocity = np.gradient(np.asarray(positions, dtype=float), axis=0) * fps
    velocity[:, 2] = 0.0
    return velocity


def path_frame(
    pelvis: np.ndarray,
    fps: float,
    *,
    smooth_s: float = 1.0,
    yaw_rate_max: float = 0.5,
) -> PathFrame:
    """Build the heading frame from the smoothed horizontal pelvis velocity."""
    pelvis = np.asarray(pelvis, dtype=float)
    velocity = _horizontal_velocity(pelvis, fps)
    window = max(1, round(smooth_s * fps))
    smooth = uniform_filter1d(velocity, size=window, axis=0, mode="nearest")
    speed = np.linalg.norm(smooth, axis=1)
    mean_speed = float(np.nanmean(speed)) if np.any(np.isfinite(speed)) else 0.0
    if not mean_speed >= MIN_FORWARD_SPEED:
        raise ValueError("path frame needs forward travel")
    moving = np.nan_to_num(speed) > 1e-9
    forward = np.zeros_like(smooth)
    forward[moving] = smooth[moving] / speed[moving, None]
    # Frames with no travel keep the previous heading (or the first valid one).
    valid = np.flatnonzero(moving)
    fill = np.maximum.accumulate(np.where(moving, np.arange(len(moving)), -1))
    fill = np.where(fill < 0, valid[0], fill)
    forward = forward[fill]
    left = np.cross([0.0, 0.0, 1.0], forward)
    yaw = np.unwrap(np.arctan2(forward[:, 1], forward[:, 0]))
    yaw_rate = np.gradient(yaw) * fps
    return PathFrame(pelvis, forward, left, np.abs(yaw_rate) > yaw_rate_max, mean_speed)


@dataclass(frozen=True)
class FootEvents:
    """Foot contact timing: frame indices of strikes and toe-offs, and the stance mask."""

    strikes: np.ndarray  # int frame indices
    toe_offs: np.ndarray  # int frame indices
    stance: np.ndarray  # (T,) bool


def _alternate(maxima: np.ndarray, minima: np.ndarray, x: np.ndarray) -> list[tuple[int, int]]:
    """Merge extrema into a strictly alternating ``(frame, +1 max | -1 min)`` sequence.

    Of two neighbours of the same kind the more extreme survives.
    """
    merged = sorted([(int(k), 1) for k in maxima] + [(int(k), -1) for k in minima])
    out: list[tuple[int, int]] = []
    for k, kind in merged:
        if out and out[-1][1] == kind:
            if kind * x[k] > kind * x[out[-1][0]]:
                out[-1] = (k, kind)
            continue
        out.append((k, kind))
    return out


def foot_events(
    foot: np.ndarray,
    frame: PathFrame,
    fps: float,
    *,
    min_peak_s: float = MIN_PEAK_S,
    prominence_frac: float = PROMINENCE_FRAC,
) -> FootEvents:
    """Strikes and toe-offs from the foot's forward position relative to the pelvis.

    Zeni, Richards & Higginson (2008): with ``x = frame.to_path(foot)[:, 0]`` (the caller
    low-passes the tracks), a strike is a local maximum of ``x`` and a toe-off the following
    local minimum; stance runs from a strike to the next toe-off. Extrema of one kind are at
    least ``min_peak_s`` apart and have a prominence of at least ``prominence_frac`` of the
    5th-95th percentile range of ``x``, so marker noise cannot create them, and the sequence
    is forced to alternate. Frames before a leading toe-off are stance (the clip starts in
    stance; that run has no strike) and so are frames after a trailing strike, so the k-th
    toe-off always follows the k-th strike. NaN rows are bridged for the search and are never
    stance.
    """
    foot = np.asarray(foot, dtype=float)
    x = frame.to_path(foot)[:, 0]
    n = len(x)
    ok = np.isfinite(x)
    empty = np.zeros(0, dtype=int)
    if np.count_nonzero(ok) < 3:
        return FootEvents(empty, empty, np.zeros(n, dtype=bool))
    k = np.arange(n)
    filled = np.interp(k, k[ok], x[ok])
    lo, hi = np.percentile(filled, [5, 95])
    if not hi > lo:
        return FootEvents(empty, empty, np.zeros(n, dtype=bool))
    kw = {"distance": max(1, round(min_peak_s * fps)), "prominence": prominence_frac * (hi - lo)}
    maxima, _ = find_peaks(filled, **kw)
    minima, _ = find_peaks(-filled, **kw)
    sequence = _alternate(maxima, minima, filled)

    stance = np.zeros(n, dtype=bool)
    strikes, toe_offs = [], []
    if sequence and sequence[0][1] == -1:
        stance[: sequence[0][0]] = True
    for i, (frame_k, kind) in enumerate(sequence):
        if kind != 1:
            continue
        strikes.append(frame_k)
        if i + 1 < len(sequence):
            stop = sequence[i + 1][0]
            toe_offs.append(stop)
        else:
            stop = n
        stance[frame_k:stop] = True
    stance &= ok
    return FootEvents(np.array(strikes, dtype=int), np.array(toe_offs, dtype=int), stance)


def cycles(left: FootEvents, right: FootEvents) -> list[tuple[int, int]]:
    """Consecutive left strikes ``(start, end)`` with exactly one right strike between."""
    pairs = []
    for start, end in zip(left.strikes[:-1], left.strikes[1:]):
        between = np.count_nonzero((right.strikes > start) & (right.strikes < end))
        if between == 1:
            pairs.append((int(start), int(end)))
    return pairs
