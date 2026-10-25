"""Path frame and gait events from joint tracks (NumPy and SciPy only).

All positions are ``(T, 3)`` world coordinates in metres, +z up. The thresholds below are
keyword defaults, frozen against the synthetic round trip in Task 3:

``smooth_s`` (1.0 s)
    Width of the moving-average window applied to the pelvis velocity before taking the
    travel heading. About one stride, so lateral sway (one period per stride) averages out;
    0.25 s flags most of a swaying straight walk as turning.
``yaw_rate_max`` (0.5 rad/s)
    A frame is ``turning`` where the heading of the path frame rotates faster than this.
``speed_frac`` (0.15)
    A foot is stationary when its horizontal world speed is below this fraction of the mean
    pelvis speed.
``height_band_m`` (0.03 m)
    A foot is on the ground when it is within this height of its own 5th-percentile height
    (markers sit on the ankle, so the floor itself is not a usable reference).
``MIN_STANCE_S`` (0.1 s)
    Stance runs shorter than this are flicker and are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

MIN_FORWARD_SPEED = 0.05  # m/s; slower mean travel means walking in place
MIN_STANCE_S = 0.1


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


def foot_events(
    foot: np.ndarray,
    frame: PathFrame,
    fps: float,
    *,
    speed_frac: float = 0.15,
    height_band_m: float = 0.03,
) -> FootEvents:
    """Detect stance as slow and low relative to the foot's own track.

    Speed is the magnitude of the foot's horizontal world velocity (not relative to the
    pelvis). A strike is a stance onset, a toe-off the first frame after a stance run. A run
    touching frame 0 has no strike and one touching the last frame has no toe-off, so the
    k-th toe-off always follows the k-th strike.
    """
    foot = np.asarray(foot, dtype=float)
    foot_speed = np.linalg.norm(_horizontal_velocity(foot, fps), axis=1)
    low = foot[:, 2] - np.nanpercentile(foot[:, 2], 5) < height_band_m
    stance = (np.isfinite(foot_speed) & (foot_speed < speed_frac * frame.mean_speed)) & low

    padded = np.concatenate([[False], stance, [False]]).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)  # first frame after each run
    keep = (stops - starts) >= MIN_STANCE_S * fps
    starts, stops = starts[keep], stops[keep]
    stance = np.zeros_like(stance)
    for a, b in zip(starts, stops):
        stance[a:b] = True
    strikes = starts[starts > 0]
    toe_offs = stops[(stops < len(stance)) & (starts > 0)]
    return FootEvents(strikes.astype(int), toe_offs.astype(int), stance)


def cycles(left: FootEvents, right: FootEvents) -> list[tuple[int, int]]:
    """Consecutive left strikes ``(start, end)`` with exactly one right strike between."""
    pairs = []
    for start, end in zip(left.strikes[:-1], left.strikes[1:]):
        between = np.count_nonzero((right.strikes > start) & (right.strikes < end))
        if between == 1:
            pairs.append((int(start), int(end)))
    return pairs
