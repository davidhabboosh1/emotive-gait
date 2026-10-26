"""Synthetic walkers shared by the event and extraction tests (0.6 m/s, 120 Hz).

``straight`` and ``make_arc_pelvis`` are Task 2's pelvis-and-feet fixtures. ``full_walker`` adds
hips, knees, a trunk, a head and arms with known posture, and a 0.6 s stance (0.4 s cubic
swing) so each step has 0.2 of double support by schedule. ``bend`` maps a straight clip onto
a curved path without moving a stance foot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emotive_gait.mocap.skeleton import Landmark, SkeletonClip

FPS = 120.0
SPEED = 0.6
STRIDE_S = 1.0  # one left-to-left cycle
SWAY_M = 0.015
CLEARANCE_M = 0.05
FOOT_REST_Z = 0.08  # the ankle never reaches the floor, as in the real markers
PELVIS_Z = 0.9


@dataclass(frozen=True)
class Walker:
    pelvis: np.ndarray
    left_foot: np.ndarray
    right_foot: np.ndarray


def foot_track(t: np.ndarray, offset_s: float, y: float, swing_s: float = 0.5) -> np.ndarray:
    """Cubic smoothstep swing for the first ``swing_s`` of each cycle, then stance (fixed x)."""
    u = (t - offset_s) / STRIDE_S
    cycle = np.floor(u)
    swing = np.clip((u - cycle) * STRIDE_S / swing_s, 0.0, 1.0)
    smooth = swing * swing * (3.0 - 2.0 * swing)
    x = SPEED * STRIDE_S * (cycle + smooth)
    z = FOOT_REST_Z + CLEARANCE_M * np.sin(np.pi * swing)
    return np.stack([x, np.full_like(t, y), z], axis=1)


def _pelvis(t: np.ndarray, sway_m: float) -> np.ndarray:
    return np.stack(
        [SPEED * t, sway_m * np.sin(2 * np.pi * t / STRIDE_S), np.full_like(t, PELVIS_Z)], axis=1
    )


def straight_walker(sway_m: float = SWAY_M) -> Walker:
    t = np.arange(int(6 * FPS)) / FPS
    return Walker(
        pelvis=_pelvis(t, sway_m),
        left_foot=foot_track(t, 0.0, 0.1),
        right_foot=foot_track(t, STRIDE_S / 2, -0.1),
    )


def make_arc_pelvis() -> np.ndarray:
    """Straight, a 90-degree left turn over 1.5 s, straight again; constant 0.6 m/s."""
    n = int(6 * FPS)
    t = np.arange(n) / FPS
    turn = np.clip((t - 2.25) / 1.5, 0.0, 1.0) * (np.pi / 2)
    velocity = SPEED * np.stack([np.cos(turn), np.sin(turn), np.zeros(n)], axis=1)
    pelvis = np.cumsum(velocity, axis=0) / FPS
    pelvis[:, 2] = PELVIS_Z
    return pelvis


# Full walker construction (all angles in the sagittal x-z plane, +x forward).
SWING_S = 0.4
TRUNK_PITCH = 0.1  # forward
NECK_PITCH = 0.3  # head on trunk, forward (gaze down)
ARM_SWING = 0.3  # upper-arm sagittal amplitude relative to the trunk
ELBOW_FLEXION = 0.8
ELBOW_SWING_RATIO = 0.4
TRUNK_M, HEAD_M, UPPER_ARM_M, FOREARM_M, THIGH_M, SHANK_M = 0.5, 0.2, 0.3, 0.25, 0.45, 0.45
HIP_Y, SHOULDER_Y, FOOT_Y = 0.1, 0.18, 0.1


@dataclass(frozen=True)
class FullWalker:
    clip: SkeletonClip
    landmarks: dict[Landmark, str]
    double_support: float


def _up_at(angle: np.ndarray | float, n: int) -> np.ndarray:
    """Unit vectors tilted ``angle`` forward from +z."""
    a = np.broadcast_to(angle, (n,))
    return np.stack([np.sin(a), np.zeros(n), np.cos(a)], axis=1)


def _knee(hip: np.ndarray, ankle: np.ndarray) -> np.ndarray:
    """Knee forward of the hip-ankle line with the thigh and shank lengths held."""
    mid = (hip + ankle) / 2
    chord = ankle - hip
    half = np.linalg.norm(chord, axis=1, keepdims=True) / 2
    normal = np.stack([-chord[:, 2], np.zeros(len(chord)), chord[:, 0]], axis=1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    normal *= np.where(normal[:, :1] < 0, -1.0, 1.0)  # forward
    return mid + normal * np.sqrt(THIGH_M**2 - half**2)


def full_walker(seconds: float = 6.0) -> FullWalker:
    t = np.arange(int(seconds * FPS)) / FPS
    n = len(t)
    pelvis = _pelvis(t, SWAY_M)
    ankles = {
        "l": foot_track(t, 0.0, FOOT_Y, SWING_S),
        "r": foot_track(t, STRIDE_S / 2, -FOOT_Y, SWING_S),
    }
    neck = pelvis + TRUNK_M * _up_at(TRUNK_PITCH, n)
    head = neck + HEAD_M * _up_at(TRUNK_PITCH + NECK_PITCH, n)
    joints = {"pelvis": pelvis, "neck": neck, "head": head}
    for side, sign in (("l", 1.0), ("r", -1.0)):
        hip = pelvis + [0.0, sign * HIP_Y, 0.0]
        joints[f"hip_{side}"] = hip
        joints[f"knee_{side}"] = _knee(hip, ankles[side])
        joints[f"ankle_{side}"] = ankles[side]
        # Arms swing opposite to their own leg: left arm forward while the left foot is back.
        arm = -sign * ARM_SWING * np.sin(2 * np.pi * t / STRIDE_S)
        flex = ELBOW_FLEXION + ELBOW_SWING_RATIO * arm
        shoulder = neck + [0.0, sign * SHOULDER_Y, 0.0]
        # Down along the trunk line, then forward by ``arm``; the forearm bends forward further.
        elbow = shoulder - UPPER_ARM_M * _up_at(TRUNK_PITCH - arm, n)
        wrist = elbow - FOREARM_M * _up_at(TRUNK_PITCH - arm - flex, n)
        joints[f"shoulder_{side}"] = shoulder
        joints[f"elbow_{side}"] = elbow
        joints[f"wrist_{side}"] = wrist
    names = tuple(joints)
    clip = SkeletonClip(
        joint_names=names,
        parents=None,
        positions=np.stack([joints[k] for k in names], axis=1),
        fps=FPS,
        source="synthetic",
    )
    return FullWalker(
        clip=clip,
        landmarks={Landmark(name): name for name in names},
        double_support=(STRIDE_S - 2 * SWING_S) / STRIDE_S,
    )


def extremum_offset_s(swing_s: float) -> float:
    """Time from swing start (or before swing end) to the position extremum of the foot.

    The position-extremum rule (``events.foot_events``) finds the extremum of the foot's forward
    position relative to the pelvis, i.e. where the foot's speed equals the pelvis speed. The
    cubic smoothstep foot covers ``SPEED * STRIDE_S`` in ``swing_s``, so its speed is
    ``SPEED * STRIDE_S / swing_s * 6 u (1 - u)``; equal to ``SPEED`` where
    ``u (1 - u) = swing_s / (6 STRIDE_S)``, the smaller root ``u = (1 - sqrt(1 - 4c)) / 2``.
    """
    c = swing_s / (6.0 * STRIDE_S)
    return swing_s * (1.0 - np.sqrt(1.0 - 4.0 * c)) / 2.0


def bend(positions: np.ndarray, start_m: float, length_m: float, angle: float) -> np.ndarray:
    """Map world ``x`` to arc length along a path that turns by ``angle`` over ``length_m``.

    The path is straight before ``start_m`` and after ``start_m + length_m``; world ``y`` is the
    offset along the path's left normal, so a stance foot (fixed x, y) stays fixed.
    """
    grid = np.linspace(positions[..., 0].min() - 1.0, positions[..., 0].max() + 1.0, 20001)
    heading = angle * np.clip((grid - start_m) / length_m, 0.0, 1.0)
    step = np.diff(grid, prepend=grid[0])
    cx = np.cumsum(np.cos(heading) * step)
    cy = np.cumsum(np.sin(heading) * step)
    x, y = positions[..., 0], positions[..., 1]
    h = np.interp(x, grid, heading)
    px, py = np.interp(x, grid, cx), np.interp(x, grid, cy)
    return np.stack([px - y * np.sin(h), py + y * np.cos(h), positions[..., 2]], axis=-1)
