"""Synthetic walkers shared by the event and extraction tests (0.6 m/s, 120 Hz).

``straight`` and ``make_arc_pelvis`` are Task 2's pelvis-and-feet fixtures. ``full_walker`` adds
hips, knees, a trunk, a head and arms with known posture, and a 0.6 s stance (0.4 s quintic
swing, as the generator's) so each step has 0.2 of double support.
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


def foot_track(
    t: np.ndarray, offset_s: float, y: float, swing_s: float = 0.5, *, quintic: bool = False
) -> np.ndarray:
    """Smoothstep swing for the first ``swing_s`` of each cycle, then stance (fixed world x).

    ``quintic`` uses the generator's C2 smoothstep (``gait.footsteps.smoothstep5``), whose
    swing foot eases into and out of stance as the G1's does, instead of the cubic.
    """
    u = (t - offset_s) / STRIDE_S
    cycle = np.floor(u)
    swing = np.clip((u - cycle) * STRIDE_S / swing_s, 0.0, 1.0)
    if quintic:
        smooth = swing**3 * (10.0 + swing * (-15.0 + 6.0 * swing))
    else:
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
        "l": foot_track(t, 0.0, FOOT_Y, SWING_S, quintic=True),
        "r": foot_track(t, STRIDE_S / 2, -FOOT_Y, SWING_S, quintic=True),
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
