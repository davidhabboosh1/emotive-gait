"""Foot targets and pelvis path as pure functions of ``GaitParameters`` and phase.

Convention ("treadmill style"): foot targets are expressed in the pelvis frame. The stance foot
slides backward under the pelvis at the walking speed while the world pelvis moves forward at that
speed, so the stance foot is stationary in the world by construction.

Phase ``0`` starts the LEFT swing; the right foot runs half a cycle later. One cycle is two steps,
so the phase advances at ``cadence / 2`` per second. Swing occupies ``swing_fraction`` of the cycle
and stance the rest, which leaves ``double_support_frac`` of each step with both feet down.

``x`` is +forward in the pelvis frame. ``y`` is +left, so sway toward the stance side is negative
while the left foot swings. ``z`` is clearance above the floor-contact height.

This module is pure math: it imports nothing from ``robot/`` and nothing from MuJoCo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from emotive_gait.gait.params import GaitParameters


class Side(str, Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class FootPoint:
    x: float  # pelvis frame, metres, +forward
    z: float  # clearance above the floor-contact height, 0 in stance
    in_stance: bool


def SWING_X_EASING(u: float) -> float:
    """Quintic smoothstep ``6u^5 - 15u^4 + 10u^3``: C2 at u = 0 and u = 1."""
    return u**3 * (10.0 + u * (-15.0 + 6.0 * u))


def _clearance(u: float) -> float:
    """Clearance profile ``sin(pi u)^3``: zero value, slope and curvature at both ends."""
    return math.sin(math.pi * u) ** 3


def swing_fraction(params: GaitParameters) -> float:
    return (1.0 - params.double_support_frac) / 2.0


def foot_travel(params: GaitParameters) -> float:
    return params.step_length * (1.0 + params.double_support_frac)


def foot_point(params: GaitParameters, phase: float, side: Side) -> FootPoint:
    p = (phase + (0.5 if side is Side.RIGHT else 0.0)) % 1.0
    sf = swing_fraction(params)
    half = foot_travel(params) / 2.0
    if p < sf:
        u = p / sf
        return FootPoint(
            x=-half + 2.0 * half * SWING_X_EASING(u),
            z=params.step_height * _clearance(u),
            in_stance=False,
        )
    u = (p - sf) / (1.0 - sf)
    return FootPoint(x=half - 2.0 * half * u, z=0.0, in_stance=True)


def pelvis_offset(params: GaitParameters, phase: float) -> tuple[float, float]:
    """Pelvis ``(dy, dz)`` offsets: sway toward the stance side and a two-per-cycle bounce."""
    dy = -params.lateral_sway_amp * math.sin(2.0 * math.pi * phase)
    dz = params.com_bounce_amp * math.cos(4.0 * math.pi * phase)
    return dy, dz


def pelvis_x(params: GaitParameters, t: float) -> float:
    return params.speed * t


def phase_at(params: GaitParameters, t: float) -> float:
    return (t * params.cadence / 2.0) % 1.0
