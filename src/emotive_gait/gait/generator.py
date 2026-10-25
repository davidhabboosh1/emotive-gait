"""Parameterised in-place gait: per-role joint angles over normalised phase.

Composition order (design 4.3.1): procedural base, then a per-role override replaces or an
offset adds, then the result is clamped to the joint range last. ``expansiveness`` scales only
the procedural amplitude group, never hand-authored curves. Foot placement and root motion
parameters are deliberately not consumed here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from emotive_gait.gait.params import GaitParameters
from emotive_gait.gait.spline import PeriodicCurve
from emotive_gait.robot.loader import RobotModel
from emotive_gait.robot.roles import JointRole

T = TypeVar("T")

_BASE_KNEE_AMPLITUDE = 0.25
# The Z-rail bob and contact bounds in test_z_rail_survives_six_seconds_on_every_preset were
# calibrated against exactly these leg values: changing them is a physics decision, not a
# tuning tweak.
_HIP_PITCH_OFFSET = -0.10
_KNEE_OFFSET = 0.20
_HIP_SWING_GAIN = 0.8
_ANKLE_PITCH_OFFSET = -0.10
_ANKLE_SWING_GAIN = 0.3

_ARM_PITCH = {
    JointRole.SHOULDER_PITCH_L: JointRole.HIP_PITCH_R,
    JointRole.SHOULDER_PITCH_R: JointRole.HIP_PITCH_L,
}
_LEG_SIDES = (
    (JointRole.HIP_PITCH_L, JointRole.KNEE_L, JointRole.ANKLE_PITCH_L, 0.0),
    (JointRole.HIP_PITCH_R, JointRole.KNEE_R, JointRole.ANKLE_PITCH_R, 0.5),
)
_ROLL = (JointRole.SHOULDER_ROLL_L, JointRole.SHOULDER_ROLL_R)
_ELBOW_FOR_ARM = {
    JointRole.ELBOW_L: JointRole.SHOULDER_PITCH_L,
    JointRole.ELBOW_R: JointRole.SHOULDER_PITCH_R,
}


def _pairs(mapping: Mapping[JointRole, T]) -> tuple[tuple[JointRole, T], ...]:
    return tuple(sorted(mapping.items(), key=lambda kv: kv[0].name))


@dataclass(frozen=True)
class GaitTrajectory:
    roles: tuple[JointRole, ...]
    params: GaitParameters
    # Everything angles_at needs, captured at generate() time so it takes no robot argument.
    # Captured state is stored as role-name-sorted tuples of (role, value): immutable, and part
    # of equality and hashing, so the trajectory is a true value type.
    _ranges: tuple[tuple[JointRole, tuple[float, float]], ...] = ()
    _stand: tuple[tuple[JointRole, float], ...] = ()
    _signs: tuple[tuple[JointRole, float], ...] = ()
    _offsets: tuple[tuple[JointRole, PeriodicCurve], ...] = ()
    _overrides: tuple[tuple[JointRole, PeriodicCurve], ...] = ()

    def _procedural(self, phase: float, stand: dict[JointRole, float]) -> dict[JointRole, float]:
        p, e = self.params, self.params.expansiveness
        k = e * (_BASE_KNEE_AMPLITUDE + p.knee_flexion_bias)
        arm_swing = e * p.arm_swing_amp
        abduction = e * p.arm_abduction

        signs = dict(self._signs)
        out = dict(stand)
        swing = {}
        for hip, knee, ankle, shift in _LEG_SIDES:
            s = math.sin(2.0 * math.pi * (phase + shift))
            swing[hip] = s
            out[hip] = _HIP_PITCH_OFFSET + _HIP_SWING_GAIN * k * s
            out[knee] = _KNEE_OFFSET + k * max(0.0, s)
            out[ankle] = _ANKLE_PITCH_OFFSET - _ANKLE_SWING_GAIN * k * s
        for arm, leg in _ARM_PITCH.items():
            # Contralateral: the arm opposes its own side's leg, i.e. follows the other leg.
            out[arm] = stand[arm] + arm_swing * swing[leg]
        for roll in _ROLL:
            out[roll] = signs[roll] * abduction
        for elbow, arm in _ELBOW_FOR_ARM.items():
            # In phase with the shoulder deviation (elbow flexes more as the arm swings
            # forward) is a tuning choice; the amplitude semantics (elbow swing / shoulder
            # swing = elbow_swing_ratio) are what the emotion model relies on.
            out[elbow] = p.elbow_flexion + p.elbow_swing_ratio * (out[arm] - stand[arm])
        out[JointRole.WAIST_PITCH] = p.trunk_pitch
        out[JointRole.NECK_PITCH] = p.neck_pitch
        return out

    def angles_at(self, phase: float) -> dict[JointRole, float]:
        stand = dict(self._stand)
        ranges = dict(self._ranges)
        offsets = dict(self._offsets)
        overrides = dict(self._overrides)
        base = self._procedural(float(phase), stand)
        angles: dict[JointRole, float] = {}
        for role in self.roles:
            value = base[role]
            if role in overrides:
                value = float(overrides[role](phase))
            elif role in offsets:
                value += float(offsets[role](phase))
            lo, hi = ranges[role]
            angles[role] = min(max(value, lo), hi)
        return angles

    def qpos_at(self, robot: RobotModel, phase: float) -> np.ndarray:
        qpos = robot.stand_qpos.copy()
        for role, value in self.angles_at(phase).items():
            qpos[robot.qpos_adr(role)] = value
        return qpos

    def ctrl_at(self, robot: RobotModel, phase: float) -> np.ndarray:
        model = robot.model
        joint_ids = model.actuator_trnid[:, 0]
        ctrl = robot.stand_qpos[model.jnt_qposadr[joint_ids]].astype(float)
        for role, value in self.angles_at(phase).items():
            ctrl[robot.ctrl_index(role)] = value
        return ctrl


def generate(
    params: GaitParameters,
    robot: RobotModel,
    *,
    offsets: Mapping[JointRole, PeriodicCurve] | None = None,
    overrides: Mapping[JointRole, PeriodicCurve] | None = None,
) -> GaitTrajectory:
    profile = robot.profile
    roles = tuple(r for r in JointRole if profile.has(r))
    return GaitTrajectory(
        roles=roles,
        params=params,
        _ranges=_pairs({r: tuple(float(x) for x in robot.joint_range(r)) for r in roles}),
        _stand=_pairs(
            {r: float(robot.stand_qpos[robot.qpos_adr(r)]) for r in roles}
            | {r: 0.0 for r in JointRole if not profile.has(r)}
        ),
        _signs=_pairs({r: float(profile.sign(r)) for r in JointRole}),
        _offsets=_pairs(offsets or {}),
        _overrides=_pairs(overrides or {}),
    )
