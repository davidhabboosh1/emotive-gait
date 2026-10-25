"""Parameterised in-place gait: per-role joint angles over normalised phase.

Composition order (design 4.3.1): procedural base, then a per-role override replaces or an
offset adds, then the result is clamped to the joint range last. ``expansiveness`` scales only
the procedural amplitude group, never hand-authored curves. Foot placement and root motion
parameters are deliberately not consumed here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from emotive_gait.gait.params import GaitParameters
from emotive_gait.gait.spline import PeriodicCurve
from emotive_gait.robot.loader import RobotModel
from emotive_gait.robot.roles import JointRole

_BASE_KNEE_AMPLITUDE = 0.25

_ARM_PITCH = {
    JointRole.SHOULDER_PITCH_L: JointRole.HIP_PITCH_R,
    JointRole.SHOULDER_PITCH_R: JointRole.HIP_PITCH_L,
}
_LEG_SIDES = (
    (JointRole.HIP_PITCH_L, JointRole.KNEE_L, JointRole.ANKLE_PITCH_L, 0.0),
    (JointRole.HIP_PITCH_R, JointRole.KNEE_R, JointRole.ANKLE_PITCH_R, 0.5),
)
_ROLL = (JointRole.SHOULDER_ROLL_L, JointRole.SHOULDER_ROLL_R)
_ELBOWS = (JointRole.ELBOW_L, JointRole.ELBOW_R)


@dataclass(frozen=True)
class GaitTrajectory:
    roles: tuple[JointRole, ...]
    params: GaitParameters
    # Everything angles_at needs, captured at generate() time so it takes no robot argument.
    _ranges: Mapping[JointRole, tuple[float, float]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _stand: Mapping[JointRole, float] = field(default_factory=dict, repr=False, compare=False)
    _signs: Mapping[JointRole, float] = field(default_factory=dict, repr=False, compare=False)
    _offsets: Mapping[JointRole, PeriodicCurve] = field(
        default_factory=dict, repr=False, compare=False
    )
    _overrides: Mapping[JointRole, PeriodicCurve] = field(
        default_factory=dict, repr=False, compare=False
    )

    def _procedural(self, phase: float) -> dict[JointRole, float]:
        p, e = self.params, self.params.expansiveness
        k = e * (_BASE_KNEE_AMPLITUDE + p.knee_flexion_bias)
        arm_swing = e * p.arm_swing_amp
        abduction = e * p.arm_abduction

        out = dict(self._stand)
        swing = {}
        for hip, knee, ankle, shift in _LEG_SIDES:
            s = math.sin(2.0 * math.pi * (phase + shift))
            swing[hip] = s
            out[hip] = -0.10 + 0.8 * k * s
            out[knee] = 0.20 + k * max(0.0, s)
            out[ankle] = -0.10 - 0.3 * k * s
        for arm, leg in _ARM_PITCH.items():
            # Contralateral: the arm opposes its own side's leg, i.e. follows the other leg.
            out[arm] = self._stand[arm] + arm_swing * swing[leg]
        for roll in _ROLL:
            out[roll] = self._signs[roll] * abduction
        for elbow in _ELBOWS:
            out[elbow] = p.elbow_flexion
        out[JointRole.WAIST_PITCH] = p.trunk_pitch
        out[JointRole.NECK_PITCH] = p.neck_pitch
        return out

    def angles_at(self, phase: float) -> dict[JointRole, float]:
        base = self._procedural(float(phase))
        angles: dict[JointRole, float] = {}
        for role in self.roles:
            value = base[role]
            if role in self._overrides:
                value = float(self._overrides[role](phase))
            elif role in self._offsets:
                value += float(self._offsets[role](phase))
            lo, hi = self._ranges[role]
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
        _ranges={r: robot.joint_range(r) for r in roles},
        _stand={r: float(robot.stand_qpos[robot.qpos_adr(r)]) for r in roles}
        | {r: 0.0 for r in JointRole if not profile.has(r)},
        _signs={r: profile.sign(r) for r in JointRole},
        _offsets=dict(offsets or {}),
        _overrides=dict(overrides or {}),
    )
