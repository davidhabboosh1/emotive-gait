"""Parameterised gait: per-role joint angles over normalised phase, plus the pelvis path.

The legs come from task-space IK (``gait.ik``): one cycle of foot targets from
``gait.footsteps`` is solved on a welded robot and each leg role is fitted with a periodic C2
spline. The upper body is procedural: each arm follows the OPPOSITE leg's solved hip-pitch curve,
mean-removed and normalised by its peak deviation, so ``arm_swing_amp`` is the arm's amplitude
whatever the stride (an in-place gait still swings its arms). The swing fades in with the hip
excursion below ``_HIP_SWING_FULL``, so a static gait does not swing its arms on solver noise.

Composition order (design 4.3.1): solved legs and procedural upper body, then a per-role override
replaces or an offset adds, then the result is clamped to the joint range last.
``expansiveness`` scales arm swing, arm abduction and the swing-foot clearance (``step_height``),
never ``step_length`` and never hand-authored curves.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TypeVar

import numpy as np

from emotive_gait.gait.footsteps import pelvis_offset, pelvis_x
from emotive_gait.gait.ik import FeasibilityReport, solve_leg_cycle, weld_robot_for
from emotive_gait.gait.params import GaitParameters
from emotive_gait.gait.spline import PeriodicCurve
from emotive_gait.robot.loader import RobotModel
from emotive_gait.robot.roles import JointRole

T = TypeVar("T")

# Arm swing is a reaction to leg swing: it fades in linearly up to this peak hip-pitch excursion
# (rad) and is full above it. Every walking preset exceeds 0.1 rad, so arm_swing_amp is the arm
# amplitude there; a static gait (whose hip curve is only solver noise) gets no arm swing.
_HIP_SWING_FULL = 0.1

_ARM_PITCH = {
    JointRole.SHOULDER_PITCH_L: JointRole.HIP_PITCH_R,
    JointRole.SHOULDER_PITCH_R: JointRole.HIP_PITCH_L,
}
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
    feasibility: FeasibilityReport
    # Everything angles_at needs, captured at generate() time so it takes no robot argument.
    # Captured state is stored as role-name-sorted tuples of (role, value): immutable, and part
    # of equality and hashing, so the trajectory is a true value type.
    _legs: tuple[tuple[JointRole, PeriodicCurve], ...] = ()
    _nominal_height: float = 0.0
    _hip_norm: tuple[tuple[JointRole, tuple[float, float]], ...] = ()  # (mean, peak) per hip
    _ranges: tuple[tuple[JointRole, tuple[float, float]], ...] = ()
    _stand: tuple[tuple[JointRole, float], ...] = ()
    _signs: tuple[tuple[JointRole, float], ...] = ()
    _offsets: tuple[tuple[JointRole, PeriodicCurve], ...] = ()
    _overrides: tuple[tuple[JointRole, PeriodicCurve], ...] = ()

    @property
    def offsets(self) -> Mapping[JointRole, PeriodicCurve]:
        """Hand-authored additive curves, so a projected trajectory can be regenerated."""
        return dict(self._offsets)

    @property
    def overrides(self) -> Mapping[JointRole, PeriodicCurve]:
        """Hand-authored replacement curves, so a projected trajectory can be regenerated."""
        return dict(self._overrides)

    def root_offset_at(self, phase: float) -> np.ndarray:
        """Pelvis ``(x, y, z)`` relative to the walking line: sway, crouch and bounce.

        ``x`` is always 0 here; forward travel is :meth:`root_x_at`.
        """
        dy, dz = pelvis_offset(self.params, float(phase))
        return np.array([0.0, dy, self._nominal_height * self.params.com_height_scale + dz])

    def root_x_at(self, t: float) -> float:
        """World forward travel of the pelvis at time ``t`` (seconds)."""
        return pelvis_x(self.params, t)

    def _procedural(self, phase: float, stand: dict[JointRole, float]) -> dict[JointRole, float]:
        p, e = self.params, self.params.expansiveness
        arm_swing = e * p.arm_swing_amp
        abduction = e * p.arm_abduction

        signs = dict(self._signs)
        legs = dict(self._legs)
        hip_norm = dict(self._hip_norm)
        out = dict(stand)
        for role, curve in legs.items():
            out[role] = float(curve(phase))
        for arm, hip in _ARM_PITCH.items():
            # Contralateral: the arm follows the OPPOSITE leg's hip, normalised so that
            # arm_swing_amp is the arm amplitude whatever the stride.
            out[arm] = stand[arm] + arm_swing * _arm_drive(legs.get(hip), hip_norm.get(hip), phase)
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


def _hip_norm(curve: PeriodicCurve) -> tuple[float, float]:
    """``(mean, peak absolute deviation)`` of a hip curve over its solved samples (the knots)."""
    samples = curve.knots_value[:-1]
    mean = float(samples.mean())
    return mean, float(np.abs(samples - mean).max())


def _arm_drive(
    curve: PeriodicCurve | None, norm: tuple[float, float] | None, phase: float
) -> float:
    """The hip curve mean-removed and divided by its peak, faded by the hip excursion.

    Zero when the curve is absent or exactly flat; ``min(1, peak / _HIP_SWING_FULL)`` keeps
    solver noise on a static gait from being amplified into a full arm swing.
    """
    if curve is None or norm is None:
        return 0.0
    mean, peak = norm
    if peak == 0.0:
        return 0.0
    scale = min(1.0, peak / _HIP_SWING_FULL)
    return scale * (float(curve(phase)) - mean) / peak


def generate(
    params: GaitParameters,
    robot: RobotModel,
    *,
    offsets: Mapping[JointRole, PeriodicCurve] | None = None,
    overrides: Mapping[JointRole, PeriodicCurve] | None = None,
) -> GaitTrajectory:
    """Solve the legs on the profile's welded robot and compose the upper body.

    Works for a robot of any root mode: the leg solve always runs on
    ``weld_robot_for(robot.profile)`` and never alters ``params``. An infeasible stride is
    flagged in ``feasibility``, not rescaled.
    """
    profile = robot.profile
    roles = tuple(r for r in JointRole if profile.has(r))
    solved = replace(params, step_height=params.expansiveness * params.step_height)
    cycle = solve_leg_cycle(solved, weld_robot_for(profile))
    legs = {
        role: PeriodicCurve.from_knots(cycle.phases, cycle.angles[role]) for role in cycle.angles
    }
    return GaitTrajectory(
        roles=roles,
        params=params,
        feasibility=cycle.report,
        _legs=_pairs(legs),
        _nominal_height=float(profile.nominal_pelvis_height),
        _hip_norm=_pairs({hip: _hip_norm(legs[hip]) for hip in _ARM_PITCH.values() if hip in legs}),
        _ranges=_pairs({r: tuple(float(x) for x in robot.joint_range(r)) for r in roles}),
        _stand=_pairs(
            {r: float(robot.stand_qpos[robot.qpos_adr(r)]) for r in roles}
            | {r: 0.0 for r in JointRole if not profile.has(r)}
        ),
        _signs=_pairs({r: float(profile.sign(r)) for r in JointRole}),
        _offsets=_pairs(offsets or {}),
        _overrides=_pairs(overrides or {}),
    )
