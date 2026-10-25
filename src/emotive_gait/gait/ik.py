"""Task-space leg IK: solve one gait cycle of foot targets for joint angles with mink.

Targets come from ``gait.footsteps`` in the "treadmill" convention: feet are placed in the pelvis
frame, so the solve runs on a WELD-rooted robot whose pelvis never moves. The target of each foot
site, relative to the pelvis origin, is ``(foot.x, +-feet_spacing/2 - dy, floor - (nominal *
com_height_scale + dz) + foot.z)``: lateral sway and vertical bounce of the pelvis appear as the
opposite displacement of the feet. The target orientation is the orientation of the site at the
seed pose (a level sole): level in stance; swing pitch is emergent, not modelled. Orientation is
a hard task in stance (``STANCE_ORIENTATION_COST``) and a soft preference in swing
(``SWING_ORIENTATION_COST``), so a lifted foot behind the body may plantarflex as at toe-off
instead of driving the ankle into its dorsiflexion limit (rulings R7/R9: a level lifted sole
behind the body exceeds the G1 ankle range, so swing tilt is traded for foot position).

The seed is ``SEED_LEG_POSE`` (bent hips, knees and ankles), not the stand keyframe: the stand
pose has straight knees, a singular configuration from which the differential QP cannot flex the
knee. Two laps are solved with the configuration carried across; the second, warm-started lap is
kept, so the cycle closes on itself. The feasibility report is measured afterwards with
``mj_forward`` on the kept angles, never read from the solver.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import mink
import mujoco
import numpy as np

from emotive_gait.gait.footsteps import Side, foot_point, pelvis_offset
from emotive_gait.gait.params import GaitParameters
from emotive_gait.robot.loader import RobotModel, load_robot
from emotive_gait.robot.roles import JointRole, RobotProfile
from emotive_gait.robot.root import RootMode

QP_SOLVER = "daqp"
N_SAMPLES = 120
ITERATIONS = 8
IK_DT = 0.05
POSITION_TOL_M = 0.002
ORIENTATION_TOL_RAD = 0.01
LIMIT_MARGIN_RAD = 1e-3
POSITION_COST = 10.0
STANCE_ORIENTATION_COST = 5.0  # hard: the sole is level in stance
SWING_ORIENTATION_COST = 0.1  # soft: swing pitch is emergent, not modelled

SEED_LEG_POSE: Mapping[JointRole, float] = MappingProxyType(
    {
        JointRole.HIP_PITCH_L: -0.2,
        JointRole.HIP_PITCH_R: -0.2,
        JointRole.KNEE_L: 0.4,
        JointRole.KNEE_R: 0.4,
        JointRole.ANKLE_PITCH_L: -0.2,
        JointRole.ANKLE_PITCH_R: -0.2,
    }
)

LEG_ROLES: tuple[JointRole, ...] = (
    JointRole.HIP_PITCH_L,
    JointRole.HIP_ROLL_L,
    JointRole.HIP_YAW_L,
    JointRole.KNEE_L,
    JointRole.ANKLE_PITCH_L,
    JointRole.ANKLE_ROLL_L,
    JointRole.HIP_PITCH_R,
    JointRole.HIP_ROLL_R,
    JointRole.HIP_YAW_R,
    JointRole.KNEE_R,
    JointRole.ANKLE_PITCH_R,
    JointRole.ANKLE_ROLL_R,
)

_PELVIS = "pelvis"
_SIDES = ((Side.LEFT, 1.0), (Side.RIGHT, -1.0))  # order matches RobotProfile.foot_sites


@dataclass(frozen=True)
class FeasibilityReport:
    """Measured after the solve. Position error is over all samples; orientation error is over
    stance samples only (level in stance; swing pitch is emergent, not modelled)."""

    max_foot_position_error_m: float
    max_foot_orientation_error_rad: float
    roles_at_limit: tuple[JointRole, ...]
    worst_phase: float  # phase of the sample with the largest foot position error

    @property
    def feasible(self) -> bool:
        return (
            self.max_foot_position_error_m <= POSITION_TOL_M
            and self.max_foot_orientation_error_rad <= ORIENTATION_TOL_RAD
        )


@dataclass(frozen=True)
class LegCycle:
    phases: np.ndarray  # (n,), k / n, read-only
    angles: Mapping[JointRole, np.ndarray]  # (n,) per leg role, read-only
    report: FeasibilityReport


_WELD_CACHE: dict[tuple[str, bool], RobotModel] = {}


def weld_robot_for(profile: RobotProfile) -> RobotModel:
    """The WELD-rooted robot for ``profile``, built once per ``(profile.name, has neck)``.

    A dict rather than ``functools.cache``: profiles hold dicts and are not hashable.
    """
    with_neck = profile.has(JointRole.NECK_PITCH)
    key = (profile.name, with_neck)
    robot = _WELD_CACHE.get(key)
    if robot is None:
        robot = load_robot(profile, with_neck=with_neck, root=RootMode.WELD)
        _WELD_CACHE[key] = robot
    return robot


def _is_welded(model: mujoco.MjModel) -> bool:
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _PELVIS)
    return pelvis >= 0 and int(model.body_jntnum[pelvis]) == 0


def _rotation_error(target: np.ndarray, actual: np.ndarray) -> float:
    """Angle (rad) of the rotation taking ``target`` to ``actual`` (both 3x3)."""
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, (target.T @ actual).ravel())
    return 2.0 * math.atan2(float(np.linalg.norm(quat[1:])), abs(float(quat[0])))


def solve_leg_cycle(
    params: GaitParameters,
    robot: RobotModel,
    *,
    n_samples: int = N_SAMPLES,
    iterations: int = ITERATIONS,
) -> LegCycle:
    """Solve the leg joint angles of one gait cycle at phases ``k / n_samples``."""
    model = robot.model
    if not _is_welded(model):
        raise ValueError("solve_leg_cycle needs a WELD-rooted robot (see weld_robot_for)")
    profile = robot.profile
    adr = {role: robot.qpos_adr(role) for role in LEG_ROLES}

    seed = np.array(robot.stand_qpos, dtype=float)
    for role, value in SEED_LEG_POSE.items():
        seed[robot.qpos_adr(role)] = value

    scratch = mujoco.MjData(model)
    scratch.qpos[:] = seed
    mujoco.mj_forward(model, scratch)
    pelvis_pos = scratch.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _PELVIS)].copy()
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in profile.foot_sites]
    level = [scratch.site_xmat[i].reshape(3, 3).copy() for i in site_ids]

    phases = np.arange(n_samples) / n_samples
    targets = np.empty((n_samples, 2, 3))
    stance = np.empty((n_samples, 2), dtype=bool)
    for k, phase in enumerate(phases):
        dy, dz = pelvis_offset(params, float(phase))
        drop = profile.nominal_pelvis_height * params.com_height_scale + dz
        for i, (side, sign) in enumerate(_SIDES):
            foot = foot_point(params, float(phase), side)
            stance[k, i] = foot.in_stance
            targets[k, i] = pelvis_pos + (
                foot.x,
                sign * params.feet_spacing / 2.0 - dy,
                profile.foot_site_floor_height - drop + foot.z,
            )

    cfg = mink.Configuration(model)
    cfg.update(seed)
    feet = [
        mink.FrameTask(
            s, "site", position_cost=POSITION_COST, orientation_cost=STANCE_ORIENTATION_COST
        )
        for s in profile.foot_sites
    ]
    posture = mink.PostureTask(model, cost=1e-3)
    posture.set_target(seed)
    tasks = [*feet, posture]
    limits = [mink.ConfigurationLimit(model)]
    rotations = [mink.SO3.from_matrix(r) for r in level]

    kept = np.empty((n_samples, model.nq))
    for _lap in range(2):
        for k in range(n_samples):
            for i, task in enumerate(feet):
                task.set_target(mink.SE3.from_rotation_and_translation(rotations[i], targets[k, i]))
                task.set_orientation_cost(
                    STANCE_ORIENTATION_COST if stance[k, i] else SWING_ORIENTATION_COST
                )
            for _ in range(iterations):
                v = mink.solve_ik(cfg, tasks, IK_DT, QP_SOLVER, damping=1e-4, limits=limits)
                cfg.integrate_inplace(v, IK_DT)
            kept[k] = cfg.q

    pos_err = np.zeros(n_samples)
    ori_err = 0.0
    at_limit: set[JointRole] = set()
    ranges = {role: robot.joint_range(role) for role in LEG_ROLES}
    for k in range(n_samples):
        scratch.qpos[:] = kept[k]
        mujoco.mj_forward(model, scratch)
        for i, sid in enumerate(site_ids):
            err = float(np.linalg.norm(scratch.site_xpos[sid] - targets[k, i]))
            pos_err[k] = max(pos_err[k], err)
            actual = scratch.site_xmat[sid].reshape(3, 3)
            if stance[k, i]:
                ori_err = max(ori_err, _rotation_error(level[i], actual))
        for role in LEG_ROLES:
            lo, hi = ranges[role]
            q = kept[k, adr[role]]
            if q - lo <= LIMIT_MARGIN_RAD or hi - q <= LIMIT_MARGIN_RAD:
                at_limit.add(role)

    worst = int(np.argmax(pos_err))
    report = FeasibilityReport(
        max_foot_position_error_m=float(pos_err[worst]),
        max_foot_orientation_error_rad=ori_err,
        roles_at_limit=tuple(r for r in LEG_ROLES if r in at_limit),
        worst_phase=float(phases[worst]),
    )
    angles: dict[JointRole, np.ndarray] = {}
    for role in LEG_ROLES:
        column = kept[:, adr[role]].copy()
        column.flags.writeable = False
        angles[role] = column
    phases.flags.writeable = False
    return LegCycle(phases=phases, angles=MappingProxyType(angles), report=report)
