"""Physical playback: step the trajectory through MuJoCo with position servos only.

There is deliberately no balance authority here. Open-loop stepping on a free base is an
unstable equilibrium, so the supported physics tier is the vertical-slide ``Z_RAIL`` root; the
root mode is fixed when the robot is loaded (``load_robot(root=...)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from emotive_gait.gait.footsteps import phase_at
from emotive_gait.gait.generator import GaitTrajectory, generate
from emotive_gait.gait.params import GaitParameters
from emotive_gait.robot.loader import RobotModel
from emotive_gait.robot.root import (
    ROOT_SLIDE_JOINT,
    ROOT_SLIDE_X_JOINT,
    RootMode,
    to_root_mode,
)
from emotive_gait.sim.metrics import (
    center_of_pressure,
    foot_contact_flags,
    uprightness,
    zmp_margin,
)

__all__ = [
    "GaitReport",
    "RootMode",
    "SimResult",
    "project_for_root",
    "simulate",
    "summarize",
    "to_root_mode",
]

_FALL_UPRIGHTNESS = 0.6
# Constrained-root (Z_RAIL / XZ_RAIL / WELD) failure criteria. On those roots the pelvis cannot rotate, so
# uprightness is floored at cos(0.52)^2 = 0.753 by the waist limits and can never cross 0.6.
_RAIL_STOP_MARGIN = 0.02  # m: slide joint this close to a range limit is resting on the stop
_CONTACT_WINDOW = 0.5  # s: sliding window for the foot-contact check
_MAX_NO_CONTACT_FRAC = 0.25  # fraction of a window's steps allowed with no foot on the floor
_MAX_PELVIS_DROP = 0.15  # m below the pelvis height at the end of the ramp


@dataclass
class SimResult:
    """Per-physics-step records, truncated to the steps actually run."""

    fell: bool
    fell_at: float | None
    pelvis_z: np.ndarray  # (N,) world z of the pelvis body
    uprightness: np.ndarray  # (N,)
    cop: np.ndarray  # (N, 2) world xy; NaN rows where no foot contact
    root_x: np.ndarray  # (N,) world x of the pelvis body
    played_params: GaitParameters  # the projected parameters actually simulated
    fail_reason: str | None = None  # which criterion fired; None when not fell
    zmp_margin: np.ndarray = field(default_factory=lambda: np.empty(0))  # (N,) m; NaN if None
    # (N, 2) horizontal site speed (m/s) of (left, right) foot, NaN when that foot has no contact
    foot_slip: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    # (N,) commanded pelvis height of the played trajectory at each sample time
    pelvis_z_cmd: np.ndarray = field(default_factory=lambda: np.empty(0))
    root_mode: RootMode = RootMode.FREE


def project_for_root(params: GaitParameters, mode: RootMode) -> GaitParameters:
    """Zero the parameters a constrained root cannot express; unchanged when nothing differs.

    ``FREE`` plays everything. ``XZ_RAIL`` is driven forward, so only ``lateral_sway_amp``
    is zeroed. ``Z_RAIL`` has no horizontal freedom, so ``step_length`` and
    ``lateral_sway_amp`` are zeroed. ``WELD`` also cannot move vertically, so
    ``com_bounce_amp`` is zeroed and ``com_height_scale`` forced to 1.0.
    """
    if mode is RootMode.FREE:
        return params
    changes: dict[str, float] = {"lateral_sway_amp": 0.0}
    if mode is not RootMode.XZ_RAIL:
        changes["step_length"] = 0.0
    if mode is RootMode.WELD:
        changes |= {"com_bounce_amp": 0.0, "com_height_scale": 1.0}
    if all(getattr(params, k) == v for k, v in changes.items()):
        return params
    return replace(params, **changes)


def _root_mode(model: mujoco.MjModel, pelvis: int) -> RootMode:
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ROOT_SLIDE_X_JOINT) >= 0:
        return RootMode.XZ_RAIL
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ROOT_SLIDE_JOINT) >= 0:
        return RootMode.Z_RAIL
    for j in range(
        model.body_jntadr[pelvis], model.body_jntadr[pelvis] + model.body_jntnum[pelvis]
    ):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return RootMode.FREE
    return RootMode.WELD


def simulate(
    robot: RobotModel, traj: GaitTrajectory, *, duration: float, ramp: float = 0.5
) -> SimResult:
    """Step ``traj`` open-loop for ``duration`` seconds from the standing pose.

    For ``t < ramp`` the control blends linearly from the standing pose to the trajectory target
    so the position servos are not yanked. Stepping stops at the first fall, and
    ``fail_reason`` names the criterion that fired. The criterion depends on the root mode
    (read from the compiled model):

    - ``FREE``: ``uprightness < 0.6`` (never height).
    - ``Z_RAIL`` / ``XZ_RAIL`` / ``WELD``: the pelvis cannot rotate, so uprightness depends only on the waist
      joints and their limits floor it above 0.6; it cannot signal a fall. A fall is instead ANY
      of: (a) ``"rail stop"``: the ``root_slide_z`` joint within 2 cm of either range limit
      (rails only); (b) ``"foot contact lost"``: no foot-floor contact (``center_of_pressure``
      is None) on more than 25% of the steps of any sliding 0.5 s window; (c) ``"pelvis drop"``:
      pelvis z more than 15 cm below its value at the end of the ramp (checked after the ramp).
    """
    model = robot.model
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis < 0:
        raise ValueError("body 'pelvis' not found in model")
    mode = _root_mode(model, pelvis)
    played = project_for_root(traj.params, mode)
    if played is not traj.params:
        traj = generate(played, robot, offsets=traj.offsets, overrides=traj.overrides)

    start_qpos = robot.stand_qpos
    if mode in (RootMode.WELD, RootMode.XZ_RAIL):
        # Start inside the gait, not from the straight-leg stand pose: a welded pelvis sits at the
        # nominal height where the stand pose would bury the feet, and on XZ_RAIL the pelvis sits
        # at the commanded height so the ramp blend is a near no-op.
        start_qpos = traj.qpos_at(robot, 0.0)
    if mode is RootMode.XZ_RAIL:
        if model.body_parentid[pelvis] == 0:  # direct child of the world on slide roots
            keyframe_z = float(model.body_pos[pelvis][2])
        else:
            probe = mujoco.MjData(model)
            probe.qpos[:] = robot.stand_qpos
            mujoco.mj_forward(model, probe)
            keyframe_z = float(probe.xpos[pelvis][2])
        z_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ROOT_SLIDE_JOINT)
        start_qpos[int(model.jnt_qposadr[z_joint])] = traj.root_offset_at(0.0)[2] - keyframe_z
    data = mujoco.MjData(model)
    data.qpos[:] = start_qpos
    mujoco.mj_forward(model, data)

    stand_ctrl = start_qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]].astype(float)
    slide_adr: int | None = None
    slide_lo = slide_hi = 0.0
    rail_x_adr = rail_x_dof = -1
    if mode is RootMode.XZ_RAIL:
        x_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ROOT_SLIDE_X_JOINT)
        rail_x_adr = int(model.jnt_qposadr[x_joint])
        rail_x_dof = int(model.jnt_dofadr[x_joint])
    if mode in (RootMode.Z_RAIL, RootMode.XZ_RAIL):
        slide = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ROOT_SLIDE_JOINT)
        slide_adr = int(model.jnt_qposadr[slide])
        slide_lo, slide_hi = (float(v) for v in model.jnt_range[slide])
    window = max(1, round(_CONTACT_WINDOW / model.opt.timestep))

    n_steps = round(duration / model.opt.timestep)
    pelvis_z = np.empty(n_steps)
    upright = np.empty(n_steps)
    cop = np.full((n_steps, 2), np.nan)
    root_x = np.empty(n_steps)
    margin = np.full(n_steps, np.nan)
    slip = np.full((n_steps, 2), np.nan)
    pelvis_cmd = np.empty(n_steps)
    foot_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in robot.profile.foot_sites
    ]
    if min(foot_ids) < 0:
        raise ValueError(f"foot sites {robot.profile.foot_sites} not found in model")
    prev_xy = data.site_xpos[foot_ids][:, :2].copy()
    no_contact = np.zeros(n_steps, dtype=bool)
    fell_at: float | None = None
    fail_reason: str | None = None
    ramp_z: float | None = None
    lost_in_window = 0
    done = 0
    for k in range(n_steps):
        t = data.time
        phase = phase_at(traj.params, t)
        target = traj.ctrl_at(robot, phase)
        w = t / ramp if t < ramp else 1.0
        if w < 1.0:
            data.ctrl[:] = stand_ctrl * (1.0 - w) + target * w
        else:
            data.ctrl[:] = target
        if rail_x_adr >= 0:
            # The IK legs assume the pelvis is exactly at speed * t, so x is prescribed, not servoed.
            data.qpos[rail_x_adr] = traj.root_x_at(t)
            data.qvel[rail_x_dof] = traj.params.speed
        mujoco.mj_step(model, data)

        pelvis_z[k] = data.xpos[pelvis][2]
        root_x[k] = data.xpos[pelvis][0]
        upright[k] = uprightness(model, data)
        c = center_of_pressure(model, data)
        if c is not None:
            cop[k] = c
        no_contact[k] = c is None
        m = zmp_margin(model, data)
        if m is not None:
            margin[k] = m
        xy = data.site_xpos[foot_ids][:, :2].copy()
        speed = np.linalg.norm(xy - prev_xy, axis=1) / model.opt.timestep
        prev_xy = xy
        slip[k] = np.where(foot_contact_flags(model, data), speed, np.nan)
        pelvis_cmd[k] = traj.root_offset_at(phase_at(traj.params, data.time))[2]
        done = k + 1

        if mode is RootMode.FREE:
            if upright[k] < _FALL_UPRIGHTNESS:
                fail_reason = f"uprightness {upright[k]:.3f} < {_FALL_UPRIGHTNESS}"
        else:
            lost_in_window += int(no_contact[k])
            if k >= window:
                lost_in_window -= int(no_contact[k - window])
            if slide_adr is not None:
                s = float(data.qpos[slide_adr])
                if s <= slide_lo + _RAIL_STOP_MARGIN or s >= slide_hi - _RAIL_STOP_MARGIN:
                    fail_reason = f"rail stop: root slide at {s:.3f} m"
            if fail_reason is None and k + 1 >= window:
                frac = lost_in_window / window
                if frac > _MAX_NO_CONTACT_FRAC:
                    fail_reason = f"foot contact lost on {frac:.0%} of a {_CONTACT_WINDOW} s window"
            if data.time >= ramp:
                if ramp_z is None:
                    ramp_z = float(pelvis_z[k])
                drop = ramp_z - float(pelvis_z[k])
                if fail_reason is None and drop > _MAX_PELVIS_DROP:
                    fail_reason = f"pelvis drop: {drop:.3f} m below its post-ramp height"
        if fail_reason is not None:
            fell_at = data.time
            break

    return SimResult(
        fell=fell_at is not None,
        fell_at=fell_at,
        pelvis_z=pelvis_z[:done],
        uprightness=upright[:done],
        cop=cop[:done],
        root_x=root_x[:done],
        played_params=played,
        fail_reason=fail_reason,
        zmp_margin=margin[:done],
        foot_slip=slip[:done],
        pelvis_z_cmd=pelvis_cmd[:done],
        root_mode=mode,
    )


@dataclass(frozen=True)
class GaitReport:
    """Physics-run summary over post-ramp steps; NaN where no such steps (or contacts) exist.

    ``pelvis_bob_m`` is the raw peak-to-peak pelvis height and includes the COMMANDED bounce
    (``com_bounce_amp`` is a rendered parameter and must never be presented as instability).
    ``pelvis_tracking_ptp_m`` is the peak-to-peak of pelvis height minus its commanded height:
    the uncommanded motion. On ``WELD`` the pelvis is held, so the ZMP numbers describe a held
    body (``pelvis_held``). Foot slip is the RMS horizontal foot-site speed over the steps where
    that foot is in contact.
    """

    zmp_outside_frac: float
    zmp_mean_margin_m: float
    foot_slip_rms_m_s: tuple[float, float]
    pelvis_bob_m: float
    pelvis_tracking_ptp_m: float
    pelvis_held: bool
    played_params: GaitParameters


def _nan_rms(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x**2))) if x.size else float("nan")


def summarize(res: SimResult, *, ramp: float = 0.5, timestep: float) -> GaitReport:
    """Reduce ``res`` to a :class:`GaitReport` over steps sampled at ``t >= ramp``."""
    t = (np.arange(len(res.pelvis_z)) + 1) * timestep
    post = t >= ramp - 1e-9
    margin = res.zmp_margin[post]
    margin = margin[np.isfinite(margin)]
    z = res.pelvis_z[post]
    resid = z - res.pelvis_z_cmd[post]
    return GaitReport(
        zmp_outside_frac=float(np.mean(margin < 0)) if margin.size else float("nan"),
        zmp_mean_margin_m=float(np.mean(margin)) if margin.size else float("nan"),
        foot_slip_rms_m_s=(_nan_rms(res.foot_slip[post, 0]), _nan_rms(res.foot_slip[post, 1])),
        pelvis_bob_m=float(np.ptp(z)) if z.size else float("nan"),
        pelvis_tracking_ptp_m=float(np.ptp(resid)) if resid.size else float("nan"),
        pelvis_held=res.root_mode is RootMode.WELD,
        played_params=res.played_params,
    )
