"""Physical playback: step the trajectory through MuJoCo with position servos only.

There is deliberately no balance authority here. Open-loop stepping on a free base is an
unstable equilibrium, so the supported physics tier is the vertical-slide ``Z_RAIL`` root; the
root mode is fixed when the robot is loaded (``load_robot(root=...)``).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from emotive_gait.gait.generator import GaitTrajectory
from emotive_gait.robot.loader import RobotModel
from emotive_gait.robot.root import ROOT_SLIDE_JOINT, RootMode, to_root_mode
from emotive_gait.sim.metrics import center_of_pressure, uprightness

__all__ = ["RootMode", "SimResult", "simulate", "to_root_mode"]

_FALL_UPRIGHTNESS = 0.6
# Constrained-root (Z_RAIL / WELD) failure criteria. On those roots the pelvis cannot rotate, so
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
    fail_reason: str | None = None  # which criterion fired; None when not fell


def _root_mode(model: mujoco.MjModel, pelvis: int) -> RootMode:
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
    - ``Z_RAIL`` / ``WELD``: the pelvis cannot rotate, so uprightness depends only on the waist
      joints and their limits floor it above 0.6; it cannot signal a fall. A fall is instead ANY
      of: (a) ``"rail stop"``: the ``root_slide_z`` joint within 2 cm of either range limit
      (Z_RAIL only); (b) ``"foot contact lost"``: no foot-floor contact (``center_of_pressure``
      is None) on more than 25% of the steps of any sliding 0.5 s window; (c) ``"pelvis drop"``:
      pelvis z more than 15 cm below its value at the end of the ramp (checked after the ramp).
    """
    model = robot.model
    data = mujoco.MjData(model)
    data.qpos[:] = robot.stand_qpos
    mujoco.mj_forward(model, data)

    stand_ctrl = robot.stand_qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]].astype(float)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis < 0:
        raise ValueError("body 'pelvis' not found in model")
    mode = _root_mode(model, pelvis)
    slide_adr: int | None = None
    slide_lo = slide_hi = 0.0
    if mode is RootMode.Z_RAIL:
        slide = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ROOT_SLIDE_JOINT)
        slide_adr = int(model.jnt_qposadr[slide])
        slide_lo, slide_hi = (float(v) for v in model.jnt_range[slide])
    window = max(1, round(_CONTACT_WINDOW / model.opt.timestep))

    n_steps = round(duration / model.opt.timestep)
    pelvis_z = np.empty(n_steps)
    upright = np.empty(n_steps)
    cop = np.full((n_steps, 2), np.nan)
    no_contact = np.zeros(n_steps, dtype=bool)
    fell_at: float | None = None
    fail_reason: str | None = None
    ramp_z: float | None = None
    lost_in_window = 0
    done = 0
    for k in range(n_steps):
        t = data.time
        phase = (t * traj.params.cadence / 2.0) % 1.0
        target = traj.ctrl_at(robot, phase)
        if t < ramp:
            w = t / ramp
            data.ctrl[:] = stand_ctrl * (1.0 - w) + target * w
        else:
            data.ctrl[:] = target
        mujoco.mj_step(model, data)

        pelvis_z[k] = data.xpos[pelvis][2]
        upright[k] = uprightness(model, data)
        c = center_of_pressure(model, data)
        if c is not None:
            cop[k] = c
        no_contact[k] = c is None
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
        fail_reason=fail_reason,
    )
