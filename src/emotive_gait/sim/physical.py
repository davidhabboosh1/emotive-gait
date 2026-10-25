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
from emotive_gait.robot.root import RootMode, to_root_mode
from emotive_gait.sim.metrics import center_of_pressure, uprightness

__all__ = ["RootMode", "SimResult", "simulate", "to_root_mode"]

_FALL_UPRIGHTNESS = 0.6


@dataclass
class SimResult:
    """Per-physics-step records, truncated to the steps actually run."""

    fell: bool
    fell_at: float | None
    pelvis_z: np.ndarray  # (N,) world z of the pelvis body
    uprightness: np.ndarray  # (N,)
    cop: np.ndarray  # (N, 2) world xy; NaN rows where no foot contact


def simulate(
    robot: RobotModel, traj: GaitTrajectory, *, duration: float, ramp: float = 0.5
) -> SimResult:
    """Step ``traj`` open-loop for ``duration`` seconds from the standing pose.

    For ``t < ramp`` the control blends linearly from the standing pose to the trajectory target
    so the position servos are not yanked. A fall is ``uprightness < 0.6`` (never height), and
    stepping stops at the first one.
    """
    model = robot.model
    data = mujoco.MjData(model)
    data.qpos[:] = robot.stand_qpos
    mujoco.mj_forward(model, data)

    stand_ctrl = robot.stand_qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]].astype(float)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis < 0:
        raise ValueError("body 'pelvis' not found in model")

    n_steps = round(duration / model.opt.timestep)
    pelvis_z = np.empty(n_steps)
    upright = np.empty(n_steps)
    cop = np.full((n_steps, 2), np.nan)
    fell_at: float | None = None
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
        done = k + 1
        if upright[k] < _FALL_UPRIGHTNESS:
            fell_at = data.time
            break

    return SimResult(
        fell=fell_at is not None,
        fell_at=fell_at,
        pelvis_z=pelvis_z[:done],
        uprightness=upright[:done],
        cop=cop[:done],
    )
