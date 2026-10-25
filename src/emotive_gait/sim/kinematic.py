"""Kinematic playback: pose the robot from the trajectory with no physics.

This is the stimulus source for perception work. It only ever calls ``mj_forward``, so
``data.time`` stays 0.0 and the commanded angles are reproduced exactly.
"""

from __future__ import annotations

from collections.abc import Iterator

import mujoco
import numpy as np

from emotive_gait.gait.footsteps import phase_at
from emotive_gait.gait.generator import GaitTrajectory
from emotive_gait.robot.loader import RobotModel


def play_kinematic(
    robot: RobotModel, traj: GaitTrajectory, *, duration: float, fps: float
) -> Iterator[mujoco.MjData]:
    """Yield ``int(duration * fps)`` posed frames of ``traj``, walking forward in the world.

    The root must be FREE: the pelvis is placed at ``(root_x_at(t), sway, height)`` and upright,
    so the treadmill-style stance foot is stationary on the floor. The same ``MjData`` object is
    yielded every time and mutated in place, so the consumer must read (or copy) it before
    advancing the iterator.
    """
    model = robot.model
    if model.njnt == 0 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("kinematic playback needs a FREE root; load the robot with RootMode.FREE")
    data = mujoco.MjData(model)
    for k in range(int(duration * fps)):
        t = k / fps
        phase = phase_at(traj.params, t)
        qpos = np.array(traj.qpos_at(robot, phase), dtype=float)
        qpos[0:3] = (traj.root_x_at(t), *traj.root_offset_at(phase)[1:])
        qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        yield data
