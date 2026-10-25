"""Kinematic playback: pose the robot from the trajectory with no physics.

This is the stimulus source for perception work. It only ever calls ``mj_forward``, so
``data.time`` stays 0.0 and the commanded angles are reproduced exactly.
"""

from __future__ import annotations

from collections.abc import Iterator

import mujoco

from emotive_gait.gait.generator import GaitTrajectory
from emotive_gait.robot.loader import RobotModel


def play_kinematic(
    robot: RobotModel, traj: GaitTrajectory, *, duration: float, fps: float
) -> Iterator[mujoco.MjData]:
    """Yield ``int(duration * fps)`` posed frames of ``traj``.

    The same ``MjData`` object is yielded every time and mutated in place, so the consumer must
    read (or copy) it before advancing the iterator. One gait cycle is two steps, so the phase
    advances by ``cadence / 2`` per second.
    """
    model = robot.model
    data = mujoco.MjData(model)
    for k in range(int(duration * fps)):
        phase = ((k / fps) * traj.params.cadence / 2.0) % 1.0
        data.qpos[:] = traj.qpos_at(robot, phase)
        mujoco.mj_forward(model, data)
        yield data
