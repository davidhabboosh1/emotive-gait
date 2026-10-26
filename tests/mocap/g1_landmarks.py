"""G1 landmark map and a recorder that turns the kinematic tier into a ``SkeletonClip``."""

from __future__ import annotations

import numpy as np

from emotive_gait.gait.generator import GaitTrajectory
from emotive_gait.mocap.skeleton import Landmark, LandmarkMap, SkeletonClip
from emotive_gait.robot.loader import RobotModel
from emotive_gait.sim.kinematic import play_kinematic

HEAD_TOP = "head_top"  # synthetic: 0.10 m along head_link's local +z
HEAD_TOP_OFFSET = np.array([0.0, 0.0, 0.10])

G1_LANDMARKS: LandmarkMap = {
    Landmark.PELVIS: "pelvis",
    Landmark.HIP_L: "left_hip_pitch_link",
    Landmark.HIP_R: "right_hip_pitch_link",
    Landmark.KNEE_L: "left_knee_link",
    Landmark.KNEE_R: "right_knee_link",
    Landmark.ANKLE_L: "left_ankle_pitch_link",
    Landmark.ANKLE_R: "right_ankle_pitch_link",
    Landmark.SHOULDER_L: "left_shoulder_roll_link",
    Landmark.SHOULDER_R: "right_shoulder_roll_link",
    Landmark.ELBOW_L: "left_elbow_link",
    Landmark.ELBOW_R: "right_elbow_link",
    Landmark.WRIST_L: "left_wrist_roll_link",
    Landmark.WRIST_R: "right_wrist_roll_link",
    Landmark.NECK: "head_link",  # the neck joint sits at the head body origin
    Landmark.HEAD: HEAD_TOP,
}

_BODIES = tuple(name for name in G1_LANDMARKS.values() if name != HEAD_TOP)


def record_clip(
    robot: RobotModel, traj: GaitTrajectory, *, duration: float, fps: float
) -> SkeletonClip:
    """Play ``traj`` kinematically and stack the landmark bodies' world positions per frame."""
    model = robot.model
    ids = [model.body(name).id for name in _BODIES]
    head = model.body("head_link").id
    frames = []
    for data in play_kinematic(robot, traj, duration=duration, fps=fps):
        top = data.xpos[head] + data.xmat[head].reshape(3, 3) @ HEAD_TOP_OFFSET
        frames.append(np.vstack([data.xpos[ids], top]))
    return SkeletonClip(
        joint_names=(*_BODIES, HEAD_TOP),
        parents=None,
        positions=np.array(frames),
        fps=float(fps),
        source=f"g1-kinematic:{traj.params}",
    )
