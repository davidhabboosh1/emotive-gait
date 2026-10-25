"""Unitree G1 (MuJoCo Menagerie ``unitree_g1``) role mapping.

Every G1 actuator carries the same name as the joint it drives. The model has no neck, so
``NECK_PITCH`` is deliberately unmapped here.
"""

from __future__ import annotations

from emotive_gait.robot.roles import JointRole, RobotProfile

_ACTUATORS = {
    JointRole.HIP_PITCH_L: "left_hip_pitch_joint",
    JointRole.HIP_PITCH_R: "right_hip_pitch_joint",
    JointRole.HIP_ROLL_L: "left_hip_roll_joint",
    JointRole.HIP_ROLL_R: "right_hip_roll_joint",
    JointRole.HIP_YAW_L: "left_hip_yaw_joint",
    JointRole.HIP_YAW_R: "right_hip_yaw_joint",
    JointRole.KNEE_L: "left_knee_joint",
    JointRole.KNEE_R: "right_knee_joint",
    JointRole.ANKLE_PITCH_L: "left_ankle_pitch_joint",
    JointRole.ANKLE_PITCH_R: "right_ankle_pitch_joint",
    JointRole.ANKLE_ROLL_L: "left_ankle_roll_joint",
    JointRole.ANKLE_ROLL_R: "right_ankle_roll_joint",
    JointRole.WAIST_YAW: "waist_yaw_joint",
    JointRole.WAIST_ROLL: "waist_roll_joint",
    JointRole.WAIST_PITCH: "waist_pitch_joint",
    JointRole.SHOULDER_PITCH_L: "left_shoulder_pitch_joint",
    JointRole.SHOULDER_PITCH_R: "right_shoulder_pitch_joint",
    JointRole.SHOULDER_ROLL_L: "left_shoulder_roll_joint",
    JointRole.SHOULDER_ROLL_R: "right_shoulder_roll_joint",
    JointRole.SHOULDER_YAW_L: "left_shoulder_yaw_joint",
    JointRole.SHOULDER_YAW_R: "right_shoulder_yaw_joint",
    JointRole.ELBOW_L: "left_elbow_joint",
    JointRole.ELBOW_R: "right_elbow_joint",
}

G1_PROFILE = RobotProfile(
    name="unitree_g1",
    scene_entry="scene",
    actuator_for=_ACTUATORS,
    sign_for={role: -1.0 if role.name.endswith("_ROLL_R") else 1.0 for role in _ACTUATORS},
)
