"""Semantic joint roles and the per-robot mapping from roles to actuator names.

Emotion and gait code is written against :class:`JointRole`; each robot supplies a
:class:`RobotProfile` saying which actuator realises each role. This module is pure data and
deliberately does not import mujoco.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum


class JointRole(str, Enum):
    HIP_PITCH_L = "hip_pitch_l"
    HIP_PITCH_R = "hip_pitch_r"
    HIP_ROLL_L = "hip_roll_l"
    HIP_ROLL_R = "hip_roll_r"
    HIP_YAW_L = "hip_yaw_l"
    HIP_YAW_R = "hip_yaw_r"
    KNEE_L = "knee_l"
    KNEE_R = "knee_r"
    ANKLE_PITCH_L = "ankle_pitch_l"
    ANKLE_PITCH_R = "ankle_pitch_r"
    ANKLE_ROLL_L = "ankle_roll_l"
    ANKLE_ROLL_R = "ankle_roll_r"
    WAIST_YAW = "waist_yaw"
    WAIST_ROLL = "waist_roll"
    WAIST_PITCH = "waist_pitch"
    SHOULDER_PITCH_L = "shoulder_pitch_l"
    SHOULDER_PITCH_R = "shoulder_pitch_r"
    SHOULDER_ROLL_L = "shoulder_roll_l"
    SHOULDER_ROLL_R = "shoulder_roll_r"
    SHOULDER_YAW_L = "shoulder_yaw_l"
    SHOULDER_YAW_R = "shoulder_yaw_r"
    ELBOW_L = "elbow_l"
    ELBOW_R = "elbow_r"
    NECK_PITCH = "neck_pitch"

    def mirror(self) -> JointRole:
        """Return the opposite-side role, or ``self`` for midline roles."""
        name = self.name
        if name.endswith("_L"):
            return JointRole[name[:-2] + "_R"]
        if name.endswith("_R"):
            return JointRole[name[:-2] + "_L"]
        return self


class MissingRoleError(KeyError):
    """Raised when a profile has no actuator for a requested role."""

    def __init__(self, role: JointRole, profile_name: str) -> None:
        super().__init__(f"robot profile {profile_name!r} has no actuator for role {role.name}")
        self.role = role
        self.profile_name = profile_name


@dataclass(frozen=True)
class RobotProfile:
    """Describes how a specific robot realises each :class:`JointRole`.

    ``sign_for`` carries a per-role multiplier for robots whose joint axes are mirrored
    (e.g. left/right roll joints), so symmetric modulations can negate rather than copy.
    """

    name: str
    scene_entry: str
    actuator_for: Mapping[JointRole, str]
    sign_for: Mapping[JointRole, float]

    def has(self, role: JointRole) -> bool:
        return role in self.actuator_for

    def missing(self, roles: Iterable[JointRole]) -> tuple[JointRole, ...]:
        return tuple(r for r in roles if r not in self.actuator_for)

    def actuator(self, role: JointRole) -> str:
        try:
            return self.actuator_for[role]
        except KeyError:
            raise MissingRoleError(role, self.name) from None
