"""Load a robot from MuJoCo Menagerie and resolve semantic roles against the compiled model."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import mujoco_menagerie
import numpy as np

from emotive_gait.robot.profiles.g1 import G1_PROFILE
from emotive_gait.robot.roles import JointRole, RobotProfile

_STAND_KEYFRAME = "stand"


class ModelUnavailableError(RuntimeError):
    """The robot model could not be fetched or extracted."""


@dataclass
class RobotModel:
    model: mujoco.MjModel
    profile: RobotProfile
    stand_qpos: np.ndarray

    def ctrl_index(self, role: JointRole) -> int:
        name = self.profile.actuator(role)
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if idx < 0:
            raise KeyError(f"actuator {name!r} not found in model for role {role.name}")
        return idx

    def _joint_id(self, role: JointRole) -> int:
        # An actuator's transmission target is the joint it drives.
        return int(self.model.actuator_trnid[self.ctrl_index(role), 0])

    def qpos_adr(self, role: JointRole) -> int:
        return int(self.model.jnt_qposadr[self._joint_id(role)])

    def joint_range(self, role: JointRole) -> tuple[float, float]:
        lo, hi = self.model.jnt_range[self._joint_id(role)]
        return float(lo), float(hi)


def _stand_qpos(model: mujoco.MjModel) -> np.ndarray:
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, _STAND_KEYFRAME)
    if key >= 0:
        return np.array(model.key_qpos[key], dtype=float)
    return np.array(model.qpos0, dtype=float)


def load_robot(profile: RobotProfile | None = None, *, with_neck: bool = False) -> RobotModel:
    """Load ``profile``'s robot (default: G1) and wrap it with role lookups."""
    if with_neck:
        raise NotImplementedError("with_neck=True is provided by Task 5 (neck injection)")
    if profile is None:
        profile = G1_PROFILE
    try:
        model = mujoco_menagerie.get(profile.name).model(profile.scene_entry)
    except (OSError, TypeError, mujoco_menagerie.MenagerieError) as exc:
        raise ModelUnavailableError(
            f"Could not load robot model {profile.name!r}: {exc}. "
            f"Fetch it with `python -m mujoco_menagerie prefetch {profile.name}`; "
            "the first download requires Python >= 3.11.4 (3.12 recommended) because "
            "mujoco-menagerie extracts archives with tarfile's `filter=` argument."
        ) from exc
    return RobotModel(model=model, profile=profile, stand_qpos=_stand_qpos(model))
