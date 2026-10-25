"""Load a robot from MuJoCo Menagerie and resolve semantic roles against the compiled model."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import mujoco_menagerie
import numpy as np

from emotive_gait.robot.neck import build_with_neck, standing_qpos
from emotive_gait.robot.profiles.g1 import G1_PROFILE, G1_PROFILE_WITH_NECK
from emotive_gait.robot.roles import JointRole, RobotProfile
from emotive_gait.robot.root import RootMode

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


def load_robot(
    profile: RobotProfile | None = None,
    *,
    with_neck: bool = True,
    root: RootMode = RootMode.FREE,
) -> RobotModel:
    """Load ``profile``'s robot (default: G1) and wrap it with role lookups.

    With ``with_neck`` the robot gets an injected pitch neck; ``profile=None`` then resolves to
    ``G1_PROFILE_WITH_NECK``, otherwise to ``G1_PROFILE``. ``root`` constrains the pelvis (see
    ``robot.root``); non-FREE roots need the neck path, which is the only one built from a spec.
    """
    if root is not RootMode.FREE and not with_neck:
        raise ValueError("non-FREE root modes require with_neck=True in this version")
    if profile is None:
        profile = G1_PROFILE_WITH_NECK if with_neck else G1_PROFILE
    try:
        robot = mujoco_menagerie.get(profile.name)
        model = robot.model(profile.scene_entry)
        entry_xml = str(robot.xml(profile.scene_entry))
    except (OSError, TypeError, mujoco_menagerie.MenagerieError) as exc:
        raise ModelUnavailableError(
            f"Could not load robot model {profile.name!r}: {exc}. "
            f"Fetch it with `python -m mujoco_menagerie prefetch {profile.name}`; "
            "the first download requires Python >= 3.11.4 (3.12 recommended) because "
            "mujoco-menagerie extracts archives with tarfile's `filter=` argument. "
            "If prefetch is not possible, clone the model directly:\n"
            "  git clone --filter=blob:none --no-checkout --depth 1 "
            "https://github.com/google-deepmind/mujoco_menagerie.git _menagerie\n"
            f"  cd _menagerie && git sparse-checkout set {profile.name} && git checkout\n"
            "then point mujoco-menagerie at the clone:\n"
            "  set MENAGERIE_ROOT=<absolute path to _menagerie>   (Windows)\n"
            "  export MENAGERIE_ROOT=<absolute path to _menagerie> (POSIX)"
        ) from exc
    if with_neck:
        neck_model = build_with_neck(entry_xml, root=root)
        return RobotModel(
            model=neck_model, profile=profile, stand_qpos=standing_qpos(model, neck_model)
        )
    return RobotModel(model=model, profile=profile, stand_qpos=_stand_qpos(model))
