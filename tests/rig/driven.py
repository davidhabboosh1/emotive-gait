"""Test helpers: a synthetic rig carried through map, reference, bind, ``compile_driven`` and
``rig_robot``, and a probe of one role's hinge on the driven chain."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from emotive_gait.mocap.skeleton import Landmark
from emotive_gait.rig.calibration import ReferencePose, reference_pose
from emotive_gait.rig.compile import CompiledRig, compile_driven, compile_full
from emotive_gait.rig.mapping import RigMapping, bind_roles, map_landmarks
from emotive_gait.rig.profile import rig_profile, rig_robot
from emotive_gait.robot.conventions import PROBE_RAD
from emotive_gait.robot.loader import RobotModel
from emotive_gait.robot.roles import JointRole
from rig.rigs import RigSpec, walk_and_clip


def spec_id(spec: RigSpec) -> str:
    default = RigSpec()
    changed = [f"{k}={v}" for k, v in vars(spec).items() if getattr(default, k) != v]
    return "-".join(changed) or "default"


def driven_rig(
    spec: RigSpec, directory: Path
) -> tuple[RobotModel, RigMapping, ReferencePose, CompiledRig, CompiledRig]:
    """``(robot, mapping, reference, driven, full)`` for ``spec`` walked in ``directory``."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    f, clip = walk_and_clip(spec, directory)
    landmarks = map_landmarks(f, clip)
    reference = reference_pose(f, clip, landmarks.landmarks)
    full = compile_full(f, pelvis=landmarks.landmarks[Landmark.PELVIS], pelvis_height=1.0)
    mapping = bind_roles(full, reference, landmarks)
    driven = compile_driven(f, mapping, reference)
    profile = rig_profile(f"synthetic-{spec_id(spec)}", mapping, reference, posture_map={})
    return rig_robot(driven, profile), mapping, reference, driven, full


def landmark_points(rig: CompiledRig, mapping: RigMapping, data: mujoco.MjData) -> dict:
    """Landmark value -> world point of its site."""
    return {
        lm.value: data.site_xpos[rig.model.site(rig.landmark_sites[bone]).id].copy()
        for lm, bone in mapping.landmarks.items()
    }


def probe_driven(
    robot: RobotModel, driven: CompiledRig, mapping: RigMapping, role: JointRole
) -> tuple[dict, dict]:
    """Landmark points at qpos zero, then with ``role``'s hinge at ``+PROBE_RAD``."""
    data = mujoco.MjData(robot.model)
    data.qpos[:] = 0.0
    mujoco.mj_forward(robot.model, data)
    before = landmark_points(driven, mapping, data)
    data.qpos[robot.qpos_adr(role)] = PROBE_RAD
    mujoco.mj_forward(robot.model, data)
    return before, landmark_points(driven, mapping, data)


X, Y, Z = np.eye(3)
