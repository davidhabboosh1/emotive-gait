"""Test helpers: a synthetic rig carried through map, reference, bind, ``compile_driven`` and
``rig_robot``, a probe of one role's hinge on the driven chain, and the posture calibration
(``posture_map``) with an independent static posture reading."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from emotive_gait.mocap.events import PathFrame
from emotive_gait.mocap.extract import posture_angles
from emotive_gait.mocap.skeleton import Landmark
from emotive_gait.rig.calibration import (
    ReferencePose,
    landmark_site_names,
    posture_map,
    reference_pose,
)
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


def calibrated_rig(
    spec: RigSpec, directory: Path
) -> tuple[RobotModel, RigMapping, ReferencePose, dict, dict[str, float]]:
    """``(robot, mapping, reference, posture_map rows, p_ref)``: ``driven_rig`` with the
    measured posture map in its profile."""
    robot, mapping, reference, driven, _ = driven_rig(spec, directory)
    rows, p_ref = posture_map(
        robot, mapping, landmark_sites=landmark_site_names(driven, mapping.landmarks)
    )
    profile = rig_profile(f"synthetic-{spec_id(spec)}", mapping, reference, posture_map=rows)
    return rig_robot(driven, profile), mapping, reference, rows, p_ref


def hinges_from_posture(robot: RobotModel, rows: dict, posture: dict[str, float]) -> np.ndarray:
    """qpos zero (the reference) with each mapped posture role at
    ``offset + sum(coef * posture[k])``, as the generator's ``_procedural`` writes it."""
    qpos = np.zeros(robot.model.nq)
    for role, row in rows.items():
        value = row["offset"] + sum(c * posture[k] for k, c in row.items() if k != "offset")
        qpos[robot.qpos_adr(role)] = value
    return qpos


def static_posture(robot: RobotModel, qpos: np.ndarray, mapping: RigMapping) -> dict[str, float]:
    """The extractor's static posture reading of the driven chain at ``qpos``, in a path frame
    with forward +x, left +y (the driven chain's reference heading is +x). Sites are read by
    the clip joint name, which is how ``compile_driven`` names them; a landmark that maps to
    several joints is their mean."""
    data = mujoco.MjData(robot.model)
    data.qpos[:] = qpos
    mujoco.mj_forward(robot.model, data)
    tracks = {}
    for lm, bones in mapping.landmarks.items():
        names = (bones,) if isinstance(bones, str) else tuple(bones)
        points = [data.site_xpos[robot.model.site(n).id] for n in names]
        tracks[lm] = np.mean(points, axis=0)[None, :]
    frame = PathFrame(
        origin=tracks[Landmark.PELVIS],
        forward=X[None, :],
        left=Y[None, :],
        turning=np.zeros(1, dtype=bool),
    )
    return posture_angles(tracks, frame)


X, Y, Z = np.eye(3)
