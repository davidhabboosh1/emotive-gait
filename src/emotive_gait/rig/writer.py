"""Write a generated trajectory into a BVH file's own channels (spec 4.7, the writer).

Per output frame at the source fps:

- every compiled bone's local rotation is ``R_parent^T @ R_bone`` of the MuJoCo ``xmat`` of the
  bodies carrying their frames (``CompiledRig.frame_of``), brought into the file's axes as
  ``P^T R P`` (``P = z_up_matrix(up_axis)``) and decomposed in the bone's channel order
  (``channel_angles``, ``euler_from_matrix`` for three axes); undriven channels come out at
  their reference values because they are baked into the body quaternions. The pelvis's parent
  frame is the world turned by the source's first-frame heading ``H``, so its world rotation is
  ``H @ xmat(carrier)``; when the pelvis is not the root its local rotation is that world
  rotation under its ancestors held at the reference;
- the ancestors of the pelvis and every bone the chain does not compile take their reference
  channel values; their position channels keep the source's first-frame values;
- the root's position channels put the pelvis at ``H @ (pelvis_x(t), sway, nominal *
  com_height_scale + bounce)`` plus the source's first-frame horizontal pelvis position and
  ``floor_z``, less the constant root-to-pelvis offset of the held ancestors (zero when the
  pelvis is the root), in file units and axes. The held row's zero root columns leave the
  root OFFSET standing, so the written channels carry it.
"""

from __future__ import annotations

import mujoco
import numpy as np

from emotive_gait.gait.footsteps import pelvis_x, phase_at
from emotive_gait.gait.generator import GaitTrajectory
from emotive_gait.gait.params import GaitParameters
from emotive_gait.mocap.extract import gait_timing
from emotive_gait.mocap.readers.bvh import BvhFile
from emotive_gait.mocap.skeleton import LandmarkMap, SkeletonClip
from emotive_gait.rig.calibration import ReferencePose
from emotive_gait.rig.compile import CompiledRig, channel_angles, heading_matrix, z_up_matrix
from emotive_gait.rig.errors import RigError
from emotive_gait.robot.loader import RobotModel

_AXIS = {"X": 0, "Y": 1, "Z": 2}


def _rotation_columns(file: BvhFile, bone: int) -> tuple[list[str], list[int]]:
    joint = file.joints[bone]
    channels = [c for c in joint.channels if c.endswith("rotation")]
    return channels, [file.channel_index(joint.name, c) for c in channels]


def _held_row(file: BvhFile, reference: ReferencePose) -> np.ndarray:
    """One motion row: every rotation channel at its reference value, every position channel
    at the source's first frame, the root's position channels at zero (which leaves the root
    OFFSET standing under the reader's replace-the-OFFSET rule)."""
    row = np.array(file.motion[0], dtype=float)
    for i, joint in enumerate(file.joints):
        channels, columns = _rotation_columns(file, i)
        if channels:
            row[columns] = np.degrees(channel_angles(reference.local_rotations[i], channels))
        if joint.parent < 0:
            for c in joint.channels:
                if c.endswith("position"):
                    row[file.channel_index(joint.name, c)] = 0.0
    return row


def _ancestors_world(file: BvhFile, reference: ReferencePose, pelvis: int) -> np.ndarray:
    """World rotation (file axes) of the pelvis's parent with the ancestors at the reference."""
    chain, i = [], file.joints[pelvis].parent
    while i >= 0:
        chain.append(i)
        i = file.joints[i].parent
    world = np.eye(3)
    for i in reversed(chain):
        world = world @ reference.local_rotations[i]
    return world


def write_trajectory(
    traj: GaitTrajectory,
    p1: GaitParameters,
    *,
    robot: RobotModel,
    compiled: CompiledRig,
    reference: ReferencePose,
    clip: SkeletonClip,
    landmarks: LandmarkMap,
    duration_s: float,
) -> BvhFile:
    """``traj`` (generated from ``p1`` on ``robot``, the ``compiled`` driven chain) written into
    ``compiled.file``'s channels: ``round(duration_s * fps)`` frames at the source frame time,
    the source hierarchy, channel order, units and up-axis. ``clip`` is the source's FK clip
    (first-frame pelvis position and heading)."""
    file = compiled.file
    model = robot.model
    data = mujoco.MjData(model)
    fps = file.fps
    n = max(1, round(duration_s * fps))
    times = np.arange(n) / fps

    names = [j.name for j in file.joints]
    pelvis = names.index(compiled.pelvis)
    p = z_up_matrix(file.up_axis)
    timing = gait_timing(clip, landmarks)
    heading = heading_matrix(timing.frame.forward[0])
    start = clip.positions[0, clip.joint_names.index(compiled.pelvis)]

    held = _held_row(file, reference)
    pelvis_held = file.with_motion(held[None, :]).to_clip().positions[0, pelvis]
    root = file.joints[0]
    root_position = {
        _AXIS[c[0]]: file.channel_index(root.name, c)
        for c in root.channels
        if c.endswith("position")
    }
    if len(root_position) != 3:
        raise RigError(f"root {root.name!r} needs X, Y and Z position channels to walk")
    ancestors = _ancestors_world(file, reference, pelvis)

    body = {bone: model.body(frame).id for bone, frame in compiled.frame_of.items()}
    xmat = np.empty((n, model.nbody, 3, 3))
    targets = np.empty((n, 3))
    for k, t in enumerate(times):
        phase = phase_at(p1, float(t))
        data.qpos[:] = traj.qpos_at(robot, phase)
        mujoco.mj_forward(model, data)
        xmat[k] = data.xmat.reshape(-1, 3, 3)
        offset = traj.root_offset_at(phase)
        local = np.array([pelvis_x(p1, float(t)), offset[1], offset[2]])
        targets[k] = heading @ local + np.array([start[0], start[1], reference.floor_z])

    motion = np.tile(held, (n, 1))
    # Root translation: the held ancestors carry the pelvis by a constant offset.
    # A position column replaces the OFFSET, and the held row's zero columns left it standing in
    # pelvis_held, so the channels carry the root OFFSET back.
    root_file = np.einsum("ij,tj->ti", p.T, targets - pelvis_held) / file.scale + np.asarray(
        root.offset, dtype=float
    )
    for axis, column in root_position.items():
        motion[:, column] = root_file[:, axis]

    for bone, frame_id in body.items():
        i = names.index(bone)
        channels, columns = _rotation_columns(file, i)
        if not channels:
            continue
        if i == pelvis:
            world_z = np.einsum("ij,tjk->tik", heading, xmat[:, frame_id])
            world_file = np.einsum("ji,tjk,kl->til", p, world_z, p)
            local = np.einsum("ji,tjk->tik", ancestors, world_file)
        else:
            parent = body[names[file.joints[i].parent]]
            local_z = np.einsum("tji,tjk->tik", xmat[:, parent], xmat[:, frame_id])
            local = np.einsum("ji,tjk,kl->til", p, local_z, p)
        motion[:, columns] = np.degrees([channel_angles(r, channels) for r in local])
    return file.with_motion(motion)
