"""The actor's reference pose (the calibrated zero) and the hinge probes that bind roles.

Reference pose (spec 4.3): for every bone, the chordal mean of its local rotation over the
usable cycle frames (the union of the cycles ``extract_cycles`` keeps): the rotation matrices
are averaged and projected back onto SO(3) with an SVD, ``U @ diag(1, 1, det(U Vt)) @ Vt``;
Euler channels are never averaged. The pelvis uses its WORLD rotation (its ancestors composed,
so a rotating reference bone above it is absorbed) after removing each frame's path-frame
heading, so it carries the tilt and no yaw. The two ankle bones are the exception: their
reference (``foot_stance_rotations``) gives the foot its stance WORLD orientation, so "level"
is the foot as it rests on the floor. It is the chordal mean over the stance frames of the
foot's heading-removed world rotation, re-expressed through the reference shank (the ankle's
parent at the reference pose), so the chain at the reference holds the foot exactly as the
actor's feet sit in stance, whatever the cycle-mean hip and knee do. Stance frames
are the clip's (``foot_events``), not only the usable cycles': the floor is the floor, and the
foot rests on it the same way outside a cycle the extractor keeps. Rotations are
stored in the file's own axes, like the channels they come from.

Timing is the extractor's own (``mocap.extract.gait_timing``): its path frame gives each
frame's heading and its foot events on the foot point (toe landmark, else ankle) the stance
masks; its thresholds are frozen and used as they are.

Probes (spec 4.4): with the ``full`` chain posed at the reference, each hinge is turned by
``+PROBE_RAD`` and the landmark points are read before and after; the
hinge's world axis is recorded too. The full chain's pelvis carries the heading-removed
rotation, so its world frame is the reference path frame: forward +x, left +y, up +z.

Posture map (spec 4.3, "Posture gain"): on the ``driven`` chain at qpos zero (the reference,
heading +x) the extractor's own static reading (``mocap.extract.posture_angles``, path frame
forward +x, left +y) is ``p_ref``. Each posture parameter has a role group (waist pitch; neck
pitch; both elbows; both shoulder rolls), turned together by ``+PROBE_RAD``; the reading's
change over ``PROBE_RAD`` is the Jacobian column of that group, ``J[parameter, group]``. A
group with no bound role drops its parameter's row and column from ``J``. ``K = J^-1`` and
every bound role of a group gets the row ``{parameter: K[group, parameter], "offset":
-(K @ p_ref)[group]}`` over all four posture parameters (0.0 for a dropped one), so commanding
``p_ref`` writes hinge 0 (the reference pose) and commanding ``p_ref + delta`` turns the group
by ``K @ delta``. See ``posture_map`` for the elbow and half-bound cases.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from emotive_gait.mocap.events import PathFrame
from emotive_gait.mocap.extract import extract_cycles, gait_timing, posture_angles, summarise
from emotive_gait.mocap.readers.bvh import BvhFile, rotation_about
from emotive_gait.mocap.skeleton import Landmark, LandmarkMap, SkeletonClip
from emotive_gait.rig.compile import (
    CompiledRig,
    bone_offset,
    channel_angles,
    z_up_matrix,
)
from emotive_gait.rig.errors import RigError
from emotive_gait.robot.conventions import PROBE_RAD
from emotive_gait.robot.roles import POSTURE_PARAMETERS, JointRole

if TYPE_CHECKING:  # rig.mapping imports this module
    from emotive_gait.rig.mapping import RigMapping
    from emotive_gait.robot.loader import RobotModel

L = Landmark
_AXIS = {"X": 0, "Y": 1, "Z": 2}
_FEET = ((L.ANKLE_L, L.TOE_L), (L.ANKLE_R, L.TOE_R))
# Posture parameter -> the roles turned together to probe it (spec 4.3).
POSTURE_GROUPS: Mapping[str, tuple[JointRole, ...]] = MappingProxyType(
    {
        "trunk_pitch": (JointRole.WAIST_PITCH,),
        "neck_pitch": (JointRole.NECK_PITCH,),
        "elbow_flexion": (JointRole.ELBOW_L, JointRole.ELBOW_R),
        "arm_abduction": (JointRole.SHOULDER_ROLL_L, JointRole.SHOULDER_ROLL_R),
    }
)
# The landmark whose absence leaves a two-arm measure on the other arm alone: hiding it drops
# that role's side from the measure (elbow flexion needs the wrist, abduction the elbow).
_SIDE_LANDMARK: Mapping[JointRole, Landmark] = MappingProxyType(
    {
        JointRole.ELBOW_L: L.WRIST_L,
        JointRole.ELBOW_R: L.WRIST_R,
        JointRole.SHOULDER_ROLL_L: L.ELBOW_L,
        JointRole.SHOULDER_ROLL_R: L.ELBOW_R,
    }
)


@dataclass(frozen=True, eq=False)
class ReferencePose:
    """The actor's mean walking pose; every rotation in the file's own axes."""

    # (J, 3, 3) per bone in file order: the chordal mean of the local rotation over the usable
    # cycle frames; the pelvis's is its world rotation with each frame's heading removed.
    local_rotations: np.ndarray
    heading: np.ndarray  # unit horizontal forward (z-up world), the mean over usable frames
    floor_z: float  # median foot-point height over the stance frames, both feet pooled
    pelvis_height: float  # mean pelvis height above floor_z over the usable frames
    # ankle bone name -> its local reference rotation: the stance-mean heading-removed world
    # foot rotation re-expressed through the reference shank frame
    foot_stance_rotations: Mapping[str, np.ndarray]
    stance_frames: tuple[np.ndarray, np.ndarray]  # (T,) bool, left and right, from foot_events
    # Diagnostic only: mean height of the two foot points above floor_z with the chain at the
    # reference pose. The profile's ``foot_site_floor_height`` is 0 by construction (the floor
    # is the stance height of the very point the foot site sits on).
    foot_height: float = 0.0

    def baked_rotations(self, names: Sequence[str]) -> np.ndarray:
        """``local_rotations`` with the ankle bones' stance rotations swapped in (a copy).

        ``names`` are the file's bone names in order; this is the rotation the ``driven`` chain
        bakes into every body and the one the ``full`` chain is posed at.
        """
        out = np.array(self.local_rotations, dtype=float)
        for bone, rotation in self.foot_stance_rotations.items():
            out[list(names).index(bone)] = rotation
        return out


def chordal_mean(rotations: np.ndarray) -> np.ndarray:
    """The rotation nearest (Frobenius) to the mean of ``rotations`` (N, 3, 3)."""
    u, _, vt = np.linalg.svd(np.mean(np.asarray(rotations, dtype=float), axis=0))
    return u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt


def _local_rotations(file: BvhFile) -> list[np.ndarray]:
    """Per bone, (T, 3, 3) local rotation composed from its channels in order (file axes)."""
    out = []
    t = file.motion.shape[0]
    for joint in file.joints:
        rot = np.tile(np.eye(3), (t, 1, 1))
        for channel in joint.channels:
            if channel.endswith("rotation"):
                column = file.motion[:, file.channel_index(joint.name, channel)]
                rot = rot @ rotation_about(_AXIS[channel[0]], column)
        out.append(rot)
    return out


def _heading_matrices(forward: np.ndarray) -> np.ndarray:
    """(T, 3, 3) rotations about +z taking +x to each horizontal unit ``forward``."""
    f = np.asarray(forward, dtype=float)[:, :2]
    f = f / np.linalg.norm(f, axis=1, keepdims=True)
    h = np.zeros((len(f), 3, 3))
    h[:, 0, 0], h[:, 1, 0] = f[:, 0], f[:, 1]
    h[:, 0, 1], h[:, 1, 1] = -f[:, 1], f[:, 0]
    h[:, 2, 2] = 1.0
    return h


def _bone(landmarks: LandmarkMap, lm: Landmark) -> str | None:
    bone = landmarks.get(lm)
    if bone is None:
        return None
    if not isinstance(bone, str):
        raise RigError(f"landmark {lm.value} averages several joints {bone}; bind one bone")
    return bone


def _reference_points(file: BvhFile, rotations: np.ndarray, pelvis: str) -> dict[str, np.ndarray]:
    """z-up positions relative to the pelvis of every bone (and End Site) below it, with the
    chain at ``rotations`` (file axes; the pelvis's is its world rotation)."""
    p = z_up_matrix(file.up_axis)
    names = [j.name for j in file.joints]
    root = names.index(pelvis)
    world: dict[int, np.ndarray] = {}
    points: dict[str, np.ndarray] = {}
    for i, joint in enumerate(file.joints):
        if i == root:
            pos = np.zeros(3)
        elif joint.parent in world:
            offset = p @ (bone_offset(file, i) * file.scale)
            pos = points[names[joint.parent]] + world[joint.parent] @ offset
        else:
            continue
        rot = p @ rotations[i] @ p.T
        world[i] = rot if i == root else world[joint.parent] @ rot
        points[joint.name] = pos
        if joint.end_site is not None:
            end = p @ (np.asarray(joint.end_site, dtype=float) * file.scale)
            points[f"{joint.name}_End"] = pos + world[i] @ end
    return points


def reference_pose(file: BvhFile, clip: SkeletonClip, landmarks: LandmarkMap) -> ReferencePose:
    """The reference pose over the usable cycles of ``clip`` (``file``'s own FK clip).

    Raises ``RigError("no usable cycle: ...")`` with the extractor's discard reasons when the
    clip has no usable cycle (standing, in place, turning throughout).
    """
    usable_cycles = extract_cycles(clip, landmarks)
    if not usable_cycles:
        reasons = summarise(clip, landmarks).discard_reasons
        text = ", ".join(f"{why} ({n})" for why, n in reasons.items()) or "no cycle found"
        raise RigError(f"no usable cycle: {text}")
    usable = np.zeros(clip.frames(), dtype=bool)
    for c in usable_cycles:
        usable[c.start_frame : c.end_frame + 1] = True
    timing = gait_timing(clip, landmarks)  # never raises: extract_cycles found cycles
    forward = timing.frame.forward
    stance = (timing.events[0].stance, timing.events[1].stance)

    names = [j.name for j in file.joints]
    pelvis = _bone(landmarks, L.PELVIS)
    root = names.index(pelvis)
    local = _local_rotations(file)
    world: list[np.ndarray] = []
    for i, joint in enumerate(file.joints):
        world.append(local[i] if joint.parent < 0 else world[joint.parent] @ local[i])
    p = z_up_matrix(file.up_axis)
    h = _heading_matrices(forward)

    def level(i: int) -> np.ndarray:
        """Bone ``i``'s world rotation with each frame's heading removed, in file axes:
        P^T H^T (P W P^T) P = P^T H^T P W."""
        return np.einsum("ij,tkj,kl,tlm->tim", p.T, h, p, world[i])

    means = [chordal_mean(r[usable]) for r in local]
    means[root] = chordal_mean(level(root)[usable])
    reference_world: dict[int, np.ndarray] = {root: means[root]}
    for i in range(root + 1, len(file.joints)):
        if file.joints[i].parent in reference_world:
            reference_world[i] = reference_world[file.joints[i].parent] @ means[i]

    stance_rotations: dict[str, np.ndarray] = {}
    for (ankle, _), mask in zip(_FEET, stance, strict=True):
        i = names.index(_bone(landmarks, ankle))
        # A usable cycle has stance on both feet, so the mask is never empty here.
        foot_world = chordal_mean(level(i)[mask])
        stance_rotations[names[i]] = reference_world[file.joints[i].parent].T @ foot_world

    tracks = clip.landmark_positions(landmarks)
    feet = [tracks.get(toe, tracks[ankle]) for ankle, toe in _FEET]
    floor_z = float(
        np.nanmedian(np.concatenate([f[m, 2] for f, m in zip(feet, stance, strict=True)]))
    )
    pelvis_height = float(np.nanmean(tracks[L.PELVIS][usable, 2])) - floor_z
    mean_forward = np.array([*forward[usable, :2].mean(axis=0), 0.0])
    heading = mean_forward / np.linalg.norm(mean_forward)

    pose = ReferencePose(
        local_rotations=np.stack(means),
        heading=heading,
        floor_z=floor_z,
        pelvis_height=pelvis_height,
        foot_stance_rotations=MappingProxyType(stance_rotations),
        stance_frames=stance,
    )
    points = _reference_points(file, pose.baked_rotations(names), pelvis)
    foot_points = [_bone(landmarks, toe) or _bone(landmarks, ankle) for ankle, toe in _FEET]
    foot_height = pelvis_height + float(np.mean([points[n][2] for n in foot_points]))
    return replace(pose, foot_height=foot_height)


def reference_qpos(full: CompiledRig, reference: ReferencePose) -> np.ndarray:
    """The ``full`` chain's hinge values (rad) that pose it at the reference (heading removed)."""
    names = [j.name for j in full.file.joints]
    rotations = reference.baked_rotations(names)
    angles: dict[str, dict[str, float]] = {}
    for i, joint in enumerate(full.file.joints):
        rot = [c for c in joint.channels if c.endswith("rotation")]
        if rot:
            angles[joint.name] = dict(zip(rot, channel_angles(rotations[i], rot), strict=True))
    qpos = np.zeros(full.model.nq)
    for hinge in full.hinges:
        bone, channel = hinge.rsplit("/", 1)
        qpos[full.model.joint(hinge).qposadr[0]] = angles[bone][channel]
    return qpos


@dataclass(frozen=True, eq=False)
class HingeProbe:
    """One hinge turned by ``+PROBE_RAD`` from the reference on the ``full`` chain."""

    hinge: str
    axis: np.ndarray  # world unit axis at the reference: forward +x, left +y, up +z
    before: Mapping[str, np.ndarray]  # landmark value -> world point at the reference
    after: Mapping[str, np.ndarray]  # at +PROBE_RAD


def probe_hinges(
    full: CompiledRig, reference: ReferencePose, landmarks: LandmarkMap, hinges: Sequence[str]
) -> dict[str, HingeProbe]:
    """Probe each of ``hinges`` (``full`` chain hinge names) at the reference pose."""
    model = full.model
    data = mujoco.MjData(model)
    q0 = reference_qpos(full, reference)
    sites = {
        lm.value: model.site(full.landmark_sites[bone]).id
        for lm, bone in landmarks.items()
        if isinstance(bone, str) and bone in full.landmark_sites
    }

    def points(qpos: np.ndarray) -> dict[str, np.ndarray]:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        return {name: data.site_xpos[sid].copy() for name, sid in sites.items()}

    before = points(q0)
    axes = {h: data.xaxis[model.joint(h).id].copy() for h in hinges}
    out = {}
    for hinge in hinges:
        adr = model.joint(hinge).qposadr[0]
        turned = q0.copy()
        turned[adr] += PROBE_RAD
        out[hinge] = HingeProbe(
            hinge=hinge,
            axis=axes[hinge] / np.linalg.norm(axes[hinge]),
            before=MappingProxyType(before),
            after=MappingProxyType(points(turned)),
        )
    return out


def landmark_site_names(
    rig: CompiledRig, landmarks: LandmarkMap
) -> dict[Landmark, str | tuple[str, ...]]:
    """Landmark -> the site name(s) of ``rig`` carrying its joint(s); landmarks whose joints
    have no site are left out."""
    out: dict[Landmark, str | tuple[str, ...]] = {}
    for lm, bones in landmarks.items():
        names = (bones,) if isinstance(bones, str) else tuple(bones)
        if names and all(n in rig.landmark_sites for n in names):
            sites = tuple(rig.landmark_sites[n] for n in names)
            out[lm] = sites[0] if isinstance(bones, str) else sites
    return out


def posture_map(
    robot: RobotModel,
    mapping: RigMapping,
    *,
    landmark_sites: Mapping[Landmark, str | tuple[str, ...]],
) -> tuple[Mapping[JointRole, Mapping[str, float]], dict[str, float]]:
    """``(rows, p_ref)``: the measured affine posture map of the ``driven`` chain in ``robot``.

    ``landmark_sites`` maps each landmark to its site name (several are averaged, as the clip's
    landmark tracks are). Every row carries all four ``POSTURE_PARAMETERS`` and ``"offset"``.

    Elbow: ``elbow_flexion`` is unsigned (``pi`` minus the angle at the elbow, the frozen
    extractor definition), so its gain is taken on the flexion side, the magnitude of the
    measured diagonal entry; the bound sign already makes ``+hinge`` anatomical flexion. On a
    straight reference elbow the one-sided probe reads that side directly. On a reference
    hyperextended beyond ``PROBE_RAD`` the probe reduces ``|bend|``; the gain magnitude is kept
    and a flexion command first unfolds the arm, so the extractor's read-back on such a rig is
    off by the fold (a documented limitation of the unsigned measure, never retuned).

    Half-bound group: when one side of a two-role group (elbows, shoulder rolls) is unbound,
    that group's measure is probed and read on the bound arm only, so the bound arm moves by the
    commanded delta and the unbound arm stays at the reference. The extractor's two-arm mean on
    the output then reads half the delta, a round-trip residual such a rig shows.

    Raises ``RigError`` when a bound posture group's measure cannot be read (a landmark is
    absent) or the Jacobian is singular.
    """
    model = robot.model
    data = mujoco.MjData(model)
    site_ids = {
        lm: [model.site(n).id for n in ((s,) if isinstance(s, str) else s)]
        for lm, s in landmark_sites.items()
    }
    frame = PathFrame(
        origin=np.zeros((1, 3)),
        forward=np.array([[1.0, 0.0, 0.0]]),
        left=np.array([[0.0, 1.0, 0.0]]),
        turning=np.zeros(1, dtype=bool),
    )
    groups = {
        k: tuple(r for r in POSTURE_GROUPS[k] if r in mapping.roles) for k in POSTURE_PARAMETERS
    }
    # Per measure, the landmarks hidden so a half-bound group reads its bound arm alone.
    hidden = {
        k: frozenset(_SIDE_LANDMARK[r] for r in POSTURE_GROUPS[k] if r not in groups[k])
        if groups[k]
        else frozenset()
        for k in POSTURE_PARAMETERS
    }

    def reading(qpos: np.ndarray) -> np.ndarray:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        tracks = {lm: data.site_xpos[ids].mean(axis=0)[None, :] for lm, ids in site_ids.items()}
        out, cache = [], {}
        for k in POSTURE_PARAMETERS:
            if hidden[k] not in cache:
                shown = {lm: v for lm, v in tracks.items() if lm not in hidden[k]}
                cache[hidden[k]] = posture_angles(shown, frame)
            out.append(cache[hidden[k]][k])
        return np.array(out)

    q0 = np.zeros(model.nq)
    p_all = reading(q0)
    used = [i for i, k in enumerate(POSTURE_PARAMETERS) if groups[k]]
    for i in used:
        if not np.isfinite(p_all[i]):
            k = POSTURE_PARAMETERS[i]
            raise RigError(f"posture {k} is bound but cannot be read: a landmark is absent")
    jacobian = np.zeros((len(used), len(used)))
    for col, i in enumerate(used):
        turned = q0.copy()
        for role in groups[POSTURE_PARAMETERS[i]]:
            turned[robot.qpos_adr(role)] += PROBE_RAD
        jacobian[:, col] = (reading(turned)[used] - p_all[used]) / PROBE_RAD
    elbow = POSTURE_PARAMETERS.index("elbow_flexion")
    if elbow in used:
        e = used.index(elbow)
        jacobian[e, e] = abs(jacobian[e, e])  # the flexion side (docstring)
    if used and np.linalg.cond(jacobian) > 1e6:
        raise RigError(f"posture Jacobian is singular: {np.round(jacobian, 3).tolist()}")
    gains = np.linalg.inv(jacobian) if used else np.zeros((0, 0))
    p_ref = {k: float(v) for k, v in zip(POSTURE_PARAMETERS, p_all, strict=True)}
    offsets = -gains @ p_all[used]
    rows: dict[JointRole, Mapping[str, float]] = {}
    for row, i in enumerate(used):
        coefficients = dict.fromkeys(POSTURE_PARAMETERS, 0.0)
        for c, j in enumerate(used):
            coefficients[POSTURE_PARAMETERS[j]] = float(gains[row, c])
        coefficients["offset"] = float(offsets[row])
        for role in groups[POSTURE_PARAMETERS[i]]:
            rows[role] = MappingProxyType(dict(coefficients))
    return MappingProxyType(rows), p_ref
