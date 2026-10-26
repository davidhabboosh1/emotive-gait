"""The actor's reference pose (the calibrated zero) and the hinge probes that bind roles.

Reference pose (spec 4.3): for every bone, the chordal mean of its local rotation over the
usable cycle frames (the union of the cycles ``extract_cycles`` keeps): the rotation matrices
are averaged and projected back onto SO(3) with an SVD, ``U @ diag(1, 1, det(U Vt)) @ Vt``;
Euler channels are never averaged. The pelvis uses its WORLD rotation (its ancestors composed,
so a rotating reference bone above it is absorbed) after removing each frame's path-frame
heading, so it carries the tilt and no yaw. The two ankle bones also get a stance-only mean
(``foot_stance_rotations``), so "level" is the foot as it rests on the floor. Stance frames
are the clip's (``foot_events``), not only the usable cycles': the floor is the floor, and the
foot rests on it the same way outside a cycle the extractor keeps. Rotations are
stored in the file's own axes, like the channels they come from.

Timing replicates the extractor exactly (``mocap.extract``): the pelvis and foot tracks are
low-passed at ``EVENT_FILTER_HZ``, the path frame is rebuilt with a heading window of one
measured stride, and ``foot_events`` on the foot point (toe landmark, else ankle) gives the
stance masks. Its thresholds are frozen and used as they are.

Probes (spec 4.4): with the ``full`` chain posed at the reference, each hinge is turned by
``+PROBE_RAD`` (and ``-PROBE_RAD``) and the landmark points are read before and after; the
hinge's world axis is recorded too. The full chain's pelvis carries the heading-removed
rotation, so its world frame is the reference path frame: forward +x, left +y, up +z.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

import mujoco
import numpy as np

from emotive_gait.mocap.events import cycles, foot_events, path_frame
from emotive_gait.mocap.extract import EVENT_FILTER_HZ, _lowpass, extract_cycles, summarise
from emotive_gait.mocap.readers.bvh import BvhFile, rotation_about
from emotive_gait.mocap.skeleton import Landmark, LandmarkMap, SkeletonClip
from emotive_gait.rig.compile import (
    CompiledRig,
    _z_up_matrix,
    channel_angles,
)
from emotive_gait.rig.errors import RigError
from emotive_gait.robot.conventions import PROBE_RAD

L = Landmark
_AXIS = {"X": 0, "Y": 1, "Z": 2}
_FEET = ((L.ANKLE_L, L.TOE_L), (L.ANKLE_R, L.TOE_R))


@dataclass(frozen=True, eq=False)
class ReferencePose:
    """The actor's mean walking pose; every rotation in the file's own axes."""

    # (J, 3, 3) per bone in file order: the chordal mean of the local rotation over the usable
    # cycle frames; the pelvis's is its world rotation with each frame's heading removed.
    local_rotations: np.ndarray
    heading: np.ndarray  # unit horizontal forward (z-up world), the mean over usable frames
    floor_z: float  # median foot-point height over the stance frames, both feet pooled
    pelvis_height: float  # mean pelvis height above floor_z over the usable frames
    # ankle bone name -> chordal mean of its local rotation over its stance frames
    foot_stance_rotations: Mapping[str, np.ndarray]
    stance_frames: tuple[np.ndarray, np.ndarray]  # (T,) bool, left and right, from foot_events
    # Mean height of the two foot points above floor_z with the chain at the reference pose
    # (ankles at their stance rotation): the profile's ``foot_site_floor_height``.
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


def _timing(
    clip: SkeletonClip, landmarks: LandmarkMap
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The extractor's path-frame forward (T, 3) and the left and right stance masks."""
    fps = float(clip.fps)
    raw = clip.landmark_positions(landmarks)
    timing = {
        k: _lowpass(v, fps, EVENT_FILTER_HZ)
        for k, v in raw.items()
        if k in (L.PELVIS, L.ANKLE_L, L.ANKLE_R, L.TOE_L, L.TOE_R)
    }
    feet = [timing.get(toe, timing[ankle]) for ankle, toe in _FEET]
    frame = path_frame(timing[L.PELVIS], fps)
    pairs = cycles(*(foot_events(foot, frame, fps) for foot in feet))
    stride = float(np.median([e - s for s, e in pairs]))
    frame = path_frame(timing[L.PELVIS], fps, smooth_s=stride / fps)
    left, right = (foot_events(foot, frame, fps).stance for foot in feet)
    return frame.forward, left, right


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
    p = _z_up_matrix(file.up_axis)
    names = [j.name for j in file.joints]
    root = names.index(pelvis)
    world: dict[int, np.ndarray] = {}
    points: dict[str, np.ndarray] = {}
    for i, joint in enumerate(file.joints):
        if i == root:
            pos = np.zeros(3)
        elif joint.parent in world:
            offset = p @ (np.asarray(joint.offset, dtype=float) * file.scale)
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
    forward, stance_l, stance_r = _timing(clip, landmarks)
    stance = (stance_l, stance_r)

    names = [j.name for j in file.joints]
    pelvis = _bone(landmarks, L.PELVIS)
    root = names.index(pelvis)
    local = _local_rotations(file)
    world = local[root]
    i = file.joints[root].parent
    while i >= 0:
        world = local[i] @ world
        i = file.joints[i].parent
    p = _z_up_matrix(file.up_axis)
    # Heading removed in the z-up world, back to file axes: P^T H^T (P W P^T) P = P^T H^T P W.
    h = _heading_matrices(forward)
    level = np.einsum("ij,tkj,kl,tlm->tim", p.T, h, p, world)
    means = [chordal_mean(r[usable]) for r in local]
    means[root] = chordal_mean(level[usable])

    stance_rotations: dict[str, np.ndarray] = {}
    for (ankle, _), mask in zip(_FEET, stance, strict=True):
        bone = _bone(landmarks, ankle)
        # A usable cycle has stance on both feet, so the mask is never empty here.
        stance_rotations[bone] = chordal_mean(local[names.index(bone)][mask])

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
    """One hinge turned by ``+-PROBE_RAD`` from the reference on the ``full`` chain."""

    hinge: str
    axis: np.ndarray  # world unit axis at the reference: forward +x, left +y, up +z
    before: Mapping[str, np.ndarray]  # landmark value -> world point at the reference
    after: Mapping[str, np.ndarray]  # at +PROBE_RAD
    minus: Mapping[str, np.ndarray]  # at -PROBE_RAD


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
        turned = [q0.copy(), q0.copy()]
        turned[0][adr] += PROBE_RAD
        turned[1][adr] -= PROBE_RAD
        out[hinge] = HingeProbe(
            hinge=hinge,
            axis=axes[hinge] / np.linalg.norm(axes[hinge]),
            before=MappingProxyType(before),
            after=MappingProxyType(points(turned[0])),
            minus=MappingProxyType(points(turned[1])),
        )
    return out
