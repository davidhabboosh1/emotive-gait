"""Compile a BVH hierarchy into a MuJoCo kinematic chain.

``full`` mode makes every rotation channel a hinge, so posing the chain with the file's own
channel values reproduces the reader's forward kinematics (the FK-equivalence gate).
``driven`` mode (``compile_driven``) bakes the reference pose into the body quaternions and has
a hinge only for the channels bound to a role, so qpos zero is the reference pose; posed at
zero it agrees with ``full`` posed at the reference channel values.

Conventions:

- Everything is mapped to +z up with the same permutation ``to_z_up`` applies to positions
  (y-up: ``(x, y, z) -> (x, -z, y)``, a proper rotation ``P``). A file rotation ``R`` becomes
  ``P R P^T``, the rotation about ``P a`` for a rotation about file axis ``a``, so the hinge
  axes go through the same ``P`` as the offsets.
- Several hinges on one body compose intrinsically in definition order (MuJoCo 3.17), matching
  the reader's ``R = R_c1 @ R_c2 @ R_c3``; hinges are defined in channel order.
- The root body is ``pelvis`` with no joint (what ``gait.ik`` calls welded). When the pelvis
  bone has hinges they sit on a zero-offset carrier body (``PELVIS_CARRIER``) between ``pelvis``
  and the pelvis's children; a pelvis body quaternion (``driven`` mode's baked tilt) also goes
  on the carrier, so the root stays world-aligned. ``body_of`` gives the body a bone hangs from,
  ``frame_of`` the body whose frame is the bone's rotated frame (the carrier for the pelvis).
  Ancestors of the pelvis are not compiled, so their rotations are not represented.
- The pelvis's position channels are irrelevant: everything is measured relative to the pelvis.
  Translation channels below the pelvis are refused (``RigError``) unless their motion column is
  identically zero.
- Hinges and their position actuators are named ``f"{bone}/{channel}"``; bodies are named after
  their bones; a site sits at every bone origin (named like the bone) and every End Site
  (``f"{bone}_End"``); ``driven`` adds ``foot_l`` / ``foot_r`` on the ankle bodies. Ranges are
  radians (``compiler.degree`` off). Bodies carry a tiny explicit inertia so MuJoCo compiles
  them; the model is only ever ``mj_forward``-ed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from emotive_gait.mocap.readers.bvh import BvhFile, euler_from_matrix, rotation_about
from emotive_gait.mocap.skeleton import Landmark, to_z_up
from emotive_gait.rig.errors import RigError
from emotive_gait.rig.ranges import RANGES

if TYPE_CHECKING:
    from emotive_gait.rig.calibration import ReferencePose
    from emotive_gait.rig.mapping import RigMapping

PELVIS_BODY = "pelvis"
PELVIS_CARRIER = "pelvis_hinges"
_AXIS = {"X": 0, "Y": 1, "Z": 2}
_MASS = 1e-3
_INERTIA = (1e-6, 1e-6, 1e-6)
_KP = 100.0  # arbitrary: the chain is only ever mj_forward-ed, never stepped


@dataclass(frozen=True)
class _Hinge:
    channel: str  # e.g. "Xrotation"
    axis: np.ndarray  # in the body frame, +z up


@dataclass(frozen=True)
class CompiledRig:
    """A compiled chain and the names that tie it back to the file."""

    model: mujoco.MjModel
    file: BvhFile
    pelvis: str  # bone name
    hinges: tuple[str, ...]  # hinge names in qpos order
    # bone name -> the body it hangs from ("pelvis" for the pelvis bone), which is not
    # necessarily the body carrying its rotation: see ``frame_of``.
    body_of: Mapping[str, str]
    landmark_sites: Mapping[str, str]  # clip joint name (incl. _End) -> site name
    frame_of: Mapping[str, str]  # bone name -> body whose xmat is the bone's rotated frame

    def qpos_from_motion(self, frame: int) -> np.ndarray:
        """Every hinge's channel value at ``frame``, in radians (``full`` mode only)."""
        columns = []
        for hinge in self.hinges:
            bone, channel = hinge.rsplit("/", 1)
            columns.append(self.file.channel_index(bone, channel))
        return np.radians(self.file.motion[frame, columns])


def z_up_matrix(up_axis: str) -> np.ndarray:
    """``P`` with ``to_z_up(v) == P @ v`` (rows of ``to_z_up(I)`` are the columns of ``P``)."""
    return to_z_up(np.eye(3), up_axis).T


def _subtree(file: BvhFile, root: int) -> list[int]:
    """``root`` and its descendants, in file order (parents before children)."""
    inside = {root}
    for i in range(root + 1, len(file.joints)):
        if file.joints[i].parent in inside:
            inside.add(i)
    return sorted(inside)


def _refuse_translation(file: BvhFile, bones: Sequence[int], root: int) -> None:
    """Raise ``RigError`` for a non-zero position channel on a bone below the pelvis."""
    for bone in bones:
        if bone == root:
            continue
        name = file.joints[bone].name
        for channel in file.joints[bone].channels:
            if not channel.endswith("position"):
                continue
            if np.any(file.motion[:, file.channel_index(name, channel)] != 0.0):
                raise RigError(
                    f"bone {name!r} has a non-zero translation channel {channel!r}; "
                    "only the pelvis may translate"
                )


def _build(
    file: BvhFile,
    *,
    pelvis: str,
    pelvis_height: float,
    hinges_for: Callable[[int], Sequence[_Hinge]],
    quat_for: Callable[[int], np.ndarray | None],
    hinge_range: Callable[[str], tuple[float, float] | None],
    extra_sites: Mapping[str, Sequence[tuple[str, np.ndarray]]] | None = None,
) -> CompiledRig:
    """Shared spec builder: bodies, sites, hinges and actuators from the pelvis down.

    ``hinges_for(bone index)`` lists that bone's hinges in definition order, ``quat_for`` gives
    the body quaternion (``None`` for identity) and ``hinge_range(hinge name)`` the range
    (``None`` for unlimited). ``extra_sites`` maps a bone name to ``(site name, position)``
    pairs added on the body carrying its rotated frame (not landmark sites).
    """
    names = [j.name for j in file.joints]
    if pelvis not in names:
        raise KeyError(f"no bone named {pelvis!r}")
    clash = {PELVIS_BODY, PELVIS_CARRIER} & (set(names) - {pelvis})
    if clash:
        raise ValueError(f"bone names {sorted(clash)} collide with the compiled root bodies")
    p = z_up_matrix(file.up_axis)
    root = names.index(pelvis)
    bones = _subtree(file, root)
    _refuse_translation(file, bones, root)

    spec = mujoco.MjSpec()
    spec.compiler.degree = False  # hinge ranges are radians
    hangs_from: dict[int, mujoco.MjsBody] = {}  # bone index -> body its children hang from
    body_of: dict[str, str] = {}
    frame_of: dict[str, str] = {}
    sites: dict[str, str] = {}
    hinge_names: list[str] = []

    def new_body(parent, name: str, pos: np.ndarray, quat: np.ndarray | None):
        body = parent.add_body(name=name, pos=list(pos))
        if quat is not None:
            body.quat = list(quat)
        body.mass = _MASS
        body.inertia = list(_INERTIA)
        body.explicitinertial = True
        return body

    def add_hinges(body, bone: int) -> None:
        for hinge in hinges_for(bone):
            name = f"{names[bone]}/{hinge.channel}"
            limits = hinge_range(name)
            joint = body.add_joint(
                name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=list(hinge.axis)
            )
            if limits is None:
                joint.limited = mujoco.mjtLimited.mjLIMITED_FALSE
            else:
                joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
                joint.range = list(limits)
            actuator = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            actuator.set_to_position(kp=_KP)
            hinge_names.append(name)

    for bone in bones:
        joint = file.joints[bone]
        if bone == root:
            pos = np.array([0.0, 0.0, pelvis_height])
            body = new_body(spec.worldbody, PELVIS_BODY, pos, None)  # stays world-aligned
            body_of[joint.name] = PELVIS_BODY
            quat = quat_for(bone)
            if hinges_for(bone) or quat is not None:
                body = new_body(body, PELVIS_CARRIER, np.zeros(3), quat)
            frame_of[joint.name] = body.name
        else:
            offset = p @ (np.asarray(joint.offset, dtype=float) * file.scale)
            body = new_body(hangs_from[joint.parent], joint.name, offset, quat_for(bone))
            body_of[joint.name] = joint.name
            frame_of[joint.name] = joint.name
        add_hinges(body, bone)
        hangs_from[bone] = body
        body.add_site(name=joint.name, pos=[0.0, 0.0, 0.0])
        sites[joint.name] = joint.name
        if joint.end_site is not None:
            end = f"{joint.name}_End"
            end_pos = p @ (np.asarray(joint.end_site, dtype=float) * file.scale)
            body.add_site(name=end, pos=list(end_pos))
            sites[end] = end
        for site, site_pos in (extra_sites or {}).get(joint.name, ()):
            body.add_site(name=site, pos=list(site_pos))

    return CompiledRig(
        model=spec.compile(),
        file=file,
        pelvis=pelvis,
        hinges=tuple(hinge_names),
        body_of=MappingProxyType(body_of),
        landmark_sites=MappingProxyType(sites),
        frame_of=MappingProxyType(frame_of),
    )


def compile_full(file: BvhFile, *, pelvis: str, pelvis_height: float = 1.0) -> CompiledRig:
    """Every rotation channel from ``pelvis`` down as an unlimited hinge (the FK-gate chain)."""
    p = z_up_matrix(file.up_axis)

    def hinges_for(bone: int) -> list[_Hinge]:
        return [
            _Hinge(channel=c, axis=p[:, _AXIS[c[0]]].copy())
            for c in file.joints[bone].channels
            if c.endswith("rotation")
        ]

    return _build(
        file,
        pelvis=pelvis,
        pelvis_height=pelvis_height,
        hinges_for=hinges_for,
        quat_for=lambda bone: None,
        hinge_range=lambda name: None,
    )


def heading_matrix(forward: np.ndarray) -> np.ndarray:
    """Rotation about +z taking +x to the horizontal unit ``forward``."""
    f = np.array([forward[0], forward[1]], dtype=float)
    norm = float(np.hypot(f[0], f[1]))
    if norm < 1e-9:
        raise ValueError(f"forward has no horizontal component: {forward!r}")
    c, s = f / norm
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotation(axis: int, angle: float) -> np.ndarray:
    """Rotation by ``angle`` (rad) about coordinate axis ``axis`` (as ``rotation_about``)."""
    return rotation_about(axis, np.array([np.degrees(angle)]))[0]


def channel_angles(rotation: np.ndarray, channels: Sequence[str]) -> np.ndarray:
    """Radians per rotation channel (in order) whose composition ``R_c1 @ R_c2 ...`` is
    ``rotation``: exact for three distinct axes and for one axis when ``rotation`` is about it;
    with two axes the third (missing) angle of the three-axis decomposition is dropped."""
    rot = [c for c in channels if c.endswith("rotation")]
    axes = [_AXIS[c[0]] for c in rot]
    if len(set(axes)) != len(axes):
        raise RigError(f"repeated rotation axes {rot} are not supported")
    r = np.asarray(rotation, dtype=float)
    if not rot:
        return np.zeros(0)
    if len(rot) == 1:
        i, j = [(1, 2), (2, 0), (0, 1)][axes[0]]
        return np.array([np.arctan2(r[j, i] - r[i, j], r[i, i] + r[j, j])])
    full = rot + [f"{'XYZ'[a]}rotation" for a in range(3) if a not in axes]
    return np.radians(euler_from_matrix(r, tuple(full)))[: len(rot)]


def _quat(matrix: np.ndarray) -> np.ndarray:
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(matrix, dtype=float).ravel())
    return quat


def _point_in_frame(
    file: BvhFile, ancestor: str, target: str, rotations_z: Mapping[str, np.ndarray]
) -> np.ndarray:
    """``target`` (a bone or ``"<bone>_End"``) in ``ancestor``'s rotated frame, z-up, metres,
    with the bones between at ``rotations_z`` (z-up local rotations)."""
    p = z_up_matrix(file.up_axis)
    names = [j.name for j in file.joints]
    end = target not in names and target.endswith("_End")
    bone = target[: -len("_End")] if end else target
    chain, i = [], names.index(bone)
    while names[i] != ancestor:
        chain.append(i)
        i = file.joints[i].parent
        if i < 0:
            raise RigError(f"{target!r} is not below {ancestor!r}")
    pos, rot = np.zeros(3), np.eye(3)
    for i in reversed(chain):
        pos = pos + rot @ (p @ (np.asarray(file.joints[i].offset, dtype=float) * file.scale))
        rot = rot @ rotations_z[names[i]]
    if end:
        pos = pos + rot @ (p @ (np.asarray(file.joints[names.index(bone)].end_site) * file.scale))
    return pos


def compile_driven(file: BvhFile, mapping: RigMapping, reference: ReferencePose) -> CompiledRig:
    """The IK and synthesis chain: the reference pose baked into the bodies, a hinge per role.

    Every body's ``quat`` is its bone's reference rotation (the ankles' stance rotation; the
    pelvis's is heading-removed, so world +x is the walking direction), hence qpos zero is the
    reference pose. A bound channel ``k`` of a bone with reference channel angles
    ``theta`` gets the axis ``R_post.T @ e_k`` (times the binding's sign), ``R_post`` the
    product of the rotations of the channels after ``k`` at ``theta``: then
    ``R_ref @ Rot(axis, d) == R_pre @ Rot(e_k, theta_k + d) @ R_post``, the BVH's own Euler
    semantics for that channel. Ranges come from ``RANGES``, shifted by the binding's
    ``zero_offset`` (non-zero for the knees and elbows only: their ranges start at straight).
    ``foot_l`` / ``foot_r`` sit on the ankle bodies at the toe landmark (toe bone or End Site;
    the ankle origin without one); the root ``pelvis`` is at ``(0, 0,
    reference.pelvis_height)``.
    """
    p = z_up_matrix(file.up_axis)
    names = [j.name for j in file.joints]
    rotations = reference.baked_rotations(names)
    rotations_z = {name: p @ rotations[i] @ p.T for i, name in enumerate(names)}
    bound = {b.hinge: (role, b) for role, b in mapping.roles.items()}
    known = {f"{j.name}/{c}" for j in file.joints for c in j.channels if c.endswith("rotation")}
    unknown = sorted(set(bound) - known)
    if unknown:
        raise RigError(f"bound hinges {unknown} are not rotation channels of the file")

    def hinges_for(bone: int) -> list[_Hinge]:
        joint = file.joints[bone]
        rot = [c for c in joint.channels if c.endswith("rotation")]
        theta = channel_angles(rotations[bone], rot)
        out = []
        for k, channel in enumerate(rot):
            hit = bound.get(f"{joint.name}/{channel}")
            if hit is None:
                continue
            post = np.eye(3)
            for c, angle in zip(rot[k + 1 :], theta[k + 1 :], strict=True):
                post = post @ _rotation(_AXIS[c[0]], float(angle))
            axis = post.T @ np.eye(3)[_AXIS[channel[0]]] * hit[1].sign
            out.append(_Hinge(channel=channel, axis=p @ axis))
        return out

    def hinge_range(name: str) -> tuple[float, float] | None:
        role, binding = bound[name]
        limits = RANGES.get(role)
        if limits is None:
            return None
        # Knee and elbow ranges are about the straight limb (RoleBinding.zero_offset).
        return (limits[0] + binding.zero_offset, limits[1] + binding.zero_offset)

    sites: dict[str, list[tuple[str, np.ndarray]]] = {}
    for site, ankle_lm, toe_lm in (
        ("foot_l", Landmark.ANKLE_L, Landmark.TOE_L),
        ("foot_r", Landmark.ANKLE_R, Landmark.TOE_R),
    ):
        ankle = mapping.landmarks[ankle_lm]
        toe = mapping.landmarks.get(toe_lm)
        pos = np.zeros(3) if toe is None else _point_in_frame(file, ankle, toe, rotations_z)
        sites.setdefault(ankle, []).append((site, pos))

    return _build(
        file,
        pelvis=mapping.landmarks[Landmark.PELVIS],
        pelvis_height=reference.pelvis_height,
        hinges_for=hinges_for,
        quat_for=lambda bone: _quat(rotations_z[names[bone]]),
        hinge_range=hinge_range,
        extra_sites=sites,
    )
