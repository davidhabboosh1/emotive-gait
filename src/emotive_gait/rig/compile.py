"""Compile a BVH hierarchy into a MuJoCo kinematic chain.

``full`` mode makes every rotation channel a hinge, so posing the chain with the file's own
channel values reproduces the reader's forward kinematics (the FK-equivalence gate).

Conventions:

- Everything is mapped to +z up with the same permutation ``to_z_up`` applies to positions
  (y-up: ``(x, y, z) -> (x, -z, y)``, a proper rotation ``P``). A file rotation ``R`` becomes
  ``P R P^T``, the rotation about ``P a`` for a rotation about file axis ``a``, so the hinge
  axes go through the same ``P`` as the offsets.
- Several hinges on one body compose intrinsically in definition order (MuJoCo 3.17), matching
  the reader's ``R = R_c1 @ R_c2 @ R_c3``; hinges are defined in channel order.
- The root body is ``pelvis`` with no joint (what ``gait.ik`` calls welded). When the pelvis
  bone has hinges they sit on a zero-offset carrier body (``PELVIS_CARRIER``) between ``pelvis``
  and the pelvis's children. Ancestors of the pelvis are not compiled, so their rotations are
  not represented.
- Hinges and their position actuators are named ``f"{bone}/{channel}"``; bodies are named after
  their bones; a site sits at every bone origin (named like the bone) and every End Site
  (``f"{bone}_End"``). Bodies carry a tiny explicit inertia so MuJoCo compiles them; the model
  is only ever ``mj_forward``-ed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import mujoco
import numpy as np

from emotive_gait.mocap.readers.bvh import BvhFile

PELVIS_BODY = "pelvis"
PELVIS_CARRIER = "pelvis_hinges"
_AXIS = {"X": 0, "Y": 1, "Z": 2}
_MASS = 1e-3
_INERTIA = (1e-6, 1e-6, 1e-6)
_KP = 100.0


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
    body_of: Mapping[str, str]  # bone name -> body name
    landmark_sites: Mapping[str, str]  # clip joint name (incl. _End) -> site name

    def qpos_from_motion(self, frame: int) -> np.ndarray:
        """Every hinge's channel value at ``frame``, in radians (``full`` mode only)."""
        columns = []
        for hinge in self.hinges:
            bone, channel = hinge.rsplit("/", 1)
            columns.append(self.file.channel_index(bone, channel))
        return np.radians(self.file.motion[frame, columns])


def _z_up_matrix(up_axis: str) -> np.ndarray:
    """``P`` with ``to_z_up(v) == P @ v``."""
    if up_axis == "z":
        return np.eye(3)
    if up_axis == "y":
        return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    raise ValueError(f"up_axis must be 'y' or 'z', got {up_axis!r}")


def _subtree(file: BvhFile, root: int) -> list[int]:
    """``root`` and its descendants, in file order (parents before children)."""
    inside = {root}
    for i in range(root + 1, len(file.joints)):
        if file.joints[i].parent in inside:
            inside.add(i)
    return sorted(inside)


def _build(
    file: BvhFile,
    *,
    pelvis: str,
    pelvis_height: float,
    hinges_for: Callable[[int], Sequence[_Hinge]],
    quat_for: Callable[[int], np.ndarray | None],
    hinge_range: Callable[[str], tuple[float, float] | None],
) -> CompiledRig:
    """Shared spec builder: bodies, sites, hinges and actuators from the pelvis down.

    ``hinges_for(bone index)`` lists that bone's hinges in definition order, ``quat_for`` gives
    the body quaternion (``None`` for identity) and ``hinge_range(hinge name)`` the range
    (``None`` for unlimited).
    """
    names = [j.name for j in file.joints]
    if pelvis not in names:
        raise KeyError(f"no bone named {pelvis!r}")
    clash = {PELVIS_BODY, PELVIS_CARRIER} & (set(names) - {pelvis})
    if clash:
        raise ValueError(f"bone names {sorted(clash)} collide with the compiled root bodies")
    p = _z_up_matrix(file.up_axis)
    root = names.index(pelvis)

    spec = mujoco.MjSpec()
    hangs_from: dict[int, mujoco.MjsBody] = {}  # bone index -> body its children hang from
    body_of: dict[str, str] = {}
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

    for bone in _subtree(file, root):
        joint = file.joints[bone]
        if bone == root:
            pos = np.array([0.0, 0.0, pelvis_height])
            body = new_body(spec.worldbody, PELVIS_BODY, pos, quat_for(bone))
            body_of[joint.name] = PELVIS_BODY
            if hinges_for(bone):
                body = new_body(body, PELVIS_CARRIER, np.zeros(3), None)
        else:
            offset = p @ (np.asarray(joint.offset, dtype=float) * file.scale)
            body = new_body(hangs_from[joint.parent], joint.name, offset, quat_for(bone))
            body_of[joint.name] = joint.name
        add_hinges(body, bone)
        hangs_from[bone] = body
        body.add_site(name=joint.name, pos=[0.0, 0.0, 0.0])
        sites[joint.name] = joint.name
        if joint.end_site is not None:
            end = f"{joint.name}_End"
            end_pos = p @ (np.asarray(joint.end_site, dtype=float) * file.scale)
            body.add_site(name=end, pos=list(end_pos))
            sites[end] = end

    return CompiledRig(
        model=spec.compile(),
        file=file,
        pelvis=pelvis,
        hinges=tuple(hinge_names),
        body_of=MappingProxyType(body_of),
        landmark_sites=MappingProxyType(sites),
    )


def compile_full(file: BvhFile, *, pelvis: str, pelvis_height: float = 1.0) -> CompiledRig:
    """Every rotation channel from ``pelvis`` down as an unlimited hinge (the FK-gate chain)."""
    p = _z_up_matrix(file.up_axis)

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
