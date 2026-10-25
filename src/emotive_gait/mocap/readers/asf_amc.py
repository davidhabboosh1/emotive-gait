"""Reader for CMU ASF (skeleton) plus AMC (motion) pairs.

Lengths convert with ``(1 / 0.45) * 0.0254`` (CMU units to metres). Each bone has a rest-frame
rotation ``C`` from its ``axis`` block and a motion ``M`` from the AMC channels in the bone's
``dof`` order; the world rotation is ``parent @ C @ M @ C^-1`` and the bone tip is the parent
tip plus that rotation applied to ``length * direction``. A joint's position is its bone tip,
so ``lfemur`` is the knee and the root is the pelvis. Euler triples are composed so the first
letter is applied first (``XYZ`` is ``Rz @ Ry @ Rx``), the usual Acclaim convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from emotive_gait.mocap.readers.bvh import rotation_about
from emotive_gait.mocap.skeleton import MocapFormatError, SkeletonClip, to_z_up

_CMU_TO_METRES = (1.0 / 0.45) * 0.0254
_CMU_FPS = 120.0
_AXIS = {"X": 0, "Y": 1, "Z": 2}


@dataclass
class _Bone:
    name: str
    direction: np.ndarray = field(default_factory=lambda: np.zeros(3))
    length: float = 0.0
    axis: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis_order: str = "XYZ"
    dof: list[str] = field(default_factory=list)  # lower-case channel names (rx, ry, tx, ...)
    parent: int = -1


def _nums(path: Path, line: int, tokens: list[str], count: int, what: str) -> list[float]:
    try:
        values = [float(t) for t in tokens]
    except ValueError as exc:
        raise MocapFormatError(path, line, f"{what} has a non-numeric value ({exc})") from exc
    if len(values) != count:
        raise MocapFormatError(path, line, f"{what} needs {count} numbers, got {len(values)}")
    return values


def _euler(order: str, degrees: np.ndarray) -> np.ndarray:
    """Rotation matrices (T, 3, 3) from per-letter angles; the first letter acts first."""
    out = np.tile(np.eye(3), (degrees.shape[0], 1, 1))
    for letter, column in zip(order, degrees.T, strict=True):
        out = rotation_about(_AXIS[letter], column) @ out
    return out


def _axis_order(path: Path, number: int, tokens: list[str]) -> str:
    order = tokens[-1].upper() if tokens else ""
    if len(order) != 3 or set(order) - set(_AXIS):
        raise MocapFormatError(path, number, f"bad axis order {tokens}")
    return order


def _bone_field(path: Path, number: int, bone: _Bone, key: str, tokens: list[str]) -> None:
    if key == "name":
        bone.name = tokens[0] if tokens else ""
    elif key == "direction":
        bone.direction = np.asarray(_nums(path, number, tokens, 3, "direction"))
    elif key == "length":
        bone.length = _nums(path, number, tokens, 1, "length")[0]
    elif key == "axis":
        bone.axis = np.asarray(_nums(path, number, tokens[:3], 3, "axis"))
        bone.axis_order = _axis_order(path, number, tokens[3:])
    elif key == "dof":
        bone.dof = [t.lower() for t in tokens]


def _parse_asf(path: Path) -> tuple[list[_Bone], str]:
    """Return bones (root first, parents before children) and the angle unit."""
    bones: list[_Bone] = []
    angle_unit = "deg"
    section = ""
    current: _Bone | None = None
    root = _Bone("root", dof=["tx", "ty", "tz", "rx", "ry", "rz"])
    hierarchy: list[tuple[int, list[str]]] = []
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        tokens = raw.split()
        if not tokens or tokens[0].startswith("#"):
            continue
        if tokens[0].startswith(":"):
            section = tokens[0][1:].lower()
            continue
        key = tokens[0].lower()
        if section == "units" and key == "angle":
            angle_unit = "rad" if tokens[1:2] and tokens[1].lower().startswith("rad") else "deg"
        elif section == "root":
            if key == "order":
                root.dof = [t.lower() for t in tokens[1:]]
            elif key == "axis":
                root.axis_order = _axis_order(path, number, tokens[1:])
        elif section == "bonedata":
            if key == "begin":
                current = _Bone("")
            elif key == "end":
                if current is None or not current.name:
                    raise MocapFormatError(path, number, "bone block without a name")
                bones.append(current)
                current = None
            elif current is not None:
                _bone_field(path, number, current, key, tokens[1:])
        elif section == "hierarchy" and key not in ("begin", "end"):
            hierarchy.append((number, tokens))

    everyone = [root, *bones]
    index = {b.name: i for i, b in enumerate(everyone)}
    for number, tokens in hierarchy:
        for name in tokens:
            if name not in index:
                raise MocapFormatError(path, number, f"hierarchy names unknown bone {name!r}")
        for child in tokens[1:]:
            everyone[index[child]].parent = index[tokens[0]]
    for bone in everyone[1:]:
        if bone.parent < 0:
            raise MocapFormatError(path, None, f"bone {bone.name!r} has no parent in :hierarchy")

    ordered: list[_Bone] = []
    placed: dict[int, int] = {}
    remaining = list(range(len(everyone)))
    while remaining:
        ready = [i for i in remaining if everyone[i].parent < 0 or everyone[i].parent in placed]
        if not ready:
            raise MocapFormatError(path, None, "hierarchy contains a cycle")
        for i in ready:
            placed[i] = len(ordered)
            ordered.append(everyone[i])
            remaining.remove(i)
    for bone in ordered:
        if bone.parent >= 0:
            bone.parent = placed[bone.parent]
    return ordered, angle_unit


def read_asf_amc(asf: Path, amc: Path) -> SkeletonClip:
    """Read a CMU ASF/AMC pair; output is metres with +z up, 120 fps."""
    asf, amc = Path(asf), Path(amc)
    bones, asf_angle = _parse_asf(asf)
    index = {b.name: i for i, b in enumerate(bones)}

    angle_unit = asf_angle
    frames: list[dict[int, np.ndarray]] = []
    for number, raw in enumerate(amc.read_text().splitlines(), start=1):
        tokens = raw.split()
        if not tokens or tokens[0].startswith("#"):
            continue
        if tokens[0].startswith(":"):
            flag = tokens[0].upper()
            if flag == ":DEGREES":
                angle_unit = "deg"
            elif flag == ":RADIANS":
                angle_unit = "rad"
            continue
        if len(tokens) == 1:
            try:
                int(tokens[0])
            except ValueError:
                raise MocapFormatError(amc, number, f"unexpected token {tokens[0]!r}") from None
            frames.append({})
            continue
        if not frames:
            raise MocapFormatError(amc, number, "bone values before the first frame number")
        if tokens[0] not in index:
            raise MocapFormatError(amc, number, f"unknown bone {tokens[0]!r}")
        bone_id = index[tokens[0]]
        values = _nums(amc, number, tokens[1:], len(bones[bone_id].dof), tokens[0])
        frames[-1][bone_id] = np.asarray(values)
    if not frames:
        raise MocapFormatError(amc, None, "no frames")

    to_deg = np.degrees if angle_unit == "rad" else (lambda a: a)
    t = len(frames)
    positions = np.zeros((t, len(bones), 3))
    rotations: list[np.ndarray] = []
    for i, bone in enumerate(bones):
        values = np.zeros((t, len(bone.dof)))
        for f, frame in enumerate(frames):
            if i in frame:
                values[f] = frame[i]
        rot_cols = [k for k, d in enumerate(bone.dof) if d.startswith("r")]
        letters = "".join(bone.dof[k][1].upper() for k in rot_cols)
        motion = _euler(letters, to_deg(values[:, rot_cols]))
        c = _euler(bone.axis_order, np.tile(to_deg(bone.axis), (t, 1)))
        local = c @ motion @ np.swapaxes(c, 1, 2)
        if bone.parent < 0:
            for k, d in enumerate(bone.dof):
                if d.startswith("t"):
                    positions[:, i, _AXIS[d[1].upper()]] = values[:, k] * _CMU_TO_METRES
            rotations.append(local)
        else:
            world = rotations[bone.parent] @ local
            rotations.append(world)
            vec = bone.direction * bone.length * _CMU_TO_METRES
            positions[:, i] = positions[:, bone.parent] + np.einsum("tij,j->ti", world, vec)

    return SkeletonClip(
        joint_names=tuple(b.name for b in bones),
        parents=tuple(b.parent for b in bones),
        positions=to_z_up(positions, "y"),
        fps=_CMU_FPS,
        source=f"{asf}+{amc}",
    )
