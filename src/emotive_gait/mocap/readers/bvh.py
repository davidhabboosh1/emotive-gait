"""Reader for BVH files: forward kinematics over the hierarchy.

Euler channels compose in file order, ``R = R_c1 @ R_c2 @ R_c3``; a joint's world rotation is
``R_parent_world @ R_local`` and its position is the parent position plus the parent world
rotation applied to the (scaled) offset. End Sites are not joints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from emotive_gait.mocap.skeleton import MocapFormatError, SkeletonClip, to_z_up

_AXIS = {"X": 0, "Y": 1, "Z": 2}


@dataclass
class _Joint:
    name: str
    parent: int
    offset: np.ndarray
    channels: list[str] = field(default_factory=list)
    first_channel: int = 0


def rotation_about(axis: int, degrees: np.ndarray) -> np.ndarray:
    """Rotation matrices (T, 3, 3) about one coordinate axis."""
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    m = np.zeros((len(a), 3, 3))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    m[:, axis, axis] = 1.0
    m[:, i, i] = c
    m[:, j, j] = c
    m[:, i, j] = -s
    m[:, j, i] = s
    return m


def _floats(path: Path, line: int, tokens: list[str], count: int | None, what: str) -> list[float]:
    if count is not None and len(tokens) != count:
        raise MocapFormatError(path, line, f"{what} needs {count} numbers, got {len(tokens)}")
    try:
        return [float(t) for t in tokens]
    except ValueError as exc:
        raise MocapFormatError(path, line, f"{what} has a non-numeric value ({exc})") from exc


def _parse_hierarchy(path: Path, lines: list[str]) -> tuple[list[_Joint], int]:
    """Parse up to the ``MOTION`` line; return the joints and that line's 0-based index."""
    joints: list[_Joint] = []
    stack: list[int | None] = []  # joint index, or None for an End Site block
    pending: _Joint | None = None  # declared, awaiting its opening brace
    in_end_site = False
    channel_total = 0
    for idx, raw in enumerate(lines):
        number = idx + 1
        tokens = raw.split()
        if not tokens:
            continue
        key = tokens[0]
        if key == "HIERARCHY":
            continue
        if key == "MOTION":
            if stack or not joints:
                raise MocapFormatError(path, number, "MOTION before the hierarchy is closed")
            return joints, idx
        if key in ("ROOT", "JOINT"):
            if len(tokens) != 2:
                raise MocapFormatError(path, number, f"{key} needs exactly one name")
            parent = -1
            if key == "JOINT":
                live = [s for s in stack if s is not None]
                if not live:
                    raise MocapFormatError(path, number, "JOINT outside a ROOT")
                parent = live[-1]
            pending = _Joint(tokens[1], parent, np.zeros(3))
        elif key == "End":
            in_end_site = True
        elif key == "{":
            if in_end_site:
                stack.append(None)
            elif pending is not None:
                joints.append(pending)
                stack.append(len(joints) - 1)
                pending = None
            else:
                raise MocapFormatError(path, number, "unexpected opening brace")
        elif key == "}":
            if not stack:
                raise MocapFormatError(path, number, "unbalanced closing brace")
            if stack.pop() is None:
                in_end_site = False
        elif key == "OFFSET":
            values = _floats(path, number, tokens[1:], 3, "OFFSET")
            if in_end_site:
                continue
            if not stack or stack[-1] is None:
                raise MocapFormatError(path, number, "OFFSET outside a joint")
            joints[stack[-1]].offset = np.asarray(values)
        elif key == "CHANNELS":
            if not stack or stack[-1] is None:
                raise MocapFormatError(path, number, "CHANNELS outside a joint")
            try:
                count = int(tokens[1])
            except (IndexError, ValueError):
                raise MocapFormatError(path, number, "CHANNELS needs a count") from None
            names = tokens[2:]
            valid = all(
                len(n) > 4 and n[0] in _AXIS and n[1:] in ("position", "rotation") for n in names
            )
            if len(names) != count or not valid:
                raise MocapFormatError(path, number, f"bad channel list {names}")
            joint = joints[stack[-1]]
            joint.channels = names
            joint.first_channel = channel_total
            channel_total += count
        else:
            raise MocapFormatError(path, number, f"unexpected token {key!r} in hierarchy")
    raise MocapFormatError(path, len(lines), "no MOTION section (file ends in the hierarchy)")


def read_bvh(path: Path, *, scale: float, up_axis: Literal["y", "z"]) -> SkeletonClip:
    """Read a BVH file; ``scale`` converts its length unit to metres, output is +z up."""
    path = Path(path)
    lines = path.read_text().splitlines()
    joints, motion_idx = _parse_hierarchy(path, lines)
    total = sum(len(j.channels) for j in joints)

    frames_declared: int | None = None
    frame_time: float | None = None
    rows: list[list[float]] = []
    for idx in range(motion_idx + 1, len(lines)):
        number = idx + 1
        text = lines[idx].strip()
        if not text:
            continue
        if text.lower().startswith("frames:"):
            frames_declared = int(_floats(path, number, text.split()[1:], 1, "Frames")[0])
        elif text.lower().startswith("frame time:"):
            frame_time = _floats(path, number, text.split()[2:], 1, "Frame Time")[0]
        else:
            rows.append(_floats(path, number, text.split(), total, "frame row"))
    if frame_time is None or frame_time <= 0:
        raise MocapFormatError(path, None, "missing or non-positive 'Frame Time'")
    if frames_declared is not None and frames_declared != len(rows):
        raise MocapFormatError(path, None, f"'Frames: {frames_declared}' but {len(rows)} rows")
    if not rows:
        raise MocapFormatError(path, None, "no frames")
    data = np.asarray(rows)
    t = len(rows)

    world_rot: list[np.ndarray] = []
    world_pos = np.zeros((t, len(joints), 3))
    for i, joint in enumerate(joints):
        local_t = np.tile(joint.offset, (t, 1))
        rot = np.tile(np.eye(3), (t, 1, 1))
        for k, name in enumerate(joint.channels):
            column = data[:, joint.first_channel + k]
            axis = _AXIS[name[0]]
            if name.endswith("position"):
                local_t[:, axis] += column
            else:
                rot = rot @ rotation_about(axis, column)
        local_t = local_t * scale
        if joint.parent < 0:
            world_pos[:, i] = local_t
            world_rot.append(rot)
        else:
            parent_rot = world_rot[joint.parent]
            world_pos[:, i] = world_pos[:, joint.parent] + np.einsum(
                "tij,tj->ti", parent_rot, local_t
            )
            world_rot.append(parent_rot @ rot)

    return SkeletonClip(
        joint_names=tuple(j.name for j in joints),
        parents=tuple(j.parent for j in joints),
        positions=to_z_up(world_pos, up_axis),
        fps=1.0 / frame_time,
        source=str(path),
    )
