"""Reader for BVH files: forward kinematics over the hierarchy.

Euler channels compose in file order, ``R = R_c1 @ R_c2 @ R_c3``; a joint's world rotation is
``R_parent_world @ R_local`` and its position is the parent position plus the parent world
rotation applied to the (scaled) offset. ``read_bvh_file`` keeps everything the file says (hierarchy,
channels, End Sites, motion in file units and degrees, frame time); End Sites become clip joints
only when asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    end_site: np.ndarray | None = None


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
                if len(stack) >= 2 and stack[-2] is not None:
                    joints[stack[-2]].end_site = np.asarray(values)
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


@dataclass(frozen=True, eq=False)
class BvhJoint:
    """One joint as the file declares it; ``offset`` and ``end_site`` are in file units."""

    name: str
    parent: int
    offset: np.ndarray
    channels: tuple[str, ...]
    end_site: np.ndarray | None = None


@dataclass(frozen=True, eq=False)
class BvhFile:
    """A whole BVH file: hierarchy, channels, End Sites and motion (file units, degrees)."""

    joints: tuple[BvhJoint, ...]
    motion: np.ndarray  # (T, C)
    frame_time: float
    scale: float
    up_axis: Literal["y", "z"]
    source: str = ""

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time

    def _joint_index(self, joint: str) -> int:
        for i, j in enumerate(self.joints):
            if j.name == joint:
                return i
        raise KeyError(f"no joint named {joint!r}")

    def channel_slice(self, joint: str) -> slice:
        """Columns of ``motion`` that belong to ``joint``."""
        index = self._joint_index(joint)
        start = sum(len(j.channels) for j in self.joints[:index])
        return slice(start, start + len(self.joints[index].channels))

    def channel_index(self, joint: str, channel: str) -> int:
        """Column of ``motion`` holding ``channel`` (e.g. ``"Xrotation"``) of ``joint``."""
        names = self.joints[self._joint_index(joint)].channels
        if channel not in names:
            raise KeyError(f"joint {joint!r} has no channel {channel!r}")
        return self.channel_slice(joint).start + names.index(channel)

    def with_motion(self, motion: np.ndarray) -> BvhFile:
        """The same hierarchy with different motion rows."""
        motion = np.asarray(motion, dtype=float)
        if motion.ndim != 2 or motion.shape[1] != self.motion.shape[1]:
            raise ValueError(f"motion must be (T, {self.motion.shape[1]}), got {motion.shape}")
        return replace(self, motion=motion)

    def _kinematics(self) -> tuple[np.ndarray, list[np.ndarray]]:
        """Scaled world positions (T, J, 3) and world rotations, in the file's own axes."""
        data = self.motion
        t = data.shape[0]
        world_rot: list[np.ndarray] = []
        world_pos = np.zeros((t, len(self.joints), 3))
        first = 0
        for i, joint in enumerate(self.joints):
            local_t = np.tile(joint.offset, (t, 1))
            rot = np.tile(np.eye(3), (t, 1, 1))
            for k, name in enumerate(joint.channels):
                column = data[:, first + k]
                axis = _AXIS[name[0]]
                if name.endswith("position"):
                    local_t[:, axis] += column
                else:
                    rot = rot @ rotation_about(axis, column)
            first += len(joint.channels)
            local_t = local_t * self.scale
            if joint.parent < 0:
                world_pos[:, i] = local_t
                world_rot.append(rot)
            else:
                parent_rot = world_rot[joint.parent]
                world_pos[:, i] = world_pos[:, joint.parent] + np.einsum(
                    "tij,tj->ti", parent_rot, local_t
                )
                world_rot.append(parent_rot @ rot)
        return world_pos, world_rot

    def world_rotations(self, frame: int) -> np.ndarray:
        """World rotation of every joint at ``frame``, (J, 3, 3), in the file's own axes."""
        _, world_rot = self._kinematics()
        return np.stack([r[frame] for r in world_rot])

    def to_clip(self, *, end_sites: bool = False) -> SkeletonClip:
        """Forward kinematics to a +z-up clip; End Sites optionally follow as ``<joint>_End``."""
        world_pos, world_rot = self._kinematics()
        names = [j.name for j in self.joints]
        parents = [j.parent for j in self.joints]
        if end_sites:
            extra = []
            for i, joint in enumerate(self.joints):
                if joint.end_site is None:
                    continue
                offset = np.tile(np.asarray(joint.end_site) * self.scale, (len(world_pos), 1))
                extra.append(world_pos[:, i] + np.einsum("tij,tj->ti", world_rot[i], offset))
                names.append(f"{joint.name}_End")
                parents.append(i)
            if extra:
                world_pos = np.concatenate([world_pos, np.stack(extra, axis=1)], axis=1)
        return SkeletonClip(
            joint_names=tuple(names),
            parents=tuple(parents),
            positions=to_z_up(world_pos, self.up_axis),
            fps=1.0 / self.frame_time,
            source=self.source,
        )


def euler_from_matrix(r: np.ndarray, channels: tuple[str, ...]) -> np.ndarray:
    """Degrees per rotation channel so that composing them in channel order gives ``r``.

    Inverts ``R = R_c1 @ R_c2 @ R_c3`` (``rotation_about``). Position channels are ignored.
    With ``i, j, k`` the three axes and ``e = +1`` for a cyclic order, ``R[i, k] = e sin(b)``;
    at gimbal lock (``|sin b| > 1 - 1e-9``) the last angle is set to 0 and the first is solved
    from what remains.
    """
    rotation_channels = [c for c in channels if c.endswith("rotation")]
    axes = [_AXIS[c[0]] for c in rotation_channels]
    if len(axes) != 3 or len(set(axes)) != 3:
        raise ValueError(f"need three distinct rotation channels, got {channels}")
    i, j, k = axes
    e = 1.0 if (j - i) % 3 == 1 else -1.0
    sin_b = e * r[i, k]
    if abs(sin_b) > 1.0 - 1e-9:
        b = np.arcsin(np.clip(sin_b, -1.0, 1.0))
        a = np.arctan2(e * r[k, j], r[j, j])
        c = 0.0
    else:
        b = np.arcsin(sin_b)
        a = np.arctan2(-e * r[j, k], r[k, k])
        c = np.arctan2(-e * r[i, j], r[i, i])
    return np.degrees(np.array([a, b, c]))


def read_bvh_file(path: Path, *, scale: float, up_axis: Literal["y", "z"]) -> BvhFile:
    """Read everything a BVH file says; ``scale`` converts its length unit to metres."""
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
    return BvhFile(
        joints=tuple(
            BvhJoint(j.name, j.parent, j.offset, tuple(j.channels), j.end_site) for j in joints
        ),
        motion=np.asarray(rows),
        frame_time=frame_time,
        scale=scale,
        up_axis=up_axis,
        source=str(path),
    )


def read_bvh(path: Path, *, scale: float, up_axis: Literal["y", "z"]) -> SkeletonClip:
    """Read a BVH file; ``scale`` converts its length unit to metres, output is +z up."""
    return read_bvh_file(path, scale=scale, up_axis=up_axis).to_clip()
