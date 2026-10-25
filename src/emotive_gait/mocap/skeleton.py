"""A rig-free skeleton clip: joint positions over time, in metres with +z up.

Readers (BVH, ASF/AMC, PTD) all produce a :class:`SkeletonClip`; later stages ask it for
named :class:`Landmark` points through a per-dataset :data:`LandmarkMap`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np


class MocapFormatError(ValueError):
    """A motion-capture file is malformed; the message names the path and the line."""

    def __init__(self, path: Path | str, line: int | None, message: str) -> None:
        where = f"{path}" if line is None else f"{path}: line {line}"
        super().__init__(f"{where}: {message}")


class Landmark(str, Enum):
    PELVIS = "pelvis"
    HIP_L = "hip_l"
    HIP_R = "hip_r"
    KNEE_L = "knee_l"
    KNEE_R = "knee_r"
    ANKLE_L = "ankle_l"
    ANKLE_R = "ankle_r"
    TOE_L = "toe_l"
    TOE_R = "toe_r"
    SHOULDER_L = "shoulder_l"
    SHOULDER_R = "shoulder_r"
    ELBOW_L = "elbow_l"
    ELBOW_R = "elbow_r"
    WRIST_L = "wrist_l"
    WRIST_R = "wrist_r"
    NECK = "neck"
    HEAD = "head"


# A tuple of names means: average those joints.
LandmarkMap = Mapping[Landmark, str | tuple[str, ...]]


def to_z_up(positions: np.ndarray, up_axis: Literal["y", "z"]) -> np.ndarray:
    """Return ``positions`` with +z up; a y-up file maps (x, y, z) -> (x, -z, y)."""
    if up_axis == "z":
        return positions
    if up_axis == "y":
        return np.stack([positions[..., 0], -positions[..., 2], positions[..., 1]], axis=-1)
    raise ValueError(f"up_axis must be 'y' or 'z', got {up_axis!r}")


@dataclass(frozen=True)
class SkeletonClip:
    """Joint positions over time: ``positions`` is (T, J, 3) in metres, +z up."""

    joint_names: tuple[str, ...]
    parents: tuple[int, ...] | None
    positions: np.ndarray
    fps: float
    source: str

    def frames(self) -> int:
        return int(self.positions.shape[0])

    def landmark_positions(self, landmarks: LandmarkMap) -> dict[Landmark, np.ndarray]:
        """Resolve landmarks to (T, 3) tracks, omitting any whose joints are absent."""
        index = {name: i for i, name in enumerate(self.joint_names)}
        out: dict[Landmark, np.ndarray] = {}
        for landmark, spec in landmarks.items():
            names = (spec,) if isinstance(spec, str) else tuple(spec)
            if not names or any(n not in index for n in names):
                continue
            out[landmark] = np.mean([self.positions[:, index[n]] for n in names], axis=0)
        return out
