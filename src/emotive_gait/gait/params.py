"""Gait parameters named after the affective-gait literature."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from enum import Enum
from pathlib import Path

import yaml


class SpeedMode(str, Enum):
    """How ``with_speed`` distributes a speed change over cadence and stride."""

    PROPORTIONAL = "proportional"
    HOLD_CADENCE = "hold_cadence"
    HOLD_STEP_LENGTH = "hold_step_length"


class SpeedConstraintError(ValueError):
    """Raised when a requested walking speed cannot be realised."""


@dataclass(frozen=True)
class GaitParameters:
    cadence: float = 1.8  # steps per second
    step_length: float = 0.35  # metres
    step_height: float = 0.05
    double_support_frac: float = 0.20
    feet_spacing: float = 0.15
    com_height_scale: float = 1.0
    com_bounce_amp: float = 0.02
    lateral_sway_amp: float = 0.015
    trunk_pitch: float = 0.0
    neck_pitch: float = 0.0
    knee_flexion_bias: float = 0.0
    arm_swing_amp: float = 0.35
    arm_abduction: float = 0.05
    elbow_flexion: float = 0.25
    elbow_swing_ratio: float = 0.5
    expansiveness: float = 1.0

    @property
    def speed(self) -> float:
        return self.step_length * self.cadence

    def with_speed(
        self, target: float, mode: SpeedMode = SpeedMode.PROPORTIONAL
    ) -> GaitParameters:
        # Validate first so no division or sqrt can produce NaN/ZeroDivisionError.
        if self.cadence <= 0 or self.step_length <= 0 or target <= 0:
            raise SpeedConstraintError(
                f"cannot reach speed {target} with cadence={self.cadence}, "
                f"step_length={self.step_length}; all must be positive"
            )
        if mode is SpeedMode.HOLD_CADENCE:
            return replace(self, step_length=target / self.cadence)
        if mode is SpeedMode.HOLD_STEP_LENGTH:
            return replace(self, cadence=target / self.step_length)
        k = math.sqrt(target / (self.cadence * self.step_length))
        return replace(self, cadence=self.cadence * k, step_length=self.step_length * k)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))

    @classmethod
    def from_yaml(cls, path: str | Path) -> GaitParameters:
        data = yaml.safe_load(Path(path).read_text()) or {}
        unknown = sorted(set(data) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown gait parameter(s): {', '.join(unknown)}")
        return cls(**data)
