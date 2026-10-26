"""Re-dimensioning dimensionless gait deltas onto a robot's ``GaitParameters``.

Two scalings (design spec 4.2):

``froude`` (default, Hof 1996, ``G = 9.81``)
    A delta measured in leg lengths and Froude time units is turned back into metres and steps
    per second with the robot's leg length ``l``: lengths x ``l``, ``cadence`` x
    ``sqrt(G / l)``. ``com_height_scale`` carries a pelvis-height delta in leg lengths, so it is
    multiplied by ``l / nominal_pelvis_height``. Angles and fractions are added as-is.
``ratio``
    A delta measured as a fraction of the neutral value: lengths, ``cadence`` and
    ``com_height_scale`` become ``base * (1 + delta)``; angles and fractions are added as-is.

This module imports only ``gait.params``, so both ``mocap/`` and ``emotion/`` may use it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Literal

from emotive_gait.gait.params import GaitParameters

G = 9.81  # m/s^2

Scaling = Literal["froude", "ratio"]

MEASURED_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(GaitParameters) if f.name != "expansiveness"
)
LENGTH_FIELDS = ("step_length", "step_height", "feet_spacing", "com_bounce_amp", "lateral_sway_amp")


def to_gait_parameters(
    delta: Mapping[str, float],
    base: GaitParameters,
    *,
    scaling: Scaling,
    leg_length: float,
    nominal_pelvis_height: float,
) -> GaitParameters:
    """Apply a dimensionless ``delta`` (keyed by ``MEASURED_FIELDS`` names) to ``base``."""
    unknown = sorted(set(delta) - set(MEASURED_FIELDS))
    if unknown:
        raise ValueError(f"not a measured gait parameter: {', '.join(unknown)}")
    if scaling not in ("froude", "ratio"):
        raise ValueError(f"scaling must be 'froude' or 'ratio', got {scaling!r}")
    if not (leg_length > 0 and nominal_pelvis_height > 0):
        raise ValueError("leg_length and nominal_pelvis_height must be positive")
    changes: dict[str, float] = {}
    for name, d in delta.items():
        value = float(getattr(base, name))
        if scaling == "ratio":
            scaled = name in LENGTH_FIELDS or name in ("cadence", "com_height_scale")
            changes[name] = value * (1.0 + d) if scaled else value + d
        elif name in LENGTH_FIELDS:
            changes[name] = value + d * leg_length
        elif name == "cadence":
            changes[name] = value + d * math.sqrt(G / leg_length)
        elif name == "com_height_scale":
            changes[name] = value + d * leg_length / nominal_pelvis_height
        else:
            changes[name] = value + d
    return replace(base, **changes)
