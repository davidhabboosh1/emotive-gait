"""Dimensionless gait measurements, keyed by ``MEASURED_FIELDS`` names.

``normalise`` is Hof (1996) scaling with the subject's leg length ``l`` and ``G = 9.81``:
lengths / ``l``, ``cadence`` x ``sqrt(l / G)``; angles, ``double_support_frac`` and
``elbow_swing_ratio`` unchanged; ``pelvis_height / l`` is stored under ``com_height_scale``.
``gait.scaling.to_gait_parameters(..., scaling="froude")`` inverts it.

``ratio_delta`` is the alternative: lengths, ``cadence`` and pelvis height (under
``com_height_scale``) as a fraction of the neutral value minus 1, the rest as differences;
``scaling="ratio"`` inverts it.
"""

from __future__ import annotations

import math

from emotive_gait.gait.scaling import LENGTH_FIELDS, MEASURED_FIELDS, G
from emotive_gait.mocap.extract import GaitMeasurement


def _measured(m: GaitMeasurement, name: str) -> float:
    return float(m.pelvis_height if name == "com_height_scale" else getattr(m, name))


def normalise(m: GaitMeasurement, *, leg_length: float) -> dict[str, float]:
    """Hof-normalised measurement; ``speed`` (not a parameter) is left out."""
    if not leg_length > 0:
        raise ValueError(f"leg_length must be positive, got {leg_length}")
    out: dict[str, float] = {}
    for name in MEASURED_FIELDS:
        value = _measured(m, name)
        if name in LENGTH_FIELDS or name == "com_height_scale":
            out[name] = value / leg_length
        elif name == "cadence":
            out[name] = value * math.sqrt(leg_length / G)
        else:
            out[name] = value
    return out


def ratio_delta(m: GaitMeasurement, neutral: GaitMeasurement) -> dict[str, float]:
    """``m`` relative to ``neutral``: fractional for lengths, cadence and pelvis height."""
    out: dict[str, float] = {}
    for name in MEASURED_FIELDS:
        value, base = _measured(m, name), _measured(neutral, name)
        if name in LENGTH_FIELDS or name in ("cadence", "com_height_scale"):
            out[name] = value / base - 1.0 if base != 0 else math.nan
        else:
            out[name] = value - base
    return out
