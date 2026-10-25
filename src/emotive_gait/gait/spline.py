"""C2-continuous periodic curves over normalized gait phase."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline

_CLOSE_TOL = 1e-12


class NonPeriodicKnotsError(ValueError):
    """Raised when an explicit closing knot at phase 1.0 disagrees with the first value."""


@dataclass(frozen=True)
class PeriodicCurve:
    """Cubic spline over phase in [0, 1) that wraps with matching value, velocity and acceleration.

    ``knots_phase`` always starts at 0.0 and ends at 1.0, and ``knots_value`` has equal
    first and last entries; use :meth:`from_knots` to build one from open knots.
    """

    knots_phase: np.ndarray
    knots_value: np.ndarray
    _spline: CubicSpline = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        phase = np.array(self.knots_phase, dtype=float, copy=True)
        value = np.array(self.knots_value, dtype=float, copy=True)
        phase.flags.writeable = False
        value.flags.writeable = False
        object.__setattr__(self, "knots_phase", phase)
        object.__setattr__(self, "knots_value", value)
        object.__setattr__(self, "_spline", CubicSpline(phase, value, bc_type="periodic"))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PeriodicCurve):
            return NotImplemented
        return np.array_equal(self.knots_phase, other.knots_phase) and np.array_equal(
            self.knots_value, other.knots_value
        )

    def __hash__(self) -> int:
        return hash((tuple(self.knots_phase.tolist()), tuple(self.knots_value.tolist())))

    @classmethod
    def from_knots(cls, phase: Sequence[float], value: Sequence[float]) -> PeriodicCurve:
        p = np.asarray(phase, dtype=float)
        v = np.asarray(value, dtype=float)
        if p.ndim != 1 or p.shape != v.shape or p.size == 0:
            raise ValueError("phase and value must be non-empty 1-D sequences of equal length")
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(v))):
            raise ValueError("knots must be finite")
        if p[0] != 0.0:
            raise ValueError(f"first knot phase must be exactly 0.0, got {p[0]}")
        if p[-1] > 1.0 or np.any(np.diff(p) <= 0.0):
            raise ValueError("knot phases must be strictly increasing within [0.0, 1.0]")
        if p[-1] == 1.0:
            if abs(v[-1] - v[0]) > _CLOSE_TOL:
                raise NonPeriodicKnotsError(
                    f"closing knot at phase 1.0 has value {v[-1]} but first value is {v[0]}"
                )
            # Within tolerance: snap to exactly v[0] so SciPy's strict periodic check passes.
            v = v.copy()
            v[-1] = v[0]
        else:
            p = np.append(p, 1.0)
            v = np.append(v, v[0])
        return cls(knots_phase=p, knots_value=v)

    def __call__(self, phase: ArrayLike) -> np.ndarray:
        return np.asarray(self._spline(np.mod(phase, 1.0)))

    def derivative(self, phase: ArrayLike, order: int = 1) -> np.ndarray:
        return np.asarray(self._spline.derivative(order)(np.mod(phase, 1.0)))
