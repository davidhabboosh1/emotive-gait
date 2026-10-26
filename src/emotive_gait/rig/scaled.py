"""``RigScaledModel``: an emotion model's deltas re-dimensioned for a rig (spec 4.1).

The analytic model's deltas are G1 absolutes and ignore the profile; the learned model's
already go through the profile's ``leg_length``. On a rig:

``froude``
    A profile-aware inner model (``profile_aware``) passes through. Otherwise each G1 delta is
    made dimensionless as if measured on the G1 (lengths ``/ 0.69``, ``cadence`` x
    ``sqrt(0.69 / G)``, ``com_height_scale`` x ``0.68 / 0.69``, angles and fractions unchanged)
    and re-dimensioned to the rig as ``to_gait_parameters(..., scaling="froude")`` does
    (lengths x ``l``, ``cadence`` x ``sqrt(G / l)``, ``com_height_scale`` x ``l / h``). The two
    steps are applied as one factor per field (``l / 0.69``, ``sqrt(0.69 / l)``,
    ``(0.68 * l) / (0.69 * h)``), which is the same arithmetic and exactly 1.0 on the G1's own
    dimensions, so the wrapper is bit-identical there.
``ratio``
    Every length, ``cadence`` and ``com_height_scale`` delta is a fraction of the G1 default
    re-based on the rig's own neutral ``p0``: ``delta / default_g1 * p0_value`` (as the factor
    ``p0_value / default_g1``); angles and fractions pass through.

Keys that are not gait measurements (``expansiveness``) pass through unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from emotive_gait.emotion.evidence import Emotion
from emotive_gait.emotion.model import EmotionModel
from emotive_gait.gait.params import GaitParameters
from emotive_gait.gait.scaling import LENGTH_FIELDS, Scaling
from emotive_gait.robot.roles import RobotProfile

G1_LEG_LENGTH = 0.69  # m, RobotProfile.leg_length of the G1
G1_PELVIS_HEIGHT = 0.68  # m, the G1's nominal pelvis height
_SCALED = (*LENGTH_FIELDS, "cadence", "com_height_scale")
_DEFAULTS = GaitParameters()


class RigScaledModel(EmotionModel):
    """``inner``'s deltas in the rig's units (module docstring); itself profile-aware."""

    profile_aware = True

    def __init__(self, inner: EmotionModel, *, scaling: Scaling, p0: GaitParameters) -> None:
        if scaling not in ("froude", "ratio"):
            raise ValueError(f"scaling must be 'froude' or 'ratio', got {scaling!r}")
        self.inner = inner
        self.scaling: Scaling = scaling
        self.p0 = p0

    def _factor(self, name: str, profile: RobotProfile) -> float:
        if name not in _SCALED:
            return 1.0
        if self.scaling == "ratio":
            return float(getattr(self.p0, name)) / float(getattr(_DEFAULTS, name))
        l, h = profile.leg_length, profile.nominal_pelvis_height
        if name == "cadence":
            return math.sqrt(G1_LEG_LENGTH / l)
        if name == "com_height_scale":
            return (G1_PELVIS_HEIGHT * l) / (G1_LEG_LENGTH * h)
        return l / G1_LEG_LENGTH

    def deltas(
        self, intensities: Mapping[Emotion, float], profile: RobotProfile
    ) -> Mapping[str, float]:
        inner = self.inner.deltas(intensities, profile)
        if self.scaling == "froude" and getattr(self.inner, "profile_aware", False):
            return inner
        return {name: d * self._factor(name, profile) for name, d in inner.items()}
