"""Emotion intensities to gait parameters.

Six emotion intensities become changes to the gait *parameters* (never to joint angles
directly). An :class:`EmotionModel` turns intensities into parameter deltas in the robot's
units; :class:`AnalyticEmotionModel` (``ANALYTIC``, the default) does it through three affective
axes plus a few category offsets, and ``emotion/learned.py`` supplies a data-derived one. This
module is the one sanctioned meeting point of the emotion and robot layers: it asks a
:class:`RobotProfile` which cues the robot can physically show and drops the rest, whichever
model produced them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Protocol

from emotive_gait.emotion.axes import CATEGORY_OFFSETS, AffectiveAxes, axes_from
from emotive_gait.emotion.evidence import Emotion
from emotive_gait.gait.params import GaitParameters, SpeedMode
from emotive_gait.robot.profiles.g1 import G1_PROFILE_WITH_NECK
from emotive_gait.robot.roles import JointRole, RobotProfile


class IntensityError(ValueError):
    """Raised when an emotion intensity is not a finite number in [0, 1]."""


AXIS_GAINS: Mapping[str, tuple[str, float]] = MappingProxyType(
    {
        "cadence": ("activation", 0.60),  # steps/s per unit
        "step_length": ("activation", 0.12),  # m
        "arm_swing_amp": ("activation", 0.25),  # rad
        "expansiveness": ("activation", 0.35),
        "com_bounce_amp": ("activation", 0.012),  # m
        "trunk_pitch": ("valence", -0.30),  # rad; positive valence = erect = negative pitch
        "neck_pitch": ("valence", -0.45),  # rad
        "elbow_flexion": ("tension", 0.45),  # rad
        "arm_abduction": ("tension", 0.10),  # rad
    }
)
"""Parameter delta per unit of the named axis: ``parameter -> (axis, gain)``. Tuned, not cited.

Only the three-axis structure is literature-backed. Fear's cadence and neck_pitch are
overridden by ``CATEGORY_OFFSETS``, and anger's neck_pitch offset cancels its valence term;
re-check the claim cross-check test if a gain changes.
"""

_LEG_ROLES = (
    JointRole.HIP_PITCH_L,
    JointRole.HIP_PITCH_R,
    JointRole.KNEE_L,
    JointRole.KNEE_R,
    JointRole.ANKLE_PITCH_L,
    JointRole.ANKLE_PITCH_R,
)
_PARAMETER_ROLES: Mapping[str, tuple[JointRole, ...]] = MappingProxyType(
    {
        "neck_pitch": (JointRole.NECK_PITCH,),
        "trunk_pitch": (JointRole.WAIST_PITCH,),
        "elbow_flexion": (JointRole.ELBOW_L, JointRole.ELBOW_R),
        "elbow_swing_ratio": (JointRole.ELBOW_L, JointRole.ELBOW_R),
        "arm_swing_amp": (JointRole.SHOULDER_PITCH_L, JointRole.SHOULDER_PITCH_R),
        "arm_abduction": (JointRole.SHOULDER_ROLL_L, JointRole.SHOULDER_ROLL_R),
    }
    | dict.fromkeys(
        (
            "com_height_scale",
            "step_length",
            "step_height",
            "feet_spacing",
            "double_support_frac",
            "com_bounce_amp",
            "lateral_sway_amp",
        ),
        _LEG_ROLES,
    )
)
# Parameters absent here are whole-body timing/amplitude and are never unexpressible.

_NONNEGATIVE_AMPLITUDES = (
    "arm_swing_amp",
    "arm_abduction",
    "expansiveness",
    "com_bounce_amp",
    "lateral_sway_amp",
    "step_height",
    "step_length",
)


def validate_intensities(intensities: Mapping[Emotion, float]) -> dict[Emotion, float]:
    """Normalise keys to :class:`Emotion` and check each intensity is finite and in [0, 1]."""
    clean: dict[Emotion, float] = {}
    for key, value in intensities.items():
        try:
            emotion = Emotion(key)
        except ValueError:
            raise IntensityError(f"unknown emotion {key!r}") from None
        try:
            x = float(value)
        except (TypeError, ValueError):
            raise IntensityError(
                f"intensity for {emotion.value} is not a number: {value!r}"
            ) from None
        if not math.isfinite(x) or not 0.0 <= x <= 1.0:
            raise IntensityError(
                f"intensity for {emotion.value} must be finite and in [0, 1], got {value!r}"
            )
        clean[emotion] = x
    return clean


def _deltas(axes: AffectiveAxes, offsets: Mapping[Emotion, float]) -> dict[str, float]:
    """Parameter deltas from axis values plus intensity-scaled category offsets."""
    deltas = {param: gain * getattr(axes, axis) for param, (axis, gain) in AXIS_GAINS.items()}
    for emotion, intensity in offsets.items():
        for param, offset in CATEGORY_OFFSETS[emotion].items():
            deltas[param] = deltas.get(param, 0.0) + intensity * offset
    return deltas


class EmotionModel(Protocol):
    """Anything that turns emotion intensities into gait-parameter deltas.

    ``deltas`` validates ``intensities`` (as :func:`validate_intensities` does) and returns
    deltas keyed by ``GaitParameters`` field name, in the robot's units. Cue dropping, the
    amplitude floor and ``hold_speed`` are applied by :func:`modulate`, not by the model.
    """

    def deltas(
        self, intensities: Mapping[Emotion, float], profile: RobotProfile
    ) -> Mapping[str, float]: ...


class AnalyticEmotionModel:
    """The hand-tuned model: ``AXIS_GAINS`` on the three axes plus ``CATEGORY_OFFSETS``.

    Its deltas do not depend on the profile (the axis gains are already in the G1's units).
    """

    def deltas(
        self, intensities: Mapping[Emotion, float], profile: RobotProfile
    ) -> Mapping[str, float]:
        clean = validate_intensities(intensities)
        return _deltas(axes_from(clean), clean)


ANALYTIC = AnalyticEmotionModel()


def _plan(
    intensities: Mapping[Emotion, float], profile: RobotProfile, model: EmotionModel | None
) -> tuple[dict[str, float], tuple[tuple[Emotion, str], ...]]:
    """Validate, build parameter deltas, and drop those the profile cannot carry.

    Returns the surviving deltas and the dropped ``(emotion, parameter)`` pairs.
    """
    model = ANALYTIC if model is None else model
    clean = validate_intensities(intensities)

    def carried(param: str) -> bool:
        roles = _PARAMETER_ROLES.get(param, ())
        return not roles or any(profile.has(r) for r in roles)

    total = model.deltas(clean, profile)
    dropped = tuple(
        (emotion, param)
        for emotion, intensity in clean.items()
        if intensity != 0.0
        for param, delta in model.deltas({emotion: intensity}, profile).items()
        if delta != 0.0 and not carried(param)
    )
    return {p: d for p, d in total.items() if carried(p)}, dropped


def modulate(
    params: GaitParameters,
    intensities: Mapping[Emotion, float],
    profile: RobotProfile,
    *,
    model: EmotionModel | None = None,
    hold_speed: float | None = None,
    speed_mode: SpeedMode = SpeedMode.PROPORTIONAL,
) -> GaitParameters:
    """Return ``params`` shifted by the given emotion intensities.

    ``model`` produces the deltas (default: the analytic model, ``ANALYTIC``). Cues the profile cannot show are dropped (see :func:`unexpressible`). Amplitude parameters
    (``arm_swing_amp``, ``arm_abduction``, ``expansiveness``, ``com_bounce_amp``,
    ``lateral_sway_amp``, ``step_height``, ``step_length``) are floored at 0.0: "less arm swing"
    saturates at none, and a negative amplitude is a phase flip, not a smaller motion. When
    ``hold_speed`` is given, speed is pinned to it last, so posture cues survive.
    """
    deltas, _ = _plan(intensities, profile, model)
    result = replace(params, **{p: getattr(params, p) + d for p, d in deltas.items()})
    # "Less arm swing" saturates at none: a negative amplitude is a phase flip (e.g. ipsilateral
    # arm swing), not a smaller motion. Only negative values are touched, so a value that is
    # already >= 0 passes through bit-exact.
    floored = {f: 0.0 for f in _NONNEGATIVE_AMPLITUDES if getattr(result, f) < 0.0}
    if floored:
        result = replace(result, **floored)
    if hold_speed is not None:
        result = result.with_speed(hold_speed, speed_mode)
    return result


def unexpressible(
    intensities: Mapping[Emotion, float],
    profile: RobotProfile,
    model: EmotionModel | None = None,
) -> tuple[tuple[Emotion, str], ...]:
    """``(emotion, parameter)`` pairs the model would move but the profile cannot show."""
    return _plan(intensities, profile, model)[1]


PRESET_PARAMS: Mapping[str, GaitParameters] = MappingProxyType(
    {
        "neutral": GaitParameters(),
        "sad": modulate(GaitParameters(), {Emotion.SADNESS: 1.0}, G1_PROFILE_WITH_NECK),
        "angry": modulate(GaitParameters(), {Emotion.ANGER: 1.0}, G1_PROFILE_WITH_NECK),
        "happy": modulate(GaitParameters(), {Emotion.HAPPINESS: 1.0}, G1_PROFILE_WITH_NECK),
    }
)


def learned_presets(model: EmotionModel) -> Mapping[str, GaitParameters]:
    """The four ``PRESET_PARAMS`` presets produced by ``model`` instead of the analytic one."""
    return MappingProxyType(
        {
            "neutral": GaitParameters(),
            "sad": modulate(
                GaitParameters(), {Emotion.SADNESS: 1.0}, G1_PROFILE_WITH_NECK, model=model
            ),
            "angry": modulate(
                GaitParameters(), {Emotion.ANGER: 1.0}, G1_PROFILE_WITH_NECK, model=model
            ),
            "happy": modulate(
                GaitParameters(), {Emotion.HAPPINESS: 1.0}, G1_PROFILE_WITH_NECK, model=model
            ),
        }
    )
