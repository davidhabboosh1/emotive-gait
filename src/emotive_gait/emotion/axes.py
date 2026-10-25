"""Three-axis affective space: activation, valence, tension.

Emotion intensities are projected onto three axes rather than driving six
independent offsets. The literature's perceptual confusions are organised by
activation, with posture cues discriminating within an activation level and limb
flexion separating anger/fear from happiness/sadness. Averaging six independent
per-emotion joint offsets would cancel those posture cues toward neutral when
opposing emotions mix. See design 4.4 and Appendix A.6 for the provenance of the
three-axis structure; the numeric weights here are tuned, not cited.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from emotive_gait.emotion.evidence import Emotion


@dataclass(frozen=True)
class AffectiveAxes:
    """Position in affective space; each component nominally in [-1, 1] per emotion."""

    activation: float
    valence: float
    tension: float


_WEIGHTS: dict[Emotion, AffectiveAxes] = {
    Emotion.ANGER: AffectiveAxes(1.00, -0.30, 1.00),
    Emotion.HAPPINESS: AffectiveAxes(0.40, 1.00, -0.60),
    Emotion.FEAR: AffectiveAxes(-0.50, -0.40, 1.00),
    Emotion.SADNESS: AffectiveAxes(-0.80, -1.00, -0.50),
    # Disgust and surprise placements are unsourced (no measured gait data).
    Emotion.DISGUST: AffectiveAxes(-0.30, -0.60, 0.20),
    Emotion.SURPRISE: AffectiveAxes(0.60, 0.20, 0.00),
}

assert set(_WEIGHTS) == set(Emotion)
assert all(
    -1.0 <= v <= 1.0 for w in _WEIGHTS.values() for v in (w.activation, w.valence, w.tension)
)

AXIS_WEIGHTS: Mapping[Emotion, AffectiveAxes] = MappingProxyType(_WEIGHTS)
"""Axis values contributed by each emotion at intensity 1.0.

The three-axis structure is documented in design Appendix A.6. The weight values
are tuned, not cited. Only the signs of activation (anger > happiness > 0 > fear
> sadness, mirroring measured walking speeds) and the grouping of tension
(anger/fear positive, happiness/sadness negative) are literature-grounded.
Disgust and surprise are unsourced placements. Happiness and sadness valence are
exact negatives so that mixing them cancels to neutral.
"""

_OFFSETS: dict[Emotion, Mapping[str, float]] = {
    Emotion.ANGER: MappingProxyType(
        {
            # Cancels the valence contribution (AXIS_GAINS["neck_pitch"] -0.45 x anger valence
            # weight -0.30 = +0.135, head down) because the evidence reports no consistent head
            # change for anger. Must track both of those numbers if either changes.
            "neck_pitch": -0.135,
        }
    ),
    Emotion.DISGUST: MappingProxyType({}),
    Emotion.FEAR: MappingProxyType(
        {
            "com_height_scale": -0.02,
            "elbow_swing_ratio": 0.30,
            "cadence": 0.45,
            "neck_pitch": -0.25,
        }
    ),
    Emotion.HAPPINESS: MappingProxyType({}),
    Emotion.SADNESS: MappingProxyType(
        {
            "elbow_swing_ratio": -0.30,
            "lateral_sway_amp": 0.010,
        }
    ),
    Emotion.SURPRISE: MappingProxyType({}),
}

CATEGORY_OFFSETS: Mapping[Emotion, Mapping[str, float]] = MappingProxyType(_OFFSETS)
"""Per-emotion parameter offsets (keyed by GaitParameters field name) that no axis carries.

Tuned, not cited; the three-axis structure these sit on top of is described in
design Appendix A.6. Fear has negative activation but fast cadence, and negative
valence but an erect head, so two of its cues cannot come from the axes and are
carried as offsets sized to override them. Fear's knee flexion and elbow swing
ratio, and sadness's elbow swing ratio and lateral sway, are likewise not
expressible on any axis. Anger's neck_pitch offset exists only to cancel its valence
contribution, since the evidence reports no head change for anger. Emotions without offsets
map to an empty mapping.
"""


def axes_from(intensities: Mapping[Emotion, float]) -> AffectiveAxes:
    """Intensity-weighted linear sum of AXIS_WEIGHTS.

    Emotions absent from ``intensities`` contribute nothing. Intensities are not
    clamped or validated here; callers validate ranges first.
    """
    activation = valence = tension = 0.0
    for emotion, intensity in intensities.items():
        w = AXIS_WEIGHTS[emotion]
        activation += intensity * w.activation
        valence += intensity * w.valence
        tension += intensity * w.tension
    return AffectiveAxes(activation, valence, tension)
