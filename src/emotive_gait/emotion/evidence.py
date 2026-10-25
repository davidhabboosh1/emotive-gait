"""Evidence base for emotion-to-gait claims.

Four honesty rules apply. Every claim is a signed direction relative to a neutral gait, never a
magnitude, because the literature publishes no joint angles in degrees. The only measured values
are walking speeds in m/s and swing-phase percentages. Disgust and surprise have no gait
kinematics in any source, so they ship as unsourced placements by arousal and valence. The
parameters ``jerk`` and ``swing_phase_fraction`` are kept as inert claims with no gait parameter.
Provenance for every row is in design Appendix A.
"""

from dataclasses import dataclass
from enum import Enum


class Emotion(str, Enum):
    ANGER = "anger"
    DISGUST = "disgust"
    FEAR = "fear"
    HAPPINESS = "happiness"
    SADNESS = "sadness"
    SURPRISE = "surprise"


class Grounding(str, Enum):
    MEASURED = "measured"
    DIRECTIONAL = "directional"
    UNSOURCED = "unsourced"


@dataclass(frozen=True)
class Claim:
    """One signed direction (-1, 0 or +1) of a gait parameter versus a neutral gait."""

    emotion: Emotion
    parameter: str
    direction: int
    grounding: Grounding
    citation: str
    note: str = ""


_M = Grounding.MEASURED
_D = Grounding.DIRECTIONAL
_ROETHER = "Roether 2011 ch.3"
_BARLIYA = "Barliya 2013"
_HK = "Halovic & Kroos 2018"
_MONTEPARE = "Montepare 1987"
_MICHALAK = "Michalak 2009"
_GROSS = "Gross 2012"
_RANDHAVANE = "Randhavane 2019"
_NONE = "none - placed by arousal/valence (design 10.2)"

# (parameter, direction, grounding, citation, note) per emotion, in claims-table order.
_GROUNDED_ROWS: dict[Emotion, tuple[tuple[str, int, Grounding, str, str], ...]] = {
    Emotion.ANGER: (
        ("speed", 1, _M, _ROETHER, "1.82 +/- 0.22 m/s, highest of the four"),
        ("cadence", 1, _D, _HK, "fastest cadence"),
        ("step_length", 1, _D, _MONTEPARE, "long strides"),
        ("neck_pitch", 0, _D, _ROETHER, "no consistent head change reported"),
        ("trunk_pitch", 1, _D, _ROETHER, "spine inclination for all negative affects"),
        ("arm_swing_amp", 1, _D, _ROETHER, "larger than speed-matched neutral"),
        (
            "elbow_flexion",
            1,
            _D,
            _ROETHER,
            "pronounced increase; top predictor of anger expressiveness",
        ),
        ("knee_flexion_bias", 0, _D, _ROETHER, "hip/knee flexion not significant"),
        ("expansiveness", 1, _D, _ROETHER, "larger joint amplitudes overall"),
        ("com_bounce_amp", 0, _D, _GROSS, "reported only as smoother than sadness, not vs neutral"),
        (
            "jerk",
            1,
            _D,
            _RANDHAVANE,
            "higher-arousal emotions have higher jerk; no GaitParameters field",
        ),
        (
            "swing_phase_fraction",
            0,
            _M,
            _BARLIYA,
            "38 percent of cycle; neutral value not published; no GaitParameters field",
        ),
    ),
    Emotion.FEAR: (
        ("speed", -1, _M, _ROETHER, "0.83 +/- 0.31 m/s"),
        ("cadence", 1, _D, _HK, "fast short steps, scurrying"),
        ("step_length", -1, _D, _HK, "short steps"),
        ("neck_pitch", -1, _D, _ROETHER, "head held more erect than normal"),
        ("trunk_pitch", 1, _D, _ROETHER, "spine inclination"),
        ("arm_swing_amp", -1, _D, _HK, "reduced; mostly forearm movement"),
        (
            "elbow_flexion",
            1,
            _D,
            _ROETHER,
            "pronounced increase; top predictor of fear expressiveness",
        ),
        ("elbow_swing_ratio", 1, _D, _HK, "mainly the lower arm moves"),
        (
            "knee_flexion_bias",
            1,
            _D,
            _ROETHER,
            "increased hip and knee flexion, widespread postural tension",
        ),
        ("expansiveness", -1, _D, _ROETHER, "smaller and slower than speed-matched neutral"),
        ("swing_phase_fraction", 0, _M, _BARLIYA, "34 percent of cycle; no GaitParameters field"),
    ),
    Emotion.HAPPINESS: (
        (
            "speed",
            1,
            _M,
            _ROETHER,
            (
                "1.31 +/- 0.36 m/s (thesis also states 1.41 +/- 0.19; "
                "inconsistency flagged in Appendix A.2)"
            ),
        ),
        ("cadence", 0, _D, _HK, "reported only as slower than anger, not vs neutral"),
        ("step_length", 1, _D, _HK, "long; Montepare notes shorter than proud or angry"),
        ("neck_pitch", -1, _D, _ROETHER, "head held more erect"),
        (
            "trunk_pitch",
            -1,
            _D,
            _ROETHER,
            "unusually erect spine; vertical extent cues positive affect",
        ),
        (
            "arm_swing_amp",
            1,
            _D,
            _ROETHER,
            "larger; shoulder amplitude effect larger than for anger",
        ),
        ("elbow_flexion", -1, _D, _ROETHER, "reduced; ratings rose with larger elbow angle"),
        ("expansiveness", 1, _D, _GROSS, "more expanded than anger"),
        ("com_bounce_amp", 1, _D, _HK, "perceivers describe bouncing"),
        ("jerk", 1, _D, _RANDHAVANE, "higher arousal, higher jerk; no GaitParameters field"),
    ),
    Emotion.SADNESS: (
        (
            "speed",
            -1,
            _M,
            _ROETHER,
            "0.68 +/- 0.21 m/s, lowest; Barliya: slower than all other conditions",
        ),
        ("cadence", -1, _D, _HK, "slowest"),
        ("step_length", -1, _D, _MONTEPARE, "short strides"),
        (
            "neck_pitch",
            1,
            _D,
            _ROETHER,
            (
                "strongly increased head inclination; the single most important posture "
                "feature, discriminant loading 0.56"
            ),
        ),
        ("trunk_pitch", 1, _D, _MICHALAK, "slumped posture; Roether: spine inclination"),
        ("lateral_sway_amp", 1, _D, _MICHALAK, "larger lateral upper-body sway"),
        ("arm_swing_amp", -1, _D, _MONTEPARE, "less arm swing"),
        ("elbow_flexion", -1, _D, _ROETHER, "reduced elbow flexion"),
        ("elbow_swing_ratio", -1, _D, _HK, "the whole arm moves"),
        ("expansiveness", -1, _D, _GROSS, "contracted torso and limbs"),
        ("com_bounce_amp", -1, _D, _MICHALAK, "reduced vertical head movement"),
        ("jerk", -1, _D, _RANDHAVANE, "lower; no GaitParameters field"),
        ("swing_phase_fraction", 0, _M, _BARLIYA, "33 percent of cycle; no GaitParameters field"),
    ),
}

# (parameter, direction, note) per unsourced emotion; every claim is a modelling choice.
_UNSOURCED_ROWS: dict[Emotion, tuple[tuple[str, int, str], ...]] = {
    Emotion.DISGUST: (
        ("speed", -1, "low-arousal placement; modelling choice, not a finding"),
        ("trunk_pitch", 1, "negative-valence placement (inclined spine); modelling choice"),
        ("arm_swing_amp", -1, "low-arousal placement; modelling choice"),
        ("neck_pitch", 0, "no basis to place; deliberately neutral"),
    ),
    Emotion.SURPRISE: (
        ("speed", 1, "high-arousal placement; modelling choice, not a finding"),
        ("expansiveness", 1, "high-arousal placement; modelling choice"),
        ("neck_pitch", -1, "head-up placement; modelling choice"),
        ("trunk_pitch", -1, "erect placement; modelling choice"),
    ),
}

EVIDENCE: tuple[Claim, ...] = (
    *(
        Claim(e, p, d, g, c, n)
        for e in (Emotion.ANGER, Emotion.FEAR, Emotion.HAPPINESS, Emotion.SADNESS)
        for p, d, g, c, n in _GROUNDED_ROWS[e]
    ),
    *(
        Claim(e, p, d, Grounding.UNSOURCED, _NONE, n)
        for e in (Emotion.DISGUST, Emotion.SURPRISE)
        for p, d, n in _UNSOURCED_ROWS[e]
    ),
)

GROUNDED_EMOTIONS: frozenset[Emotion] = frozenset(_GROUNDED_ROWS)
UNSOURCED_EMOTIONS: frozenset[Emotion] = frozenset(_UNSOURCED_ROWS)

# Mean walking speeds in m/s (Roether 2011 ch.3).
MEASURED_SPEEDS: dict[Emotion, float] = {
    Emotion.ANGER: 1.82,
    Emotion.HAPPINESS: 1.31,
    Emotion.FEAR: 0.83,
    Emotion.SADNESS: 0.68,
}


def claims_for(emotion: Emotion) -> tuple[Claim, ...]:
    """Return every claim for ``emotion`` in table order."""
    return tuple(c for c in EVIDENCE if c.emotion is emotion)
