import dataclasses
import math

import pytest

from emotive_gait.emotion.evidence import GROUNDED_EMOTIONS, Emotion, claims_for
from emotive_gait.emotion.model import (
    PRESET_PARAMS,
    IntensityError,
    modulate,
    unexpressible,
)
from emotive_gait.gait.params import GaitParameters
from emotive_gait.robot.profiles.g1 import G1_PROFILE, G1_PROFILE_WITH_NECK


def test_neutral_is_the_identity():
    p = GaitParameters()
    assert modulate(p, {e: 0.0 for e in Emotion}, G1_PROFILE_WITH_NECK) == p


def test_sadness_slows_shortens_slumps_and_drops_the_head():
    p = GaitParameters()
    q = modulate(p, {Emotion.SADNESS: 1.0}, G1_PROFILE_WITH_NECK)
    assert q.cadence < p.cadence and q.step_length < p.step_length
    assert q.neck_pitch > p.neck_pitch and q.trunk_pitch > p.trunk_pitch
    assert q.arm_swing_amp < p.arm_swing_amp


def test_anger_speeds_up_and_flexes_the_elbows():
    p = GaitParameters()
    q = modulate(p, {Emotion.ANGER: 1.0}, G1_PROFILE_WITH_NECK)
    assert q.speed > p.speed and q.elbow_flexion > p.elbow_flexion


def test_modulation_direction_agrees_with_every_grounded_claim():
    p = GaitParameters()
    for emotion in GROUNDED_EMOTIONS:
        q = modulate(p, {emotion: 1.0}, G1_PROFILE_WITH_NECK)
        for c in claims_for(emotion):
            if c.direction == 0 or not hasattr(p, c.parameter):
                continue
            delta = getattr(q, c.parameter) - getattr(p, c.parameter)
            assert delta * c.direction >= 0, f"{emotion}.{c.parameter} contradicts {c.citation}"


def test_hold_speed_pins_speed_while_posture_still_varies():
    p = GaitParameters()
    q = modulate(p, {Emotion.SADNESS: 1.0}, G1_PROFILE_WITH_NECK, hold_speed=p.speed)
    assert q.speed == pytest.approx(p.speed)
    assert q.neck_pitch > p.neck_pitch and q.trunk_pitch > p.trunk_pitch


def test_profile_without_a_neck_reports_the_lost_cue():
    lost = unexpressible({Emotion.SADNESS: 1.0}, G1_PROFILE)  # no neck
    assert (Emotion.SADNESS, "neck_pitch") in lost


def test_nothing_is_reported_when_the_profile_can_show_everything():
    assert unexpressible({Emotion.SADNESS: 1.0}, G1_PROFILE_WITH_NECK) == ()


def test_zero_intensity_reports_nothing():
    assert unexpressible({Emotion.SADNESS: 0.0}, G1_PROFILE) == ()


def test_modulation_degrades_rather_than_raising_without_a_neck():
    p = GaitParameters()
    q = modulate(p, {Emotion.SADNESS: 1.0}, G1_PROFILE)
    assert q.cadence < p.cadence  # other cues still applied
    assert q.neck_pitch == p.neck_pitch  # dropped cue left untouched


def test_string_keys_are_normalised_to_emotions():
    p = GaitParameters()
    assert modulate(p, {"anger": 1.0}, G1_PROFILE_WITH_NECK) == modulate(
        p, {Emotion.ANGER: 1.0}, G1_PROFILE_WITH_NECK
    )


def test_unknown_emotion_raises():
    with pytest.raises(IntensityError):
        modulate(GaitParameters(), {"boredom": 0.5}, G1_PROFILE_WITH_NECK)


# Review Focus 1: out-of-range and non-finite intensities
@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_intensity_raises_before_any_nan_escapes(bad):
    with pytest.raises(IntensityError, match="anger"):
        modulate(GaitParameters(), {Emotion.ANGER: bad}, G1_PROFILE_WITH_NECK)
    with pytest.raises(IntensityError):
        unexpressible({Emotion.ANGER: bad}, G1_PROFILE)


def test_preset_keys():
    assert set(PRESET_PARAMS) == {"neutral", "sad", "angry", "happy"}


def test_every_preset_produces_finite_parameters():
    for name, p in PRESET_PARAMS.items():
        for f in dataclasses.fields(p):
            assert math.isfinite(getattr(p, f.name)), f"{name}.{f.name}"


_AMPLITUDES = (
    "arm_swing_amp",
    "arm_abduction",
    "expansiveness",
    "com_bounce_amp",
    "lateral_sway_amp",
    "step_height",
)


def test_amplitudes_saturate_at_zero_instead_of_flipping_phase():
    q = modulate(
        GaitParameters(),
        {Emotion.FEAR: 1.0, Emotion.SADNESS: 1.0, Emotion.DISGUST: 1.0},
        G1_PROFILE_WITH_NECK,
    )
    assert q.arm_swing_amp == 0.0


def test_every_preset_has_non_negative_amplitudes():
    for name, p in PRESET_PARAMS.items():
        for field in _AMPLITUDES:
            assert getattr(p, field) >= 0.0, f"{name}.{field}"
