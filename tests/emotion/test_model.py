import dataclasses
import math

import pytest
from artefacts import FULL_AGREEING, make_artefact
from directions import check_directions

from emotive_gait.emotion.evidence import (
    NO_CHANGE_CLAIMS,
    Emotion,
)
from emotive_gait.emotion.learned import LearnedEmotionModel
from emotive_gait.emotion.model import (
    ANALYTIC,
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


_LEARNED_FROUDE = LearnedEmotionModel(make_artefact(FULL_AGREEING))
_LEARNED_RATIO = LearnedEmotionModel(make_artefact(FULL_AGREEING), scaling="ratio")


@pytest.mark.parametrize(
    "model",
    [ANALYTIC, _LEARNED_FROUDE, _LEARNED_RATIO],
    ids=["analytic", "learned_froude", "learned_ratio"],
)
def test_modulation_direction_agrees_with_every_grounded_claim(model):
    check_directions(model)


def test_an_unreported_contradiction_fails_the_direction_test():
    m = LearnedEmotionModel(make_artefact({**FULL_AGREEING, ("sadness", "arm_swing_amp"): +0.3}))
    with pytest.raises(AssertionError, match="sadness.arm_swing_amp"):
        check_directions(m)


def test_a_reported_contradiction_passes():
    m = LearnedEmotionModel(
        make_artefact(
            {**FULL_AGREEING, ("sadness", "arm_swing_amp"): +0.3},
            contradictions=[{"emotion": "sadness", "parameter": "arm_swing_amp"}],
        )
    )
    check_directions(m)


def test_an_unreported_no_change_violation_fails_the_direction_test():
    m = LearnedEmotionModel(make_artefact({**FULL_AGREEING, ("anger", "neck_pitch"): 0.2}))
    with pytest.raises(AssertionError, match="anger.neck_pitch"):
        check_directions(m)


def test_a_no_change_check_ignores_the_artefacts_own_verdict():
    m = LearnedEmotionModel(make_artefact({**FULL_AGREEING, ("anger", "neck_pitch"): 0.2}))
    cell = m.artefact.cells[("froude", "anger", "neck_pitch")]
    lying = dataclasses.replace(cell, direction_check="agrees_ci_spans_zero")
    art = dataclasses.replace(
        m.artefact, cells={**m.artefact.cells, ("froude", "anger", "neck_pitch"): lying}
    )
    with pytest.raises(AssertionError, match="anger.neck_pitch"):
        check_directions(LearnedEmotionModel(art))


def test_a_not_reported_direction_zero_claim_is_exempt_for_the_learned_model():
    # happiness cadence is direction 0 "not reported vs neutral", not a no-change claim
    assert (Emotion.HAPPINESS, "cadence") not in NO_CHANGE_CLAIMS
    check_directions(
        LearnedEmotionModel(make_artefact({**FULL_AGREEING, ("happiness", "cadence"): 0.5}))
    )


# Fear's claims: cadence up, step length down, speed down. A big cadence rise with a small
# step-length drop makes the modulated speed rise although both cells agree with their claims.
_FEAR_FAST = {**FULL_AGREEING, ("fear", "cadence"): 0.3, ("fear", "step_length"): -0.01}


def test_an_unreported_speed_contradiction_fails():
    with pytest.raises(AssertionError, match="(?i)fear.speed contradicts"):
        check_directions(LearnedEmotionModel(make_artefact(_FEAR_FAST)))


def test_a_reported_speed_contradiction_passes():
    reported = [{"emotion": "fear", "parameter": "speed", "scaling": "froude"}]
    check_directions(LearnedEmotionModel(make_artefact(_FEAR_FAST, contradictions=reported)))


def test_missing_cells_of_the_learned_model_are_checked_against_the_analytic_delta():
    effects = {k: v for k, v in FULL_AGREEING.items() if k != ("sadness", "trunk_pitch")}
    flipped = LearnedEmotionModel(make_artefact(effects), fallback=_NegatedSadness())
    assert ("sadness", "trunk_pitch") in flipped.analytic_cells
    with pytest.raises(AssertionError, match="(?i)sadness.trunk_pitch contradicts"):
        check_directions(flipped)  # the analytic share now has the wrong sign


class _NegatedSadness:
    """The analytic model with every sadness delta's sign flipped."""

    def deltas(self, intensities, profile):
        d = ANALYTIC.deltas(intensities, profile)
        return {k: -v for k, v in d.items()} if Emotion.SADNESS in intensities else d


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


def test_analytic_model_object_matches_modulate_defaults():
    for e in Emotion:
        a = modulate(GaitParameters(), {e: 0.7}, G1_PROFILE_WITH_NECK)
        b = modulate(GaitParameters(), {e: 0.7}, G1_PROFILE_WITH_NECK, model=ANALYTIC)
        assert a == b
    assert PRESET_PARAMS["angry"] == modulate(
        GaitParameters(), {Emotion.ANGER: 1.0}, G1_PROFILE_WITH_NECK
    )
