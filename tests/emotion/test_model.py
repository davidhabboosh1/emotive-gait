import dataclasses
import math

import pytest
from artefacts import FULL_AGREEING, make_artefact

from emotive_gait.emotion.evidence import GROUNDED_EMOTIONS, Emotion, claims_for
from emotive_gait.emotion.learned import LearnedEmotionModel
from emotive_gait.emotion.model import (
    ANALYTIC,
    PRESET_PARAMS,
    EmotionModel,
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


# Two kinds of direction-0 claim. These are "no change" findings: the source measured the cue
# against neutral and found no effect, so the model must not move it. Every other direction-0
# claim means "not reported vs neutral" (e.g. only compared with another emotion): the evidence
# is silent, so the model is free there and the claim stays exempt.
NO_CHANGE = {(Emotion.ANGER, "neck_pitch"), (Emotion.ANGER, "com_height_scale")}


def _reported(model: LearnedEmotionModel, emotion: str, parameter: str) -> bool:
    return any(
        c.get("emotion") == emotion
        and c.get("parameter") == parameter
        and c.get("scaling", model.scaling) == model.scaling
        for c in model.artefact.contradictions
    )


def _check_learned_cell(model: LearnedEmotionModel, claim) -> None:
    """Spec 4.5: a fitted or thin cell agrees in sign, or is a reported contradiction."""
    emotion, parameter = claim.emotion.value, claim.parameter
    if _reported(model, emotion, parameter):
        return
    cell = model.cell(emotion, parameter)
    name = f"{emotion}.{parameter}"
    if claim.direction == 0:
        assert cell.direction_check in ("agrees_ci_spans_zero", "thin_unchecked"), (
            f"{name} moved off zero ({cell.direction_check}) and is not a reported "
            f"contradiction ({claim.citation})"
        )
        return
    assert cell.estimate * claim.direction > 0, (
        f"{name} estimate {cell.estimate} contradicts {claim.citation} and is not reported"
    )


def check_directions(model: EmotionModel) -> None:
    """Every grounded claim against ``model``; fails only on an unreported contradiction."""
    learned = isinstance(model, LearnedEmotionModel)
    p = GaitParameters()
    for emotion in GROUNDED_EMOTIONS:
        q = modulate(p, {emotion: 1.0}, G1_PROFILE_WITH_NECK, model=model)
        for c in claims_for(emotion):
            if not hasattr(p, c.parameter):
                continue
            if learned:
                if c.parameter == "expansiveness":
                    continue  # no estimate: fixed, carried by arm swing, abduction, step height
                if model.cell(emotion, c.parameter) is not None:
                    _check_learned_cell(model, c)
                    continue
                if _reported(model, emotion.value, c.parameter):
                    continue  # e.g. speed, derived from two cells: checked as a delta below
            delta = getattr(q, c.parameter) - getattr(p, c.parameter)
            if (emotion, c.parameter) in NO_CHANGE:
                assert c.direction == 0, f"{emotion}.{c.parameter} is no longer a no-change claim"
                assert abs(delta) < 1e-9, f"{emotion}.{c.parameter} moved by {delta} ({c.citation})"
                continue
            if c.direction == 0:
                continue
            assert delta * c.direction >= 0, f"{emotion}.{c.parameter} contradicts {c.citation}"


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
