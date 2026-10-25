import pytest

from emotive_gait.emotion.axes import (
    AXIS_WEIGHTS,
    CATEGORY_OFFSETS,
    AffectiveAxes,
    axes_from,
)
from emotive_gait.emotion.evidence import Emotion


def test_all_zero_intensities_give_zero_axes():
    a = axes_from({e: 0.0 for e in Emotion})
    assert (a.activation, a.valence, a.tension) == (0.0, 0.0, 0.0)


def test_activation_ordering_follows_measured_speeds():
    def act(e):
        return axes_from({e: 1.0}).activation

    assert (
        act(Emotion.ANGER) > act(Emotion.HAPPINESS) > 0 > act(Emotion.FEAR) > act(Emotion.SADNESS)
    )


def test_valence_separates_sadness_from_happiness():
    sad = axes_from({Emotion.SADNESS: 1.0}).valence
    happy = axes_from({Emotion.HAPPINESS: 1.0}).valence
    assert sad < 0 < happy


def test_tension_groups_anger_and_fear_against_happiness_and_sadness():
    def t(e):
        return axes_from({e: 1.0}).tension

    assert t(Emotion.ANGER) > 0 and t(Emotion.FEAR) > 0
    assert t(Emotion.HAPPINESS) < 0 and t(Emotion.SADNESS) < 0


def test_opposing_emotions_cancel_toward_neutral():
    """Averaging happy and sad yields neutral, not 'mixed' (Appendix A.6)."""
    a = axes_from({Emotion.HAPPINESS: 1.0, Emotion.SADNESS: 1.0})
    assert a.valence == pytest.approx(0.0, abs=1e-9)


def test_intensity_scales_each_axis_monotonically():
    prev = -1e9
    for i in (0.0, 0.25, 0.5, 0.75, 1.0):
        cur = axes_from({Emotion.ANGER: i}).activation
        assert cur > prev
        prev = cur


def test_fear_carries_a_category_offset_for_crouch():
    assert CATEGORY_OFFSETS[Emotion.FEAR]["com_height_scale"] < 0


def test_axes_are_a_linear_sum_of_weights():
    a = axes_from({Emotion.ANGER: 0.5, Emotion.FEAR: 0.25})
    w_a, w_f = AXIS_WEIGHTS[Emotion.ANGER], AXIS_WEIGHTS[Emotion.FEAR]
    assert a.activation == pytest.approx(0.5 * w_a.activation + 0.25 * w_f.activation)
    assert a.valence == pytest.approx(0.5 * w_a.valence + 0.25 * w_f.valence)
    assert a.tension == pytest.approx(0.5 * w_a.tension + 0.25 * w_f.tension)


def test_every_emotion_has_weights_in_range_and_an_offsets_entry():
    for e in Emotion:
        w = AXIS_WEIGHTS[e]
        for v in (w.activation, w.valence, w.tension):
            assert -1.0 <= v <= 1.0
        assert e in CATEGORY_OFFSETS


def test_tables_are_read_only():
    with pytest.raises(TypeError):
        AXIS_WEIGHTS[Emotion.ANGER] = AffectiveAxes(0, 0, 0)  # type: ignore[index]
    with pytest.raises(TypeError):
        CATEGORY_OFFSETS[Emotion.FEAR]["cadence"] = 0.0  # type: ignore[index]
