import numpy as np
import pytest

from emotive_gait.gait.spline import NonPeriodicKnotsError, PeriodicCurve


def test_curve_interpolates_its_knots():
    c = PeriodicCurve.from_knots([0.0, 0.25, 0.5, 0.75], [0.0, 1.0, 0.0, -1.0])
    assert c(0.25) == pytest.approx(1.0)
    assert c(0.75) == pytest.approx(-1.0)


def test_phase_wraps_modulo_one():
    c = PeriodicCurve.from_knots([0.0, 0.25, 0.5, 0.75], [0.0, 1.0, 0.0, -1.0])
    assert c(1.25) == pytest.approx(c(0.25))
    assert c(-0.75) == pytest.approx(c(0.25))


def test_c2_continuous_across_the_wrap():
    """bc_type='periodic' matches first AND second derivatives, unlike the old
    GUI's hand-shared endpoint Y which only ever bought C0."""
    c = PeriodicCurve.from_knots([0.0, 0.25, 0.5, 0.75], [0.0, 1.0, 0.0, -1.0])
    eps = 1e-6
    for order in (1, 2):
        before = c.derivative(1.0 - eps, order)
        after = c.derivative(0.0 + eps, order)
        assert before == pytest.approx(after, abs=1e-3)


def test_vectorized_call_matches_scalar_calls():
    c = PeriodicCurve.from_knots([0.0, 0.5], [0.0, 1.0])
    xs = np.linspace(0, 1, 11)
    assert np.allclose(c(xs), [c(float(x)) for x in xs])


def test_explicitly_non_periodic_closing_knot_is_rejected():
    with pytest.raises(NonPeriodicKnotsError):
        PeriodicCurve.from_knots([0.0, 0.5, 1.0], [0.0, 1.0, 0.7])


def test_explicit_matching_closing_knot_is_accepted():
    c = PeriodicCurve.from_knots([0.0, 0.5, 1.0], [0.0, 1.0, 0.0])
    assert c(0.5) == pytest.approx(1.0)
    assert c(1.0) == pytest.approx(c(0.0))


@pytest.mark.parametrize(
    "phase",
    [[0.1, 0.5], [0.0, 0.5, 0.25], [0.0, 0.5, 0.5], [0.0, 1.5], [-0.1, 0.5]],
)
def test_invalid_phases_are_rejected(phase):
    with pytest.raises(ValueError):
        PeriodicCurve.from_knots(phase, [0.0] * len(phase))


def _curve(values=(0.0, 1.0, 0.0, -1.0)):
    return PeriodicCurve.from_knots([0.0, 0.25, 0.5, 0.75], list(values))


def test_curves_with_identical_knots_are_equal():
    a, b = _curve(), _curve()
    assert a == b
    assert hash(a) == hash(b)


def test_curves_with_different_knots_are_not_equal():
    assert _curve() != _curve((0.0, 2.0, 0.0, -1.0))


def test_equality_does_not_raise_against_other_types():
    assert (_curve() == 42) is False


def test_curve_is_usable_as_dict_key_and_set_member():
    a, b = _curve(), _curve()
    assert {a: "x"}[b] == "x"
    assert b in {a}
