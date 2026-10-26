import math

import numpy as np
import pytest

from emotive_gait.gait.footsteps import (
    Side,
    foot_point,
    foot_travel,
    pelvis_offset,
    pelvis_x,
    phase_at,
    rule_double_support,
    swing_fraction,
)
from emotive_gait.gait.params import GaitParameters

P = GaitParameters()  # step_length 0.35, cadence 1.8, double_support_frac 0.20


def test_left_swings_first_and_right_half_a_cycle_later():
    sf = swing_fraction(P)
    assert not foot_point(P, 0.1 * sf, Side.LEFT).in_stance
    assert foot_point(P, 0.1 * sf, Side.RIGHT).in_stance
    assert not foot_point(P, 0.5 + 0.1 * sf, Side.RIGHT).in_stance


def test_both_feet_are_down_through_double_support():
    sf = swing_fraction(P)
    for ph in np.linspace(sf, 0.5, 20, endpoint=False):
        assert foot_point(P, ph, Side.LEFT).in_stance and foot_point(P, ph, Side.RIGHT).in_stance


def test_stance_foot_moves_back_at_walking_speed():
    dt = 1e-4
    t0 = 0.6 / (P.cadence / 2)  # a time inside left stance
    x0 = foot_point(P, phase_at(P, t0), Side.LEFT).x
    x1 = foot_point(P, phase_at(P, t0 + dt), Side.LEFT).x
    assert (x1 - x0) / dt == pytest.approx(-P.speed, rel=1e-3)


def test_foot_x_runs_between_half_travel_each_way_with_bounded_overshoot():
    # Swing x inherits the stance drift (C1 handoff), so it overshoots +-D/2 slightly near the
    # swing ends; the handoff values themselves are exactly +-D/2.
    half = foot_travel(P) / 2
    sf = swing_fraction(P)
    assert foot_point(P, 0.0, Side.LEFT).x == pytest.approx(-half)
    assert foot_point(P, sf, Side.LEFT).x == pytest.approx(half)
    assert foot_point(P, sf - 1e-12, Side.LEFT).x == pytest.approx(half)
    xs = [foot_point(P, ph, Side.LEFT).x for ph in np.linspace(0, 1, 2001)]
    assert half <= max(xs) <= 1.15 * half
    assert -1.15 * half <= min(xs) <= -half


def test_swing_apex_is_step_height_and_stance_is_on_the_floor():
    sf = swing_fraction(P)
    assert foot_point(P, sf / 2, Side.LEFT).z == pytest.approx(P.step_height)
    assert all(
        foot_point(P, ph, Side.LEFT).z == 0.0 for ph in np.linspace(sf, 1, 50, endpoint=False)
    )


@pytest.mark.parametrize("ph", [0.0, swing_fraction(P), 0.5, 0.5 + swing_fraction(P)])
def test_paths_are_c2_at_every_handoff(ph):
    h = 1e-4  # the stencil samples z'' at +-2h, so a smaller h tightens the C2 check
    for side in Side:
        for attr in ("x", "z"):

            def g(p, side=side, attr=attr):
                return getattr(foot_point(P, p % 1.0, side), attr)

            left2 = (g(ph - h) - 2 * g(ph - 2 * h) + g(ph - 3 * h)) / h**2
            right2 = (g(ph + 3 * h) - 2 * g(ph + 2 * h) + g(ph + h)) / h**2
            assert g(ph - 1e-9) == pytest.approx(g(ph + 1e-9), abs=1e-6)
            left1 = (g(ph - h) - g(ph - 2 * h)) / h
            right1 = (g(ph + 2 * h) - g(ph + h)) / h
            assert left1 == pytest.approx(right1, abs=1e-2)  # first derivative continuous
            assert left2 == pytest.approx(right2, abs=0.5)  # second derivative continuous


def test_in_stance_at_the_exact_boundaries():
    assert foot_point(P, swing_fraction(P), Side.LEFT).in_stance
    assert not foot_point(P, 1.0, Side.LEFT).in_stance  # wraps to swing start
    assert not foot_point(P, 0.5, Side.RIGHT).in_stance


def test_in_place_when_step_length_is_zero():
    q = GaitParameters(step_length=0.0)
    pts = [foot_point(q, ph, Side.LEFT) for ph in np.linspace(0, 1, 101)]
    assert all(p.x == 0.0 and math.isfinite(p.z) for p in pts)
    assert max(p.z for p in pts) == pytest.approx(q.step_height)


def test_sway_goes_to_the_stance_side_and_bounce_has_two_periods():
    sf = swing_fraction(P)
    dy, _ = pelvis_offset(P, sf / 2)  # left swinging, right in stance: pelvis toward -y
    assert dy < 0
    for ph in (0.1, 0.3):
        assert pelvis_offset(P, ph)[1] == pytest.approx(pelvis_offset(P, ph + 0.5)[1])


def test_pelvis_x_is_speed_times_time():
    assert pelvis_x(P, 2.0) == pytest.approx(2.0 * P.speed)


def test_rule_double_support_reads_above_the_parameter_on_the_generator_shape():
    assert rule_double_support(GaitParameters()) == pytest.approx(0.41, abs=0.02)  # 0.20 + ~0.21
