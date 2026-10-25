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


def test_foot_x_range_is_half_the_travel_each_way():
    xs = [foot_point(P, ph, Side.LEFT).x for ph in np.linspace(0, 1, 2001)]
    assert max(xs) == pytest.approx(foot_travel(P) / 2, abs=1e-6)
    assert min(xs) == pytest.approx(-foot_travel(P) / 2, abs=1e-6)


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
            assert g(ph - h) == pytest.approx(g(ph + h), abs=5e-3)
            assert left2 == pytest.approx(right2, abs=0.5)  # second derivative continuous


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
