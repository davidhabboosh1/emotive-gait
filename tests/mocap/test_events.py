"""Path frame and gait events on a synthetic walker (0.6 m/s, 120 Hz) and a 90-degree arc."""

from __future__ import annotations

import numpy as np
import pytest

from emotive_gait.mocap.events import cycles, foot_events, path_frame
from mocap.walker import FPS, make_arc_pelvis, straight_walker

straight = straight_walker()
arc_pelvis = make_arc_pelvis()


def test_path_frame_forward_follows_travel_and_marks_turns():
    pf = path_frame(straight.pelvis, FPS)
    assert np.allclose(pf.forward[60:-60], [1.0, 0.0, 0.0], atol=0.02)
    assert not pf.turning.any()
    pf2 = path_frame(arc_pelvis, FPS)
    assert pf2.turning[len(arc_pelvis) // 2]
    assert not pf2.turning[0] and not pf2.turning[-1]


def test_a_quarter_second_window_mistakes_sway_for_turning():
    pf = path_frame(straight.pelvis, FPS, smooth_s=0.25)
    assert pf.turning.mean() > 0.3  # why the default window is about one stride


def test_nan_foot_row_does_not_kill_events():
    clean = foot_events(straight.left_foot, path_frame(straight.pelvis, FPS), FPS)
    foot = straight.left_foot.copy()
    foot[30] = np.nan  # mid-swing
    ev = foot_events(foot, path_frame(straight.pelvis, FPS), FPS)
    assert abs(len(ev.strikes) - len(clean.strikes)) <= 1 and len(ev.strikes) >= 4
    assert not ev.stance[30]


def test_noisy_in_place_motion_is_refused():
    rng = np.random.default_rng(0)
    pelvis = np.array([0.0, 0.0, 0.9]) + rng.normal(0.0, 0.001, (720, 3))
    with pytest.raises(ValueError, match="forward travel"):
        path_frame(pelvis, FPS)


def test_path_frame_axes_are_orthonormal_and_to_path_projects():
    pf = path_frame(arc_pelvis, FPS)
    assert np.allclose(np.linalg.norm(pf.forward, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(pf.left, axis=1), 1.0)
    assert np.allclose(np.sum(pf.forward * pf.left, axis=1), 0.0)
    assert np.allclose(pf.forward[:, 2], 0.0)
    # A point 1 m ahead along the heading and 0.3 m to its left, 0.2 m up.
    world = pf.origin + pf.forward + 0.3 * pf.left + np.array([0.0, 0.0, 0.2])
    assert np.allclose(pf.to_path(world), [1.0, 0.3, 0.2])


def test_strikes_and_toe_offs_match_the_synthetic_schedule():
    pf = path_frame(straight.pelvis, FPS)
    ev = foot_events(straight.left_foot, pf, FPS)
    assert len(ev.strikes) >= 5
    assert np.allclose(ev.strikes[1:] - ev.strikes[:-1], 120, atol=6)  # one stride per second
    assert len(ev.toe_offs) >= 5
    # smoothstep ends are slow
    assert np.allclose(ev.toe_offs - ev.strikes[: len(ev.toe_offs)], 60, atol=6)
    assert ev.stance.dtype == bool and ev.stance.shape == (len(straight.pelvis),)


def test_cycles_pair_left_strikes_that_bracket_one_right_strike():
    pf = path_frame(straight.pelvis, FPS)
    left = foot_events(straight.left_foot, pf, FPS)
    right = foot_events(straight.right_foot, pf, FPS)
    cyc = cycles(left, right)
    assert len(cyc) == 5 and all(e - s == pytest.approx(120, abs=6) for s, e in cyc)


def test_cycles_skip_pairs_without_exactly_one_right_strike():
    pf = path_frame(straight.pelvis, FPS)
    left = foot_events(straight.left_foot, pf, FPS)
    right = foot_events(straight.right_foot, pf, FPS)
    assert cycles(left, type(right)(right.strikes[:0], right.toe_offs[:0], right.stance)) == []


def test_in_place_motion_is_refused():
    with pytest.raises(ValueError, match="forward travel"):
        path_frame(np.tile([0.0, 0.0, 0.9], (600, 1)), FPS)
