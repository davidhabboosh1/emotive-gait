"""Per-cycle measurements on the synthetic full walker, whose posture is known by construction."""

from __future__ import annotations

import math
from dataclasses import fields, replace

import numpy as np
import pytest

from emotive_gait.gait.params import GaitParameters
from emotive_gait.gait.scaling import MEASURED_FIELDS, to_gait_parameters
from emotive_gait.mocap.extract import GaitMeasurement, extract_cycles, leg_length, summarise
from emotive_gait.mocap.normalise import normalise, ratio_delta
from emotive_gait.mocap.skeleton import Landmark
from mocap.walker import FPS, SWING_S, bend, extremum_offset_s, full_walker

walker = full_walker()
summary = summarise(walker.clip, walker.landmarks)
ZERO_BASE = GaitParameters(**{**dict.fromkeys(MEASURED_FIELDS, 0.0), "com_height_scale": 1.0})


def _with_positions(positions: np.ndarray):
    return replace(walker.clip, positions=positions)


def test_cadence_step_length_and_double_support_from_the_schedule():
    m = summary.mean
    assert m.cadence == pytest.approx(2.0, rel=0.02)  # two steps per 1 s cycle
    assert m.step_length == pytest.approx(0.3, rel=0.02)  # 0.6 m per cycle
    assert m.speed == pytest.approx(0.6, rel=0.02)
    # The position-extremum rule reads each stance longer than the 0.6 s schedule by the time
    # from swing start (and before swing end) to where foot speed equals pelvis speed:
    # stance 0.6 + 2 x 0.0287 = 0.657 s, so double support per 0.5 s step is 2 x 0.657 - 1.
    stance = 1.0 - SWING_S + 2.0 * extremum_offset_s(SWING_S)
    assert stance == pytest.approx(0.657, abs=0.001)
    assert m.double_support_frac == pytest.approx(2.0 * stance - 1.0, abs=0.05)
    assert m.swing_fraction == pytest.approx(1.0 - stance, abs=0.05)


def test_posture_angles_follow_the_construction():
    m = summary.mean
    assert m.trunk_pitch == pytest.approx(0.1, abs=0.01)
    assert m.neck_pitch == pytest.approx(0.3, abs=0.01)
    assert m.arm_swing_amp == pytest.approx(0.3, abs=0.01)
    assert m.elbow_flexion == pytest.approx(0.8, abs=0.01)
    assert m.elbow_swing_ratio == pytest.approx(0.4, abs=0.03)


def test_lengths_and_heights_follow_the_construction():
    m = summary.mean
    assert m.step_height == pytest.approx(0.05, abs=0.003)
    assert m.feet_spacing == pytest.approx(0.2, abs=0.005)
    assert m.lateral_sway_amp == pytest.approx(0.015, abs=0.002)
    assert m.com_bounce_amp == pytest.approx(0.0, abs=0.001)
    assert m.arm_abduction == pytest.approx(0.0, abs=0.01)
    assert m.pelvis_height == pytest.approx(0.9, abs=0.005)
    assert summary.leg_length == pytest.approx(0.45 + 0.45 + 0.08, abs=0.005)


def test_every_cycle_is_one_stride_and_none_is_turning():
    cyc = extract_cycles(walker.clip, walker.landmarks)
    assert len(cyc) >= 4 and summary.n_cycles == len(cyc) and summary.n_discarded == 0
    assert all(c.end_frame - c.start_frame == pytest.approx(FPS, abs=3) for c in cyc)


def test_missing_head_gives_nan_neck_pitch_and_nothing_else():  # Review Focus 1
    lm = {k: v for k, v in walker.landmarks.items() if k is not Landmark.HEAD}
    m = summarise(walker.clip, lm).mean
    assert math.isnan(m.neck_pitch) and not math.isnan(m.trunk_pitch)
    nan = [f.name for f in fields(GaitMeasurement) if math.isnan(float(getattr(m, f.name)))]
    assert nan == ["neck_pitch"]


def test_toes_are_used_for_the_foot_when_present():
    # A toe 0.1 m ahead of the ankle and 0.06 m lower lifts exactly as the ankle does.
    names = walker.clip.joint_names
    pos = walker.clip.positions
    toes = [pos[:, names.index(f"ankle_{s}")] + [0.1, 0.0, -0.06] for s in "lr"]
    clip = replace(
        walker.clip,
        joint_names=(*names, "toe_l", "toe_r"),
        positions=np.concatenate([pos, np.stack(toes, axis=1)], axis=1),
    )
    lm = {**walker.landmarks, Landmark.TOE_L: "toe_l", Landmark.TOE_R: "toe_r"}
    m = summarise(clip, lm).mean
    assert m.step_height == pytest.approx(0.05, abs=0.003)
    assert m.cadence == pytest.approx(2.0, rel=0.02)


def test_floor_height_does_not_move_relative_measures():
    a = summary.mean
    b = summarise(_with_positions(walker.clip.positions + [0.0, 0.0, 0.3]), walker.landmarks).mean
    for name in ("step_height", "com_bounce_amp", "trunk_pitch", "cadence", "step_length"):
        assert getattr(b, name) == pytest.approx(getattr(a, name), abs=1e-6), name


def test_mirrored_coordinates_keep_abduction_outward_positive():
    # Abduct both arms by moving the elbows outward by 0.03 m (upper arm 0.3 m), then mirror y.
    names = walker.clip.joint_names
    pos = walker.clip.positions.copy()
    for s, sign in (("l", 1.0), ("r", -1.0)):
        pos[:, names.index(f"elbow_{s}"), 1] += sign * 0.03
    out = summarise(_with_positions(pos), walker.landmarks).mean
    mirrored = summarise(_with_positions(pos * [1.0, -1.0, 1.0]), walker.landmarks).mean
    assert out.arm_abduction == pytest.approx(math.asin(0.1), abs=0.01)
    assert mirrored.arm_abduction == pytest.approx(out.arm_abduction, abs=1e-6)


def test_leg_length_is_nan_without_knees():
    lm = {k: v for k, v in walker.landmarks.items() if k not in (Landmark.KNEE_L, Landmark.KNEE_R)}
    assert math.isnan(leg_length(walker.clip, lm))


def test_too_short_clip_summarises_to_zero_cycles_with_a_reason():  # Review Focus 2
    short = replace(walker.clip, positions=walker.clip.positions[:100])
    s = summarise(short, walker.landmarks)
    assert s.n_cycles == 0 and math.isnan(s.mean.cadence) and sum(s.discard_reasons.values()) >= 1


def test_one_complete_cycle_is_summarised():
    one = _with_positions(walker.clip.positions[: int(2.3 * FPS)])
    s = summarise(one, walker.landmarks)
    assert s.n_cycles == 1 and s.n_discarded == 0
    assert s.mean.step_length == pytest.approx(0.3, rel=0.02)


def test_a_cycle_over_a_turn_is_discarded_and_straight_cycles_are_kept():
    # Straight, then a 90-degree left turn over 0.9 m (1.5 s), then straight again.
    start, length = 1.5, 0.9
    turned = summarise(
        _with_positions(bend(walker.clip.positions, start, length, math.pi / 2)), walker.landmarks
    )
    assert turned.discard_reasons.get("turning", 0) >= 1
    kept = extract_cycles(
        _with_positions(bend(walker.clip.positions, start, length, math.pi / 2)), walker.landmarks
    )
    assert len(kept) >= 2 and turned.n_cycles == len(kept)
    pelvis_x = walker.clip.positions[:, walker.clip.joint_names.index("pelvis"), 0]
    for c in kept:  # no kept cycle lies mostly on the turn
        x = pelvis_x[c.start_frame : c.end_frame]
        assert ((x > start) & (x < start + length)).mean() < 0.5
    # A cycle whose feet are all on a straight stretch measures as on the straight walk (the
    # one next to the turn places a foot on the curve, so its steps are legitimately shorter).
    clear = [
        c
        for c in kept
        if pelvis_x[c.end_frame] + 0.2 < start or pelvis_x[c.start_frame] - 0.2 > start + length
    ]
    assert len(clear) >= 2
    assert all(c.step_length == pytest.approx(0.3, rel=0.02) for c in clear)


def test_n_discarded_counts_every_discarded_cycle():
    # Freezing the whole body for 40 frames inside one cycle makes it 33 % long: irregular.
    pos = walker.clip.positions
    frozen = np.concatenate([pos[:300], np.repeat(pos[300:301], 40, axis=0), pos[300:]])
    s = summarise(_with_positions(frozen), walker.landmarks)
    assert s.discard_reasons.get("irregular", 0) >= 1
    cycle_level = s.discard_reasons.get("irregular", 0) + s.discard_reasons.get("turning", 0)
    assert s.n_discarded == cycle_level and s.n_cycles >= 3


def test_turning_throughout_summarises_to_zero_cycles():  # Review Focus 2
    # Bend the straight walk onto a circle of radius 0.8 m: forward x becomes arc length, so a
    # stance foot stays put and the heading turns at 0.6 / 0.8 = 0.75 rad/s, above the threshold.
    radius = 0.8
    pos = walker.clip.positions
    theta, r = pos[..., 0] / radius, radius - pos[..., 1]
    bent = np.stack([r * np.sin(theta), radius - r * np.cos(theta), pos[..., 2]], axis=-1)
    out = summarise(_with_positions(bent), walker.landmarks)
    assert out.n_cycles == 0 and out.discard_reasons.get("turning", 0) >= 1


def test_standing_still_summarises_to_zero_cycles_without_raising():
    still = _with_positions(np.repeat(walker.clip.positions[:1], 600, axis=0))
    s = summarise(still, walker.landmarks)
    assert s.n_cycles == 0 and s.discard_reasons


def test_missing_pelvis_summarises_to_zero_cycles_without_raising():
    lm = {k: v for k, v in walker.landmarks.items() if k is not Landmark.PELVIS}
    s = summarise(walker.clip, lm)
    assert s.n_cycles == 0 and s.discard_reasons


def test_normalise_then_to_gait_parameters_returns_the_measured_value():
    m = summary.mean
    n = normalise(m, leg_length=0.9)
    assert set(n) == set(MEASURED_FIELDS)
    p = to_gait_parameters(
        n, ZERO_BASE, scaling="froude", leg_length=0.9, nominal_pelvis_height=m.pelvis_height
    )
    assert p.step_length == pytest.approx(m.step_length) and p.cadence == pytest.approx(m.cadence)
    assert p.com_height_scale == pytest.approx(2.0)  # 1.0 base + pelvis_height / nominal


def test_ratio_delta_is_fractional_for_lengths_and_a_difference_for_angles():
    m = summary.mean
    other = replace(m, step_length=0.8 * m.step_length, trunk_pitch=m.trunk_pitch + 0.1)
    d = ratio_delta(other, m)
    assert set(d) == set(MEASURED_FIELDS)
    assert d["step_length"] == pytest.approx(-0.2)
    assert d["trunk_pitch"] == pytest.approx(0.1)
    assert d["cadence"] == pytest.approx(0.0) and d["com_height_scale"] == pytest.approx(0.0)
