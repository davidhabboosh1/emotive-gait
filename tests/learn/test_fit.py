from __future__ import annotations

import math

import pytest

from emotive_gait.emotion.evidence import Emotion
from emotive_gait.emotion.model import modulate
from emotive_gait.gait.params import GaitParameters
from emotive_gait.gait.scaling import G
from emotive_gait.learn.fit import fit_all, fit_parameter, hand_tuned_deltas, leave_one_actor_out
from emotive_gait.learn.synthetic import synthetic_table
from emotive_gait.robot.profiles.g1 import G1_PROFILE_WITH_NECK

EFFECTS = {
    ("anger", "cadence"): 0.08,
    ("sadness", "cadence"): -0.10,
    ("anger", "trunk_pitch"): 0.12,
}


def test_fixed_effects_are_recovered_within_their_intervals():
    df = synthetic_table(
        n_actors=20,
        emotions=("anger", "sadness"),
        effects=EFFECTS,
        actor_sd=0.03,
        cycle_sd=0.02,
        cycles_per_clip=10,
        seed=1,
    )
    cells = {c.emotion: c for c in fit_parameter(df, "cadence")}
    assert cells["anger"].ci_low < 0.08 < cells["anger"].ci_high
    assert cells["sadness"].estimate == pytest.approx(-0.10, abs=0.02)
    assert cells["anger"].grounding == "fitted" and cells["anger"].n_actors == 20
    assert cells["anger"].between_actor_sd == pytest.approx(0.03, abs=0.015)


def test_two_actor_emotion_is_thin_without_an_interval():
    df = synthetic_table(
        n_actors=2,
        emotions=("fear",),
        effects={("fear", "cadence"): 0.05},
        actor_sd=0.02,
        cycle_sd=0.02,
        cycles_per_clip=3,
        seed=2,
    )
    cell = fit_parameter(df, "cadence")[0]
    assert cell.grounding == "thin" and cell.ci_low is None and cell.n_actors == 2


def test_missing_parameter_rows_make_a_missing_cell():
    df = synthetic_table(
        n_actors=5,
        emotions=("anger",),
        effects={},
        actor_sd=0.01,
        cycle_sd=0.01,
        cycles_per_clip=4,
        seed=3,
    )
    df.loc[:, "neck_pitch"] = float("nan")
    assert fit_parameter(df, "neck_pitch")[0].grounding == "missing"


def test_hand_tuned_deltas_fold_expansiveness_and_normalise():
    h = hand_tuned_deltas("froude")
    assert h[("anger", "cadence")] == pytest.approx(0.60 * math.sqrt(0.69 / G))
    angry = modulate(GaitParameters(), {Emotion.ANGER: 1.0}, G1_PROFILE_WITH_NECK)
    assert h[("anger", "arm_swing_amp")] == pytest.approx(
        angry.expansiveness * angry.arm_swing_amp - 0.35
    )


def test_held_out_fit_beats_zero_and_a_wrong_hand_tuning_on_synthetic_data():
    df = synthetic_table(
        n_actors=12,
        emotions=("anger", "sadness"),
        effects=EFFECTS,
        actor_sd=0.03,
        cycle_sd=0.02,
        cycles_per_clip=8,
        seed=4,
    )
    wrong = {k: -v for k, v in EFFECTS.items()}
    out = {h.parameter: h for h in leave_one_actor_out(df, fit_all(df), wrong)}
    assert out["cadence"].e_fit < out["cadence"].e_zero < out["cadence"].e_hand
    assert 0.8 < out["cadence"].e_fit < 1.5  # near the between-actor floor
    assert out["overall"].n_pairs == 24
