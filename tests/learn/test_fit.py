from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from emotive_gait.emotion.evidence import Emotion
from emotive_gait.emotion.model import modulate
from emotive_gait.gait.params import GaitParameters
from emotive_gait.gait.scaling import G
from emotive_gait.learn.fit import (
    _rejection,
    fit_all,
    fit_parameter,
    hand_tuned_deltas,
    leave_one_actor_out,
)
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


def test_healthy_fit_keeps_the_maximal_structure_and_per_emotion_slope_sds():
    df = synthetic_table(
        n_actors=20,
        emotions=("anger", "sadness"),
        effects=EFFECTS,
        actor_sd={"anger": 0.06, "sadness": 0.02},
        cycle_sd=0.02,
        cycles_per_clip=8,
        seed=3,
    )
    cells = {c.emotion: c for c in fit_parameter(df, "cadence")}
    for cell in cells.values():
        assert cell.converged_maximal is True and cell.fallback_reason is None
    assert cells["anger"].between_actor_sd > 1.5 * cells["sadness"].between_actor_sd


def test_degenerate_fit_falls_back_to_the_random_intercept():
    # Every actor's anger mean is the same constant, so the anger slope is exactly minus the
    # intercept: the maximal covariance is degenerate and the optimiser does not converge.
    df = synthetic_table(
        n_actors=20,
        emotions=("anger",),
        effects={},
        actor_sd=0.03,
        cycle_sd=0.02,
        cycles_per_clip=6,
        seed=1,
    )
    angry = df["emotion"] == "anger"
    df.loc[angry, "cadence"] = 2.0 + np.where(df.loc[angry, "cycle"] % 2 == 0, 0.02, -0.02)
    cell = fit_parameter(df, "cadence")[0]
    assert cell.converged_maximal is False and cell.fallback_reason == "not_converged"
    assert cell.grounding == "fitted" and cell.ci_low < cell.estimate < cell.ci_high


def test_converged_but_collinear_slope_is_singular():
    # As above with 10 actors: this seed converges, eigenvalue ratio ~1e-5, but the intercept
    # and slope correlate at about -0.9999, so the maximal structure is abandoned.
    df = synthetic_table(
        n_actors=10,
        emotions=("anger",),
        effects={},
        actor_sd=0.03,
        cycle_sd=0.02,
        cycles_per_clip=6,
        seed=3,
    )
    angry = df["emotion"] == "anger"
    df.loc[angry, "cadence"] = 2.0 + np.where(df.loc[angry, "cycle"] % 2 == 0, 0.02, -0.02)
    cell = fit_parameter(df, "cadence")[0]
    assert cell.converged_maximal is False and cell.fallback_reason == "singular_cov_re"


def _fake(converged=True, bse=(0.01, 0.02), cov=((1e-3, 0.0), (0.0, 2e-3))):
    return SimpleNamespace(converged=converged, bse_fe=np.array(bse), cov_re=np.array(cov))


@pytest.mark.parametrize(
    ("result", "messages", "reason"),
    [
        (None, [], "fit_error"),
        (_fake(converged=False), [], "not_converged"),
        (
            _fake(),
            ["MixedLM optimization failed, trying a different optimizer may help."],
            "terminal_warning",
        ),
        (_fake(), ["Gradient optimization failed, |grad| = 10.3"], "terminal_warning"),
        (_fake(bse=(0.01, float("nan"))), [], "nonfinite_se"),
        (_fake(cov=((1e-3, 0.0), (0.0, 1e-10))), [], "singular_cov_re"),
        (_fake(cov=((1e-3, 1e-3), (1e-3, 1e-3))), [], "singular_cov_re"),
        (_fake(cov=((1e-3, float("inf")), (0.0, 1e-3))), [], "singular_cov_re"),
        (_fake(cov=((1e-3, -0.9999e-3), (-0.9999e-3, 1e-3))), [], "singular_cov_re"),
        (
            _fake(),
            [
                "Maximum Likelihood optimization failed to converge. Check mle_retvals",
                "Retrying MixedLM optimization with lbfgs",
                "The MLE may be on the boundary of the parameter space.",
            ],
            None,
        ),
    ],
)
def test_fallback_trigger_branches(result, messages, reason):
    assert _rejection(result, messages) == reason
