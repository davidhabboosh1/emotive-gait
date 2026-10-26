from __future__ import annotations

import math

import numpy as np
import pandas as pd

from emotive_gait.emotion.model import AXIS_GAINS
from emotive_gait.learn.fit import FITTED_EMOTIONS, between_actor_sd
from emotive_gait.learn.structure import STRUCTURES, _s3, _weights, structure_test

PARAMS = [*AXIS_GAINS, "com_height_scale", "elbow_swing_ratio", "step_height"]
SD = {p: 1.0 for p in PARAMS}
COLUMNS = ["dataset", "actor", "emotion", "parameter", "delta"]


def _deltas(n_actors=30, seed=0, drop=()):
    rng = np.random.default_rng(seed)
    w_true = rng.normal(size=(4, 3))
    g_true = rng.normal(size=(3, len(PARAMS)))
    means = w_true @ g_true
    rows = []
    for a in range(n_actors):
        for e, emotion in enumerate(FITTED_EMOTIONS):
            if (a, emotion) in drop:
                continue
            for p, name in enumerate(PARAMS):
                noise = rng.normal(scale=0.3)
                rows.append(("syn", f"a{a}", emotion, name, means[e, p] + noise))
    return pd.DataFrame(rows, columns=COLUMNS)


deltas_rank3 = _deltas()


def _res(deltas=deltas_rank3, sd=SD):
    return {r.name: r for r in structure_test(deltas, sd, parameters=PARAMS)}


def test_full_model_fits_in_sample_at_least_as_well_as_every_constraint():
    res = _res()
    assert (
        res["S4_full"].r2_in_sample
        >= res["S3_rank3"].r2_in_sample
        >= res["S2_fixed_axes_free_loadings"].r2_in_sample
    )
    assert res["S4_full"].n_free == 4 * len(PARAMS)
    assert res["S1_analytic"].n_free == len(AXIS_GAINS)


def test_results_come_in_structure_order_with_documented_free_counts():
    results = structure_test(deltas_rank3, SD, parameters=PARAMS)
    assert tuple(r.name for r in results) == STRUCTURES
    res = _res()
    assert res["S2_fixed_axes_free_loadings"].n_free == 3 * len(PARAMS)
    assert res["S3_rank3"].n_free == 3 * (4 + len(PARAMS)) - 9


def test_rank3_truth_is_recovered_by_rank3_but_rank4_overfits_held_out():
    res = _res()
    assert res["S3_rank3"].e_heldout <= res["S4_full"].e_heldout + 0.05
    assert res["S4_full"].r2_in_sample > 0.9


def test_fear_offsets_row_frees_exactly_four_more_parameters():
    res = _res()
    assert res["S1_fear_offsets"].n_free == res["S1_analytic"].n_free + 4
    assert res["S1_fear_offsets"].r2_in_sample >= res["S1_analytic"].r2_in_sample


def test_missing_actor_emotion_pairs_are_fitted_on_the_rows_that_exist():
    drop = {(a, "fear") for a in range(20)} | {(25, "anger")}
    for r in _res(_deltas(drop=drop)).values():
        assert math.isfinite(r.r2_in_sample) and math.isfinite(r.e_heldout)


def test_nan_deltas_and_unusable_sd_are_excluded():
    df = deltas_rank3.copy()
    df.loc[df.index[:5], "delta"] = np.nan
    sd = {**SD, PARAMS[0]: float("nan")}
    for r in structure_test(df, sd, parameters=PARAMS):
        assert math.isfinite(r.r2_in_sample) and math.isfinite(r.e_heldout)


def test_exact_analytic_structure_is_recovered_by_s1():
    w = _weights()
    axes = ("activation", "valence", "tension")
    rows = []
    for a in range(10):
        for e, emotion in enumerate(FITTED_EMOTIONS):
            for name in PARAMS:
                axis, gain = AXIS_GAINS.get(name, (None, 0.0))
                mean = gain * w[e, axes.index(axis)] if axis else 0.0
                rows.append(("syn", f"a{a}", emotion, name, mean + 0.01 * ((a * 7 + e) % 3 - 1)))
    df = pd.DataFrame(rows, columns=COLUMNS)
    res = {r.name: r for r in structure_test(df, between_actor_sd(df), parameters=PARAMS)}
    assert res["S1_analytic"].e_heldout < 1.5 * res["S4_full"].e_heldout


def test_s3_fit_has_rank_three_where_the_full_means_have_rank_four():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(4, len(PARAMS)))
    assert np.linalg.matrix_rank(y) == 4
    emo = np.arange(4)
    assert np.linalg.matrix_rank(_s3(emo, y)) == 3


def test_rank3_is_strictly_worse_than_full_in_sample_on_noisy_data():
    res = _res()
    assert res["S3_rank3"].r2_in_sample < res["S4_full"].r2_in_sample


def test_s3_does_not_depend_on_parameter_units():
    scaled = deltas_rank3.copy()
    name = PARAMS[0]
    scaled.loc[scaled["parameter"] == name, "delta"] *= 1000.0
    a = _res()["S3_rank3"]
    b = _res(scaled, {**SD, name: 1000.0})["S3_rank3"]
    assert math.isclose(a.r2_in_sample, b.r2_in_sample, rel_tol=1e-9)
    assert math.isclose(a.e_heldout, b.e_heldout, rel_tol=1e-9)


def test_fear_offset_structure_is_recovered_exactly_where_plain_s1_is_not():
    w = _weights()
    axes = ("activation", "valence", "tension")
    rows = []
    for a in range(8):
        for e, emotion in enumerate(FITTED_EMOTIONS):
            for name in PARAMS:
                axis, gain = AXIS_GAINS.get(name, (None, 0.0))
                mean = gain * w[e, axes.index(axis)] if axis else 0.0
                if emotion == "fear" and name == "cadence":
                    mean += 0.5
                rows.append(("syn", f"a{a}", emotion, name, mean))
    df = pd.DataFrame(rows, columns=COLUMNS)
    res = _res(df, {p: 0.1 for p in PARAMS})
    assert res["S1_fear_offsets"].e_heldout < 1e-9
    assert res["S1_analytic"].e_heldout > 0.01
