"""The three-axis structure test (design spec 4.3).

Five structures of the emotion-by-parameter effect matrix ``B`` (4 emotions x P parameters) are
compared on actor-by-emotion mean deltas (``fit.actor_deltas``, one row per actor and emotion,
equal weight) by in-sample R^2 and by leave-one-actor-out error, both standardised by the
between-actor SD of each parameter:

* ``S1_analytic``: ``B = W G`` with ``W`` the fixed ``AXIS_WEIGHTS`` and ``G`` sparse as
  ``AXIS_GAINS`` assigns one axis per parameter; only the gains are fitted.
* ``S1_fear_offsets``: S1 plus a free fear entry for each ``CATEGORY_OFFSETS[FEAR]`` parameter.
* ``S2_fixed_axes_free_loadings``: ``W`` fixed, ``G`` dense 3 x P.
* ``S3_rank3``: rank-3 truncated SVD of the per-emotion mean matrix.
* ``S4_full``: the per-emotion mean (saturated).

Caveat, part of the result: four emotions span at most a four-dimensional space, so rank 3 removes
a single dimension and the test has low power. An actor may lack emotions; every fit uses the
rows that exist, and a held-out actor is predicted only for the emotions they performed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from emotive_gait.emotion.axes import AXIS_WEIGHTS, CATEGORY_OFFSETS
from emotive_gait.emotion.evidence import Emotion
from emotive_gait.emotion.model import AXIS_GAINS
from emotive_gait.learn.fit import FITTED_EMOTIONS

STRUCTURES = (
    "S1_analytic",
    "S1_fear_offsets",
    "S2_fixed_axes_free_loadings",
    "S3_rank3",
    "S4_full",
)
_AXES = ("activation", "valence", "tension")
_RANK = 3


@dataclass(frozen=True)
class StructureResult:
    name: str
    n_free: int
    r2_in_sample: float
    e_heldout: float


def _weights() -> np.ndarray:
    """``W``: 4 x 3 ``AXIS_WEIGHTS`` in ``FITTED_EMOTIONS`` order (activation, valence, tension)."""
    return np.array(
        [[getattr(AXIS_WEIGHTS[Emotion(e)], a) for a in _AXES] for e in FITTED_EMOTIONS]
    )


def _means(emo: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-emotion mean of the rows that exist (4 x P); an empty cell is 0."""
    out = np.zeros((len(FITTED_EMOTIONS), y.shape[1]))
    for e in range(len(FITTED_EMOTIONS)):
        for p in range(y.shape[1]):
            v = y[(emo == e) & ~np.isnan(y[:, p]), p]
            out[e, p] = v.mean() if v.size else 0.0
    return out


def _gain(w: np.ndarray, emo: np.ndarray, y: np.ndarray, skip: int | None = None) -> float:
    """Least-squares gain of ``y ~ gain * w[emo]`` over the existing rows (not ``skip``'s)."""
    ok = ~np.isnan(y)
    if skip is not None:
        ok &= emo != skip
    x = w[emo[ok]]
    den = float(x @ x)
    return float(x @ y[ok]) / den if den > 0.0 else 0.0


def _s1(parameters: Sequence[str], offsets: bool) -> Callable:
    w_all = _weights()
    fear = FITTED_EMOTIONS.index("fear")
    free = set(CATEGORY_OFFSETS[Emotion.FEAR]) if offsets else set()

    def fit(emo: np.ndarray, y: np.ndarray) -> np.ndarray:
        b = np.zeros((len(FITTED_EMOTIONS), len(parameters)))
        for p, name in enumerate(parameters):
            axis, _ = AXIS_GAINS.get(name, (None, 0.0))
            col = w_all[:, _AXES.index(axis)] if axis else np.zeros(len(FITTED_EMOTIONS))
            drop = fear if name in free else None
            if axis:
                b[:, p] = _gain(col, emo, y[:, p], skip=drop) * col
            if drop is not None:
                rows = (emo == fear) & ~np.isnan(y[:, p])
                if rows.any():
                    b[fear, p] = y[rows, p].mean()
        return b

    return fit


def _s2(parameters: Sequence[str]) -> Callable:
    w_all = _weights()

    def fit(emo: np.ndarray, y: np.ndarray) -> np.ndarray:
        g = np.zeros((len(_AXES), len(parameters)))
        for p in range(len(parameters)):
            ok = ~np.isnan(y[:, p])
            if ok.any():
                g[:, p] = np.linalg.lstsq(w_all[emo[ok]], y[ok, p], rcond=None)[0]
        return w_all @ g

    return fit


def _s3(emo: np.ndarray, y: np.ndarray) -> np.ndarray:
    u, s, vt = np.linalg.svd(_means(emo, y), full_matrices=False)
    return (u[:, :_RANK] * s[:_RANK]) @ vt[:_RANK]


def _s4(emo: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _means(emo, y)


def _n_free(parameters: Sequence[str]) -> dict[str, int]:
    p = len(parameters)
    s1 = sum(1 for name in parameters if name in AXIS_GAINS)
    fear = sum(1 for name in parameters if name in CATEGORY_OFFSETS[Emotion.FEAR])
    n_emotions = len(FITTED_EMOTIONS)
    return {
        "S1_analytic": s1,
        "S1_fear_offsets": s1 + fear,
        "S2_fixed_axes_free_loadings": len(_AXES) * p,
        # rank-3 factorisation W G of a 4 x P matrix: 3 (4 + P) minus the 3 x 3 gauge freedom
        "S3_rank3": _RANK * (n_emotions + p) - _RANK * _RANK,
        "S4_full": n_emotions * p,
    }


def structure_test(
    deltas: pd.DataFrame, sd: Mapping[str, float], *, parameters: Sequence[str]
) -> list[StructureResult]:
    """The five structures, in ``STRUCTURES`` order, scored in sample and on held-out actors.

    ``deltas`` is ``actor_deltas`` output. Parameters whose ``sd`` is not a positive finite
    number are fitted but excluded from both scores. ``e_heldout`` is the unweighted mean over
    parameters of the RMS of (prediction - observed) / sd over held-out (actor, emotion) pairs;
    ``r2_in_sample`` is 1 - SS_res / SS_tot of the standardised deltas (centred per parameter)
    from the fit on all actors. NaN deltas are excluded from every sum.
    """
    parameters = list(parameters)
    index = {e: i for i, e in enumerate(FITTED_EMOTIONS)}
    rows = deltas[deltas["emotion"].isin(index) & deltas["parameter"].isin(parameters)]
    wide = rows.pivot_table(
        index=["dataset", "actor", "emotion"], columns="parameter", values="delta", aggfunc="mean"
    ).reindex(columns=parameters)
    y = wide.to_numpy(dtype=float)
    emo = np.array([index[e] for e in wide.index.get_level_values("emotion")], dtype=int)
    actor = np.array([f"{d}/{a}" for d, a, _ in wide.index], dtype=object)
    scale = np.array([float(sd.get(p, np.nan)) for p in parameters])
    scored = np.isfinite(scale) & (scale > 0.0)
    scale = np.where(scored, scale, 1.0)

    fitters = {
        "S1_analytic": _s1(parameters, False),
        "S1_fear_offsets": _s1(parameters, True),
        "S2_fixed_axes_free_loadings": _s2(parameters),
        "S3_rank3": _s3,
        "S4_full": _s4,
    }
    n_free = _n_free(parameters)
    centre = np.nanmean(y, axis=0)
    ss_tot = float(np.nansum((((y - centre) / scale)[:, scored]) ** 2))
    actors = list(dict.fromkeys(actor))

    results = []
    for name in STRUCTURES:
        fit = fitters[name]
        resid = (y - fit(emo, y)[emo]) / scale
        ss_res = float(np.nansum(resid[:, scored] ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

        sq: list[list[float]] = [[] for _ in parameters]
        for held in actors:
            test = actor == held
            pred = fit(emo[~test], y[~test])[emo[test]]
            err = (pred - y[test]) / scale
            for p in range(len(parameters)):
                sq[p].extend((err[~np.isnan(err[:, p]), p] ** 2).tolist())
        per_param = [
            float(np.sqrt(np.mean(sq[p]))) for p in range(len(parameters)) if scored[p] and sq[p]
        ]
        e_held = float(np.mean(per_param)) if per_param else float("nan")
        results.append(StructureResult(name, n_free[name], r2, e_held))
    return results
