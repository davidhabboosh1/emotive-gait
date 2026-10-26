"""The mixed-effects emotion model, its held-out-actor validation and its baselines.

Design spec 4.3. One model per parameter, fitted by REML with ``statsmodels`` ``MixedLM``::

    x[a, e, c] = mu + beta[e] + u[a] + v[a, e] + eps[a, e, c]        beta[neutral] = 0

``neutral`` is a level of ``emotion``, not a subtraction, so each ``beta[e]`` carries the
uncertainty of every actor's neutral mean. Actors are the groups (keyed by dataset and actor, so
two datasets reusing an actor name stay apart). The maximal random structure (an intercept and
one slope per emotion) falls back to the random intercept alone when it does not converge or its
covariance is singular; the fallback is recorded as ``converged_maximal=False`` and nothing is
tuned to make it converge.

Grounding per ``(emotion, parameter)`` cell: ``fitted`` with three or more actors (95 % Wald
interval), ``thin`` with one or two (estimate, no interval), ``missing`` with none (estimate 0.0,
delegated to the analytic model).

``statsmodels`` is imported lazily by the fitting functions, so the pure-pandas parts of this
module (``actor_deltas``, ``between_actor_sd``, ``hand_tuned_deltas``) import without it.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Literal

import numpy as np
import pandas as pd

from emotive_gait.emotion.evidence import Emotion
from emotive_gait.emotion.model import modulate
from emotive_gait.gait.params import GaitParameters
from emotive_gait.gait.scaling import LENGTH_FIELDS, MEASURED_FIELDS, G, Scaling
from emotive_gait.robot.profiles.g1 import G1_PROFILE_WITH_NECK

FITTED_EMOTIONS = ("anger", "fear", "happiness", "sadness")
MIN_ACTORS_FITTED = 3

_EXPANSIVE = ("arm_swing_amp", "arm_abduction", "step_height")
_RATIO_SCALED = (*LENGTH_FIELDS, "cadence", "com_height_scale")
_TERM = "C(emotion, Treatment('neutral'))"
_GROUP = "_group"
_SINGULAR_VARIANCE = 1e-10
# ConvergenceWarnings that do not signal a failed fit: statsmodels warns "on the boundary"
# whenever any random-effect variance is below 0.01 in absolute units, which every normalised
# parameter is (a variance of 1e-3 is a 0.03 SD); the boundary is judged by ``_singular``
# instead. "Retrying" announces a fall-through to the next optimiser, whose outcome is judged.
_NOT_A_FAILURE = ("The MLE may be on the boundary", "Retrying MixedLM optimization")


@dataclass(frozen=True)
class Cell:
    emotion: str
    parameter: str
    estimate: float
    ci_low: float | None
    ci_high: float | None
    between_actor_sd: float
    residual_sd: float
    n_actors: int
    n_cycles: int
    n_datasets: int
    n_mapped_actors: int
    grounding: Literal["fitted", "thin", "missing"]
    converged_maximal: bool
    per_dataset: Mapping[str, float]


@dataclass(frozen=True)
class HeldOut:
    parameter: str
    e_fit: float
    e_hand: float
    e_zero: float
    e_fit_natural: float
    e_hand_natural: float
    n_pairs: int


def _group_key(df: pd.DataFrame) -> pd.Series:
    return df["dataset"].astype(str) + "/" + df["actor"].astype(str)


def _usable(df: pd.DataFrame, parameter: str) -> pd.DataFrame:
    """Neutral and fitted-emotion rows with ``parameter`` present, from actors that have a
    neutral row for it (an actor without one cannot inform a within-actor effect), plus the
    group key."""
    keep = df["emotion"].isin(("neutral", *FITTED_EMOTIONS)) & df[parameter].notna()
    rows = df[keep].assign(**{_GROUP: _group_key(df[keep])})
    with_neutral = set(rows.loc[rows["emotion"] == "neutral", _GROUP])
    return rows[rows[_GROUP].isin(with_neutral)]


def _level(emotion: str) -> str:
    return f"{_TERM}[T.{emotion}]"


def _singular(cov_re: pd.DataFrame) -> bool:
    matrix = np.asarray(cov_re, dtype=float)
    if not np.all(np.isfinite(matrix)) or np.any(np.diag(matrix) < _SINGULAR_VARIANCE):
        return True
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return True
    return False


def _mixed_fit(rows: pd.DataFrame, formula: str):
    """Fit ``formula`` with the maximal random structure, else with the random intercept.

    Returns ``(result, converged_maximal)``. The maximal fit is rejected on a
    ``ConvergenceWarning`` (except the two in ``_NOT_A_FAILURE``), ``converged is False``, a
    fitting error, or a singular or non-finite ``cov_re`` (a variance below 1e-10 or a matrix
    that is not positive definite). The fallback is kept whatever its own convergence; its
    warnings are swallowed.
    """
    from statsmodels.formula.api import mixedlm
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    groups = rows[_GROUP]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = mixedlm(formula, rows, groups=groups, re_formula=f"~ {_TERM}").fit(reml=True)
        except (np.linalg.LinAlgError, ValueError):
            result = None
    failed = [
        w
        for w in caught
        if issubclass(w.category, ConvergenceWarning)
        and not str(w.message).startswith(_NOT_A_FAILURE)
    ]
    ok = (
        result is not None
        and result.converged is not False
        and not failed
        and not _singular(result.cov_re)
    )
    if ok:
        return result, True
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = mixedlm(formula, rows, groups=groups, re_formula="1").fit(reml=True)
    return result, False


def _missing(emotion: str, parameter: str) -> Cell:
    return Cell(
        emotion=emotion,
        parameter=parameter,
        estimate=0.0,
        ci_low=None,
        ci_high=None,
        between_actor_sd=0.0,
        residual_sd=0.0,
        n_actors=0,
        n_cycles=0,
        n_datasets=0,
        n_mapped_actors=0,
        grounding="missing",
        converged_maximal=True,
        per_dataset=MappingProxyType({}),
    )


def fit_parameter(df: pd.DataFrame, parameter: str) -> list[Cell]:
    """One cell per fitted emotion present in ``df``, in ``FITTED_EMOTIONS`` order.

    Rows with ``NaN`` in ``parameter`` are dropped first, then the rows of every actor without a
    neutral row for it. An emotion left with no actor is a ``missing`` cell; emotions absent from
    ``df`` get no cell (the caller treats them as missing). ``n_cycles`` counts the emotion's
    cycles; ``between_actor_sd`` is the emotion's random-slope SD from ``cov_re``, or the pooled
    ``between_actor_sd`` of the observed deltas when the fit fell back.
    """
    present = [e for e in FITTED_EMOTIONS if (df["emotion"] == e).any()]
    rows = _usable(df, parameter)
    levels = [e for e in present if (rows["emotion"] == e).any()]
    if not levels:
        return [_missing(e, parameter) for e in present]
    result, maximal = _mixed_fit(rows, f"{parameter} ~ {_TERM}")
    deltas = actor_deltas(rows[[*rows.columns.drop(_GROUP)]])
    deltas = deltas[deltas["parameter"] == parameter]
    pooled_sd = between_actor_sd(deltas).get(parameter, math.nan)
    interval = result.conf_int()
    residual_sd = math.sqrt(float(result.scale))
    cells = []
    for emotion in present:
        if emotion not in levels:
            cells.append(_missing(emotion, parameter))
            continue
        mine = rows[rows["emotion"] == emotion]
        mapped = mine.groupby(_GROUP)["label_kind"].agg(lambda k: bool((k == "mapped").any()))
        n_actors = len(mapped)
        grounding = "fitted" if n_actors >= MIN_ACTORS_FITTED else "thin"
        name = _level(emotion)
        low, high = (float(v) for v in interval.loc[name])
        if maximal:
            spread = math.sqrt(float(result.cov_re.loc[name, name]))
        else:
            spread = pooled_sd
        per_dataset = deltas[deltas["emotion"] == emotion].groupby("dataset")["delta"].mean()
        cells.append(
            Cell(
                emotion=emotion,
                parameter=parameter,
                estimate=float(result.fe_params[name]),
                ci_low=low if grounding == "fitted" else None,
                ci_high=high if grounding == "fitted" else None,
                between_actor_sd=spread,
                residual_sd=residual_sd,
                n_actors=n_actors,
                n_cycles=len(mine),
                n_datasets=int(mine["dataset"].nunique()),
                n_mapped_actors=int(mapped.sum()),
                grounding=grounding,
                converged_maximal=maximal,
                per_dataset=MappingProxyType({str(k): float(v) for k, v in per_dataset.items()}),
            )
        )
    return cells


def fit_all(df: pd.DataFrame) -> dict[str, list[Cell]]:
    """``fit_parameter`` for each of the 14 measured parameters."""
    return {p: fit_parameter(df, p) for p in MEASURED_FIELDS}


def actor_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Per actor, emotion and parameter: the emotion mean minus the neutral mean.

    Means are over the actor's cycles with the parameter present; an actor-emotion-parameter
    without a neutral mean has no row.
    """
    keys = ["dataset", "actor"]
    long = df.melt(
        id_vars=[*keys, "emotion"], value_vars=list(MEASURED_FIELDS), var_name="parameter"
    ).dropna(subset=["value"])
    means = long.groupby([*keys, "emotion", "parameter"], as_index=False)["value"].mean()
    neutral = means[means["emotion"] == "neutral"].drop(columns="emotion")
    emotional = means[means["emotion"] != "neutral"]
    merged = emotional.merge(neutral, on=[*keys, "parameter"], suffixes=("", "_neutral"))
    merged["delta"] = merged["value"] - merged["value_neutral"]
    return merged[["actor", "dataset", "emotion", "parameter", "delta"]].reset_index(drop=True)


def between_actor_sd(deltas: pd.DataFrame) -> dict[str, float]:
    """Per parameter, the SD of ``delta`` around its emotion mean, pooled over emotions.

    The residual SD of ``delta ~ C(emotion)`` on the actor-emotion deltas: the residual sum of
    squares over ``n - k`` (``k`` emotions), ``NaN`` when ``n <= k``.
    """
    out: dict[str, float] = {}
    for parameter, group in deltas.groupby("parameter"):
        residual = group["delta"] - group.groupby("emotion")["delta"].transform("mean")
        dof = len(group) - group["emotion"].nunique()
        out[str(parameter)] = math.sqrt(float((residual**2).sum()) / dof) if dof > 0 else math.nan
    return out


def _folded(params: GaitParameters, name: str) -> float:
    value = float(getattr(params, name))
    return value * params.expansiveness if name in _EXPANSIVE else value


def hand_tuned_deltas(
    scaling: Scaling, *, leg_length: float = 0.69, nominal_pelvis_height: float = 0.68
) -> dict[tuple[str, str], float]:
    """The analytic model's change at intensity 1.0 per ``(emotion, parameter)``, normalised.

    ``expansiveness`` is folded into arm swing, abduction and step height. ``froude`` inverts
    ``gait.scaling.to_gait_parameters``: lengths / ``leg_length``, cadence x
    ``sqrt(leg_length / G)``, ``com_height_scale`` x ``nominal_pelvis_height / leg_length``;
    ``ratio`` gives lengths, cadence and ``com_height_scale`` as ``modulated / base - 1``.
    Angles and fractions are plain differences in both.
    """
    if scaling not in ("froude", "ratio"):
        raise ValueError(f"scaling must be 'froude' or 'ratio', got {scaling!r}")
    base = GaitParameters()
    out: dict[tuple[str, str], float] = {}
    for emotion in FITTED_EMOTIONS:
        moved = modulate(base, {Emotion(emotion): 1.0}, G1_PROFILE_WITH_NECK)
        for name in MEASURED_FIELDS:
            new, old = _folded(moved, name), _folded(base, name)
            if scaling == "ratio":
                delta = new / old - 1.0 if name in _RATIO_SCALED else new - old
            elif name in LENGTH_FIELDS:
                delta = (new - old) / leg_length
            elif name == "cadence":
                delta = (new - old) * math.sqrt(leg_length / G)
            elif name == "com_height_scale":
                delta = (new - old) * nominal_pelvis_height / leg_length
            else:
                delta = new - old
            out[(emotion, name)] = delta
    return out


def _rms(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values))


def leave_one_actor_out(
    df: pd.DataFrame,
    cells: Mapping[str, list[Cell]],
    hand: Mapping[tuple[str, str], float],
) -> list[HeldOut]:
    """Held-out-actor errors per parameter with a ``fitted`` cell, plus an ``overall`` entry.

    For each actor, one refit on the other actors predicts ``beta[e]`` for every fitted emotion
    the actor performed; the target is the actor's observed delta (``actor_deltas``). Errors are
    the RMS over held-out pairs, standardised by ``between_actor_sd`` of all observed deltas for
    the parameter (``*_natural``: unstandardised). ``E_hand`` predicts ``hand[(e, p)]`` (0.0
    where ``hand`` has no entry), ``E_zero`` predicts no change. A refit that fell back is used
    as is. ``overall`` holds the unweighted means of the per-parameter errors and the number of
    distinct held-out actor-emotion pairs.
    """
    all_deltas = actor_deltas(df)
    spread = between_actor_sd(all_deltas)
    group = _group_key(df)
    entries: list[HeldOut] = []
    pairs_seen: set[tuple[str, str, str]] = set()
    for parameter, param_cells in cells.items():
        fitted = {c.emotion for c in param_cells if c.grounding == "fitted"}
        deltas = all_deltas[
            (all_deltas["parameter"] == parameter) & all_deltas["emotion"].isin(fitted)
        ]
        fit_err: list[float] = []
        hand_err: list[float] = []
        zero_err: list[float] = []
        for (dataset, actor), mine in deltas.groupby(["dataset", "actor"]):
            refit = fit_parameter(df[group != f"{dataset}/{actor}"], parameter)
            predicted = {c.emotion: c.estimate for c in refit}
            for emotion, target in zip(mine["emotion"], mine["delta"], strict=True):
                fit_err.append(predicted.get(emotion, 0.0) - target)
                hand_err.append(hand.get((emotion, parameter), 0.0) - target)
                zero_err.append(-target)
                pairs_seen.add((str(dataset), str(actor), emotion))
        if not fit_err:
            continue
        sd = spread[parameter]
        entries.append(
            HeldOut(
                parameter=parameter,
                e_fit=_rms(fit_err) / sd,
                e_hand=_rms(hand_err) / sd,
                e_zero=_rms(zero_err) / sd,
                e_fit_natural=_rms(fit_err),
                e_hand_natural=_rms(hand_err),
                n_pairs=len(fit_err),
            )
        )
    if not entries:
        return []
    means = {
        f.name: float(np.mean([getattr(h, f.name) for h in entries]))
        for f in fields(HeldOut)
        if f.name not in ("parameter", "n_pairs")
    }
    return [*entries, HeldOut(parameter="overall", n_pairs=len(pairs_seen), **means)]


def speed_adjusted(df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Emotion effects with the actor's normalised speed delta as a covariate (report only).

    Speed is ``step_length * cadence`` per row; the covariate is that minus the actor's neutral
    mean speed. Same rows, formula and fallback as ``fit_parameter``.
    """
    with_speed = df.assign(_speed=df["step_length"] * df["cadence"])
    with_speed = with_speed[with_speed["_speed"].notna()]
    out: dict[tuple[str, str], float] = {}
    for parameter in MEASURED_FIELDS:
        rows = _usable(with_speed, parameter)
        neutral = rows[rows["emotion"] == "neutral"].groupby(_GROUP)["_speed"].mean()
        rows = rows.assign(_speed_delta=rows["_speed"] - rows[_GROUP].map(neutral))
        levels = [e for e in FITTED_EMOTIONS if (rows["emotion"] == e).any()]
        if not levels:
            continue
        result, _ = _mixed_fit(rows, f"{parameter} ~ {_TERM} + _speed_delta")
        for emotion in levels:
            out[(emotion, parameter)] = float(result.fe_params[_level(emotion)])
    return out
