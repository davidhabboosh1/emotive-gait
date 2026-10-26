"""``emotive-gait fit``: manifest -> verified files -> cycle table -> fit -> artefact and report.

``run_fit`` verifies the manifest's hashes, builds the per-cycle table (written to
``derived/cycles.csv``), then for each scaling fits every cell, runs the leave-one-actor-out
validation against the hand-tuned gains, the three-axis structure test and the speed-adjusted
secondary fit, and writes the artefact and the report. Nothing is randomised; ``seed`` is
recorded in the provenance.

Committable guard: datasets with ``committable: false`` are dropped unless
``include_uncommittable`` is set, and a fit that includes them may not write inside ``src/``
(the packaged artefact is committed). A named uncommittable dataset aimed at the package path is
an error rather than a silent drop.
"""

from __future__ import annotations

import datetime
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import emotive_gait
from emotive_gait.emotion.learned import SCALINGS, LearnedEmotionModel
from emotive_gait.gait.scaling import Scaling
from emotive_gait.learn.artefact import write_artefact
from emotive_gait.learn.fit import (
    _TERM,
    HeldOut,
    actor_deltas,
    between_actor_sd,
    fit_all,
    hand_tuned_deltas,
    leave_one_actor_out,
    speed_adjusted,
)
from emotive_gait.learn.manifest import (
    DatasetEntry,
    HashProblem,
    Manifest,
    ManifestError,
    load_manifest,
    verify,
)
from emotive_gait.learn.report import write_report
from emotive_gait.learn.structure import structure_test
from emotive_gait.learn.table import build_table, normalise_table
from emotive_gait.mocap.events import MIN_PEAK_S, PROMINENCE_FRAC
from emotive_gait.mocap.extract import EVENT_FILTER_HZ, FILTER_HZ
from emotive_gait.mocap.landmarks import LANDMARKS

FIT_EXTRA_HINT = 'pip install -e ".[fit]"'
_SRC = Path(emotive_gait.__file__).resolve().parent.parent


class FitInputError(ValueError):
    """The manifest, the data or the requested datasets are wrong; the message names them."""


class FitEnvironmentError(RuntimeError):
    """A fit dependency is missing."""


@dataclass(frozen=True)
class FitSummary:
    manifest_sha256: str
    artefact_path: Path
    report_path: Path
    held_out: Mapping[Scaling, list[HeldOut]]
    thin_cells: tuple[tuple[str, str, int], ...]  # (emotion, parameter, n_actors)
    contradictions: tuple[Mapping[str, object], ...]
    fallbacks: tuple[Mapping[str, str], ...]  # {scaling, parameter, reason}
    not_maximal: tuple[Mapping[str, str], ...]  # {scaling, emotion, parameter, reason}
    n_actors: int
    n_cycles: int


def describe_problem(problem: HashProblem) -> str:
    actual = "missing" if problem.actual is None else problem.actual
    text = f"{problem.path}: expected {problem.expected}, got {actual}"
    return f"{text} [{problem.error}]" if problem.error else text


def _in_package(path: Path) -> bool:
    return Path(path).resolve().is_relative_to(_SRC)


def _select(
    manifest: Manifest,
    datasets: Iterable[str] | None,
    include_uncommittable: bool,
    target: Path,
    log: Callable[[str], None],
) -> tuple[list[DatasetEntry], list[dict[str, str]]]:
    """The datasets to fit and those left out with the reason."""
    named = None if datasets is None else list(datasets)
    try:
        chosen = list(manifest.datasets) if named is None else [manifest.dataset(i) for i in named]
    except ManifestError as exc:
        raise FitInputError(str(exc)) from None
    private = [d for d in chosen if not d.committable]
    if include_uncommittable and _in_package(target):
        raise FitInputError(
            f"--include-uncommittable would write {target} inside the package; a fit with "
            "uncommittable data may only write outside src/ (use --out); only committable "
            "numbers may enter the committed artefact"
        )
    used = list(chosen)
    if private and not include_uncommittable:
        if named is not None and _in_package(target):
            ids = ", ".join(d.id for d in private)
            raise FitInputError(
                f"dataset {ids} is not committable and {target} is the committed artefact; "
                "pass --include-uncommittable with --out outside src/, or drop the dataset"
            )
        used = [d for d in chosen if d.committable]
        for d in private:
            log(f"{d.id}: not committable, left out of this fit")
    if not used:
        raise FitInputError("no committable dataset selected; nothing to fit")
    left_out = [
        {
            "id": d.id,
            "reason": "not committable" if not d.committable and d not in used else "not selected",
        }
        for d in manifest.datasets
        if d not in used
    ]
    return used, left_out


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_SRC,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _dataset_records(used: list[DatasetEntry], table: pd.DataFrame) -> list[dict[str, object]]:
    records = []
    for d in used:
        mine = table[table["dataset"] == d.id]
        records.append(
            {
                "id": d.id,
                "name": d.name,
                "committable": d.committable,
                "permission": d.permission,
                "licence": d.licence,
                "n_actors": int(mine["actor"].nunique()),
                "n_clips": int(mine["clip"].nunique()),
                "n_cycles": len(mine),
            }
        )
    return records


def _event_rule() -> str:
    return (
        "Gait events by the position-extremum rule (Zeni, Richards & Higginson 2008) on tracks "
        f"low-passed at {EVENT_FILTER_HZ:g} Hz; extrema of one kind closer than {MIN_PEAK_S:g} s "
        f"are one event (MIN_PEAK_S); an extremum must stand out by {PROMINENCE_FRAC:g} of the "
        "5th-95th percentile range of the foot's forward coordinate (PROMINENCE_FRAC); "
        f"measurements are low-passed at {FILTER_HZ:g} Hz."
    )


def run_fit(
    manifest_path: Path,
    root: Path,
    *,
    out_artefact: Path,
    out_report: Path,
    derived: Path,
    datasets: Iterable[str] | None,
    include_uncommittable: bool,
    seed: int = 0,
    log: Callable[[str], None] = print,
    landmarks: Mapping = LANDMARKS,
) -> FitSummary:
    """Run the whole fit; see the module docstring.

    Raises :class:`FitInputError` for a manifest, selection or hash problem (the message lists
    each ``HashProblem``) and :class:`FitEnvironmentError` when ``statsmodels`` is missing.
    """
    try:
        import statsmodels
    except ImportError:
        raise FitEnvironmentError(
            f"statsmodels is not installed; install the fit extra with {FIT_EXTRA_HINT}"
        ) from None
    try:
        manifest = load_manifest(Path(manifest_path))
    except ManifestError as exc:
        raise FitInputError(str(exc)) from None
    out_artefact, out_report, derived = Path(out_artefact), Path(out_report), Path(derived)
    used, left_out = _select(manifest, datasets, include_uncommittable, out_artefact, log)
    problems = verify(manifest, Path(root), datasets=[d.id for d in used])
    if problems:
        listing = "\n  ".join(describe_problem(p) for p in problems)
        raise FitInputError(
            f"{len(problems)} manifest file(s) missing or with a different hash "
            f"(run `emotive-gait fetch`):\n  {listing}"
        )

    table = build_table(
        manifest, Path(root), datasets=[d.id for d in used], landmarks=landmarks, log=log
    )
    if table.empty or not (table["emotion"] == "neutral").any():
        raise FitInputError("the datasets yield no usable neutral cycles; nothing to fit")
    derived.mkdir(parents=True, exist_ok=True)
    table.to_csv(derived / "cycles.csv", index=False)

    cells: dict[Scaling, dict] = {}
    held_out: dict[Scaling, list[HeldOut]] = {}
    structure: dict[Scaling, list] = {}
    adjusted: dict[Scaling, dict] = {}
    for scaling in SCALINGS:
        with np.errstate(divide="ignore", invalid="ignore"):  # a zero neutral mean gives NaN/inf
            normalised = normalise_table(table, scaling)
        fitted = fit_all(normalised)
        deltas = actor_deltas(normalised)
        spread = between_actor_sd(deltas)
        held_out[scaling] = leave_one_actor_out(normalised, fitted, hand_tuned_deltas(scaling))
        parameters = [p for p, cs in fitted.items() if any(c.grounding == "fitted" for c in cs)]
        structure[scaling] = (
            structure_test(deltas, spread, parameters=parameters) if parameters else []
        )
        adjusted[scaling] = speed_adjusted(normalised)
        cells[scaling] = fitted

    emotions = table.groupby("emotion")
    fallbacks = sorted(
        {
            (scaling, parameter, c.fallback_reason)
            for scaling, by_parameter in cells.items()
            for parameter, cs in by_parameter.items()
            for c in cs
            if c.fallback_reason is not None
        }
    )
    hand_headline = {
        scaling: {
            "e_fit": h.e_fit,
            "e_hand": h.e_hand,
            "e_zero": h.e_zero,
            "e_fit_natural": h.e_fit_natural,
            "e_hand_natural": h.e_hand_natural,
            "n_pairs": h.n_pairs,
        }
        for scaling, rows in held_out.items()
        for h in rows
        if h.parameter == "overall"
    }
    provenance = {
        "schema_version": 1,
        "fit_date": datetime.date.today().isoformat(),
        "git_commit": _git_commit(),
        "statsmodels_version": statsmodels.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "manifest_sha256": manifest.sha256,
        "datasets_included": _dataset_records(used, table),
        "datasets_excluded": left_out,
        "actors_per_emotion": {
            str(e): int(g.groupby("dataset")["actor"].nunique().sum()) for e, g in emotions
        },
        "cycles_per_emotion": {str(e): len(g) for e, g in emotions},
        "model_formula": (
            f"parameter ~ {_TERM} + (1 + {_TERM} | dataset/actor), REML; falls back to "
            "(1 | dataset/actor) when the maximal structure does not converge"
        ),
        "fallbacks": [{"scaling": s, "parameter": p, "reason": r} for s, p, r in fallbacks],
        "seed": seed,
        "hand_tuned": hand_headline,
        "event_rule": _event_rule(),
    }
    artefact = write_artefact(out_artefact, cells, provenance=provenance)
    write_report(
        out_report,
        held_out=held_out,
        structure=structure,
        artefact=artefact,
        speed_adjusted=adjusted,
        provenance=provenance,
    )

    froude = LearnedEmotionModel(artefact, scaling="froude")
    thin = tuple(
        (e, p, cell.n_actors)
        for e, p in froude.thin_cells
        if (cell := froude.cell(e, p)) is not None
    )
    not_maximal = tuple(
        {"scaling": s, "emotion": e, "parameter": p, "reason": str(c.fallback_reason)}
        for (s, e, p), c in sorted(artefact.cells.items())
        if not c.converged_maximal
    )
    return FitSummary(
        manifest_sha256=manifest.sha256,
        artefact_path=out_artefact,
        report_path=out_report,
        held_out=held_out,
        thin_cells=thin,
        contradictions=artefact.contradictions,
        fallbacks=tuple(provenance["fallbacks"]),
        not_maximal=not_maximal,
        n_actors=int(table.groupby("dataset")["actor"].nunique().sum()),
        n_cycles=len(table),
    )
