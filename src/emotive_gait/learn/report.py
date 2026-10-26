"""The fit report (design spec 4.6, section 6): Markdown, aggregates only.

``write_report`` renders the headline held-out table, the per-cell effects, the per-dataset
diagnostics, the three-axis structure test with its caveat, the speed-adjusted secondary fit, the
contradictions, a synthetic self-check and the nine limitations. Nothing per-trial or per-clip is
printed; per-dataset values are means over actors.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from emotive_gait.emotion.learned import SCALINGS, Artefact
from emotive_gait.gait.scaling import MEASURED_FIELDS, Scaling
from emotive_gait.learn.fit import HeldOut
from emotive_gait.learn.structure import StructureResult

FAIRNESS_CAVEAT = (
    "The hand-tuned gains were tuned for a G1 rendering and never targeted human magnitudes, so "
    "E_hand measures how far the hand tuning sits from measured human effects, not how badly it "
    "was done."
)
STRUCTURE_CAVEAT = (
    "Caveat: five group means always lie in a four-dimensional affine space, so rank 3 removes "
    "one dimension and the test is low-powered with four sourced emotions. Read the result as "
    '"S1 loses X against S4 on held-out actors", never as "the data discovered three axes".'
)
LIMITATIONS = (
    (
        "**Human effects, not robot perception.** The fit says how acted emotion changes human gait "
        "parameters. Whether a G1 showing those changes reads as the emotion is Phase 5's question."
    ),
    (
        "**Intensity and mixing are assumptions**, not identified from these data: effects scale "
        "linearly with intensity and emotions add."
    ),
    (
        "**Acted, single-intensity portrayals by mostly non-professional actors**, pooled across "
        "capture protocols. The dataset term absorbs level shifts, not protocol differences in kind."
    ),
    (
        "**Speed is confounded with emotion** in every available set. The primary effect is "
        '"as acted"; the secondary fit shows what survives a speed covariate, and that is all that '
        "can be said."
    ),
    "{coverage}",
    (
        "**Transfer is approximate.** Froude similarity is a first-order rule (Donelan & Kram 1997); "
        "the G1 walks crouched with a bent knee and its joint limits clip some effects. The robot "
        "plays a rescaled, clipped version of a human effect; data value and played value are "
        "reported separately. Two rig constants sit between a learned posture delta and what the "
        "G1 shows: its elbow's geometric bend is 1.385 rad minus the joint angle, so a positive "
        "(more flexed) human `elbow_flexion` delta, such as happiness's, renders as straighter "
        "elbows; and the pelvis-to-neck vector pitches 0.88 of the commanded waist pitch, so a "
        "learned trunk effect renders at that gain and the measured head-on-trunk angle inherits "
        "the difference."
    ),
    (
        "**The structure test is low-powered**: four emotions give five group means, so rank 3 is "
        "one dimension short of saturated. It is a model comparison, not a discovery."
    ),
    (
        "**Label mapping is a declared assumption** for 100STYLE's Depressed and Elated and for CMU's "
        "style names; the artefact counts `mapped` actors per emotion."
    ),
    (
        "**Operational definitions are the project's.** `elbow_swing_ratio`, `arm_swing_amp` and "
        "`neck_pitch` (head on trunk) are defined to match the generator, not to match any one "
        "paper's measure, so fitted values are not directly comparable with published numbers."
    ),
)


def _coverage(provenance: Mapping[str, object]) -> str:
    """The coverage limitation, with the actor counts of this fit (thin below three actors)."""
    actors = provenance.get("actors_per_emotion") or {}
    parts = []
    for name in ("Fear", "anger"):
        n = int(actors.get(name.lower(), 0))  # type: ignore[call-overload]
        noun = "actor" if n == 1 else "actors"
        if n == 0:
            parts.append(f"**{name} has no data** in this fit")
        elif n < 3:
            parts.append(f"**{name} is thin** ({n} {noun} in this fit)")
        else:
            parts.append(f"**{name} rests on {n} actors**")
    return (
        f"{parts[0]}, {parts[1]}, and **disgust and surprise are absent**. Thin cells carry an actor count "
        "instead of an interval and enter no headline number; the CLI warns."
    )


def limitations(provenance: Mapping[str, object]) -> tuple[str, ...]:
    """``LIMITATIONS`` with the coverage item filled from ``provenance['actors_per_emotion']``."""
    coverage = _coverage(provenance)
    return tuple(coverage if text == "{coverage}" else text for text in LIMITATIONS)


def _f(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return [*lines, ""]


SELF_CHECK_SEEDS = 10


def _self_check() -> str:
    """Recover a planted effect from small synthetic tables; one line for the report."""
    from emotive_gait.learn.fit import fit_parameter
    from emotive_gait.learn.synthetic import synthetic_table

    planted = 0.08
    estimates: list[float] = []
    inside = 0
    for seed in range(SELF_CHECK_SEEDS):
        table = synthetic_table(
            n_actors=6,
            emotions=("anger",),
            effects={("anger", "cadence"): planted},
            actor_sd=0.03,
            cycle_sd=0.02,
            cycles_per_clip=4,
            seed=seed,
        )
        cell = fit_parameter(table, "cadence")[0]
        if cell.ci_low is None or cell.ci_high is None:
            continue
        estimates.append(cell.estimate)
        inside += cell.ci_low <= planted <= cell.ci_high
    if not estimates:
        return "Synthetic self-check: no planted-effect fit returned an interval."
    return (
        f"Synthetic self-check (6 actors, 4 cycles per clip, seeds 0-{SELF_CHECK_SEEDS - 1}): a "
        f"planted anger cadence effect of {planted:.3f} was recovered with mean estimate "
        f"{sum(estimates) / len(estimates):.3f}, and lay inside its 95 % Wald interval in "
        f"{inside} of {len(estimates)} datasets (with six actors the interval is expected to "
        "be somewhat optimistic)."
    )


def _provenance_section(provenance: Mapping[str, object]) -> list[str]:
    datasets = list(provenance.get("datasets_included") or [])
    excluded = list(provenance.get("datasets_excluded") or [])
    lines = [
        f"- Fit date: {provenance.get('fit_date', 'unknown')}",
        f"- Code commit: `{provenance.get('git_commit', 'unknown')}`",
        f"- Manifest SHA-256: `{provenance.get('manifest_sha256', 'unknown')}`",
        (
            f"- Software: statsmodels {provenance.get('statsmodels_version')}, "
            f"NumPy {provenance.get('numpy_version')}, pandas {provenance.get('pandas_version')}"
        ),
        f"- Seed: {provenance.get('seed')} (recorded; nothing in the fit is randomised)",
        f"- Model: `{provenance.get('model_formula')}`",
        f"- Event rule: {provenance.get('event_rule')}",
        "",
    ]
    rows = [
        (
            d["id"],
            d["name"],
            d["licence"],
            "yes" if d["committable"] else "no",
            d["permission"] or "none",
            d["n_actors"],
            d["n_clips"],
            d["n_cycles"],
        )
        for d in datasets
    ]
    lines += ["### Datasets included", ""]
    lines += _table(
        ["id", "name", "licence", "committable", "permission", "actors", "clips", "cycles"], rows
    )
    if excluded:
        lines += ["### Datasets excluded", ""]
        lines += _table(
            ["id", "reason", "committable", "permission"],
            [
                (
                    d["id"],
                    d["reason"],
                    "yes" if d.get("committable") else "no",
                    d.get("permission") or "none",
                )
                for d in excluded
            ],
        )
    bml = [d for d in [*datasets, *excluded] if "bml" in str(d["id"]).lower()]
    if bml:
        entry = bml[0]
        if entry in datasets:
            committable = "committable" if entry.get("committable") else "not committable"
            lines += [f"Body Movement Library: included in this fit; {committable}.", ""]
        else:
            permission = entry.get("permission") or "none recorded"
            lines += [
                (
                    f"Body Movement Library: excluded from this fit ({entry['reason']}; "
                    f"permission: {permission})."
                ),
                "",
            ]
    actors = provenance.get("actors_per_emotion") or {}
    cycles = provenance.get("cycles_per_emotion") or {}
    lines += ["### Actors and cycles per emotion", ""]
    lines += _table(
        ["emotion", "actors", "cycles"], [(e, actors[e], cycles.get(e, "n/a")) for e in actors]
    )
    return lines


MIXED_UNITS = "mixed units"


def all_fell_back_note(scaling: str, artefact: Artefact) -> str | None:
    """One sentence when every fitted or thin cell of ``scaling`` used the random-intercept
    fallback, else ``None``."""
    cells = [
        c for (s, _, _), c in artefact.cells.items() if s == scaling and c.grounding != "missing"
    ]
    if not cells or any(c.converged_maximal for c in cells):
        return None
    reasons = ", ".join(f"`{r}`" for r in sorted({str(c.fallback_reason) for c in cells}))
    return (
        f"Every cell of the `{scaling}` fit fell back to the random intercept ({reasons}): the "
        "Wald intervals then carry only the cycle-level residual variance, not the between-actor "
        "variance of the emotion slopes, and the fixed effects are cycle-weighted (an actor with "
        "more cycles counts for more)."
    )


def _headline_section(
    scaling: str, held: Sequence[HeldOut], provenance: Mapping[str, object], artefact: Artefact
) -> list[str]:
    lines = [f"### Held-out error, `{scaling}` scaling", ""]
    note = all_fell_back_note(scaling, artefact)
    if note is not None:
        lines += [note, ""]
    if not held:
        return [*lines, "No cell is `fitted` (three or more actors): no held-out validation.", ""]
    rows = [
        (
            h.parameter,
            _f(h.e_fit),
            _f(h.e_hand),
            _f(h.e_zero),
            MIXED_UNITS if h.parameter == "overall" else _f(h.e_fit_natural, 4),
            MIXED_UNITS if h.parameter == "overall" else _f(h.e_hand_natural, 4),
            h.n_pairs,
        )
        for h in held
    ]
    lines += _table(
        ["parameter", "E_fit", "E_hand", "E_zero", "E_fit_natural", "E_hand_natural", "n_pairs"],
        rows,
    )
    overall = next((h for h in held if h.parameter == "overall"), None)
    datasets = provenance.get("datasets_included") or []
    n_actors = sum(int(d["n_actors"]) for d in datasets)
    if overall is not None:
        lines += [
            (
                f"On {n_actors} actors from {len(datasets)} dataset(s) ({overall.n_pairs} held-out "
                f"actor-emotion pairs), the fitted map predicts per-emotion gait-parameter changes "
                f"with standardised error E_fit = {_f(overall.e_fit)}, against E_hand = "
                f"{_f(overall.e_hand)} for the hand-tuned gains and E_zero = {_f(overall.e_zero)} "
                "for no change. 1.0 is the irreducible floor of predicting an unseen person from "
                "the mean. Only `fitted` cells (three or more actors) enter these numbers."
            ),
            "",
        ]
    lines += [FAIRNESS_CAVEAT, ""]
    return lines


def _interval(cell) -> str:
    if cell.grounding == "fitted":
        return f"[{_f(cell.ci_low, 4)}, {_f(cell.ci_high, 4)}]"
    return f"n actors = {cell.n_actors}"


def _cell_section(scaling: str, artefact: Artefact) -> list[str]:
    cells = [c for (s, _, _), c in sorted(artefact.cells.items()) if s == scaling]
    shown = [c for c in cells if c.grounding != "missing"]
    rows = [
        (
            c.emotion,
            c.parameter,
            c.grounding,
            _f(c.estimate, 4),
            _interval(c),
            _f(c.between_actor_sd, 4),
            c.n_actors,
            c.n_cycles,
            c.n_datasets,
            c.n_mapped_actors,
            "yes" if c.converged_maximal else f"no ({c.fallback_reason})",
            c.direction_check,
        )
        for c in shown
    ]
    lines = [f"### Per-cell effects, `{scaling}` scaling", ""]
    lines += _table(
        [
            "emotion",
            "parameter",
            "grounding",
            "estimate",
            "95 % interval",
            "between_actor_sd",
            "n_actors",
            "n_cycles",
            "n_datasets",
            "n_mapped_actors",
            "converged_maximal",
            "direction_check",
        ],
        rows,
    )
    absent = len(cells) - len(shown)
    lines += [f"{absent} further cells are `missing` and delegated to the analytic model.", ""]
    return lines


def _dataset_section(scaling: str, artefact: Artefact) -> list[str]:
    cells = [c for (s, _, _), c in sorted(artefact.cells.items()) if s == scaling and c.per_dataset]
    ids = sorted({d for c in cells for d in c.per_dataset})
    lines = [f"### Per-dataset mean change, `{scaling}` scaling", ""]
    if not cells:
        return [*lines, "No cell has data.", ""]
    rows = [(c.emotion, c.parameter, *[_f(c.per_dataset.get(d), 4) for d in ids]) for c in cells]
    return [*lines, *_table(["emotion", "parameter", *ids], rows)]


def _structure_section(scaling: str, results: Sequence[StructureResult]) -> list[str]:
    lines = [f"### Structure test, `{scaling}` scaling", ""]
    if not results:
        return [*lines, "No fitted parameter: the structure test was not run.", ""]
    rows = [(r.name, r.n_free, _f(r.r2_in_sample), _f(r.e_heldout)) for r in results]
    lines += _table(["structure", "n_free", "R^2 (in sample)", "held-out error"], rows)
    by_name = {r.name: r for r in results}
    s1, s4 = by_name.get("S1_analytic"), by_name.get("S4_full")
    if s1 and s4 and not (math.isnan(s1.e_heldout) or math.isnan(s4.e_heldout)):
        lines += [
            (
                f"S1 loses {s1.e_heldout - s4.e_heldout:.3f} against S4 on held-out actors "
                "(standardised units)."
            ),
            "",
        ]
    lines += [STRUCTURE_CAVEAT, ""]
    return lines


def _speed_section(
    scaling: str, artefact: Artefact, adjusted: Mapping[tuple[str, str], float]
) -> list[str]:
    rows = []
    for (s, emotion, parameter), cell in sorted(artefact.cells.items()):
        if s == scaling and cell.grounding != "missing" and (emotion, parameter) in adjusted:
            rows.append(
                (emotion, parameter, _f(cell.estimate, 4), _f(adjusted[(emotion, parameter)], 4))
            )
    lines = [f"### Speed-adjusted secondary fit, `{scaling}` scaling", ""]
    if not rows:
        return [*lines, "No cell could be refitted with the speed covariate.", ""]
    return [
        *lines,
        *_table(
            ["emotion", "parameter", "estimate as acted", "estimate with speed covariate"], rows
        ),
    ]


def _contradiction_section(artefact: Artefact) -> list[str]:
    lines = ["## Contradictions with the cited claims", ""]
    if not artefact.contradictions:
        return [*lines, "None: no fitted or thin cell, and no derived speed, opposes a claim.", ""]
    rows = [
        (
            c["emotion"],
            c["parameter"],
            c["scaling"],
            _f(c["estimate"], 4),  # type: ignore[arg-type]
            "n/a" if c["ci_low"] is None else f"[{_f(c['ci_low'], 4)}, {_f(c['ci_high'], 4)}]",  # type: ignore[arg-type]
            c["citation"],
        )
        for c in artefact.contradictions
    ]
    return [
        *lines,
        *_table(["emotion", "parameter", "scaling", "estimate", "95 % interval", "citation"], rows),
    ]


def write_report(
    path: Path,
    *,
    held_out: Mapping[Scaling, list[HeldOut]],
    structure: Mapping[Scaling, list[StructureResult]],
    artefact: Artefact,
    speed_adjusted: Mapping[Scaling, Mapping[tuple[str, str], float]],
    provenance: Mapping[str, object],
) -> None:
    """Write the Markdown fit report to ``path`` (parents are created).

    ``held_out``, ``structure`` and ``speed_adjusted`` are keyed by scaling.
    """
    scalings = [s for s in SCALINGS if any(k[0] == s for k in artefact.cells)]
    lines = [
        "# Phase 4.5 fit report: the data-derived emotion model",
        "",
        "Generated by `emotive-gait fit`. Aggregates only: no per-trial numbers.",
        "",
        "## Provenance",
        "",
        *_provenance_section(provenance),
        "## Held-out validation (the headline)",
        "",
    ]
    for scaling in scalings:
        lines += _headline_section(scaling, held_out.get(scaling, []), provenance, artefact)
    lines += ["## Per-cell effects", ""]
    for scaling in scalings:
        lines += _cell_section(scaling, artefact)
    lines += ["## Per-dataset diagnostics", ""]
    for scaling in scalings:
        lines += _dataset_section(scaling, artefact)
    lines += ["## Three-axis structure test", ""]
    for scaling in scalings:
        lines += _structure_section(scaling, structure.get(scaling, []))
    lines += ["## Speed confound", ""]
    for scaling in scalings:
        lines += _speed_section(scaling, artefact, speed_adjusted.get(scaling, {}))
    lines += _contradiction_section(artefact)
    lines += ["## Self-check", "", _self_check(), ""]
    lines += ["## Limitations", ""]
    lines += [f"{i}. {text}" for i, text in enumerate(limitations(provenance), start=1)]
    lines += [""]
    fields_note = ", ".join(MEASURED_FIELDS)
    lines += [f"Measured parameters: {fields_note}.", ""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


FEASIBILITY_HEADING = "## G1 feasibility of the learned presets"


def append_feasibility(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Append the learned presets' G1 feasibility section to the report at ``path``.

    Pure text: the caller runs the IK. Each row is a plain dict with ``preset``, ``scaling``,
    ``feasible`` (bool), ``position_error_mm``, ``orientation_error_rad``, ``roles_at_limit``
    (names), ``stride`` (``{step_length, cadence, double_support_frac}``) and ``driving``: the
    cells whose return to neutral alone makes the preset feasible, each a dict with
    ``parameter``, ``value`` (played), ``estimate`` (the cell's effect) and ``per_dataset``.
    """
    table = []
    for r in rows:
        stride = r.get("stride") or {}
        driving = r.get("driving") or []
        if r["feasible"]:
            drivers = "-"
        elif not driving:
            drivers = "no single cell"
        else:
            drivers = "; ".join(
                f"`{d['parameter']}` {_f(d['value'], 4)} (effect {_f(d['estimate'], 4)}; "
                + ", ".join(f"{k} {_f(v, 4)}" for k, v in (d.get("per_dataset") or {}).items())
                + ")"
                for d in driving
            )
        table.append(
            (
                r["preset"],
                r["scaling"],
                "yes" if r["feasible"] else "no",
                _f(r["position_error_mm"], 3),  # type: ignore[arg-type]
                _f(r["orientation_error_rad"], 4),  # type: ignore[arg-type]
                ", ".join(r.get("roles_at_limit") or []) or "none",  # type: ignore[arg-type]
                (
                    f"{_f(stride.get('step_length'), 3)} m, {_f(stride.get('cadence'), 2)} steps/s, "
                    f"DS {_f(stride.get('double_support_frac'), 3)}"
                ),
                drivers,
            )
        )
    lines = [
        "",
        FEASIBILITY_HEADING,
        "",
        (
            "Each learned preset (the analytic presets' emotions at intensity 1.0, played through "
            "the learned model) solved by the G1 leg IK at the existing thresholds (2 mm foot "
            "position, 0.01 rad stance orientation). A driving cell is one whose return to the "
            "neutral value alone makes the preset feasible. Decision 2026-10-26 (spec 4.5): learned "
            "presets are flagged and rendered, not gated; the CLI warns once on infeasible "
            "parameters and still renders them, and the residuals are documented here."
        ),
        "",
        *_table(
            [
                "preset",
                "scaling",
                "feasible",
                "max foot position error (mm)",
                "stance orientation error (rad)",
                "roles at limit",
                "stride",
                "driving cells",
            ],
            table,
        ),
    ]
    path = Path(path)
    text = path.read_text(encoding="utf-8").rstrip("\n")
    path.write_text(text + "\n" + "\n".join(lines) + "\n", encoding="utf-8", newline="\n")
