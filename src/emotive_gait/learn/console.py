"""What ``emotive-gait fit`` prints after a fit: headline, thin cells, contradictions, fallbacks."""

from __future__ import annotations

from collections.abc import Callable

from emotive_gait.learn.pipeline import FitSummary


def print_summary(summary: FitSummary, out: Callable[[str], None] = print) -> None:
    out(f"actors: {summary.n_actors}, cycles: {summary.n_cycles}")
    for scaling, rows in summary.held_out.items():
        overall = next((h for h in rows if h.parameter == "overall"), None)
        if overall is None:
            out(f"{scaling}: no fitted cell, no held-out validation")
            continue
        out(
            f"{scaling}: E_fit {overall.e_fit:.3f}  E_hand {overall.e_hand:.3f}  "
            f"E_zero {overall.e_zero:.3f}  (n_pairs {overall.n_pairs})"
        )
    if summary.thin_cells:
        listing = ", ".join(f"{e}.{p} ({n} actors)" for e, p, n in summary.thin_cells)
        out(f"thin cells (no interval): {listing}")
    else:
        out("thin cells: none")
    if summary.contradictions:
        out("contradictions:")
        for c in summary.contradictions:
            out(
                f"  {c['scaling']} {c['emotion']}.{c['parameter']}: {c['estimate']:.4f} ({c['citation']})"
            )
    else:
        out("contradictions: none")
    for cell in summary.not_maximal:
        out(
            f"random-intercept fallback: {cell['scaling']} {cell['emotion']}.{cell['parameter']} "
            f"({cell['reason']})"
        )
    out(f"wrote {summary.artefact_path}")
    out(f"wrote {summary.report_path}")
