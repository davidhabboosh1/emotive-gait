"""Helpers shared by ``cli`` and ``rig.cli``: messages, emotion parsing, model options.

Imports nothing from ``rig/`` so both command modules can import it without a cycle.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from emotive_gait.emotion.evidence import EVIDENCE, UNSOURCED_EMOTIONS, Emotion
from emotive_gait.gait.params import GaitParameters
from emotive_gait.gait.scaling import MEASURED_FIELDS


def _err(msg: str) -> None:
    print(f"emotive-gait: {msg}", file=sys.stderr)


def _parse_emotions(specs: Sequence[str]) -> dict[Emotion, float]:
    names = ", ".join(e.value for e in Emotion)
    out: dict[Emotion, float] = {}
    for spec in specs:
        name, sep, raw = spec.partition("=")
        name = name.strip().lower()
        if not sep:
            raise ValueError(f"emotion {spec!r} must look like NAME=VALUE (e.g. {name}=0.8)")
        try:
            emotion = Emotion(name)
        except ValueError:
            raise ValueError(f"unknown emotion {name!r}; choose from {names}") from None
        try:
            out[emotion] = float(raw)
        except ValueError:
            raise ValueError(f"intensity for {name} is not a number: {raw!r}") from None
    return out


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        choices=["analytic", "learned"],
        default="analytic",
        help="emotion model (default analytic; learned needs the fitted artefact)",
    )
    parser.add_argument(
        "--scaling",
        choices=["froude", "ratio"],
        default="froude",
        help="re-dimensioning of the learned model's effects (default froude)",
    )


def _print_params(params: GaitParameters) -> None:
    for f in dataclasses.fields(params):
        print(f"{f.name:<22}{getattr(params, f.name): .4f}")
    print(f"{'speed (m/s)':<22}{params.speed: .4f}")


def _load_learned(scaling: str):
    """The learned model on the packaged artefact, or ``None`` after printing why not."""
    from emotive_gait.emotion.learned import (
        ArtefactMissingError,
        ArtefactSchemaError,
        LearnedEmotionModel,
    )

    try:
        return LearnedEmotionModel.default(scaling=scaling)
    except (ArtefactMissingError, ArtefactSchemaError, ValueError) as exc:
        _err(str(exc))
        return None


def _warn_learned_cells(model, intensities: dict[Emotion, float]) -> None:
    """One warning for the thin cells and one for the delegated cells of the requested emotions."""
    from emotive_gait.emotion.learned import describe_thin

    asked = {e.value for e, v in intensities.items() if v > 0}
    thin = describe_thin(
        (e, p, model.cell(e, p).n_actors) for e, p in model.thin_cells if e in asked
    )
    if thin:
        _err(
            "warning: thin cells (one or two actors, no interval) in the learned model: "
            + ", ".join(thin)
        )
    delegated: dict[str, list[str]] = {}
    for e, p in model.analytic_cells:
        if e in asked:
            delegated.setdefault(e, []).append(p)
    if delegated:
        parts = [
            f"{e} (all)" if len(ps) == len(MEASURED_FIELDS) else f"{e}.{', '.join(ps)}"
            for e, ps in sorted(delegated.items())
        ]
        _err("warning: cells taken from the analytic model, not measured: " + "; ".join(parts))


def warn_unsourced(intensities: Mapping[Emotion, float]) -> None:
    """One warning per requested emotion that no source measured (disgust, surprise)."""
    for emotion, value in intensities.items():
        if emotion in UNSOURCED_EMOTIONS and math.isfinite(value) and value > 0:
            _err(
                f"warning: {emotion.value} is unsourced: no source reports its gait kinematics, "
                "so it is a flagged arousal/valence placement, not a measured effect"
            )


def warn_infeasible(report) -> None:
    """The one-line warning for a ``FeasibilityReport`` that is not feasible."""
    if report.feasible:
        return
    roles = ", ".join(r.name.lower() for r in report.roles_at_limit) or "none"
    _err(
        "warning: parameters exceed the robot's reach: max foot error "
        f"{report.max_foot_position_error_m * 1000:.1f} mm; roles at limit: {roles}"
    )


def check_positive_finite(value: object, name: str) -> float:
    """``value`` as a positive finite float, else ``ValueError`` naming ``name`` and the value."""
    try:
        if isinstance(value, bool):
            raise TypeError
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}") from None
    if not (math.isfinite(number) and number > 0):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return number


def clean(value: Any) -> Any:
    """JSON-able: enums as values, numpy scalars as Python, NaN and infinity as ``None``."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {clean(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


_DIRECTION = {1: "+", 0: "0", -1: "-"}


def _learned_columns(cell) -> list[str]:
    if cell is None:
        return ["-", "-", "-"]
    if cell.grounding == "missing":
        return ["-", "-", cell.direction_check]
    if cell.grounding == "fitted":
        interval = f"[{cell.ci_low:.4f}, {cell.ci_high:.4f}]"
    else:
        interval = f"{cell.n_actors} actors"
    return [f"{cell.estimate:+.4f}", interval, cell.direction_check]


def evidence_lines(artefact=None, scaling: str = "froude") -> list[str]:
    """The claim table as ``evidence`` prints it, one string per row (shared with the MCP server)."""
    head = ["EMOTION", "PARAMETER", "DIR", "GROUNDING"]
    if artefact is not None:
        head += ["ESTIMATE", "CI", "CHECK"]
    rows = [(*head, "CITATION")]
    last = None
    for c in EVIDENCE:
        label = c.emotion.value if c.emotion is not last else ""
        last = c.emotion
        row = [label, c.parameter, _DIRECTION[c.direction], c.grounding.value.upper()]
        if artefact is not None:
            row += _learned_columns(artefact.cells.get((scaling, c.emotion.value, c.parameter)))
        rows.append((*row, c.citation))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]) - 1)]
    return [
        "  ".join(r[i].ljust(widths[i]) for i in range(len(widths))) + "  " + r[-1] for r in rows
    ]
