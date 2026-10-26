"""Write the learned-model artefact (design spec 4.6) and compare two of them.

``write_artefact`` is the only writer of ``emotion/data/learned_coefficients.yaml``. It turns the
fitted :class:`~emotive_gait.learn.fit.Cell` lists of both scalings into the schema documented in
``emotion.learned.load_artefact``:

* every ``(scaling, emotion, parameter)`` of ``FITTED_EMOTIONS`` x ``MEASURED_FIELDS`` gets a
  cell; those the fit produced none for (an emotion absent from the table) are ``missing`` with
  estimate 0.0, zero SDs and counts, no interval and ``converged_maximal`` true;
* ``direction_check`` is computed per cell against the cited claim: the claim's direction for a
  signed claim, ``0`` only for ``evidence.NO_CHANGE_CLAIMS``, ``None`` for every other
  direction-0 claim and for parameters without a claim;
* ``contradictions`` lists every cell whose check starts with ``contradicts`` (both scalings,
  with the claim's citation), plus a ``speed`` entry per emotion and scaling when the learned
  model's modulated speed change at intensity 1.0 has the opposite sign to the ``speed`` claim;
* keys are sorted, NaN and ``None`` are written as ``null``, ``per_dataset`` is a mapping.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path

import numpy as np
import yaml

from emotive_gait.emotion.evidence import GROUNDED_EMOTIONS, NO_CHANGE_CLAIMS, Emotion, claims_for
from emotive_gait.emotion.learned import (
    SCALINGS,
    Artefact,
    ArtefactCell,
    LearnedEmotionModel,
    direction_check,
    load_artefact,
)
from emotive_gait.emotion.model import modulate
from emotive_gait.gait.params import GaitParameters
from emotive_gait.gait.scaling import MEASURED_FIELDS, Scaling
from emotive_gait.learn.fit import FITTED_EMOTIONS, Cell
from emotive_gait.robot.profiles.g1 import G1_PROFILE_WITH_NECK

SCHEMA_VERSION = 1
_CELL_FIELDS = tuple(f.name for f in fields(ArtefactCell) if f.name not in ("emotion", "parameter"))


def _num(value: float | None) -> float | None:
    """A plain float, with ``None`` and NaN as ``None`` (YAML ``null``)."""
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _plain(value: object) -> object:
    """``value`` as plain YAML-safe Python: mappings, lists, str, int, float, bool, None."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def _claim_direction(emotion: str, parameter: str) -> tuple[int | None, str | None]:
    """The direction handed to ``direction_check`` and the claim's citation, if any."""
    claims = {c.parameter: c for c in claims_for(Emotion(emotion))}
    claim = claims.get(parameter)
    if claim is None:
        return None, None
    if claim.direction != 0:
        return claim.direction, claim.citation
    if (claim.emotion, parameter) in NO_CHANGE_CLAIMS:
        return 0, claim.citation
    return None, claim.citation


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
        fallback_reason=None,
        per_dataset={},
    )


def _artefact_cell(cell: Cell, scaling: str) -> ArtefactCell:
    low, high = _num(cell.ci_low), _num(cell.ci_high)
    where = f"{scaling}.{cell.emotion}.{cell.parameter}"
    if cell.grounding == "fitted" and (low is None or high is None):
        raise ValueError(f"fitted cell {where} has no finite interval; refusing to write it")
    if cell.grounding != "fitted":
        low = high = None
    estimate = _num(cell.estimate)
    if estimate is None:
        raise ValueError(f"cell {where} has no finite estimate; refusing to write it")
    direction, _ = _claim_direction(cell.emotion, cell.parameter)
    ci = None if low is None or high is None else (low, high)
    return ArtefactCell(
        emotion=cell.emotion,
        parameter=cell.parameter,
        estimate=estimate,
        ci_low=low,
        ci_high=high,
        between_actor_sd=float(cell.between_actor_sd),
        residual_sd=float(cell.residual_sd),
        n_actors=int(cell.n_actors),
        n_cycles=int(cell.n_cycles),
        n_datasets=int(cell.n_datasets),
        n_mapped_actors=int(cell.n_mapped_actors),
        grounding=cell.grounding,
        converged_maximal=bool(cell.converged_maximal),
        fallback_reason=cell.fallback_reason,
        per_dataset={str(k): float(v) for k, v in cell.per_dataset.items()},
        direction_check=direction_check(estimate, ci, cell.grounding, direction),
    )


def _all_cells(
    cells: Mapping[Scaling, Mapping[str, list[Cell]]],
) -> dict[tuple[str, str, str], ArtefactCell]:
    out: dict[tuple[str, str, str], ArtefactCell] = {}
    for scaling in SCALINGS:
        given: dict[tuple[str, str], Cell] = {}
        for parameter, cell_list in cells.get(scaling, {}).items():
            for cell in cell_list:
                if cell.emotion not in FITTED_EMOTIONS or parameter not in MEASURED_FIELDS:
                    raise ValueError(
                        f"not a fitted (emotion, parameter) cell: {cell.emotion}.{parameter}"
                    )
                given[(cell.emotion, parameter)] = cell
        for emotion in FITTED_EMOTIONS:
            for parameter in MEASURED_FIELDS:
                cell = given.get((emotion, parameter)) or _missing(emotion, parameter)
                out[(scaling, emotion, parameter)] = _artefact_cell(cell, scaling)
    return out


def _contradictions(cells: Mapping[tuple[str, str, str], ArtefactCell]) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    for (scaling, emotion, parameter), cell in sorted(cells.items()):
        if cell.direction_check.startswith("contradicts"):
            _, citation = _claim_direction(emotion, parameter)
            found.append(
                {
                    "emotion": emotion,
                    "parameter": parameter,
                    "scaling": scaling,
                    "estimate": cell.estimate,
                    "ci_low": cell.ci_low,
                    "ci_high": cell.ci_high,
                    "citation": citation,
                }
            )
    interim = Artefact(cells=cells)
    base = GaitParameters()
    for scaling in SCALINGS:
        model = LearnedEmotionModel(interim, scaling=scaling)
        for emotion in sorted(GROUNDED_EMOTIONS, key=lambda e: e.value):
            claim = next((c for c in claims_for(emotion) if c.parameter == "speed"), None)
            if claim is None or claim.direction == 0:
                continue
            moved = modulate(base, {emotion: 1.0}, G1_PROFILE_WITH_NECK, model=model)
            delta = moved.speed - base.speed
            if delta * claim.direction < 0:
                found.append(
                    {
                        "emotion": emotion.value,
                        "parameter": "speed",
                        "scaling": scaling,
                        "estimate": float(delta),
                        "ci_low": None,
                        "ci_high": None,
                        "citation": claim.citation,
                    }
                )
    return found


def _cell_record(cell: ArtefactCell) -> dict[str, object]:
    record = {name: getattr(cell, name) for name in _CELL_FIELDS}
    record["per_dataset"] = dict(cell.per_dataset)
    return _plain(record)  # type: ignore[return-value]


def write_artefact(
    path: Path,
    cells: Mapping[Scaling, Mapping[str, list[Cell]]],
    *,
    provenance: Mapping[str, object],
) -> Artefact:
    """Write the artefact to ``path`` and return what ``load_artefact`` reads back.

    ``cells[scaling][parameter]`` is ``fit_all`` output. Raises ``ValueError`` naming the cell
    when a ``fitted`` cell lacks a finite interval or any estimate is not finite: no invalid
    artefact is written.
    """
    built = _all_cells(cells)
    tree: dict[str, dict[str, dict[str, object]]] = {}
    for (scaling, emotion, parameter), cell in built.items():
        tree.setdefault(scaling, {}).setdefault(emotion, {})[parameter] = _cell_record(cell)
    document = {
        "schema_version": SCHEMA_VERSION,
        "provenance": _plain(provenance),
        "contradictions": _plain(_contradictions(built)),
        "cells": tree,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(document, sort_keys=True, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    return load_artefact(path)


def first_difference(
    expected: Artefact, actual: Artefact, *, tolerance: float = 1e-6
) -> str | None:
    """Describe the first cell (sorted order) whose estimate differs by more than ``tolerance``.

    A cell present in only one artefact counts as a difference. ``None`` when every estimate of
    both scalings, missing cells included, agrees.
    """
    for key in sorted(set(expected.cells) | set(actual.cells)):
        name = ".".join(key)
        if key not in expected.cells:
            return f"{name}: absent from the committed artefact"
        if key not in actual.cells:
            return f"{name}: absent from the refit"
        old, new = expected.cells[key].estimate, actual.cells[key].estimate
        if abs(old - new) > tolerance:
            return f"{name}: committed estimate {old:.9g}, refit {new:.9g}"
    return None
