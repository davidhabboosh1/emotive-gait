"""The data-derived emotion model and its artefact (design spec 4.5).

:class:`LearnedEmotionModel` reads the generated artefact
(``emotion/data/learned_coefficients.yaml``, written only by ``emotive-gait fit``) and turns its
dimensionless per-cell effects into parameter deltas in the robot's units through the profile's
``leg_length`` and ``nominal_pelvis_height``. Effects scale linearly with intensity from zero and
emotions mix additively, as in the analytic model; the two differ only in their endpoints.

Delegation is per ``(emotion, parameter)`` cell: ``fitted`` and ``thin`` cells use the measured
effect, while ``missing`` cells and every cell of the unsourced emotions (disgust, surprise) use
the fallback model's delta for that emotion alone. ``expansiveness`` is never measured and has no
cell: for the four fitted emotions its share is exactly 0.0 (it stays at the base value, 1.0 by
default, spec 4.2), because the fitted arm-swing, abduction and step-height cells already carry
the expansiveness folded into the measured motion, and adding the analytic share would count it
twice. Disgust and surprise stay fully analytic, expansiveness included.

With the ``ratio`` scaling an effect is a fraction of the neutral value, and the neutral used is
the default :class:`GaitParameters`: lengths, ``cadence`` and ``com_height_scale`` deltas are
``base_value * estimate`` for that default base, so ``ratio`` deltas are relative to the default
neutral, not to whatever ``params`` :func:`modulate` is later given.

This module needs only ``pyyaml`` at runtime; it never imports ``learn/``, ``mocap/`` or ``sim/``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, fields
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml

from emotive_gait.emotion.evidence import GROUNDED_EMOTIONS, Emotion
from emotive_gait.emotion.model import ANALYTIC, EmotionModel, validate_intensities
from emotive_gait.gait.params import GaitParameters
from emotive_gait.gait.scaling import MEASURED_FIELDS, Scaling, to_gait_parameters

if TYPE_CHECKING:
    from emotive_gait.robot.roles import RobotProfile

SCHEMA_VERSION = 1
SCALINGS: tuple[Scaling, ...] = ("froude", "ratio")
GROUNDINGS = ("fitted", "thin", "missing")
DIRECTION_CHECKS = (
    "agrees",
    "agrees_ci_spans_zero",
    "contradicts",
    "contradicts_ci_spans_zero",
    "thin_unchecked",
    "no_estimate",
    "no_claim",
)
CONTRADICTION_KEYS = (
    "emotion",
    "parameter",
    "scaling",
    "estimate",
    "ci_low",
    "ci_high",
    "citation",
)


class ArtefactMissingError(FileNotFoundError):
    """Raised when the learned-model artefact has not been generated."""


class ArtefactSchemaError(ValueError):
    """Raised when the artefact does not follow the schema; the message names the key."""


@dataclass(frozen=True)
class ArtefactCell:
    """One ``(emotion, parameter)`` cell as stored in the artefact.

    The field list duplicates ``learn.fit.Cell`` by hand (the two layers may not share a class)
    plus ``direction_check``; a test keeps the two in step.
    """

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
    fallback_reason: str | None
    per_dataset: Mapping[str, float]
    direction_check: str


@dataclass(frozen=True)
class Artefact:
    cells: Mapping[tuple[str, str, str], ArtefactCell]  # (scaling, emotion, parameter)
    contradictions: tuple[Mapping[str, object], ...] = ()
    provenance: Mapping[str, object] = field(default_factory=dict)


def direction_check(
    estimate: float,
    ci: tuple[float, float] | None,
    grounding: str,
    claim_direction: int | None,
) -> str:
    """Compare one cell with its cited direction; returns one of ``DIRECTION_CHECKS``.

    A direction-0 claim agrees when the interval spans zero, contradicts when it does not, and
    is ``thin_unchecked`` without an interval. A signed claim compares the sign of ``estimate``
    (sign only when there is no interval) and the verdict gains ``_ci_spans_zero`` when the
    interval contains zero.

    Callers pass ``claim_direction=0`` only for ``evidence.NO_CHANGE_CLAIMS`` and ``None`` for
    the direction-0 claims that mean "not reported vs neutral".
    """
    if grounding not in GROUNDINGS:
        raise ValueError(f"grounding must be one of {GROUNDINGS}, got {grounding!r}")
    if claim_direction is None:
        return "no_claim"
    if claim_direction not in (-1, 0, 1):
        raise ValueError(f"claim_direction must be -1, 0, +1 or None, got {claim_direction!r}")
    if grounding == "missing":
        return "no_estimate"
    spans = None if ci is None else ci[0] <= 0.0 <= ci[1]
    if claim_direction == 0:
        if spans is None:
            return "thin_unchecked"
        return "agrees_ci_spans_zero" if spans else "contradicts"
    sign = (estimate > 0) - (estimate < 0)
    if sign == claim_direction:
        return "agrees_ci_spans_zero" if spans else "agrees"
    return "contradicts_ci_spans_zero" if spans else "contradicts"


def _actors(counts: Iterable[int]) -> str:
    values = sorted(set(counts))
    noun = "actor" if values == [1] else "actors"
    span = str(values[0]) if len(values) == 1 else f"{values[0]}-{values[-1]}"
    return f"{span} {noun}"


def describe_thin(entries: Iterable[tuple[str, str, int]]) -> list[str]:
    """``(emotion, parameter, n_actors)`` thin cells as short phrases for a warning.

    An emotion whose every measured parameter is thin collapses to one phrase
    (``"anger: all 14 cells thin (1 actor)"``); otherwise each cell is
    ``"emotion.parameter (n actors)"``. Emotions keep their first-seen order.
    """
    by_emotion: dict[str, list[tuple[str, int]]] = {}
    for emotion, parameter, n in entries:
        by_emotion.setdefault(emotion, []).append((parameter, int(n)))
    out: list[str] = []
    for emotion, cells in by_emotion.items():
        if {p for p, _ in cells} >= set(MEASURED_FIELDS):
            out.append(f"{emotion}: all {len(cells)} cells thin ({_actors(n for _, n in cells)})")
        else:
            out += [f"{emotion}.{p} ({_actors([n])})" for p, n in cells]
    return out


def default_artefact_path() -> Path:
    """The committed artefact inside the package (it may not exist yet)."""
    return Path(str(resources.files("emotive_gait.emotion") / "data" / "learned_coefficients.yaml"))


# --- loading -------------------------------------------------------------------------------

_CELL_KEYS = tuple(f.name for f in fields(ArtefactCell) if f.name not in ("emotion", "parameter"))
_NAN = float("nan")


def _require(mapping: Mapping[str, object], key: str, where: str) -> object:
    if key not in mapping:
        raise ArtefactSchemaError(f"artefact: missing key {where + key!r}")
    return mapping[key]


def _number(value: object, key: str) -> float | None:
    """A float, or ``None`` for YAML ``null`` (how NaN is stored)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtefactSchemaError(f"artefact: {key!r} must be a number or null, got {value!r}")
    return float(value)


def _number_or_nan(value: object, key: str) -> float:
    x = _number(value, key)
    return _NAN if x is None else x


def _count(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtefactSchemaError(
            f"artefact: {key!r} must be a non-negative integer, got {value!r}"
        )
    return value


def _cell(raw: object, emotion: str, parameter: str, where: str) -> ArtefactCell:
    if not isinstance(raw, Mapping):
        raise ArtefactSchemaError(f"artefact: {where!r} must be a mapping")
    unknown = sorted(set(raw) - set(_CELL_KEYS))
    if unknown:
        raise ArtefactSchemaError(f"artefact: unknown key {where + '.' + str(unknown[0])!r}")
    v = {k: _require(raw, k, where + ".") for k in _CELL_KEYS}
    estimate = _number(v["estimate"], where + ".estimate")
    if estimate is None:
        raise ArtefactSchemaError(f"artefact: {where + '.estimate'!r} must be a number, got null")
    if v["grounding"] not in GROUNDINGS:
        raise ArtefactSchemaError(f"artefact: {where + '.grounding'!r} must be one of {GROUNDINGS}")
    if v["direction_check"] not in DIRECTION_CHECKS:
        raise ArtefactSchemaError(
            f"artefact: {where + '.direction_check'!r} must be one of {DIRECTION_CHECKS}"
        )
    ci_low = _number(v["ci_low"], where + ".ci_low")
    ci_high = _number(v["ci_high"], where + ".ci_high")
    has_interval = (ci_low, ci_high) != (None, None)
    if v["grounding"] == "fitted" and (ci_low is None or ci_high is None):
        raise ArtefactSchemaError(f"artefact: {where!r} is fitted but has no ci_low/ci_high")
    if v["grounding"] != "fitted" and has_interval:
        raise ArtefactSchemaError(f"artefact: {where!r} is {v['grounding']} but has an interval")
    reason = v["fallback_reason"]
    if reason is not None and not isinstance(reason, str):
        raise ArtefactSchemaError(
            f"artefact: {where + '.fallback_reason'!r} must be a string or null"
        )
    if not isinstance(v["converged_maximal"], bool):
        raise ArtefactSchemaError(
            f"artefact: {where + '.converged_maximal'!r} must be true or false"
        )
    per_dataset = v["per_dataset"]
    if not isinstance(per_dataset, Mapping):
        raise ArtefactSchemaError(f"artefact: {where + '.per_dataset'!r} must be a mapping")
    return ArtefactCell(
        emotion=emotion,
        parameter=parameter,
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        between_actor_sd=_number_or_nan(v["between_actor_sd"], where + ".between_actor_sd"),
        residual_sd=_number_or_nan(v["residual_sd"], where + ".residual_sd"),
        n_actors=_count(v["n_actors"], where + ".n_actors"),
        n_cycles=_count(v["n_cycles"], where + ".n_cycles"),
        n_datasets=_count(v["n_datasets"], where + ".n_datasets"),
        n_mapped_actors=_count(v["n_mapped_actors"], where + ".n_mapped_actors"),
        grounding=v["grounding"],
        converged_maximal=v["converged_maximal"],
        fallback_reason=reason,
        per_dataset={
            str(k): _number_or_nan(x, f"{where}.per_dataset.{k}") for k, x in per_dataset.items()
        },
        direction_check=v["direction_check"],
    )


def load_artefact(path: Path) -> Artefact:
    """Read the artefact; ``ArtefactMissingError`` if absent, ``ArtefactSchemaError`` if malformed.

    Schema (version 1)::

        schema_version: 1
        provenance: {...}                       # free mapping
        contradictions:                         # list, may be empty
          - {emotion, parameter, scaling, estimate, ci_low, ci_high, citation}
        cells:
          <scaling>:                            # froude, ratio
            <emotion>:
              <parameter>:                      # one of gait.scaling.MEASURED_FIELDS
                {estimate, ci_low, ci_high, between_actor_sd, residual_sd, n_actors,
                 n_cycles, n_datasets, n_mapped_actors, grounding, converged_maximal,
                 fallback_reason, per_dataset: {<dataset id>: float}, direction_check}

    NaN is stored as ``null``: a null interval bound reads back as ``None``; a null SD or
    per-dataset value reads back as ``nan``.
    """
    path = Path(path)
    if not path.is_file():
        raise ArtefactMissingError(
            f"no learned-model artefact at {path}; run `emotive-gait fit` to generate it"
        )
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ArtefactSchemaError(f"artefact: {path} is not valid YAML: {exc}") from None
    if not isinstance(data, Mapping):
        raise ArtefactSchemaError(f"artefact: {path} is not a YAML mapping")
    version = _require(data, "schema_version", "")
    if version != SCHEMA_VERSION:
        raise ArtefactSchemaError(
            f"artefact: 'schema_version' must be {SCHEMA_VERSION}, got {version!r}"
        )
    provenance = _require(data, "provenance", "")
    if not isinstance(provenance, Mapping):
        raise ArtefactSchemaError("artefact: 'provenance' must be a mapping")
    raw_contradictions = _require(data, "contradictions", "")
    if not isinstance(raw_contradictions, list):
        raise ArtefactSchemaError("artefact: 'contradictions' must be a list")
    contradictions: list[Mapping[str, object]] = []
    for i, entry in enumerate(raw_contradictions):
        if not isinstance(entry, Mapping):
            raise ArtefactSchemaError(f"artefact: 'contradictions[{i}]' must be a mapping")
        for key in CONTRADICTION_KEYS:
            _require(entry, key, f"contradictions[{i}].")
        contradictions.append(dict(entry))
    raw_cells = _require(data, "cells", "")
    if not isinstance(raw_cells, Mapping):
        raise ArtefactSchemaError("artefact: 'cells' must be a mapping")
    emotions = {e.value for e in Emotion}
    cells: dict[tuple[str, str, str], ArtefactCell] = {}
    for scaling, by_emotion in raw_cells.items():
        if scaling not in SCALINGS:
            raise ArtefactSchemaError(f"artefact: unknown scaling key 'cells.{scaling}'")
        if not isinstance(by_emotion, Mapping):
            raise ArtefactSchemaError(f"artefact: 'cells.{scaling}' must be a mapping")
        for emotion, by_parameter in by_emotion.items():
            if emotion not in emotions:
                raise ArtefactSchemaError(
                    f"artefact: unknown emotion key 'cells.{scaling}.{emotion}'"
                )
            if not isinstance(by_parameter, Mapping):
                raise ArtefactSchemaError(
                    f"artefact: 'cells.{scaling}.{emotion}' must be a mapping"
                )
            for parameter, raw in by_parameter.items():
                where = f"cells.{scaling}.{emotion}.{parameter}"
                if parameter not in MEASURED_FIELDS:
                    raise ArtefactSchemaError(f"artefact: unknown parameter key {where!r}")
                cells[(scaling, emotion, parameter)] = _cell(raw, emotion, parameter, where)
    return Artefact(
        cells=cells,
        contradictions=tuple(contradictions),
        provenance=dict(provenance),
    )


# --- the model -----------------------------------------------------------------------------

# Re-dimensioning bases. Under ``froude`` the delta does not depend on the base, so an all-zero
# base gives it exactly (no ``(x + d) - x`` rounding); under ``ratio`` it is relative to the
# default neutral.
_ZERO_BASE = GaitParameters(**dict.fromkeys((f.name for f in fields(GaitParameters)), 0.0))
_RATIO_BASE = GaitParameters()


class LearnedEmotionModel(EmotionModel):
    """Deltas from the fitted artefact, delegating missing and unsourced cells to ``fallback``."""

    profile_aware = True  # ``_effect`` re-dimensions through the profile's leg length

    def __init__(
        self,
        artefact: Artefact,
        *,
        scaling: Scaling = "froude",
        fallback: EmotionModel = ANALYTIC,
    ) -> None:
        if scaling not in SCALINGS:
            raise ValueError(f"scaling must be 'froude' or 'ratio', got {scaling!r}")
        cells = {(e, p): c for (s, e, p), c in artefact.cells.items() if s == scaling}
        if not cells:
            raise ValueError(f"the artefact has no cells for the {scaling!r} scaling")
        self.artefact = artefact
        self.scaling: Scaling = scaling
        self.fallback = fallback
        self._cells = cells

    @classmethod
    def default(cls, *, scaling: Scaling = "froude") -> LearnedEmotionModel:
        """The model on the committed artefact; ``ArtefactMissingError`` if it is absent."""
        return cls(load_artefact(default_artefact_path()), scaling=scaling)

    def cell(self, emotion: Emotion | str, parameter: str) -> ArtefactCell | None:
        """The ``fitted`` or ``thin`` cell this model uses, or ``None`` if it takes the fallback."""
        emotion = Emotion(emotion)
        if emotion not in GROUNDED_EMOTIONS:
            return None
        found = self._cells.get((emotion.value, parameter))
        if found is None or found.grounding == "missing":
            return None
        return found

    def _cells_where(
        self, keep: Callable[[ArtefactCell | None], bool]
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted((e.value, p) for e in Emotion for p in MEASURED_FIELDS if keep(self.cell(e, p)))
        )

    @property
    def thin_cells(self) -> tuple[tuple[str, str], ...]:
        """``(emotion, parameter)`` cells used with one or two actors (no interval)."""
        return self._cells_where(lambda c: c is not None and c.grounding == "thin")

    @property
    def analytic_cells(self) -> tuple[tuple[str, str], ...]:
        """Measured-field cells taken from the fallback: missing ones, and disgust and surprise."""
        return self._cells_where(lambda c: c is None)

    def _effect(self, parameter: str, estimate: float, profile: RobotProfile) -> float:
        base = _ZERO_BASE if self.scaling == "froude" else _RATIO_BASE
        moved = to_gait_parameters(
            {parameter: estimate},
            base,
            scaling=self.scaling,
            leg_length=profile.leg_length,
            nominal_pelvis_height=profile.nominal_pelvis_height,
        )
        return getattr(moved, parameter) - getattr(base, parameter)

    def deltas(
        self, intensities: Mapping[Emotion, float], profile: RobotProfile
    ) -> Mapping[str, float]:
        """Sum over emotions of each cell's share (measured, or the fallback's for that emotion).

        ``expansiveness`` gets a 0.0 share from every fitted emotion (anger, fear, happiness,
        sadness), whatever its cells' grounding, and the fallback's share from disgust and
        surprise.
        """
        clean = validate_intensities(intensities)
        total: dict[str, float] = {}
        for emotion, intensity in clean.items():
            if intensity == 0.0:
                continue
            fallback: Mapping[str, float] | None = None
            fitted_emotion = emotion in GROUNDED_EMOTIONS
            for parameter in (*MEASURED_FIELDS, "expansiveness"):
                cell = None if parameter == "expansiveness" else self.cell(emotion, parameter)
                if parameter == "expansiveness" and fitted_emotion:
                    share = 0.0  # folded into the fitted arm/step cells; fixed at the base
                elif cell is not None:
                    share = intensity * self._effect(parameter, cell.estimate, profile)
                else:
                    if fallback is None:
                        fallback = self.fallback.deltas({emotion: intensity}, profile)
                    share = fallback.get(parameter, 0.0)
                total[parameter] = total.get(parameter, 0.0) + share
        return total
