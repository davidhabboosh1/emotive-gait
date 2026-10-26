"""In-memory learned-model artefacts for the emotion tests (no YAML, no fit)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from emotive_gait.emotion.evidence import (
    GROUNDED_EMOTIONS,
    NO_CHANGE_CLAIMS,
    Emotion,
    claims_for,
)
from emotive_gait.emotion.learned import Artefact, ArtefactCell, direction_check
from emotive_gait.gait.scaling import MEASURED_FIELDS

FITTED_EMOTIONS = tuple(sorted(e.value for e in GROUNDED_EMOTIONS))
_HALF_WIDTH = 0.01


def make_artefact(
    effects: Mapping[tuple[str, str], float],
    *,
    thin: Iterable[tuple[str, str]] = (),
    contradictions: Iterable[Mapping[str, object]] = (),
) -> Artefact:
    """Both scalings with the same numbers; cells absent from ``effects`` are ``missing``.

    Cells named in ``thin`` are ``thin`` (two actors, no interval); the other ``effects`` cells
    are ``fitted`` with an interval of +-0.01 around the estimate. ``direction_check`` is
    computed from ``claims_for`` so the artefact is self-consistent.
    """
    thin = set(thin)
    unknown = (set(effects) | thin) - {(e, p) for e in FITTED_EMOTIONS for p in MEASURED_FIELDS}
    if unknown:
        raise ValueError(f"not a fitted (emotion, parameter) cell: {sorted(unknown)}")
    if thin - set(effects):
        raise ValueError(f"thin cells need an effect: {sorted(thin - set(effects))}")
    cells: dict[tuple[str, str, str], ArtefactCell] = {}
    for emotion in FITTED_EMOTIONS:
        # The writer's convention: direction 0 only for no-change claims, None for the
        # "not reported vs neutral" ones.
        directions = {
            c.parameter: c.direction
            for c in claims_for(Emotion(emotion))
            if c.direction != 0 or (c.emotion, c.parameter) in NO_CHANGE_CLAIMS
        }
        for parameter in MEASURED_FIELDS:
            key = (emotion, parameter)
            if key not in effects:
                estimate, ci, grounding, n_actors, n_cycles, n_datasets = (
                    0.0,
                    None,
                    "missing",
                    0,
                    0,
                    0,
                )
                sd = 0.0
            elif key in thin:
                estimate, ci, grounding, n_actors, n_cycles, n_datasets = (
                    effects[key],
                    None,
                    "thin",
                    2,
                    20,
                    1,
                )
                sd = 0.1
            else:
                estimate = effects[key]
                ci = (estimate - _HALF_WIDTH, estimate + _HALF_WIDTH)
                grounding, n_actors, n_cycles, n_datasets = "fitted", 5, 50, 1
                sd = 0.1
            cell = ArtefactCell(
                emotion=emotion,
                parameter=parameter,
                estimate=estimate,
                ci_low=None if ci is None else ci[0],
                ci_high=None if ci is None else ci[1],
                between_actor_sd=sd,
                residual_sd=sd,
                n_actors=n_actors,
                n_cycles=n_cycles,
                n_datasets=n_datasets,
                n_mapped_actors=0,
                grounding=grounding,
                converged_maximal=grounding != "missing",
                fallback_reason=None,
                per_dataset={},
                direction_check=direction_check(estimate, ci, grounding, directions.get(parameter)),
            )
            for scaling in ("froude", "ratio"):
                cells[(scaling, emotion, parameter)] = cell
    return Artefact(
        cells=cells,
        contradictions=tuple(dict(c) for c in contradictions),
        provenance={"source": "tests/emotion/artefacts.py"},
    )


# Agrees with every grounded claim on a measured field: direction x 0.05 for signed claims, 0.0
# (its +-0.01 interval spans zero) for the two no-change claims. ``expansiveness`` has no cell;
# direction-0 "not reported vs neutral" claims are left missing (analytic).
FULL_AGREEING: Mapping[tuple[str, str], float] = MappingProxyType(
    {
        (c.emotion.value, c.parameter): c.direction * 0.05
        for e in sorted(GROUNDED_EMOTIONS)
        for c in claims_for(e)
        if c.parameter in MEASURED_FIELDS
        and (c.direction != 0 or (c.emotion, c.parameter) in NO_CHANGE_CLAIMS)
    }
)
