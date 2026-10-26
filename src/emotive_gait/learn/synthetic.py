"""A synthetic normalised gait table with known effects, for tests and the report's self-check.

``synthetic_table`` draws the mixed model of design spec 4.3 directly, in the column layout of
``learn.table`` (``ID_COLUMNS`` + the 14 ``MEASURED_FIELDS``)::

    x[a, e, c] = base + effect[e] + u[a] + v[a, e] + eps[a, e, c]

``base`` is the ``GaitParameters()`` default of the parameter, ``effect[e]`` comes from
``effects`` (0.0 where absent, always 0.0 for neutral), ``u[a]`` is a per-actor random intercept
(SD 0.05), ``v[a, e]`` a per-actor random slope for each non-neutral emotion (SD ``actor_sd``)
and ``eps`` the cycle noise (SD ``cycle_sd``). Every actor walks one neutral clip and one clip
per emotion of ``cycles_per_clip`` cycles each.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from emotive_gait.gait.params import GaitParameters
from emotive_gait.gait.scaling import MEASURED_FIELDS
from emotive_gait.learn.table import COLUMNS

INTERCEPT_SD = 0.05
LEG_LENGTH = 0.9  # m, a nominal human leg; the table is already normalised


def synthetic_table(
    *,
    n_actors: int,
    emotions: Sequence[str],
    effects: Mapping[tuple[str, str], float],
    actor_sd: float,
    cycle_sd: float,
    cycles_per_clip: int,
    seed: int,
) -> pd.DataFrame:
    """One row per synthetic cycle; ``effects`` is keyed ``(emotion, parameter)``."""
    unknown = sorted(
        f"{e}/{p}" for e, p in effects if e not in emotions or p not in MEASURED_FIELDS
    )
    if unknown:
        raise ValueError(f"effect for an emotion or parameter not in the table: {unknown}")
    rng = np.random.default_rng(seed)
    base = GaitParameters()
    rows: list[dict] = []
    for a in range(n_actors):
        actor = f"s{a:02d}"
        intercept = rng.normal(0.0, INTERCEPT_SD, len(MEASURED_FIELDS))
        for clip, emotion in enumerate(("neutral", *emotions)):
            if emotion == "neutral":
                slope = np.zeros(len(MEASURED_FIELDS))
            else:
                slope = rng.normal(0.0, actor_sd, len(MEASURED_FIELDS))
            mean = np.array(
                [
                    getattr(base, p) + effects.get((emotion, p), 0.0) + intercept[i] + slope[i]
                    for i, p in enumerate(MEASURED_FIELDS)
                ]
            )
            for cycle in range(cycles_per_clip):
                values = mean + rng.normal(0.0, cycle_sd, len(MEASURED_FIELDS))
                rows.append(
                    {
                        "dataset": "synthetic",
                        "subject": actor,
                        "actor": actor,
                        "emotion": emotion,
                        "label_kind": "exact",
                        "clip": clip,
                        "cycle": cycle,
                        "leg_length": LEG_LENGTH,
                        **dict(zip(MEASURED_FIELDS, values.tolist(), strict=True)),
                    }
                )
    return pd.DataFrame(rows, columns=list(COLUMNS))
