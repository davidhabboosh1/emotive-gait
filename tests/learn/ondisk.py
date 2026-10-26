"""An on-disk synthetic dataset for the pipeline and CLI tests (fixture in ``tests/conftest.py``).

Three actors walk a neutral, an "anger" (the same walk played faster) and a "happiness" clip, so
anger and happiness cells are ``fitted`` and the leave-one-actor-out loop runs; actor ``a1`` also
walks one "sadness" clip (a thin cell). A second dataset, ``toy_uncommittable``, has one actor and
``committable: false``. The ids are registered in ``mocap.landmarks.LANDMARKS`` for the session so
``build_table`` (and so ``emotive-gait fit``) finds the walker's landmark map.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from mocap.walker import full_walker

from emotive_gait.mocap.landmarks import LANDMARKS
from learn.manifests import clip_dict, dataset_dict, write_manifest
from learn.synthetic import walker_clip, write_bvh

# (actor, emotion, seconds, speedup): actors differ a little so the mixed model is not singular.
_PLAN = [
    ("a1", "neutral", 8.0, 1.00),
    ("a1", "anger", 8.0, 1.20),
    ("a1", "happiness", 8.0, 1.10),
    ("a1", "sadness", 8.0, 0.85),
    ("a2", "neutral", 8.0, 0.98),
    ("a2", "anger", 8.0, 1.26),
    ("a2", "happiness", 8.0, 1.14),
    ("a3", "neutral", 8.0, 1.03),
    ("a3", "anger", 8.0, 1.17),
    ("a3", "happiness", 8.0, 1.07),
]


@dataclass(frozen=True)
class OnDisk:
    manifest: Path
    root: Path


def _clips(root: Path, dataset: str, plan) -> list[dict]:
    clips = []
    for actor, emotion, seconds, speedup in plan:
        name = f"{actor}_{emotion}.bvh"
        data = write_bvh(root / dataset / name, walker_clip(seconds, speedup))
        clips.append(
            clip_dict(
                name,
                data,
                subject=actor,
                actor=actor,
                emotion=emotion,
                label=emotion.title(),
                label_kind="exact",
            )
        )
    return clips


def write_fit_dataset(directory: Path) -> OnDisk:
    """Write both datasets under ``directory`` and return the manifest and raw root."""
    root = directory / "raw"
    toy = dataset_dict(id="toy", name="Toy", files=_clips(root, "toy", _PLAN))
    private = dataset_dict(
        id="toy_uncommittable",
        name="Toy uncommittable",
        committable=False,
        files=_clips(
            root,
            "toy_uncommittable",
            [("b1", "neutral", 8.0, 1.0), ("b1", "anger", 8.0, 1.2)],
        ),
    )
    manifest = write_manifest(directory, [toy, private])
    return OnDisk(manifest=manifest, root=root)


def register_landmarks(patch: pytest.MonkeyPatch) -> None:
    """Make ``build_table`` find the walker's landmark map for the toy dataset ids."""
    walker = full_walker().landmarks
    patch.setitem(LANDMARKS, "toy", walker)  # type: ignore[arg-type]
    patch.setitem(LANDMARKS, "toy_uncommittable", walker)  # type: ignore[arg-type]
