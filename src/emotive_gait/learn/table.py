"""One row per gait cycle across the manifest's clips, and its normalised counterpart.

``build_table`` reads every clip of the chosen datasets with the dataset's reader, cuts it to
the manifest's ``start_frame:end_frame`` when given, and appends one row per usable cycle
(``extract_cycles``) with the subject's ``leg_length``. A clip without a usable cycle is logged
(with the discard reasons) and skipped. The 14 ``MEASURED_FIELDS`` columns hold the raw
measurements in SI units and radians; ``com_height_scale`` holds the pelvis height in metres.

``normalise_table`` returns the same columns, normalised per row:

``froude``  Hof scaling by the row's ``leg_length`` (the same rule as ``mocap.normalise``).
``ratio``   lengths, ``cadence`` and ``com_height_scale`` divided by the actor's neutral mean of
            that column (within the dataset; ``NaN`` for an actor without a neutral clip);
            angles and fractions stay absolute, the fit takes their differences.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from emotive_gait.gait.scaling import LENGTH_FIELDS, MEASURED_FIELDS, G, Scaling
from emotive_gait.learn.manifest import DatasetEntry, Manifest, select
from emotive_gait.mocap.extract import extract_cycles, leg_length, summarise
from emotive_gait.mocap.landmarks import BML_PTD_JOINTS, LANDMARKS
from emotive_gait.mocap.readers.asf_amc import read_asf_amc
from emotive_gait.mocap.readers.bvh import read_bvh
from emotive_gait.mocap.readers.ptd import read_ptd
from emotive_gait.mocap.skeleton import LandmarkMap, SkeletonClip

ID_COLUMNS = ("dataset", "subject", "actor", "emotion", "label_kind", "clip", "cycle", "leg_length")
COLUMNS = (*ID_COLUMNS, *MEASURED_FIELDS)
_SCALED_BY_NEUTRAL = (*LENGTH_FIELDS, "cadence", "com_height_scale")


def _read(dataset: DatasetEntry, base: Path, clip) -> SkeletonClip:
    path = base / clip.path
    if dataset.reader == "bvh":
        skeleton = read_bvh(path, **dataset.reader_args)
    elif dataset.reader == "asf_amc":
        if "asf" not in clip.extra:
            raise ValueError(f"{dataset.id}/{clip.path}: an asf_amc clip needs extra['asf']")
        skeleton = read_asf_amc(base / clip.extra["asf"], path)
    else:
        skeleton = read_ptd(path, joint_names=BML_PTD_JOINTS, **dataset.reader_args)
    if clip.start_frame is not None or clip.end_frame is not None:
        cut = skeleton.positions[clip.start_frame : clip.end_frame]
        skeleton = dataclasses.replace(skeleton, positions=cut)
    return skeleton


def _measured(measurement, name: str) -> float:
    return float(
        measurement.pelvis_height if name == "com_height_scale" else getattr(measurement, name)
    )


def build_table(
    manifest: Manifest,
    root: Path,
    *,
    datasets: Iterable[str] | None = None,
    landmarks: Mapping[str, LandmarkMap] = LANDMARKS,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Per-cycle raw measurements of every clip; zero-cycle clips are logged and skipped."""
    rows: list[dict] = []
    for dataset in select(manifest, datasets):
        base = Path(root) / dataset.id
        lmap = landmarks[dataset.id]
        for clip in dataset.clips:
            skeleton = _read(dataset, base, clip)
            cycles = extract_cycles(skeleton, lmap)
            if not cycles:
                reasons = dict(summarise(skeleton, lmap).discard_reasons)
                log(f"{dataset.id}/{clip.path}: 0 cycles, skipped (reasons: {reasons})")
                continue
            length = leg_length(skeleton, lmap)
            for index, measurement in enumerate(cycles):
                row = {
                    "dataset": dataset.id,
                    "subject": clip.subject,
                    "actor": clip.actor,
                    "emotion": clip.emotion,
                    "label_kind": clip.label_kind,
                    "clip": clip.path,
                    "cycle": index,
                    "leg_length": length,
                }
                row.update({name: _measured(measurement, name) for name in MEASURED_FIELDS})
                rows.append(row)
    return pd.DataFrame(rows, columns=list(COLUMNS))


def normalise_table(df: pd.DataFrame, scaling: Scaling) -> pd.DataFrame:
    """The table with its measured columns normalised (see the module docstring)."""
    if scaling not in ("froude", "ratio"):
        raise ValueError(f"scaling must be 'froude' or 'ratio', got {scaling!r}")
    out = df.copy()
    if scaling == "froude":
        length = df["leg_length"]
        for name in (*LENGTH_FIELDS, "com_height_scale"):
            out[name] = df[name] / length
        out["cadence"] = df["cadence"] * np.sqrt(length / G)
        return out
    neutral = df[df["emotion"] == "neutral"]
    for name in _SCALED_BY_NEUTRAL:
        base = neutral.groupby(["dataset", "actor"])[name].mean()
        keys = pd.MultiIndex.from_frame(df[["dataset", "actor"]])
        out[name] = df[name].to_numpy() / base.reindex(keys).to_numpy()
    return out
