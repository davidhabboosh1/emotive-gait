"""Reader for PTD files: one row per frame with x, y, z for every joint.

Two layouts are accepted, decided from token counts: the real Body Movement Library layout (a
first line holding the frame count N, then N or N + 1 rows of ``3 * J`` numbers; the extra
trailing row is dropped) and a layout whose rows
carry a leading frame number (``3 * J + 1`` numbers; the number is dropped).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from emotive_gait.mocap.skeleton import MocapFormatError, SkeletonClip, to_z_up


def read_ptd(
    path: Path,
    *,
    joint_names: tuple[str, ...],
    scale: float,
    up_axis: Literal["y", "z"],
    fps: float = 60.0,
) -> SkeletonClip:
    """Read a point-trajectory file; ``joint_names`` gives the column order."""
    path = Path(path)
    width = 3 * len(joint_names)
    header: tuple[int, int] | None = None  # (declared frames, line number)
    rows: list[list[float]] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        tokens = line.split()
        if not tokens:
            continue
        if not rows and header is None and len(tokens) == 1:
            try:
                header = (int(tokens[0]), number)
            except ValueError:
                raise MocapFormatError(
                    path, number, "frame-count header is not an integer"
                ) from None
            continue
        if len(tokens) not in (width, width + 1):
            raise MocapFormatError(
                path,
                number,
                f"expected {width} columns (or {width + 1} with a frame number), got {len(tokens)}",
            )
        try:
            values = [float(t) for t in tokens]
        except ValueError as exc:
            raise MocapFormatError(path, number, f"non-numeric value ({exc})") from exc
        rows.append(values[-width:])
    if header is not None:
        declared, header_line = header
        if len(rows) not in (declared, declared + 1):
            raise MocapFormatError(
                path,
                header_line,
                f"header declares {declared} frames but the file has {len(rows)} data rows",
            )
        rows = rows[:declared]  # real BML files carry one duplicated trailing row
    if not rows:
        raise MocapFormatError(path, None, "no frames")
    data = np.asarray(rows).reshape(len(rows), len(joint_names), 3) * scale
    return SkeletonClip(
        joint_names=tuple(joint_names),
        parents=None,
        positions=to_z_up(data, up_axis),
        fps=float(fps),
        source=str(path),
    )
