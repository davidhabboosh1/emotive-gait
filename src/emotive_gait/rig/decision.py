"""One mapping or calibration choice and its source, shared by ``rig.mapping`` and
``rig.calibration`` (which ``rig.mapping`` imports, so the type cannot live there)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Source = Literal["name", "geometry", "user"]


@dataclass(frozen=True)
class Decision:
    """One mapping choice and where it came from."""

    what: str
    chosen: str
    source: Source
    note: str = ""
