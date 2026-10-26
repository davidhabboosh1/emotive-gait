"""Manifest-writing helpers shared by the learn tests (``from learn.manifests import ...``)."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import yaml


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dataset_dict(**overrides) -> dict:
    d = {
        "id": "toy",
        "name": "Toy",
        "url": "http://example.test/",
        "archive": None,
        "licence": "CC0",
        "committable": True,
        "permission": "",
        "reader": "bvh",
        "reader_args": {"scale": 1.0, "up_axis": "z"},
        "files": [],
        "excluded": [],
    }
    d.update(overrides)
    return d


def clip_dict(path: str, data: bytes, **overrides) -> dict:
    c = {
        "path": path,
        "sha256": sha(data),
        "subject": "s1",
        "actor": "a1",
        "emotion": "neutral",
        "label": "Neutral",
        "label_kind": "exact",
    }
    c.update(overrides)
    return c


def write_manifest(directory: Path, datasets: list[dict]) -> Path:
    path = directory / "manifest.yaml"
    path.write_text(yaml.safe_dump({"datasets": datasets}, sort_keys=False))
    return path


def write_minimal_manifest(directory: Path, **clip_overrides) -> Path:
    """One dataset, one clip ``walk.bvh`` whose content should be ``b"walk"``."""
    clip = clip_dict("walk.bvh", b"walk", **clip_overrides)
    return write_manifest(directory, [dataset_dict(files=[clip])])


def zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buffer.getvalue()


def md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()
