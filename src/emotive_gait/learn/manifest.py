"""The hashed data manifest: which files of which public datasets the fit may read.

YAML layout (``data/manifest.yaml``)::

    datasets:
      - id: toy                      # key used everywhere (LANDMARKS, --dataset, root/<id>/)
        name: Toy dataset
        url: http://example.test/    # the archive's URL, or the base URL when archive is null
        archive: toy.zip             # file name under root/<id>/, or null (files fetched singly)
        archive_sha256: <hex>        # optional; archive_md5 likewise; archive_bytes optional
        licence: CC BY 4.0
        committable: true            # may its numbers enter the committed artefact
        permission: ""               # who allowed it, when committable needs a permission
        reader: bvh                  # bvh | asf_amc | ptd
        reader_args: {scale: 0.01, up_axis: y}
        aux_files:                   # shared non-clip files (e.g. an ASF skeleton)
          - {path: 82/82.asf, sha256: <hex>, archive_member: null}
        files:                       # the clips
          - path: walk.bvh           # relative to root/<id>/ (and to url when archive is null)
            sha256: <hex>
            subject: s1
            actor: a1
            emotion: neutral         # a project emotion or "neutral"
            label: Neutral           # the dataset's own label
            label_kind: exact        # exact | mapped
            start_frame: null        # optional cut
            end_frame: null
            archive_member: null     # member inside the archive when it differs from path;
                                     # a nested zip is written outer_member::inner_member
            extra: {asf: 82/82.asf}  # reader extras (asf_amc: the ASF path)
        excluded:                    # clips left out, with the reason
          - {path: x.bvh, reason: turning}

``Manifest.sha256`` is the SHA-256 of the manifest file's bytes. Paths must be relative and stay
inside ``root/<id>/``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml

LABEL_KINDS = ("exact", "mapped")
READERS = ("bvh", "asf_amc", "ptd")

_REQUIRED = object()


class ManifestError(ValueError):
    """The manifest file is missing or malformed; the message names the offending key."""


@dataclass(frozen=True)
class AuxFile:
    """A shared non-clip file (CMU skeleton, 100STYLE frame cuts), verified like a clip."""

    path: str
    sha256: str
    archive_member: str | None = None


@dataclass(frozen=True)
class ClipEntry:
    path: str
    sha256: str
    subject: str
    actor: str
    emotion: str  # project emotion or "neutral"
    label: str  # the dataset's own label
    label_kind: Literal["exact", "mapped"]
    start_frame: int | None = None
    end_frame: int | None = None
    extra: Mapping[str, str] = field(default_factory=dict)
    archive_member: str | None = None


@dataclass(frozen=True)
class DatasetEntry:
    id: str
    name: str
    url: str
    archive: str | None
    archive_sha256: str | None
    archive_bytes: int | None
    licence: str
    committable: bool
    permission: str
    reader: Literal["bvh", "asf_amc", "ptd"]
    reader_args: Mapping[str, object]
    clips: tuple[ClipEntry, ...]
    excluded: tuple[Mapping[str, str], ...]
    aux_files: tuple[AuxFile, ...] = ()
    archive_md5: str | None = None


@dataclass(frozen=True)
class Manifest:
    datasets: tuple[DatasetEntry, ...]
    sha256: str  # of the manifest file itself

    def dataset(self, dataset_id: str) -> DatasetEntry:
        for d in self.datasets:
            if d.id == dataset_id:
                return d
        raise ManifestError(f"unknown dataset {dataset_id!r}")


@dataclass(frozen=True)
class HashProblem:
    path: str
    expected: str
    actual: str | None  # None = missing


def _file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm, usedforsecurity=False)
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of(path: Path) -> str:
    return _file_hash(path, "sha256")


def md5_of(path: Path) -> str:
    return _file_hash(path, "md5")


def _names(kind: type | tuple[type, ...]) -> str:
    kinds = kind if isinstance(kind, tuple) else (kind,)
    return " or ".join(k.__name__ for k in kinds)


def _get(
    d: Mapping[str, Any], key: str, kind: type | tuple[type, ...], where: str, default=_REQUIRED
):
    """``d[key]`` checked against ``kind`` (a bool never passes for int); ``default`` if absent."""
    if key not in d or (d[key] is None and default is not _REQUIRED):
        if default is _REQUIRED:
            raise ManifestError(f"{where}: missing key {key!r}")
        return default
    value = d[key]
    if not isinstance(value, kind) or (isinstance(value, bool) and bool not in _flat(kind)):
        raise ManifestError(f"{where}.{key}: expected {_names(kind)}, got {type(value).__name__}")
    return value


def _flat(kind: type | tuple[type, ...]) -> tuple[type, ...]:
    return kind if isinstance(kind, tuple) else (kind,)


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{where}: expected a mapping, got {type(value).__name__}")
    return value


def _check_path(path: str, where: str) -> str:
    p = PurePosixPath(path.replace("\\", "/"))
    if not path or p.is_absolute() or ".." in p.parts or ":" in p.parts[0]:
        raise ManifestError(f"{where}: path {path!r} must be relative and stay inside the dataset")
    return path


def _literal(value: str, allowed: tuple[str, ...], where: str) -> str:
    if value not in allowed:
        raise ManifestError(f"{where}: {value!r} is not one of {', '.join(allowed)}")
    return value


def _clip(raw: Any, where: str) -> ClipEntry:
    d = _mapping(raw, where)
    extra = _get(d, "extra", dict, where, {})
    return ClipEntry(
        path=_check_path(_get(d, "path", str, where), f"{where}.path"),
        sha256=_get(d, "sha256", str, where),
        subject=str(_get(d, "subject", (str, int), where)),
        actor=str(_get(d, "actor", (str, int), where)),
        emotion=_get(d, "emotion", str, where),
        label=str(_get(d, "label", (str, int), where)),
        label_kind=_literal(  # type: ignore[arg-type]
            _get(d, "label_kind", str, where), LABEL_KINDS, f"{where}.label_kind"
        ),
        start_frame=_get(d, "start_frame", int, where, None),
        end_frame=_get(d, "end_frame", int, where, None),
        extra={str(k): str(v) for k, v in extra.items()},
        archive_member=_get(d, "archive_member", str, where, None),
    )


def _aux(raw: Any, where: str) -> AuxFile:
    d = _mapping(raw, where)
    return AuxFile(
        path=_check_path(_get(d, "path", str, where), f"{where}.path"),
        sha256=_get(d, "sha256", str, where),
        archive_member=_get(d, "archive_member", str, where, None),
    )


def _dataset(raw: Any, where: str) -> DatasetEntry:
    d = _mapping(raw, where)
    archive = _get(d, "archive", str, where, None)
    if archive is not None:
        _check_path(archive, f"{where}.archive")
    clips = _get(d, "files", list, where, [])
    aux = _get(d, "aux_files", list, where, [])
    excluded = _get(d, "excluded", list, where, [])
    return DatasetEntry(
        id=_get(d, "id", str, where),
        name=_get(d, "name", str, where),
        url=_get(d, "url", str, where),
        archive=archive,
        archive_sha256=_get(d, "archive_sha256", str, where, None),
        archive_bytes=_get(d, "archive_bytes", int, where, None),
        licence=_get(d, "licence", str, where),
        committable=_get(d, "committable", bool, where),
        permission=_get(d, "permission", str, where, ""),
        reader=_literal(  # type: ignore[arg-type]
            _get(d, "reader", str, where), READERS, f"{where}.reader"
        ),
        reader_args=dict(_get(d, "reader_args", dict, where, {})),
        clips=tuple(_clip(c, f"{where}.files[{i}]") for i, c in enumerate(clips)),
        excluded=tuple(
            {str(k): str(v) for k, v in _mapping(e, f"{where}.excluded[{i}]").items()}
            for i, e in enumerate(excluded)
        ),
        aux_files=tuple(_aux(a, f"{where}.aux_files[{i}]") for i, a in enumerate(aux)),
        archive_md5=_get(d, "archive_md5", str, where, None),
    )


def load_manifest(path: Path) -> Manifest:
    """Parse and validate a manifest file; raises :class:`ManifestError` naming the key."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ManifestError(f"{path}: cannot read the manifest ({exc.strerror or exc})") from exc
    try:
        raw = yaml.safe_load(data)
    except yaml.YAMLError as exc:
        raise ManifestError(f"{path}: not valid YAML ({exc})") from exc
    root = _mapping(raw, "manifest")
    entries = _get(root, "datasets", list, "manifest")
    datasets = tuple(_dataset(d, f"datasets[{i}]") for i, d in enumerate(entries))
    ids = [d.id for d in datasets]
    for dataset_id in ids:
        if ids.count(dataset_id) > 1:
            raise ManifestError(f"datasets: duplicate id {dataset_id!r}")
    return Manifest(datasets=datasets, sha256=hashlib.sha256(data).hexdigest())


def select(manifest: Manifest, datasets: Iterable[str] | None) -> tuple[DatasetEntry, ...]:
    """The named datasets in manifest order (all when ``None``); an unknown id raises."""
    if datasets is None:
        return manifest.datasets
    wanted = list(datasets)
    for dataset_id in wanted:
        manifest.dataset(dataset_id)
    return tuple(d for d in manifest.datasets if d.id in wanted)


def verify(
    manifest: Manifest, root: Path, *, datasets: Iterable[str] | None = None
) -> list[HashProblem]:
    """Aux and clip files under ``root/<id>/`` whose SHA-256 differs, or which are missing."""
    problems: list[HashProblem] = []
    for dataset in select(manifest, datasets):
        entries = [*dataset.aux_files, *dataset.clips]
        for entry in entries:
            file = Path(root) / dataset.id / entry.path
            actual = sha256_of(file) if file.is_file() else None
            if actual != entry.sha256:
                problems.append(HashProblem(f"{dataset.id}/{entry.path}", entry.sha256, actual))
    return problems
