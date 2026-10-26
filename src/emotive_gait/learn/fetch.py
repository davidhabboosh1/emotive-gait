"""Fetch and verify the manifest's files under ``root/<dataset id>/``.

A dataset with an ``archive`` is obtained as that one file (copied from ``from_paths[id]`` when
the caller already has it, else downloaded from ``url``), checked against ``archive_sha256`` and
``archive_md5`` and then only the named clip and aux members are extracted (a nested zip is
``outer::inner`` and is read through memory, never unpacked whole). A dataset without an archive
downloads each file from ``url`` joined with its relative path. Whatever lands is verified
against the manifest's per-file hashes. ``fetch`` never raises on a hash mismatch and prints
nothing: it returns the problems and the caller reports them.
"""

from __future__ import annotations

import io
import os
import shutil
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from emotive_gait.learn.manifest import (
    AuxFile,
    ClipEntry,
    DatasetEntry,
    HashProblem,
    Manifest,
    md5_of,
    select,
    sha256_of,
    verify,
)

Downloader = Callable[[str, Path], None]
NESTED = "::"


def urlretrieve_to(url: str, dest: Path) -> None:
    """Stream ``url`` to a temporary file next to ``dest``, then rename it into place."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    urllib.request.urlretrieve(url, part)
    os.replace(part, dest)


def _archive_problems(dataset: DatasetEntry, archive: Path) -> list[HashProblem]:
    shown = f"{dataset.id}/{dataset.archive}"
    checks = (
        (dataset.archive_sha256, sha256_of, ""),
        (dataset.archive_md5, md5_of, "md5:"),
    )
    problems = []
    for expected, hasher, prefix in checks:
        if expected is None:
            continue
        actual = hasher(archive) if archive.is_file() else None
        if actual != expected:
            problems.append(
                HashProblem(shown, prefix + expected, None if actual is None else prefix + actual)
            )
    return problems


class _Members:
    """Extracts named members of a zip, caching the nested zips it has opened."""

    def __init__(self, archive: Path) -> None:
        self._zip = zipfile.ZipFile(archive)
        self._nested: dict[str, zipfile.ZipFile] = {}

    def close(self) -> None:
        self._zip.close()

    def extract(self, member: str, dest: Path) -> None:
        """Write ``member`` to ``dest``; an absent member is skipped (``verify`` reports it)."""
        zf, name = self._zip, member
        if NESTED in member:
            outer, name = member.split(NESTED, 1)
            if outer not in self._nested:
                try:
                    self._nested[outer] = zipfile.ZipFile(io.BytesIO(self._zip.read(outer)))
                except KeyError:
                    return
            zf = self._nested[outer]
        part = dest.with_name(dest.name + ".part")
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            source = zf.open(name)
        except KeyError:
            return
        with source, part.open("wb") as out:
            shutil.copyfileobj(source, out)
        os.replace(part, dest)


def _wanted(dataset: DatasetEntry, base: Path) -> list[AuxFile | ClipEntry]:
    """The aux and clip entries that are missing or have the wrong hash under ``base``."""
    out: list[AuxFile | ClipEntry] = []
    for entry in (*dataset.aux_files, *dataset.clips):
        file = base / entry.path
        if not file.is_file() or sha256_of(file) != entry.sha256:
            out.append(entry)
    return out


def _fetch_dataset(
    dataset: DatasetEntry, root: Path, source: Path | None, downloader: Downloader
) -> list[HashProblem]:
    base = root / dataset.id
    wanted = _wanted(dataset, base)
    if not wanted:
        return []
    base.mkdir(parents=True, exist_ok=True)
    if dataset.archive is None:
        prefix = dataset.url.rstrip("/") + "/"
        for entry in wanted:
            downloader(prefix + entry.path, base / entry.path)
        return []
    archive = base / dataset.archive
    if not archive.is_file() or _archive_problems(dataset, archive):
        if source is not None:
            if source.resolve() != archive.resolve():
                shutil.copyfile(source, archive)
        else:
            downloader(dataset.url, archive)
    problems = _archive_problems(dataset, archive)
    if problems:
        return problems
    members = _Members(archive)
    try:
        for entry in wanted:
            members.extract(entry.archive_member or entry.path, base / entry.path)
    finally:
        members.close()
    return []


def fetch(
    manifest: Manifest,
    root: Path,
    *,
    from_paths: Mapping[str, Path] | None = None,
    downloader: Downloader = urlretrieve_to,
    datasets: Iterable[str] | None = None,
) -> list[HashProblem]:
    """Make ``root/<id>/`` hold the manifest's files; return every hash problem left."""
    root = Path(root)
    local = from_paths or {}
    problems: list[HashProblem] = []
    for dataset in select(manifest, datasets):
        source = Path(local[dataset.id]) if dataset.id in local else None
        archive_problems = _fetch_dataset(dataset, root, source, downloader)
        if archive_problems:
            problems.extend(archive_problems)
        else:
            problems.extend(verify(manifest, root, datasets=[dataset.id]))
    return problems
