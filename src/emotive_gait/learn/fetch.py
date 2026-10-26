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


def _attempt(action: Callable[[], None], dest: Path) -> str | None:
    """Run ``action``; on ``OSError`` remove ``dest``'s ``.part`` file and return the error text."""
    try:
        action()
    except OSError as exc:
        dest.with_name(dest.name + ".part").unlink(missing_ok=True)
        return f"{type(exc).__name__}: {exc}"
    return None


def _copy(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    shutil.copyfile(source, part)
    os.replace(part, dest)


def _fetch_dataset(
    dataset: DatasetEntry, root: Path, source: Path | None, downloader: Downloader
) -> tuple[list[HashProblem], bool]:
    """Obtain the wanted files; return the failures and whether the archive itself is bad."""
    base = root / dataset.id
    wanted = _wanted(dataset, base)
    if not wanted:
        return [], False
    base.mkdir(parents=True, exist_ok=True)
    failures: list[HashProblem] = []
    if dataset.archive is None:
        prefix = dataset.url.rstrip("/") + "/"
        for entry in wanted:
            dest = base / entry.path
            if source is not None:
                action = lambda e=entry, d=dest: _copy(source / e.path, d)
            else:
                action = lambda e=entry, d=dest: downloader(prefix + e.path, d)
            error = _attempt(action, dest)
            if error is not None:
                failures.append(
                    HashProblem(f"{dataset.id}/{entry.path}", entry.sha256, None, error)
                )
        return failures, False
    archive = base / dataset.archive
    shown = f"{dataset.id}/{dataset.archive}"
    if not archive.is_file() or _archive_problems(dataset, archive):
        if source is not None:
            if source.resolve() == archive.resolve():
                error = None
            else:
                error = _attempt(lambda: _copy(source, archive), archive)
        else:
            error = _attempt(lambda: downloader(dataset.url, archive), archive)
        if error is not None:
            expected = dataset.archive_sha256 or f"md5:{dataset.archive_md5}"
            return [HashProblem(shown, expected, None, error)], True
    problems = _archive_problems(dataset, archive)
    if problems:
        return problems, True
    members = _Members(archive)
    try:
        for entry in wanted:
            dest = base / entry.path
            member = entry.archive_member or entry.path
            error = _attempt(lambda m=member, d=dest: members.extract(m, d), dest)
            if error is not None:
                failures.append(
                    HashProblem(f"{dataset.id}/{entry.path}", entry.sha256, None, error)
                )
    finally:
        members.close()
    return failures, False


def fetch(
    manifest: Manifest,
    root: Path,
    *,
    from_paths: Mapping[str, Path] | None = None,
    downloader: Downloader = urlretrieve_to,
    datasets: Iterable[str] | None = None,
) -> list[HashProblem]:
    """Make ``root/<id>/`` hold the manifest's files; return every problem left.

    ``from_paths[id]`` is a local copy: the archive for an archive dataset, else a directory
    laid out like ``root/<id>/``. An ``OSError`` on one file or archive (network, missing
    source, disk) becomes a ``HashProblem`` with ``error`` set and ``actual=None``; the rest
    is still fetched.
    """
    root = Path(root)
    local = from_paths or {}
    problems: list[HashProblem] = []
    for dataset in select(manifest, datasets):
        source = Path(local[dataset.id]) if dataset.id in local else None
        failures, archive_bad = _fetch_dataset(dataset, root, source, downloader)
        problems.extend(failures)
        if not archive_bad:
            failed = {p.path for p in failures}
            problems.extend(
                p for p in verify(manifest, root, datasets=[dataset.id]) if p.path not in failed
            )
    return problems
