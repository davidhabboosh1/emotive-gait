from __future__ import annotations

import pytest

from emotive_gait.learn.fetch import fetch
from emotive_gait.learn.manifest import load_manifest, verify
from learn.manifests import (
    clip_dict,
    dataset_dict,
    md5,
    sha,
    write_manifest,
    write_minimal_manifest,
    zip_bytes,
)


def fail_if_called(url, dest):
    raise AssertionError(f"unexpected download of {url}")


def write_archive_manifest(tmp_path):
    blob = zip_bytes({"walk.bvh": b"walk", "other.bvh": b"unused"})
    (tmp_path / "toy.zip").write_bytes(blob)
    d = dataset_dict(
        archive="toy.zip",
        archive_sha256=sha(blob),
        archive_md5=md5(blob),
        files=[clip_dict("walk.bvh", b"walk")],
    )
    return write_manifest(tmp_path, [d])


def test_fetch_prefers_a_local_archive_and_verifies_it(tmp_path):
    m = load_manifest(write_archive_manifest(tmp_path))
    problems = fetch(
        m, tmp_path / "raw", from_paths={"toy": tmp_path / "toy.zip"}, downloader=fail_if_called
    )
    assert problems == [] and (tmp_path / "raw" / "toy" / "walk.bvh").exists()
    assert (tmp_path / "raw" / "toy" / "toy.zip").exists()
    assert not (tmp_path / "raw" / "toy" / "other.bvh").exists()  # only the named members


def test_fetch_downloads_when_no_local_copy_and_reports_a_bad_hash(tmp_path):
    calls = []

    def dl(url, dest):
        calls.append(url)
        dest.write_bytes(b"wrong")

    m = load_manifest(write_minimal_manifest(tmp_path))
    problems = fetch(m, tmp_path / "raw", downloader=dl)
    assert calls == ["http://example.test/walk.bvh"] and problems and problems[0].actual is not None


def test_fetch_is_idempotent_once_the_files_verify(tmp_path):
    m = load_manifest(write_archive_manifest(tmp_path))
    fetch(m, tmp_path / "raw", from_paths={"toy": tmp_path / "toy.zip"}, downloader=fail_if_called)
    assert fetch(m, tmp_path / "raw", downloader=fail_if_called) == []


def test_fetch_downloads_the_archive_to_the_dataset_directory(tmp_path):
    m = load_manifest(write_archive_manifest(tmp_path))
    blob = (tmp_path / "toy.zip").read_bytes()
    (tmp_path / "toy.zip").unlink()
    calls = []

    def dl(url, dest):
        calls.append((url, dest.name))
        dest.write_bytes(blob)

    assert fetch(m, tmp_path / "raw", downloader=dl) == []
    assert calls == [("http://example.test/", "toy.zip")]


def test_a_bad_archive_hash_is_returned_and_nothing_is_extracted(tmp_path):
    m = load_manifest(write_archive_manifest(tmp_path))
    (tmp_path / "toy.zip").write_bytes(zip_bytes({"walk.bvh": b"walk", "x": b"tampered"}))
    problems = fetch(
        m, tmp_path / "raw", from_paths={"toy": tmp_path / "toy.zip"}, downloader=fail_if_called
    )
    assert {p.path for p in problems} == {"toy/toy.zip"}
    assert len(problems) == 2 and all(p.actual is not None for p in problems)  # sha256 and md5
    assert not (tmp_path / "raw" / "toy" / "walk.bvh").exists()


def test_nested_members_and_aux_files_are_extracted_without_unpacking_the_rest(tmp_path):
    inner = zip_bytes({"a_walk.ptd": b"ptd-a", "a_run.ptd": b"ptd-b"})
    blob = zip_bytes({"ptd/ptd data/a_pt.zip": inner, "readme.txt": b"hi"})
    (tmp_path / "all.zip").write_bytes(blob)
    aux = {"path": "meta/readme.txt", "sha256": sha(b"hi"), "archive_member": "readme.txt"}
    clip = clip_dict("a_walk.ptd", b"ptd-a", archive_member="ptd/ptd data/a_pt.zip::a_walk.ptd")
    d = dataset_dict(archive="all.zip", archive_sha256=sha(blob), aux_files=[aux], files=[clip])
    m = load_manifest(write_manifest(tmp_path, [d]))
    root = tmp_path / "raw"
    problems = fetch(m, root, from_paths={"toy": tmp_path / "all.zip"}, downloader=fail_if_called)
    assert problems == [] and verify(m, root) == []
    assert (root / "toy" / "meta" / "readme.txt").read_bytes() == b"hi"
    assert not (root / "toy" / "a_run.ptd").exists()


def test_a_member_missing_from_the_archive_is_reported_missing(tmp_path):
    blob = zip_bytes({"other": b"x"})
    (tmp_path / "toy.zip").write_bytes(blob)
    clip = clip_dict("walk.bvh", b"walk")
    d = dataset_dict(archive="toy.zip", archive_sha256=sha(blob), files=[clip])
    m = load_manifest(write_manifest(tmp_path, [d]))
    problems = fetch(
        m, tmp_path / "raw", from_paths={"toy": tmp_path / "toy.zip"}, downloader=fail_if_called
    )
    assert [(p.path, p.actual) for p in problems] == [("toy/walk.bvh", None)]


def test_fetch_prints_nothing_and_unknown_datasets_raise(tmp_path, capsys):
    m = load_manifest(write_minimal_manifest(tmp_path))
    fetch(m, tmp_path / "raw", downloader=lambda url, dest: dest.write_bytes(b"walk"))
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    with pytest.raises(ValueError, match="nope"):
        fetch(m, tmp_path / "raw", datasets=["nope"])
