from __future__ import annotations

from pathlib import Path

import pytest

from emotive_gait.learn.manifest import (
    ManifestError,
    load_manifest,
    sha256_of,
    verify,
)
from learn.manifests import clip_dict, dataset_dict, write_manifest, write_minimal_manifest

REPO = Path(__file__).resolve().parents[2]


def test_manifest_round_trips_and_hashes_itself(tmp_path):
    m = load_manifest(write_minimal_manifest(tmp_path))
    assert m.datasets[0].id == "toy" and m.datasets[0].clips[0].label_kind == "exact"
    assert m.sha256 == sha256_of(tmp_path / "manifest.yaml")


def test_verify_reports_missing_and_mismatched_files(tmp_path):  # Review Focus 3
    m = load_manifest(write_minimal_manifest(tmp_path))
    problems = verify(m, tmp_path / "raw")
    assert len(problems) == 1 and problems[0].actual is None
    (tmp_path / "raw" / "toy").mkdir(parents=True)
    (tmp_path / "raw" / "toy" / "walk.bvh").write_text("x")
    problems = verify(m, tmp_path / "raw")
    assert problems[0].expected != problems[0].actual and problems[0].path.endswith("walk.bvh")
    (tmp_path / "raw" / "toy" / "walk.bvh").write_bytes(b"walk")
    assert verify(m, tmp_path / "raw") == []


def test_verify_covers_aux_files_and_respects_the_dataset_filter(tmp_path):
    aux = {"path": "shared/skel.asf", "sha256": "0" * 64}
    other = dataset_dict(id="other", files=[clip_dict("b.bvh", b"b")])
    toy = dataset_dict(files=[clip_dict("walk.bvh", b"walk")], aux_files=[aux])
    m = load_manifest(write_manifest(tmp_path, [toy, other]))
    assert {p.path for p in verify(m, tmp_path)} == {
        "toy/walk.bvh",
        "toy/shared/skel.asf",
        "other/b.bvh",
    }
    assert {p.path for p in verify(m, tmp_path, datasets=["other"])} == {"other/b.bvh"}
    with pytest.raises(ManifestError, match="nope"):
        verify(m, tmp_path, datasets=["nope"])


def test_unknown_label_kind_names_the_key(tmp_path):
    with pytest.raises(ManifestError, match="label_kind"):
        load_manifest(write_minimal_manifest(tmp_path, label_kind="guessed"))


def test_unknown_reader_and_wrong_types_name_the_key(tmp_path):
    bad = write_manifest(tmp_path, [dataset_dict(reader="fbx")])
    with pytest.raises(ManifestError, match="reader"):
        load_manifest(bad)
    bad = write_manifest(tmp_path, [dataset_dict(committable="yes")])
    with pytest.raises(ManifestError, match="committable"):
        load_manifest(bad)
    clip = clip_dict("w.bvh", b"w", start_frame="1")
    with pytest.raises(ManifestError, match="start_frame"):
        load_manifest(write_manifest(tmp_path, [dataset_dict(files=[clip])]))


def test_missing_key_and_missing_file_are_manifest_errors(tmp_path):
    d = dataset_dict()
    del d["licence"]
    with pytest.raises(ManifestError, match="licence"):
        load_manifest(write_manifest(tmp_path, [d]))
    with pytest.raises(ManifestError, match="nothing.yaml"):
        load_manifest(tmp_path / "nothing.yaml")


def test_paths_may_not_escape_the_dataset_directory(tmp_path):
    bad = write_manifest(tmp_path, [dataset_dict(files=[clip_dict("../evil", b"e")])])
    with pytest.raises(ManifestError, match="path"):
        load_manifest(bad)


def test_the_committed_skeleton_names_the_three_datasets():
    m = load_manifest(REPO / "data" / "manifest.yaml")
    by_id = {d.id: d for d in m.datasets}
    assert set(by_id) == {"100style", "cmu", "bml"}
    assert by_id["bml"].committable is False and by_id["bml"].permission == ""
    assert by_id["100style"].reader == "bvh" and by_id["cmu"].archive is None
