from __future__ import annotations

import math

import pytest
from mocap.walker import full_walker

from emotive_gait.gait.scaling import MEASURED_FIELDS, G
from emotive_gait.learn.manifest import load_manifest
from emotive_gait.learn.table import build_table, normalise_table
from emotive_gait.mocap.landmarks import LANDMARKS
from learn.synthetic import write_toy_dataset

TOY = {"toy": full_walker().landmarks}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("toy")
    manifest = load_manifest(write_toy_dataset(tmp))
    lines: list[str] = []
    df = build_table(manifest, tmp / "raw", landmarks=TOY, log=lines.append)
    return df, lines


def test_build_table_has_one_row_per_cycle_and_skips_empty_clips(built, capsys):
    df, lines = built
    assert set(df.columns) >= {
        "dataset",
        "actor",
        "emotion",
        "cycle",
        "leg_length",
        *MEASURED_FIELDS,
    }
    assert (
        set(df.emotion) == {"neutral", "anger"}
        and (df.groupby(["actor", "emotion"]).size() >= 4).all()
    )
    assert any("0 cycles" in line and "a2_sadness.bvh" in line for line in lines)
    assert "sadness" not in set(df.emotion)  # the deliberately short fifth clip
    assert capsys.readouterr().out == ""  # log goes to the callable, not stdout


def test_build_table_log_defaults_to_print(tmp_path, capsys):
    manifest = load_manifest(write_toy_dataset(tmp_path))
    build_table(manifest, tmp_path / "raw", landmarks=TOY, datasets=["toy"])
    assert "0 cycles" in capsys.readouterr().out


def test_rows_carry_the_measurements_and_a_cycle_index(built):
    df, _ = built
    angry = df[df.emotion == "anger"]
    neutral = df[df.emotion == "neutral"]
    assert angry.cadence.mean() == pytest.approx(2.4, rel=0.02)
    assert neutral.cadence.mean() == pytest.approx(2.0, rel=0.02)
    first = df[(df.actor == "a1") & (df.emotion == "neutral")].cycle
    assert list(first) == list(range(len(first)))
    assert df.leg_length.between(0.5, 2.0).all() and df.label_kind.isin(["exact", "mapped"]).all()
    assert not df[list(MEASURED_FIELDS)].isna().any().any()
    assert df.com_height_scale.mean() == pytest.approx(0.9, abs=0.02)  # raw pelvis height, metres


def test_normalise_table_divides_lengths_by_leg_length(built):
    df, _ = built
    n = normalise_table(df, "froude")
    row, nrow = df.iloc[0], n.iloc[0]
    assert nrow.step_length == pytest.approx(row.step_length / row.leg_length)
    assert nrow.com_height_scale == pytest.approx(row.com_height_scale / row.leg_length)
    assert nrow.cadence == pytest.approx(row.cadence * math.sqrt(row.leg_length / G))
    assert (
        nrow.trunk_pitch == row.trunk_pitch and nrow.double_support_frac == row.double_support_frac
    )
    assert list(n.columns) == list(df.columns) and nrow.leg_length == row.leg_length


def test_ratio_scaling_divides_by_the_actors_neutral_mean(built):
    df, _ = built
    n = normalise_table(df, "ratio")
    neutral = n[n.emotion == "neutral"]
    assert neutral.cadence.groupby(neutral.actor).mean().tolist() == pytest.approx([1.0, 1.0])
    angry = n[n.emotion == "anger"]
    assert angry.cadence.mean() == pytest.approx(1.2, rel=0.02)
    assert n.trunk_pitch.tolist() == df.trunk_pitch.tolist()  # angles stay absolute


def test_ratio_scaling_without_a_neutral_gives_nan_and_bad_scaling_raises(built):
    df, _ = built
    only_angry = df[df.emotion == "anger"]
    assert normalise_table(only_angry, "ratio").cadence.isna().all()
    with pytest.raises(ValueError, match="scaling"):
        normalise_table(df, "meters")


def test_landmark_maps_cover_the_three_datasets():
    assert set(LANDMARKS) == {"100style", "cmu", "bml"}
    assert len(LANDMARKS["bml"]) == 15 and len(LANDMARKS["100style"]) == 17
