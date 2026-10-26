"""``emotive-gait rig`` (inspect, walk, validate) and the ``serve`` stub, through ``main``."""

from __future__ import annotations

import json
import sys
from dataclasses import replace

import pytest
from rig.rigs import RigSpec, standing_file, walk_file

from emotive_gait.cli import main


def test_rig_inspect_prints_decisions_and_p0(tmp_path, capsys):
    path = walk_file(RigSpec(), tmp_path)
    assert main(["rig", "inspect", str(path)]) == 0
    out = capsys.readouterr().out
    assert "mixamo" in out and "cadence" in out and "source" in out


def test_rig_walk_writes_a_bvh_and_prints_slip(tmp_path, capsys):
    path = walk_file(RigSpec(), tmp_path)
    out = tmp_path / "sad.bvh"
    assert main(["rig", "walk", str(path), "--emotion", "sadness=0.8", "--out", str(out)]) == 0
    assert out.is_file() and "slip:" in capsys.readouterr().out


def test_rig_walk_with_a_prompt_uses_the_lexicon(tmp_path, capsys):
    source = walk_file(RigSpec(), tmp_path)
    argv = ["rig", "walk", str(source), "--prompt", "very gloomy", "--out", str(tmp_path / "o.bvh")]
    assert main(argv) == 0
    assert "lexicon" in capsys.readouterr().out


def test_rig_walk_with_an_unmatched_prompt_exits_two_listing_the_words(tmp_path, capsys):
    source = walk_file(RigSpec(), tmp_path)
    argv = [
        *("rig", "walk", str(source)),
        *("--prompt", "lighthouse keeper", "--out", str(tmp_path / "o.bvh")),
    ]
    assert main(argv) == 2
    assert "lighthouse" in capsys.readouterr().err


def test_rig_walk_needs_exactly_one_of_emotion_and_prompt(tmp_path, capsys):
    source = walk_file(RigSpec(), tmp_path)
    assert main(["rig", "walk", str(source), "--out", str(tmp_path / "o.bvh")]) == 2


def test_standing_clip_exits_two_with_the_reasons(tmp_path, capsys):
    path = standing_file(RigSpec(), tmp_path)
    assert main(["rig", "inspect", str(path)]) == 2
    assert "no usable cycle" in capsys.readouterr().err


def test_uninferable_units_exit_two_asking_for_scale(tmp_path, capsys):
    path = walk_file(replace(RigSpec(), units_per_m=3.0), tmp_path)
    assert main(["rig", "inspect", str(path)]) == 2
    assert "--scale" in capsys.readouterr().err


def test_blender_style_z_up_metres_file_is_inferred(tmp_path, capsys):
    path = walk_file(replace(RigSpec(), up_axis="z", units_per_m=1.0), tmp_path)
    assert main(["rig", "inspect", str(path), "--json"]) == 0
    mapping = json.loads(capsys.readouterr().out)["mapping"]
    assert mapping["up_axis"] == "z" and mapping["scale"] == pytest.approx(1.0)


def test_mapping_override_with_an_unknown_bone_exits_two(tmp_path, capsys):
    (tmp_path / "m.yaml").write_text("landmarks:\n  head: NoSuchBone\n")
    path = walk_file(RigSpec(), tmp_path)
    assert main(["rig", "inspect", str(path), "--mapping", str(tmp_path / "m.yaml")]) == 2
    assert "NoSuchBone" in capsys.readouterr().err


def test_infeasible_emotion_warns_and_still_writes(tmp_path, capsys):
    out = tmp_path / "o.bvh"
    # a 0.6 m stride on 0.6 m legs
    source = walk_file(replace(RigSpec(), leg_length_m=0.6), tmp_path, step_length=0.6)
    assert main(["rig", "walk", str(source), "--emotion", "anger=1.0", "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert out.is_file() and "exceed the robot's reach" in captured.err and "slip:" in captured.out


def test_rig_json_output_is_parseable(tmp_path, capsys):
    assert main(["rig", "inspect", str(walk_file(RigSpec(), tmp_path)), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["p0"]["cadence"] > 0


def test_serve_without_mcp_exits_one_with_the_install_hint(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "mcp", None)
    assert main(["serve"]) == 1 and "mcp" in capsys.readouterr().err


def test_rig_validate_prints_five_rows(tmp_path, capsys):
    assert main(["rig", "validate", str(walk_file(RigSpec(), tmp_path))]) == 0
    assert capsys.readouterr().out.count("\n") >= 5
