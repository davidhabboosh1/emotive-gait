"""``emotive-gait rig`` (inspect, walk, validate) and the ``serve`` stub, through ``main``."""

from __future__ import annotations

import json
import sys
from dataclasses import replace

import pytest
import yaml
from rig.rigs import RigSpec, standing_file, walk_file

from emotive_gait.cli import main
from emotive_gait.mocap.skeleton import Landmark


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
    err = capsys.readouterr().err
    assert "--emotion" in err and "--prompt" in err


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


def test_rig_validate_prints_five_rows_and_writes_the_report(tmp_path, capsys):
    report = tmp_path / "reports" / "rig.md"
    argv = ["rig", "validate", str(walk_file(RigSpec(), tmp_path)), "--report", str(report)]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in ("neutral", "sad", "angry", "happy", "fear"))
    assert out.count("\n") >= 5
    text = report.read_bytes().decode("utf-8")
    assert "\r" not in text and "## mixamo" in text and "| fear | yes |" in text


def _inspect_mapping(path, capsys, *extra):
    assert main(["rig", "inspect", str(path), "--json", *extra]) == 0
    return json.loads(capsys.readouterr().out)["mapping"]


def _user_decisions(path, capsys, mapping_file):
    assert main(["rig", "inspect", str(path), "--json", "--mapping", str(mapping_file)]) == 0
    out = json.loads(capsys.readouterr().out)["mapping"]
    return {d["what"]: d for d in out["decisions"]}


def test_inspect_output_fed_back_marks_every_landmark_and_role_user(tmp_path, capsys):

    path = walk_file(RigSpec(), tmp_path)
    mapping = _inspect_mapping(path, capsys)
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(mapping))
    decisions = _user_decisions(path, capsys, tmp_path / "m.yaml")
    landmarks = {lm.value for lm in Landmark}
    asked = {*mapping["landmarks"], *(f"role:{r}" for r in mapping["roles"])}
    assert asked and all(decisions[w]["source"] == "user" for w in asked)
    assert asked >= landmarks & set(decisions)


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")  # a deliberately broken leg
def test_edited_knee_landmark_in_a_fed_back_file_shows_in_the_decisions(tmp_path, capsys):

    path = walk_file(RigSpec(), tmp_path)
    mapping = _inspect_mapping(path, capsys)
    other = mapping["landmarks"]["ankle_l"]
    mapping["landmarks"]["knee_l"] = other
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(mapping))
    main(["rig", "inspect", str(path), "--json", "--mapping", str(tmp_path / "m.yaml")])
    out = json.loads(capsys.readouterr().out)["mapping"]
    assert out["landmarks"]["knee_l"] == other


def test_mapping_file_scale_and_up_axis_are_honoured_flags_win(tmp_path, capsys):
    path = walk_file(RigSpec(), tmp_path)  # y-up, centimetres
    wrong = tmp_path / "m.yaml"
    wrong.write_text("up_axis: z\nscale: 5.0\nfamily: x\nleg_length: 1.0\n")
    # the file's wrong axis is used as if given on the command line, so the rig does not map
    assert main(["rig", "inspect", str(path), "--mapping", str(wrong)]) == 2
    assert "ignoring informational mapping keys: family, leg_length" in capsys.readouterr().err
    # explicit flags take precedence over the file
    argv = ["rig", "inspect", str(path), "--json", "--up", "y", "--scale", "0.01"]
    assert main([*argv, "--mapping", str(wrong)]) == 0
    mapping = json.loads(capsys.readouterr().out)["mapping"]
    assert mapping["up_axis"] == "y" and mapping["scale"] == pytest.approx(0.01)


@pytest.mark.parametrize("bad", ["-1", "0", "nan", "inf", "abc"])
def test_scale_must_be_positive_and_finite(tmp_path, capsys, bad):
    path = walk_file(RigSpec(), tmp_path)
    assert main(["rig", "inspect", str(path), f"--scale={bad}"]) == 2
    assert "--scale" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("line", "word"), [("scale: -3", "scale"), ("scale: abc", "scale"), ("up_axis: q", "up_axis")]
)
def test_bad_mapping_file_scale_or_up_axis_exits_two_without_a_traceback(
    tmp_path, capsys, line, word
):
    (tmp_path / "m.yaml").write_text(f"{line}\n")
    path = walk_file(RigSpec(), tmp_path)
    assert main(["rig", "inspect", str(path), "--mapping", str(tmp_path / "m.yaml")]) == 2
    err = capsys.readouterr().err
    assert word in err and "Traceback" not in err


def test_mapping_file_with_only_scale_and_up_axis_is_used_without_inference(tmp_path, capsys):
    (tmp_path / "m.yaml").write_text("scale: 0.01\nup_axis: y\n")
    path = walk_file(RigSpec(), tmp_path)
    assert main(["rig", "inspect", str(path), "--json", "--mapping", str(tmp_path / "m.yaml")]) == 0
    mapping = json.loads(capsys.readouterr().out)["mapping"]
    assert mapping["scale"] == 0.01 and mapping["up_axis"] == "y"
    assert not {"scale", "up_axis"} & {d["what"] for d in mapping["decisions"]}
