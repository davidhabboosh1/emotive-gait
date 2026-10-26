import sys

import pytest

from emotive_gait.cli import main


@pytest.mark.requires_gl
def test_demo_writes_a_video(tmp_path):
    out = tmp_path / "d.mp4"
    code = main(
        [
            "demo",
            "--emotion",
            "sadness=1.0",
            "--duration",
            "0.5",
            "--width",
            "320",
            "--height",
            "240",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert out.is_file() and out.stat().st_size > 1000


def test_invalid_emotion_name_is_reported_not_traced(capsys):
    assert main(["demo", "--emotion", "ennui=1.0"]) == 2
    assert "ennui" in capsys.readouterr().err


def test_out_of_range_intensity_is_reported_not_traced(capsys):
    assert main(["demo", "--emotion", "anger=5.0"]) == 2
    assert "anger" in capsys.readouterr().err


def test_unsourced_emotions_warn_the_user(capsys):
    main(["demo", "--emotion", "disgust=1.0", "--duration", "0.2", "--dry-run"])
    assert "unsourced" in capsys.readouterr().err.lower()


def test_evidence_subcommand_lists_grounding(capsys):
    assert main(["evidence"]) == 0
    out = capsys.readouterr().out
    assert "Roether" in out and "UNSOURCED" in out


def test_argparse_usage_errors_return_instead_of_exiting(capsys):
    assert main(["demo", "--fps", "abc"]) == 2
    assert capsys.readouterr().err


def test_malformed_emotion_spec_is_reported(capsys):
    assert main(["demo", "--emotion", "anger"]) == 2
    assert "anger" in capsys.readouterr().err


def test_dry_run_prints_parameters_without_writing(tmp_path, capsys):
    out = tmp_path / "x.mp4"
    assert main(["demo", "--emotion", "sadness=1.0", "--dry-run", "--out", str(out)]) == 0
    assert "cadence" in capsys.readouterr().out
    assert not out.exists()


def test_unusable_out_path_is_an_environment_failure(tmp_path, capsys):
    blocker = tmp_path / "f.txt"
    blocker.write_text("x")
    code = main(["demo", "--duration", "0.2", "--out", str(blocker / "x.mp4")])
    err = capsys.readouterr().err
    assert code == 1
    assert err.strip() and "Traceback" not in err


def test_physics_summary_names_the_fail_reason(capsys):
    code = main(["demo", "--physics", "--no-video", "--root", "free", "--duration", "3"])
    out = capsys.readouterr().out
    assert code == 0
    assert "fell: True" in out and "fail_reason: uprightness" in out


def test_width_above_the_framebuffer_is_a_user_error(tmp_path, capsys):
    code = main(["demo", "--width", "1280", "--duration", "0.2", "--out", str(tmp_path / "x.mp4")])
    err = capsys.readouterr().err
    assert code == 2
    assert "framebuffer" in err and "Traceback" not in err


def test_odd_width_is_a_user_error(tmp_path, capsys):
    code = main(["demo", "--width", "321", "--duration", "0.2", "--out", str(tmp_path / "x.mp4")])
    err = capsys.readouterr().err
    assert code == 2
    assert "even" in err and "Traceback" not in err


def test_duration_shorter_than_one_frame_is_a_user_error(tmp_path, capsys):
    out = tmp_path / "x.mp4"
    code = main(["demo", "--duration", "0.01", "--out", str(out)])
    err = capsys.readouterr().err
    assert code == 2
    assert "frame" in err and "Traceback" not in err
    assert not out.exists()


def test_model_revision_mismatch_is_an_environment_failure(monkeypatch, capsys):
    from emotive_gait.robot import loader

    def mismatched(**kwargs):
        raise ValueError("torso mass remainder 6.0 != expected 6.0155")

    monkeypatch.setattr(loader, "load_robot", mismatched)
    code = main(["demo", "--physics", "--no-video", "--duration", "0.2"])
    err = capsys.readouterr().err
    assert code == 1
    assert "revision" in err and "Traceback" not in err


def test_infeasible_parameters_warn_but_still_render(capsys):
    # --dry-run loads no model, so drive the warning through the physics path with no video
    code = main(
        ["demo", "--physics", "--no-video", "--root", "free", "--duration", "0.2"]
        + ["--hold-speed", "3.0"]
    )
    err = capsys.readouterr().err
    assert code == 0 and "exceed the robot's reach" in err


def test_xz_rail_is_a_root_choice(capsys):
    assert main(["demo", "--physics", "--no-video", "--root", "xz-rail", "--duration", "0.5"]) == 0
    assert "root: xz-rail" in capsys.readouterr().out


def test_physics_prints_the_gait_report(capsys):
    assert main(["demo", "--physics", "--no-video", "--duration", "1"]) == 0
    out = capsys.readouterr().out
    assert "zmp_outside_frac:" in out and "foot_slip_rms_m_s:" in out
    assert "pelvis_tracking_ptp_m:" in out
    line = next(ln for ln in out.splitlines() if ln.startswith("played_params:"))
    assert "step_length=0.0" in line
    assert "pelvis bob (m)" not in out


@pytest.mark.requires_gl
def test_missing_ffmpeg_is_an_environment_failure(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda *a, **k: None)
    out = tmp_path / "v.mp4"
    code = main(
        ["demo", "--duration", "0.2", "--width", "320", "--height", "240", "--out", str(out)]
    )
    assert code == 1
    assert "ffmpeg" in capsys.readouterr().err


# --- the learned model, `fit` and `fetch` ------------------------------------------------------


@pytest.fixture
def fixture_artefact_path(tmp_path):
    from dataclasses import fields

    from artefacts import FULL_AGREEING, make_artefact

    from emotive_gait.learn.artefact import write_artefact
    from emotive_gait.learn.fit import Cell

    source = make_artefact(FULL_AGREEING, thin=[("anger", "cadence")])
    cells: dict = {"froude": {}, "ratio": {}}
    for (scaling, _, parameter), c in source.cells.items():
        cells[scaling].setdefault(parameter, []).append(
            Cell(**{f.name: getattr(c, f.name) for f in fields(Cell)})
        )
    path = tmp_path / "coefficients.yaml"
    write_artefact(path, cells, provenance={"source": "test"})
    return path


def _use_artefact(monkeypatch, path):
    monkeypatch.setattr("emotive_gait.emotion.learned.default_artefact_path", lambda: path)


def _fit_args(data, out):
    return [
        "fit",
        *("--manifest", str(data.manifest), "--root", str(data.root)),
        *("--out", str(out), "--report", str(out.with_suffix(".md"))),
        *("--derived", str(out.parent / "derived")),
    ]


def test_demo_learned_without_artefact_exits_one_with_the_fit_hint(capsys, monkeypatch, tmp_path):
    _use_artefact(monkeypatch, tmp_path / "none.yaml")
    assert main(["demo", "--emotion", "sadness=1.0", "--model", "learned", "--dry-run"]) == 1
    assert "emotive-gait fit" in capsys.readouterr().err


def test_demo_learned_prints_the_model_and_warns_on_thin_cells(
    capsys, fixture_artefact_path, monkeypatch
):
    _use_artefact(monkeypatch, fixture_artefact_path)
    monkeypatch.setitem(sys.modules, "statsmodels", None)  # runtime needs no fit dependency
    monkeypatch.setitem(sys.modules, "pandas", None)
    code = main(["demo", "--emotion", "anger=1.0", "--model", "learned", "--dry-run"])
    out, err = capsys.readouterr()
    assert code == 0
    assert "model: learned (froude)" in out and "cadence" in out
    assert err.count("thin cells") == 1 and "anger.cadence (2 actors)" in err


def test_demo_learned_names_the_delegated_cells_once(capsys, fixture_artefact_path, monkeypatch):
    _use_artefact(monkeypatch, fixture_artefact_path)
    main(["demo", "--emotion", "disgust=0.5", "--model", "learned", "--dry-run"])
    err = capsys.readouterr().err
    assert err.count("analytic model, not measured") == 1 and "disgust (all)" in err
    assert "unsourced" in err


def test_demo_learned_scaling_changes_the_parameters(capsys, fixture_artefact_path, monkeypatch):
    _use_artefact(monkeypatch, fixture_artefact_path)
    main(["demo", "--emotion", "anger=1.0", "--model", "learned", "--dry-run"])
    froude = capsys.readouterr().out
    ratio_args = ["--model", "learned", "--scaling", "ratio", "--dry-run"]
    main(["demo", "--emotion", "anger=1.0", *ratio_args])
    ratio = capsys.readouterr().out
    assert "model: learned (ratio)" in ratio and froude != ratio


def test_demo_analytic_says_so(capsys):
    assert main(["demo", "--emotion", "sadness=1.0", "--dry-run"]) == 0
    assert "model: analytic" in capsys.readouterr().out


def test_evidence_learned_adds_estimate_columns(capsys, fixture_artefact_path, monkeypatch):
    _use_artefact(monkeypatch, fixture_artefact_path)
    assert main(["evidence", "--model", "learned"]) == 0
    out = capsys.readouterr().out
    assert "ESTIMATE" in out and "CHECK" in out and "2 actors" in out and "Roether" in out
    assert "agrees" in out and "UNSOURCED" in out


def test_evidence_learned_without_artefact_exits_one(capsys, monkeypatch, tmp_path):
    _use_artefact(monkeypatch, tmp_path / "none.yaml")
    assert main(["evidence", "--model", "learned"]) == 1
    assert "emotive-gait fit" in capsys.readouterr().err


def test_fit_with_a_missing_manifest_exits_two(capsys, tmp_path):
    assert main(["fit", "--manifest", str(tmp_path / "nope.yaml")]) == 2
    assert "nope.yaml" in capsys.readouterr().err


def test_fit_without_statsmodels_exits_one(
    capsys, monkeypatch, tmp_path, synthetic_dataset_on_disk
):
    monkeypatch.setitem(sys.modules, "statsmodels", None)
    args = [*_fit_args(synthetic_dataset_on_disk, tmp_path / "c.yaml"), "--include-uncommittable"]
    assert main(args) == 1
    assert "pip install" in capsys.readouterr().err


def test_include_uncommittable_needs_an_out_path(capsys, synthetic_dataset_on_disk):
    data = synthetic_dataset_on_disk
    code = main(["fit", "--manifest", str(data.manifest), "--include-uncommittable"])
    assert code == 2 and "--out" in capsys.readouterr().err


def test_fit_check_passes_on_a_matching_artefact(
    capsys, tmp_path, synthetic_dataset_on_disk, fitted_toy
):
    out = tmp_path / "c.yaml"
    out.write_bytes((fitted_toy[1] / "coef.yaml").read_bytes())
    code = main([*_fit_args(synthetic_dataset_on_disk, out), "--check"])
    assert code == 0 and "matches" in capsys.readouterr().out


def test_fit_check_fails_naming_the_first_differing_cell(
    capsys, monkeypatch, tmp_path, synthetic_dataset_on_disk, fitted_toy
):
    from emotive_gait.learn import pipeline

    good = fitted_toy[1] / "coef.yaml"
    out = tmp_path / "c.yaml"
    out.write_text(good.read_text().replace("estimate: 0.0", "estimate: 9.9", 1))

    def refit(*args, out_artefact, **kwargs):  # the real refit is covered by the passing test
        out_artefact.write_bytes(good.read_bytes())

    monkeypatch.setattr(pipeline, "run_fit", refit)
    code = main([*_fit_args(synthetic_dataset_on_disk, out), "--check"])
    err = capsys.readouterr().err
    assert code == 2 and "froude." in err and "9.9" in err


def test_fit_check_without_an_artefact_exits_one(capsys, tmp_path, synthetic_dataset_on_disk):
    code = main([*_fit_args(synthetic_dataset_on_disk, tmp_path / "none.yaml"), "--check"])
    assert code == 1 and "emotive-gait fit" in capsys.readouterr().err


def test_fit_prints_the_headline_thin_cells_and_contradictions(
    capsys, tmp_path, synthetic_dataset_on_disk
):
    args = [
        *_fit_args(synthetic_dataset_on_disk, tmp_path / "c.yaml"),
        *("--datasets", "toy_uncommittable", "--include-uncommittable"),
    ]
    code = main(args)
    out = capsys.readouterr().out
    assert code == 0
    assert "thin cells" in out and "anger" in out and "(1 actor)" in out
    assert "(1 actors)" not in out
    assert "contradictions" in out and (tmp_path / "c.md").is_file()


def test_fit_on_a_tampered_file_exits_two_naming_it(capsys, tmp_path, synthetic_dataset_on_disk):
    data = synthetic_dataset_on_disk
    target = data.root / "toy_uncommittable" / "b1_neutral.bvh"
    original = target.read_bytes()
    target.write_text("tampered")
    args = [
        *_fit_args(data, tmp_path / "c.yaml"),
        *("--datasets", "toy_uncommittable", "--include-uncommittable"),
    ]
    try:
        code = main(args)
    finally:
        target.write_bytes(original)
    err = capsys.readouterr().err
    assert code == 2 and "b1_neutral.bvh" in err and "Traceback" not in err


def test_fetch_with_a_missing_manifest_exits_two(capsys, tmp_path):
    assert main(["fetch", "--manifest", str(tmp_path / "nope.yaml")]) == 2
    assert "nope.yaml" in capsys.readouterr().err


def test_fetch_verifies_files_already_in_place(capsys, synthetic_dataset_on_disk):
    data = synthetic_dataset_on_disk
    code = main(["fetch", "--manifest", str(data.manifest), "--root", str(data.root)])
    assert code == 0 and "verified" in capsys.readouterr().out


def test_fetch_reports_a_failed_download_with_the_url_and_the_hash(
    capsys, monkeypatch, tmp_path, synthetic_dataset_on_disk
):
    def offline(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("urllib.request.urlretrieve", offline)  # no network in tests
    data = synthetic_dataset_on_disk
    args = ["fetch", "--manifest", str(data.manifest), "--root", str(tmp_path)]
    code = main([*args, "--datasets", "toy"])
    err = capsys.readouterr().err
    assert code == 2 and "toy/a1_neutral.bvh: expected" in err and "missing" in err
    assert "OSError: offline" in err and "http://example.test/" in err
    assert "fetch by hand" in err and "Traceback" not in err


def test_fetch_uses_a_local_directory(tmp_path, synthetic_dataset_on_disk):
    data = synthetic_dataset_on_disk
    args = ["fetch", "--manifest", str(data.manifest), "--root", str(tmp_path)]
    code = main([*args, "--datasets", "toy", "--from", f"toy={data.root / 'toy'}"])
    assert code == 0 and (tmp_path / "toy" / "a1_neutral.bvh").is_file()


def test_fetch_rejects_a_malformed_from(capsys):
    assert main(["fetch", "--from", "toy"]) == 2
    assert "ID=PATH" in capsys.readouterr().err


def test_uncommittable_fit_defaults_the_report_beside_the_artefact(
    capsys, tmp_path, synthetic_dataset_on_disk
):
    data = synthetic_dataset_on_disk
    out = tmp_path / "c.yaml"
    args = ["fit", "--manifest", str(data.manifest), "--root", str(data.root), "--out", str(out)]
    args += ["--derived", str(tmp_path / "d"), "--datasets", "toy_uncommittable"]
    assert main([*args, "--include-uncommittable"]) == 0
    assert (tmp_path / "fit-report.md").is_file()


def test_fetch_with_a_scheme_less_url_exits_two_without_a_traceback(capsys, tmp_path):
    from learn.manifests import clip_dict, dataset_dict, write_manifest

    d = dataset_dict(url="not-a-url", files=[clip_dict("walk.bvh", b"walk")])
    manifest = write_manifest(tmp_path, [d])
    assert main(["fetch", "--manifest", str(manifest), "--root", str(tmp_path / "raw")]) == 2
    err = capsys.readouterr().err
    assert "toy/walk.bvh" in err and "ValueError" in err and "Traceback" not in err


def test_demo_learned_with_an_artefact_lacking_the_scaling_exits_one(capsys, monkeypatch, tmp_path):
    import yaml

    path = tmp_path / "a.yaml"
    path.write_text(
        yaml.safe_dump({"schema_version": 1, "provenance": {}, "contradictions": [], "cells": {}})
    )
    _use_artefact(monkeypatch, path)
    code = main(["demo", "--emotion", "anger=1.0", "--model", "learned", "--dry-run"])
    assert code == 1 and "froude" in capsys.readouterr().err


def test_fit_with_out_defaults_the_report_beside_it_never_the_committed_one(
    capsys, monkeypatch, tmp_path, synthetic_dataset_on_disk
):
    committed = tmp_path / "docs" / "committed-report.md"
    monkeypatch.setattr("emotive_gait.cli._REPORT", str(committed))
    monkeypatch.setattr("emotive_gait.cli._report_feasibility", lambda *a: None)
    data = synthetic_dataset_on_disk
    out = tmp_path / "local" / "c.yaml"
    args = ["fit", "--manifest", str(data.manifest), "--root", str(data.root), "--out", str(out)]
    assert main([*args, "--derived", str(tmp_path / "d"), "--datasets", "toy"]) == 0
    assert (out.parent / "fit-report.md").is_file() and not committed.exists()


def test_fit_keeps_exit_zero_when_the_feasibility_step_fails(
    capsys, monkeypatch, tmp_path, synthetic_dataset_on_disk
):
    def broken(*args):
        raise RuntimeError("ik exploded")

    monkeypatch.setattr("emotive_gait.cli._report_feasibility", broken)
    out = tmp_path / "c.yaml"
    assert main([*_fit_args(synthetic_dataset_on_disk, out), "--datasets", "toy"]) == 0
    captured = capsys.readouterr()
    assert "feasibility of the learned presets not reported" in captured.out
    assert "ik exploded" in captured.out and "Traceback" not in captured.err
    assert out.is_file()


def test_fit_reports_a_value_error_as_a_failed_fit(
    capsys, monkeypatch, tmp_path, synthetic_dataset_on_disk
):
    from emotive_gait.learn import pipeline

    def broken(*args, **kwargs):
        raise ValueError("bad numbers")

    monkeypatch.setattr(pipeline, "run_fit", broken)
    code = main(_fit_args(synthetic_dataset_on_disk, tmp_path / "c.yaml"))
    err = capsys.readouterr().err
    assert code == 1 and "fit failed: bad numbers" in err and "environment failure" not in err
