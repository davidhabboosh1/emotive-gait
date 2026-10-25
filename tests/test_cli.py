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
