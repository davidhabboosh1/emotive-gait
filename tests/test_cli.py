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
