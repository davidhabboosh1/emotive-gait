"""``validate_rig`` (zero and the four presets on one rig), ``render_report`` and the spec-driven
report entry point."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

from emotive_gait.rig import validate as validate_mod
from emotive_gait.rig.validate import (
    MIN_STEPS,
    UNMEASURED,
    ValidationRow,
    ValidationTable,
    _commit,
    _commit_line,
    main,
    render_report,
    validate_rig,
)
from rig.rigs import RigSpec, walk_file


def test_validate_rig_runs_zero_and_four_presets(tmp_path):
    t = validate_rig(
        walk_file(RigSpec(), tmp_path), scale=None, up_axis=None, mapping_overrides=None
    )
    assert [r.run for r in t.rows] == ["neutral", "sad", "angry", "happy", "fear"] and t.rows[
        0
    ].within
    assert all(r.measured for r in t.rows), [r.status for r in t.rows]


def test_render_report_lists_not_run_rigs():
    text = render_report({}, not_run={"bandai": "archive not fetched"})
    assert "bandai" in text and "archive not fetched" in text


def test_a_short_source_is_written_for_at_least_four_cycles_per_run(tmp_path):
    t = validate_rig(
        walk_file(RigSpec(), tmp_path, seconds=2.0),
        scale=None,
        up_axis=None,
        mapping_overrides=None,
    )
    assert all(r.measured for r in t.rows), [r.status for r in t.rows]
    sad = next(r for r in t.rows if r.run == "sad")
    assert t.source_duration_s < sad.duration_s  # sad slows the cadence: written longer
    assert all(r.duration_s >= t.source_duration_s for r in t.rows)
    assert MIN_STEPS == 8


def _table(rows: tuple[ValidationRow, ...]) -> ValidationTable:
    return ValidationTable(
        source="x.bvh",
        family=None,
        scale=0.01,
        scale_given=False,
        up_axis="y",
        up_given=False,
        leg_length=0.9,
        cycles=1,
        source_pelvis_height=0.9,
        nominal_pelvis_height=0.85,
        source_duration_s=2.0,
        skewed=(),
        unobservable=(),
        half_bound=(),
        pitch_innermost=False,
        rows=rows,
    )


def test_unmeasured_and_error_rows_are_never_tolerance_failures():
    nan = {"cadence": math.nan, "step_length": math.nan}
    unmeasured = ValidationRow(
        run="sad",
        feasible=True,
        max_residual=MappingProxyType(nan),
        within=None,
        slip_m=0.0,
        height_m=0.0,
        duration_s=2.0,
        status=UNMEASURED,
    )
    error = replace(unmeasured, run="angry", feasible=None, status="error: boom")
    text = render_report({"rig": _table((unmeasured, error))}, not_run={})
    assert f"| sad | yes | - | n/a | {UNMEASURED} |" in text
    assert "| angry | n/a | - | n/a | error: boom |" in text
    assert "fields beyond tolerance" not in text and "runs not measured: sad" in text
    assert "- rig: none" in text  # error rows are not infeasible presets


def test_the_report_regenerates_from_a_spec(tmp_path, monkeypatch):
    walk_file(RigSpec(), tmp_path)
    (tmp_path / "spec.yaml").write_text(
        "rigs:\n"
        "  - {name: synthetic, path: mixamo.bvh}\n"
        "  - {name: gone, path: missing.bvh, scale: 0.01, up_axis: y}\n"
        "not_run:\n"
        "  - {name: other, reason: none supplied}\n"
        "setup_commands: [emotive-gait fetch]\n"
        "findings:\n"
        "  - >-\n"
        "    an authored\n"
        "    finding\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert main(["spec.yaml", "--out", "out/report.md"]) == 0
    text = (tmp_path / "out" / "report.md").read_bytes().decode("utf-8")
    assert "\r" not in text and "## synthetic" in text
    assert "python -m emotive_gait.rig.validate spec.yaml --out out/report.md" in text
    assert "emotive-gait rig validate missing.bvh --scale 0.01 --up y" in text
    assert "- gone: missing.bvh is not on disk" in text and "- other: none supplied" in text
    assert "- an authored finding" in text
    (tmp_path / "bad.yaml").write_text("rigs: [{path: x.bvh}]\n", encoding="utf-8")
    assert main(["bad.yaml", "--out", "r.md"]) == 2


def test_a_feasible_run_slipping_beyond_two_mm_is_gated_and_listed(tmp_path):
    from emotive_gait.emotion.model import ANALYTIC
    from emotive_gait.rig.synthesis import SlipReport, load_rig, walk
    from emotive_gait.rig.validate import SLIP_BOUND_M, validation_row

    rig = load_rig(walk_file(RigSpec(), tmp_path), scale=None, up_axis=None, mapping_overrides=None)
    result = walk(
        rig,
        intensities={},
        intensities_source="flags",
        model=ANALYTIC,
        model_name="analytic",
        duration_s=4.0,
    )
    assert result.feasibility.feasible and SLIP_BOUND_M == 0.002
    assert validation_row("neutral", result, 4.0).slip_within is True
    slipped = replace(
        result, slip=SlipReport(max_travel_m=0.003, max_height_error_m=0.0, stances=4)
    )
    row = validation_row("neutral", slipped, 4.0)
    assert row.feasible and row.slip_within is False
    text = render_report({"rig": _table((row,))}, not_run={})
    assert "| no |" not in text and "| NO |" in text.split("\n## Pelvis")[0]
    assert "slip beyond 2 mm on feasible runs: neutral (3.0 mm travel, 0.0 mm height)" in text
    infeasible = replace(slipped, feasibility=SimpleNamespace(feasible=False))  # a flag only
    assert validation_row("angry", infeasible, 4.0).slip_within is None


def test_commit_line_is_the_hash_alone_for_a_bare_hash():
    assert _commit_line("abc1234") == "Commit: `abc1234`."


def test_commit_line_names_the_modified_tracked_files():
    assert _commit_line("abc1234|src/a.py,tests/b.py") == (
        "Commit: `abc1234`; uncommitted changes in 2 tracked files at generation: "
        "src/a.py, tests/b.py."
    )
    assert _commit_line("abc1234|src/a.py") == (
        "Commit: `abc1234`; uncommitted changes in 1 tracked file at generation: src/a.py."
    )


def _fake_git(root, status):
    def fake(args, cwd):
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return str(root) + "\n"
        if args[:2] == ["rev-parse", "--short"]:
            return "abc1234\n"
        assert args[0] == "status"
        return status

    return fake


def test_commit_ignores_the_report_being_written(tmp_path, monkeypatch):
    monkeypatch.setattr(validate_mod, "_git", _fake_git(tmp_path, " M docs/reports/x.md\n"))
    assert _commit(report_path=tmp_path / "docs/reports/x.md") == "abc1234"
    status = " M docs/reports/x.md\n M src/b.py\nR  old.py -> src/a.py\n"
    monkeypatch.setattr(validate_mod, "_git", _fake_git(tmp_path, status))
    assert _commit(report_path=tmp_path / "docs/reports/x.md") == "abc1234|src/a.py,src/b.py"


def test_commit_line_outside_a_checkout_says_unknown(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(validate_mod.subprocess, "run", boom)
    assert _commit() == "unknown" and _commit_line("unknown") == "Commit: `unknown`."


def test_commit_ignores_a_relative_report_path_when_cwd_is_not_the_root(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    monkeypatch.chdir(tmp_path / "docs")
    monkeypatch.setattr(validate_mod, "_git", _fake_git(tmp_path, " M docs/x.md\n"))
    assert _commit(report_path=Path("x.md")) == "abc1234"
