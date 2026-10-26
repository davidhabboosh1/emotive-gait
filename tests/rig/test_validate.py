"""``validate_rig`` (zero and the four presets on one rig), ``render_report`` and the spec-driven
report entry point."""

from __future__ import annotations

import math
from dataclasses import replace
from types import MappingProxyType

from emotive_gait.rig.validate import (
    MIN_STEPS,
    UNMEASURED,
    ValidationRow,
    ValidationTable,
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
