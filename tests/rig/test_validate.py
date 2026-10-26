"""``validate_rig`` (zero and the four presets on one rig) and ``render_report``."""

from __future__ import annotations

from emotive_gait.rig.validate import render_report, validate_rig
from rig.rigs import RigSpec, walk_file


def test_validate_rig_runs_zero_and_four_presets(tmp_path):
    t = validate_rig(
        walk_file(RigSpec(), tmp_path), scale=None, up_axis=None, mapping_overrides=None
    )
    assert [r.run for r in t.rows] == ["neutral", "sad", "angry", "happy", "fear"] and t.rows[
        0
    ].within


def test_render_report_lists_not_run_rigs():
    text = render_report({}, not_run={"bandai": "archive not fetched"})
    assert "bandai" in text and "archive not fetched" in text
