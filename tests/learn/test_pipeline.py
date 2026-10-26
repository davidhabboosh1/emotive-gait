from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
import yaml
from artefacts import FULL_AGREEING, make_artefact

from emotive_gait.emotion.learned import default_artefact_path, load_artefact
from emotive_gait.gait.scaling import MEASURED_FIELDS
from emotive_gait.learn.artefact import first_difference, write_artefact
from emotive_gait.learn.fit import Cell
from emotive_gait.learn.pipeline import FitInputError, run_fit


def test_run_fit_writes_a_loadable_artefact_and_a_report(fitted_toy):
    summary, tmp = fitted_toy
    art = load_artefact(tmp / "coef.yaml")
    assert ("froude", "anger", "cadence") in art.cells and (
        "ratio",
        "anger",
        "cadence",
    ) in art.cells
    assert art.provenance["manifest_sha256"] == summary.manifest_sha256
    assert art.provenance["seed"] == 0
    text = (tmp / "report.md").read_text(encoding="utf-8")
    assert "E_fit" in text and "E_zero" in text and "five group means" in text
    assert (tmp / "derived" / "cycles.csv").is_file()


def test_the_provenance_block_has_every_documented_key(fitted_toy):
    _, tmp = fitted_toy
    provenance = load_artefact(tmp / "coef.yaml").provenance
    keys = {
        "fit_date",
        "git_commit",
        "statsmodels_version",
        "numpy_version",
        "pandas_version",
        "manifest_sha256",
        "datasets_included",
        "datasets_excluded",
        "actors_per_emotion",
        "cycles_per_emotion",
        "model_formula",
        "fallbacks",
        "seed",
        "hand_tuned",
        "event_rule",
        "schema_version",
    }
    assert keys <= set(provenance)
    included = provenance["datasets_included"]
    assert [d["id"] for d in included] == ["toy"]
    assert {
        "id",
        "name",
        "committable",
        "permission",
        "licence",
        "n_actors",
        "n_clips",
        "n_cycles",
    } <= set(included[0])
    assert provenance["datasets_excluded"] == [
        {
            "id": "toy_uncommittable",
            "reason": "not committable",
            "committable": False,
            "permission": "",
        }
    ]
    assert provenance["actors_per_emotion"]["anger"] == 3
    assert set(provenance["hand_tuned"]) == {"froude", "ratio"}


def test_cells_follow_the_grounding_of_the_data(fitted_toy):
    _, tmp = fitted_toy
    art = load_artefact(tmp / "coef.yaml")
    assert art.cells[("froude", "anger", "cadence")].grounding == "fitted"
    assert art.cells[("froude", "anger", "cadence")].ci_low is not None
    assert art.cells[("froude", "sadness", "cadence")].grounding == "thin"
    fear = art.cells[("froude", "fear", "cadence")]  # emotion absent from the table
    assert (fear.grounding, fear.estimate, fear.n_actors, fear.converged_maximal) == (
        "missing",
        0.0,
        0,
        True,
    )
    assert len({k for k in art.cells if k[0] == "froude"}) == 4 * len(MEASURED_FIELDS)


def test_the_summary_names_thin_cells_and_counts(fitted_toy):
    summary, _ = fitted_toy
    assert any(e == "sadness" and n == 1 for e, _, n in summary.thin_cells)
    assert summary.n_actors == 3 and summary.n_cycles > 0
    assert {"froude", "ratio"} == set(summary.held_out)


def test_the_report_lists_every_section_and_the_nine_limitations(fitted_toy):
    _, tmp = fitted_toy
    text = (tmp / "report.md").read_text(encoding="utf-8")
    for heading in ("Provenance", "Held-out validation", "Per-cell effects", "structure test"):
        assert heading in text
    assert "E_hand" in text and "not how badly it was done" in text
    assert (
        "Synthetic self-check" in text
        and " datasets (with six actors" in text
        and " of 10 " in text
    )
    assert "speed covariate" in text
    assert "9. **Operational definitions" in text
    assert "n actors = 1" in text  # the thin sadness cells carry a count, not an interval


def test_run_fit_refuses_uncommittable_data_into_the_package_path(
    synthetic_dataset_on_disk, tmp_path
):
    data = synthetic_dataset_on_disk
    with pytest.raises(FitInputError, match="committable"):
        run_fit(
            data.manifest,
            data.root,
            out_artefact=default_artefact_path(),
            out_report=tmp_path / "r.md",
            derived=tmp_path / "d",
            datasets=["toy_uncommittable"],
            include_uncommittable=False,
        )


def test_include_uncommittable_may_not_write_inside_src(synthetic_dataset_on_disk, tmp_path):
    data = synthetic_dataset_on_disk
    with pytest.raises(FitInputError, match="committable"):
        run_fit(
            data.manifest,
            data.root,
            out_artefact=default_artefact_path(),
            out_report=tmp_path / "r.md",
            derived=tmp_path / "d",
            datasets=None,
            include_uncommittable=True,
        )


def test_a_tampered_file_is_a_fit_input_error_naming_the_file(synthetic_dataset_on_disk, tmp_path):
    data = synthetic_dataset_on_disk
    target = data.root / "toy_uncommittable" / "b1_anger.bvh"
    original = target.read_bytes()
    target.write_text("tampered")
    try:
        with pytest.raises(FitInputError, match="b1_anger.bvh") as caught:
            run_fit(
                data.manifest,
                data.root,
                out_artefact=tmp_path / "c.yaml",
                out_report=tmp_path / "r.md",
                derived=tmp_path / "d",
                datasets=["toy_uncommittable"],
                include_uncommittable=True,
            )
    finally:
        target.write_bytes(original)
    assert "expected" in str(caught.value) and "got" in str(caught.value)
    assert not (tmp_path / "c.yaml").exists()


def test_a_missing_manifest_is_a_fit_input_error(tmp_path):
    with pytest.raises(FitInputError, match="nope.yaml"):
        run_fit(
            tmp_path / "nope.yaml",
            tmp_path,
            out_artefact=tmp_path / "c.yaml",
            out_report=tmp_path / "r.md",
            derived=tmp_path / "d",
            datasets=None,
            include_uncommittable=False,
        )


# --- the artefact writer ---------------------------------------------------------------------


def _cells_from(artefact):
    out: dict = {"froude": {}, "ratio": {}}
    for (scaling, _, parameter), c in artefact.cells.items():
        fields = {k: getattr(c, k) for k in Cell.__dataclass_fields__}
        out[scaling].setdefault(parameter, []).append(Cell(**fields))
    return out


def test_write_artefact_round_trips_and_computes_direction_checks(tmp_path):
    source = make_artefact(FULL_AGREEING, thin=[("anger", "cadence")])
    written = write_artefact(tmp_path / "a.yaml", _cells_from(source), provenance={"k": 1})
    for key, cell in source.cells.items():
        assert written.cells[key].direction_check == cell.direction_check
        assert written.cells[key].estimate == pytest.approx(cell.estimate)
    assert written.contradictions == ()
    text = (tmp_path / "a.yaml").read_text()
    assert list(yaml.safe_load(text)) == sorted(yaml.safe_load(text))  # sorted keys
    assert not re.search(r"\bnan\b", text.lower())


def test_write_artefact_reports_cell_and_speed_contradictions(tmp_path):
    effects = {**FULL_AGREEING, ("anger", "cadence"): -0.5, ("anger", "step_length"): -0.5}
    written = write_artefact(
        tmp_path / "a.yaml", _cells_from(make_artefact(effects)), provenance={}
    )
    found = {(c["scaling"], c["emotion"], c["parameter"]) for c in written.contradictions}
    assert ("froude", "anger", "cadence") in found and ("ratio", "anger", "step_length") in found
    assert ("froude", "anger", "speed") in found
    assert all(c["citation"] for c in written.contradictions)


def test_write_artefact_refuses_a_fitted_cell_without_an_interval(tmp_path):
    cells = _cells_from(make_artefact(FULL_AGREEING))
    bad = cells["froude"]["cadence"][0]
    cells["froude"]["cadence"][0] = Cell(
        **{**{k: getattr(bad, k) for k in Cell.__dataclass_fields__}, "ci_low": math.nan}
    )
    with pytest.raises(ValueError, match="froude.*cadence"):
        write_artefact(tmp_path / "a.yaml", cells, provenance={})
    assert not (tmp_path / "a.yaml").exists()


def test_first_difference_compares_missing_cells_too(tmp_path):
    cells = _cells_from(make_artefact(FULL_AGREEING))
    a = write_artefact(tmp_path / "a.yaml", cells, provenance={})
    assert first_difference(a, a) is None
    shifted = _cells_from(make_artefact(FULL_AGREEING))
    miss = shifted["froude"]["step_height"][0]  # anger step_height: a measured cell
    shifted["froude"]["step_height"][0] = Cell(
        **{
            **{k: getattr(miss, k) for k in Cell.__dataclass_fields__},
            "estimate": miss.estimate + 1e-3,
        }
    )
    b = write_artefact(tmp_path / "b.yaml", shifted, provenance={})
    assert "froude" in (first_difference(a, b) or "")


def test_include_uncommittable_may_not_write_the_report_into_docs_or_src(
    synthetic_dataset_on_disk, tmp_path
):
    data = synthetic_dataset_on_disk
    docs = Path(__file__).resolve().parents[2] / "docs" / "reports" / "x.md"
    with pytest.raises(FitInputError, match=r"x\.md"):
        run_fit(
            data.manifest,
            data.root,
            out_artefact=tmp_path / "c.yaml",
            out_report=docs,
            derived=tmp_path / "d",
            datasets=["toy_uncommittable"],
            include_uncommittable=True,
        )
    assert not docs.exists()


def test_write_artefact_refuses_an_infinite_estimate(tmp_path):
    cells = _cells_from(make_artefact(FULL_AGREEING))
    bad = cells["ratio"]["cadence"][0]
    fields = {k: getattr(bad, k) for k in Cell.__dataclass_fields__}
    cells["ratio"]["cadence"][0] = Cell(**{**fields, "estimate": math.inf})
    with pytest.raises(ValueError, match="ratio.*cadence"):
        write_artefact(tmp_path / "a.yaml", cells, provenance={})


def test_the_coverage_limitation_states_the_fits_actor_counts():
    from emotive_gait.learn.report import limitations

    thin = " ".join(limitations({"actors_per_emotion": {"fear": 1, "anger": 2}}))
    assert "**Fear is thin** (1 actor in this fit)" in thin
    assert "**anger is thin** (2 actors in this fit)" in thin
    fitted = " ".join(limitations({"actors_per_emotion": {"fear": 7, "anger": 27}}))
    assert "**Fear rests on 7 actors**" in fitted and "**anger rests on 27 actors**" in fitted


def test_append_feasibility_adds_one_row_per_preset_and_names_the_driving_cell(tmp_path):
    from emotive_gait.learn.report import FEASIBILITY_HEADING, append_feasibility

    report = tmp_path / "r.md"
    report.write_text("# Report\n", encoding="utf-8")
    stride = {"step_length": 0.5, "cadence": 2.7, "double_support_frac": 0.1}
    rows = [
        {
            "preset": "neutral",
            "scaling": "froude",
            "feasible": True,
            "position_error_mm": 0.034,
            "orientation_error_rad": 0.0,
            "roles_at_limit": ["ankle_pitch_l"],
            "stride": stride,
            "driving": [],
        },
        {
            "preset": "angry",
            "scaling": "froude",
            "feasible": False,
            "position_error_mm": 6.369,
            "orientation_error_rad": 0.0074,
            "roles_at_limit": ["ankle_pitch_l", "ankle_pitch_r"],
            "stride": stride,
            "driving": [
                {
                    "parameter": "com_height_scale",
                    "value": 0.9795,
                    "estimate": -0.0202,
                    "per_dataset": {"100style": -0.0155},
                }
            ],
        },
    ]
    append_feasibility(report, rows)
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# Report\n") and FEASIBILITY_HEADING in text
    assert "| neutral | froude | yes | 0.034 |" in text
    assert "| angry | froude | no | 6.369 |" in text
    assert "`com_height_scale` 0.9795 (effect -0.0202; 100style -0.0155)" in text
    assert "flagged and rendered" in text
