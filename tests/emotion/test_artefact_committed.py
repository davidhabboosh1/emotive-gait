"""The committed learned-model artefact: provenance, grounding invariants and cited directions."""

from pathlib import Path

import pytest
from directions import check_directions

from emotive_gait.emotion.learned import LearnedEmotionModel, default_artefact_path, load_artefact
from emotive_gait.learn.manifest import sha256_of

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "manifest.yaml"
REPORT = ROOT / "docs" / "reports" / "phase-4-5-fit-report.md"


def test_committed_artefact_loads_and_matches_the_manifest():
    art = load_artefact(default_artefact_path())
    assert art.provenance["manifest_sha256"] == sha256_of(MANIFEST)
    assert all(ds["committable"] for ds in art.provenance["datasets_included"])


def test_every_cell_has_a_grounding_and_fitted_cells_have_intervals():
    for cell in load_artefact(default_artefact_path()).cells.values():
        assert cell.grounding in ("fitted", "thin", "missing")
        assert (cell.ci_low is not None) == (cell.grounding == "fitted")
        if cell.grounding != "missing":
            # fit_error means no result at all; a cell with an estimate never carries it
            assert cell.fallback_reason != "fit_error"


def test_committed_direction_checks_are_consistent_with_evidence():
    # the parametrised direction test of Task 7 now also runs on LearnedEmotionModel.default()
    check_directions(LearnedEmotionModel.default())
    check_directions(LearnedEmotionModel.default(scaling="ratio"))


def _documented_feasibility(scaling: str) -> dict[str, tuple[bool, float]]:
    """``{preset: (feasible, max foot position error in mm)}`` from the committed report."""
    from emotive_gait.learn.report import FEASIBILITY_HEADING

    text = REPORT.read_text(encoding="utf-8")
    assert FEASIBILITY_HEADING in text, "the fit report has no learned-preset feasibility section"
    section = text.split(FEASIBILITY_HEADING, 1)[1].split("\n## ", 1)[0]
    rows = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and cells[1] == scaling:
            rows[cells[0]] = (cells[2] == "yes", float(cells[3]))
    return rows


@pytest.mark.parametrize("scaling", ["froude", "ratio"])
def test_the_reports_feasibility_section_matches_the_committed_artefact(scaling):
    # Learned presets are flagged and rendered, not gated (spec 4.5, decision 2026-10-26); this
    # pins the documented residuals of both scalings to what the IK computes on the committed
    # artefact.
    from emotive_gait.emotion.model import learned_presets
    from emotive_gait.gait.generator import generate
    from emotive_gait.robot.loader import load_robot

    robot = load_robot()
    documented = _documented_feasibility(scaling)
    computed = {
        name: generate(p, robot).feasibility
        for name, p in learned_presets(LearnedEmotionModel.default(scaling=scaling)).items()
    }
    assert set(documented) == set(computed)
    infeasible = {name for name, rep in computed.items() if not rep.feasible}
    assert {name for name, (ok, _) in documented.items() if not ok} == infeasible
    for name, rep in computed.items():
        assert abs(documented[name][1] - rep.max_foot_position_error_m * 1000.0) <= 0.1, name
