import math
import subprocess
import sys
from dataclasses import fields

import pytest
import yaml
from artefacts import make_artefact

from emotive_gait.emotion.evidence import Emotion
from emotive_gait.emotion.learned import (
    DIRECTION_CHECKS,
    ArtefactCell,
    ArtefactMissingError,
    ArtefactSchemaError,
    LearnedEmotionModel,
    default_artefact_path,
    direction_check,
    load_artefact,
)
from emotive_gait.emotion.model import (
    ANALYTIC,
    IntensityError,
    learned_presets,
    modulate,
    unexpressible,
)
from emotive_gait.gait.params import GaitParameters
from emotive_gait.gait.scaling import MEASURED_FIELDS, G
from emotive_gait.robot.profiles.g1 import G1_PROFILE, G1_PROFILE_WITH_NECK

AGREEING = {
    ("sadness", "cadence"): -0.05,
    ("sadness", "neck_pitch"): 0.3,
    ("anger", "cadence"): 0.08,
    ("anger", "elbow_flexion"): 0.4,
}


def test_learned_deltas_re_dimension_through_the_profile():
    m = LearnedEmotionModel(make_artefact(AGREEING))
    d = m.deltas({Emotion.SADNESS: 1.0}, G1_PROFILE_WITH_NECK)
    assert d["cadence"] == pytest.approx(-0.05 * math.sqrt(G / 0.69))
    assert d["neck_pitch"] == pytest.approx(0.3)
    half = m.deltas({Emotion.SADNESS: 0.5}, G1_PROFILE_WITH_NECK)
    assert half["neck_pitch"] == pytest.approx(0.15)  # linear in intensity


def test_froude_lengths_and_pelvis_height_scale_with_the_leg():
    m = LearnedEmotionModel(
        make_artefact({("fear", "step_length"): -0.1, ("fear", "com_height_scale"): -0.02})
    )
    d = m.deltas({Emotion.FEAR: 1.0}, G1_PROFILE_WITH_NECK)
    assert d["step_length"] == pytest.approx(-0.1 * 0.69)
    assert d["com_height_scale"] == pytest.approx(-0.02 * 0.69 / 0.68)


def test_ratio_scaling_is_relative_to_the_default_neutral():
    art = make_artefact({("sadness", "cadence"): -0.1, ("sadness", "trunk_pitch"): 0.2})
    m = LearnedEmotionModel(art, scaling="ratio")
    d = m.deltas({Emotion.SADNESS: 0.5}, G1_PROFILE_WITH_NECK)
    assert d["cadence"] == pytest.approx(0.5 * GaitParameters().cadence * -0.1)
    assert d["trunk_pitch"] == pytest.approx(0.5 * 0.2)  # angles are added as-is


def test_missing_cells_and_unsourced_emotions_delegate_to_the_analytic_model():
    m = LearnedEmotionModel(make_artefact(AGREEING))  # (sadness, trunk_pitch) is missing
    both = {Emotion.SADNESS: 1.0, Emotion.DISGUST: 1.0}
    d = m.deltas(both, G1_PROFILE_WITH_NECK)
    a = ANALYTIC.deltas(both, G1_PROFILE_WITH_NECK)
    assert ("sadness", "trunk_pitch") in m.analytic_cells
    assert ("disgust", "trunk_pitch") in m.analytic_cells
    assert d["trunk_pitch"] == pytest.approx(a["trunk_pitch"])  # both shares analytic
    disgust = ANALYTIC.deltas({Emotion.DISGUST: 1.0}, G1_PROFILE_WITH_NECK)
    assert d["cadence"] == pytest.approx(-0.05 * math.sqrt(G / 0.69) + disgust["cadence"])


def test_analytic_cells_cover_unsourced_emotions_and_skip_used_cells():
    m = LearnedEmotionModel(make_artefact(AGREEING))
    cells = m.analytic_cells
    assert cells == tuple(sorted(cells))
    for e in ("disgust", "surprise"):
        assert {(e, p) for p in MEASURED_FIELDS} <= set(cells)
    assert not set(AGREEING) & set(cells)
    assert all(p != "expansiveness" for _, p in cells)
    assert len(cells) == 6 * len(MEASURED_FIELDS) - len(AGREEING)


def test_expansiveness_always_comes_from_the_fallback():
    m = LearnedEmotionModel(make_artefact(AGREEING))
    for e in Emotion:
        d = m.deltas({e: 0.6}, G1_PROFILE_WITH_NECK)
        assert (
            d["expansiveness"] == ANALYTIC.deltas({e: 0.6}, G1_PROFILE_WITH_NECK)["expansiveness"]
        )


def test_thin_cells_are_used_and_listed():
    m = LearnedEmotionModel(make_artefact(AGREEING, thin=[("anger", "cadence")]))
    assert m.thin_cells == (("anger", "cadence"),)
    assert ("anger", "cadence") not in m.analytic_cells
    d = m.deltas({Emotion.ANGER: 1.0}, G1_PROFILE_WITH_NECK)["cadence"]
    assert d > 0 and d == pytest.approx(0.08 * math.sqrt(G / 0.69))  # the measured number


def test_learned_deltas_validate_intensities():
    m = LearnedEmotionModel(make_artefact(AGREEING))
    with pytest.raises(IntensityError, match="sadness"):
        m.deltas({Emotion.SADNESS: 1.5}, G1_PROFILE_WITH_NECK)


def test_negative_amplitude_from_a_fitted_delta_floors_at_zero():  # Review Focus 5
    m = LearnedEmotionModel(make_artefact({("sadness", "arm_swing_amp"): -2.0}))
    q = modulate(GaitParameters(), {Emotion.SADNESS: 1.0}, G1_PROFILE_WITH_NECK, model=m)
    assert q.arm_swing_amp == 0.0


def test_cues_the_profile_cannot_show_are_dropped_for_the_learned_model_too():
    m = LearnedEmotionModel(make_artefact(AGREEING))
    lost = unexpressible({Emotion.SADNESS: 1.0}, G1_PROFILE, m)
    assert (Emotion.SADNESS, "neck_pitch") in lost
    q = modulate(GaitParameters(), {Emotion.SADNESS: 1.0}, G1_PROFILE, model=m)
    assert q.neck_pitch == GaitParameters().neck_pitch


def test_learned_presets_have_the_preset_keys():
    presets = learned_presets(LearnedEmotionModel(make_artefact(AGREEING)))
    assert set(presets) == {"neutral", "sad", "angry", "happy"}
    assert presets["neutral"] == GaitParameters()


def test_default_without_an_artefact_raises_a_named_error(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "emotive_gait.emotion.learned.default_artefact_path", lambda: tmp_path / "none.yaml"
    )
    with pytest.raises(ArtefactMissingError, match="emotive-gait fit"):
        LearnedEmotionModel.default()


def test_default_artefact_path_is_package_data():
    path = default_artefact_path()
    assert path.name == "learned_coefficients.yaml"
    assert path.parent.name == "data" and (path.parent / "__init__.py").is_file()


def test_direction_check_vocabulary():
    assert direction_check(0.2, (0.1, 0.3), "fitted", +1) == "agrees"
    assert direction_check(0.2, (-0.1, 0.3), "fitted", -1) == "contradicts_ci_spans_zero"
    assert direction_check(0.2, (-0.1, 0.3), "fitted", 0) == "agrees_ci_spans_zero"
    assert direction_check(0.2, (0.1, 0.3), "fitted", 0) == "contradicts"
    assert direction_check(0.2, None, "thin", 0) == "thin_unchecked"
    assert direction_check(0.2, None, "thin", +1) == "agrees"  # sign only
    assert direction_check(0.2, (0.1, 0.3), "fitted", None) == "no_claim"


def test_direction_check_remaining_verdicts():
    assert direction_check(0.2, (-0.1, 0.3), "fitted", +1) == "agrees_ci_spans_zero"
    assert direction_check(0.2, (0.1, 0.3), "fitted", -1) == "contradicts"
    assert direction_check(0.0, None, "missing", +1) == "no_estimate"
    assert direction_check(0.2, None, "thin", -1) == "contradicts"
    verdicts = {
        direction_check(est, ci, g, d)
        for est in (-0.2, 0.2)
        for ci in (None, (-0.1, 0.3), (0.1, 0.3), (-0.3, -0.1))
        for g in ("fitted", "thin", "missing")
        for d in (None, -1, 0, 1)
    }
    assert verdicts == set(DIRECTION_CHECKS)


def test_artefact_cell_fields_match_the_fit_cell():
    pytest.importorskip("pandas")
    from emotive_gait.learn.fit import Cell

    names = {f.name for f in fields(ArtefactCell)}
    assert names == {f.name for f in fields(Cell)} | {"direction_check"}


def _cell_yaml(**overrides):
    cell = {
        "estimate": -0.05,
        "ci_low": -0.06,
        "ci_high": -0.04,
        "between_actor_sd": 0.1,
        "residual_sd": None,
        "n_actors": 4,
        "n_cycles": 40,
        "n_datasets": 2,
        "n_mapped_actors": 1,
        "grounding": "fitted",
        "converged_maximal": False,
        "fallback_reason": "singular_cov_re",
        "per_dataset": {"ds_a": -0.04, "ds_b": None},
        "direction_check": "agrees",
    }
    return cell | overrides


def _artefact_yaml(cell=None, **top):
    data = {
        "schema_version": 1,
        "provenance": {"git": "abc123"},
        "contradictions": [
            {
                "emotion": "sadness",
                "parameter": "arm_swing_amp",
                "scaling": "froude",
                "estimate": 0.3,
                "ci_low": None,
                "ci_high": None,
                "citation": "Montepare 1987",
            }
        ],
        "cells": {
            "froude": {"sadness": {"cadence": _cell_yaml() if cell is None else cell}},
            "ratio": {"sadness": {"cadence": _cell_yaml()}},
        },
    }
    return data | top


def _write(tmp_path, data):
    path = tmp_path / "coef.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
    return path


def test_load_artefact_reads_the_schema(tmp_path):
    art = load_artefact(_write(tmp_path, _artefact_yaml()))
    cell = art.cells[("froude", "sadness", "cadence")]
    assert (cell.emotion, cell.parameter, cell.estimate) == ("sadness", "cadence", -0.05)
    assert (cell.ci_low, cell.ci_high) == (-0.06, -0.04)
    assert math.isnan(cell.residual_sd) and cell.between_actor_sd == 0.1
    assert cell.per_dataset["ds_a"] == -0.04 and math.isnan(cell.per_dataset["ds_b"])
    assert cell.fallback_reason == "singular_cov_re" and cell.converged_maximal is False
    assert set(art.cells) == {("froude", "sadness", "cadence"), ("ratio", "sadness", "cadence")}
    assert art.contradictions[0]["parameter"] == "arm_swing_amp"
    assert art.provenance == {"git": "abc123"}
    m = LearnedEmotionModel(art)
    assert m.deltas({Emotion.SADNESS: 1.0}, G1_PROFILE_WITH_NECK)["cadence"] < 0


def test_null_interval_reads_back_as_none(tmp_path):
    cell = _cell_yaml(ci_low=None, ci_high=None, grounding="thin", direction_check="agrees")
    art = load_artefact(_write(tmp_path, _artefact_yaml(cell)))
    assert art.cells[("froude", "sadness", "cadence")].ci_low is None


def test_missing_file_raises_artefact_missing_error(tmp_path):
    with pytest.raises(ArtefactMissingError, match="emotive-gait fit"):
        load_artefact(tmp_path / "absent.yaml")
    assert issubclass(ArtefactMissingError, FileNotFoundError)


@pytest.mark.parametrize(
    ("data", "key"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"cells": None}, "cells"),
        ({"provenance": []}, "provenance"),
        ({"contradictions": [{"emotion": "sadness"}]}, "contradictions[0].parameter"),
    ],
)
def test_schema_problems_name_the_key(tmp_path, data, key):
    with pytest.raises(ValueError, match=key.replace("[", r"\[").replace("]", r"\]")):
        load_artefact(_write(tmp_path, _artefact_yaml(**data)))


def test_invalid_yaml_is_a_schema_error(tmp_path):
    path = tmp_path / "coef.yaml"
    path.write_text("cells: [unclosed", encoding="utf-8")
    with pytest.raises(ArtefactSchemaError, match="not valid YAML"):
        load_artefact(path)
    assert issubclass(ArtefactSchemaError, ValueError)


def test_a_missing_top_level_key_is_named(tmp_path):
    data = _artefact_yaml()
    del data["contradictions"]
    with pytest.raises(ValueError, match="contradictions"):
        load_artefact(_write(tmp_path, data))


@pytest.mark.parametrize(
    ("cell", "key"),
    [
        ({k: v for k, v in _cell_yaml().items() if k != "n_actors"}, "n_actors"),
        (_cell_yaml(grounding="solid"), "grounding"),
        (_cell_yaml(direction_check="maybe"), "direction_check"),
        (_cell_yaml(estimate="big"), "estimate"),
        (_cell_yaml(extra=1), "extra"),
    ],
)
def test_cell_problems_name_the_cell_and_key(tmp_path, cell, key):
    with pytest.raises(ValueError, match=rf"cells\.froude\.sadness\.cadence\.{key}"):
        load_artefact(_write(tmp_path, _artefact_yaml(cell)))


@pytest.mark.parametrize(
    "cell",
    [
        _cell_yaml(ci_low=None),
        _cell_yaml(grounding="thin"),
        _cell_yaml(grounding="missing", ci_high=None),
    ],
    ids=["fitted_without_interval", "thin_with_interval", "missing_with_half_interval"],
)
def test_interval_must_match_the_grounding(tmp_path, cell):
    with pytest.raises(ArtefactSchemaError, match=r"cells\.froude\.sadness\.cadence"):
        load_artefact(_write(tmp_path, _artefact_yaml(cell)))


def test_unknown_parameter_is_rejected(tmp_path):
    data = _artefact_yaml()
    data["cells"]["froude"]["sadness"]["expansiveness"] = _cell_yaml()
    with pytest.raises(ValueError, match="cells.froude.sadness.expansiveness"):
        load_artefact(_write(tmp_path, data))


def test_learned_module_needs_no_fit_dependency():  # Review Focus 4
    code = (
        "import sys; import emotive_gait.emotion.learned; "
        "bad = sorted(m for m in ('statsmodels', 'pandas', 'emotive_gait.learn', "
        "'emotive_gait.mocap', 'emotive_gait.sim') if m in sys.modules); "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
