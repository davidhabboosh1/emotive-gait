import math

import pytest

from emotive_gait.gait.params import GaitParameters
from emotive_gait.gait.scaling import LENGTH_FIELDS, MEASURED_FIELDS, G, to_gait_parameters


def test_measured_fields_are_every_parameter_except_expansiveness():
    assert len(MEASURED_FIELDS) == 14 and "expansiveness" not in MEASURED_FIELDS
    assert set(LENGTH_FIELDS) <= set(MEASURED_FIELDS)


def test_froude_re_dimensioning_uses_leg_length_and_nominal_height():
    base = GaitParameters()
    d = {"step_length": 0.1, "cadence": 0.05, "trunk_pitch": 0.2, "com_height_scale": -0.03}
    p = to_gait_parameters(d, base, scaling="froude", leg_length=0.69, nominal_pelvis_height=0.68)
    assert p.step_length == pytest.approx(0.35 + 0.1 * 0.69)
    assert p.cadence == pytest.approx(1.8 + 0.05 * math.sqrt(G / 0.69))
    assert p.trunk_pitch == pytest.approx(0.2)
    assert p.com_height_scale == pytest.approx(1.0 - 0.03 * 0.69 / 0.68)


def test_ratio_re_dimensioning_multiplies_the_base():
    p = to_gait_parameters(
        {"step_length": -0.2, "cadence": 0.1, "neck_pitch": 0.3},
        GaitParameters(),
        scaling="ratio",
        leg_length=0.69,
        nominal_pelvis_height=0.68,
    )
    assert p.step_length == pytest.approx(0.35 * 0.8)
    assert p.cadence == pytest.approx(1.8 * 1.1)
    assert p.neck_pitch == pytest.approx(0.3)


def test_unknown_delta_names_are_refused():
    with pytest.raises(ValueError, match="expansiveness"):
        to_gait_parameters(
            {"expansiveness": 0.1},
            GaitParameters(),
            scaling="froude",
            leg_length=0.69,
            nominal_pelvis_height=0.68,
        )
