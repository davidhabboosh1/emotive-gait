import dataclasses
import math

import pytest

from emotive_gait.gait.params import GaitParameters, SpeedConstraintError, SpeedMode


def test_speed_is_derived_and_read_only():
    p = GaitParameters(cadence=2.0, step_length=0.4)
    assert p.speed == pytest.approx(0.8)
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        p.speed = 1.0


@pytest.mark.parametrize("mode", list(SpeedMode))
def test_with_speed_hits_the_target_in_every_mode(mode):
    p = GaitParameters(cadence=2.4, step_length=0.5)
    assert p.with_speed(0.8, mode).speed == pytest.approx(0.8)


def test_proportional_mode_preserves_cadence_to_stride_ratio():
    p = GaitParameters(cadence=2.4, step_length=0.5)
    q = p.with_speed(0.8, SpeedMode.PROPORTIONAL)
    assert q.cadence / q.step_length == pytest.approx(p.cadence / p.step_length)


def test_hold_modes_pin_the_named_quantity():
    p = GaitParameters(cadence=2.4, step_length=0.5)
    assert p.with_speed(0.8, SpeedMode.HOLD_CADENCE).cadence == pytest.approx(2.4)
    assert p.with_speed(0.8, SpeedMode.HOLD_STEP_LENGTH).step_length == pytest.approx(0.5)


@pytest.mark.parametrize("mode", list(SpeedMode))
@pytest.mark.parametrize(
    "cadence,step_length,target",
    [
        (0.0, 0.35, 0.8),
        (1.8, 0.0, 0.8),
        (0.0, 0.0, 0.8),
        (1.8, 0.35, 0.0),
        (1.8, 0.35, -1.0),
        (float("nan"), 0.35, 0.8),
        (float("inf"), 0.35, 0.8),
        (1.8, float("nan"), 0.8),
        (1.8, float("inf"), 0.8),
        (1.8, 0.35, float("nan")),
        (1.8, 0.35, float("inf")),
    ],
)
def test_degenerate_speed_constraints_raise_not_nan(cadence, step_length, target, mode):
    if not (math.isfinite(cadence) and math.isfinite(step_length)):
        # Non-finite fields can no longer be constructed (GaitParameters.__post_init__).
        with pytest.raises(ValueError, match="finite"):
            GaitParameters(cadence=cadence, step_length=step_length)
        return
    p = GaitParameters(cadence=cadence, step_length=step_length)
    with pytest.raises(SpeedConstraintError):
        p.with_speed(target, mode)


def test_yaml_round_trip_preserves_every_field(tmp_path):
    values = {f.name: 0.1 + 0.01 * i for i, f in enumerate(dataclasses.fields(GaitParameters))}
    p = GaitParameters(**values)
    assert all(getattr(p, k) != getattr(GaitParameters(), k) for k in values)
    f = tmp_path / "g.yaml"
    p.to_yaml(f)
    text = f.read_text()
    for name in values:
        assert name in text
    assert GaitParameters.from_yaml(f) == p


def test_string_mode_matches_enum_mode():
    p = GaitParameters(cadence=2.4, step_length=0.5)
    assert p.with_speed(0.8, "hold_cadence") == p.with_speed(0.8, SpeedMode.HOLD_CADENCE)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        GaitParameters().with_speed(0.8, "not_a_mode")


def test_from_yaml_rejects_non_mapping(tmp_path):
    f = tmp_path / "g.yaml"
    f.write_text("- cadence: 2.0\n")
    with pytest.raises(ValueError, match="expected a mapping of gait parameters, got list"):
        GaitParameters.from_yaml(f)


def test_from_yaml_rejects_unknown_keys(tmp_path):
    f = tmp_path / "g.yaml"
    f.write_text("cadence: 2.0\nnot_a_param: 1\n")
    with pytest.raises(ValueError, match="not_a_param"):
        GaitParameters.from_yaml(f)


def test_from_yaml_missing_keys_take_defaults(tmp_path):
    f = tmp_path / "g.yaml"
    f.write_text("cadence: 2.0\n")
    assert GaitParameters.from_yaml(f) == GaitParameters(cadence=2.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_fields_are_rejected(bad):
    with pytest.raises(ValueError, match="trunk_pitch"):
        GaitParameters(trunk_pitch=bad)


def test_from_yaml_rejects_nan(tmp_path):
    f = tmp_path / "g.yaml"
    f.write_text("cadence: .nan\n")
    with pytest.raises(ValueError, match="cadence"):
        GaitParameters.from_yaml(f)
