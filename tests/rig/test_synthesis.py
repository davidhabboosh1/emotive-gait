"""End-to-end synthesis on a rig: load, modulate, generate, write back, read back, measure.

Module-scoped: one walk file per spec (``loaded``) and one ``LoadedRig`` per spec (``rigs``),
each built once; every ``walk`` generates once, writes the output and reads it back for the two
claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import pytest

from emotive_gait.emotion.evidence import Emotion
from emotive_gait.emotion.learned import LearnedEmotionModel, load_artefact
from emotive_gait.emotion.model import ANALYTIC, PRESET_PARAMS, modulate
from emotive_gait.gait.footsteps import rule_double_support
from emotive_gait.gait.generator import generate
from emotive_gait.gait.params import GaitParameters
from emotive_gait.gait.scaling import LENGTH_FIELDS, G, to_gait_parameters
from emotive_gait.mocap.extract import gait_timing
from emotive_gait.mocap.readers.bvh import read_bvh_file
from emotive_gait.mocap.writers.bvh import write_bvh
from emotive_gait.rig.errors import RigError
from emotive_gait.rig.scaled import RigScaledModel
from emotive_gait.rig.synthesis import (
    LoadedRig,
    emotional_walk,
    invert_double_support,
    load_rig,
    measure_slip,
    rendered,
    resynthesise,
    round_trip,
    walk,
    within_tolerance,
)
from emotive_gait.rig.validate import PITCH_INNERMOST_EXCLUDED, pitch_innermost_exclusion
from emotive_gait.robot.profiles.g1 import G1_PROFILE_WITH_NECK
from emotive_gait.robot.roles import JointRole
from rig.driven import spec_id
from rig.rigs import (
    FAMILIES,
    FAMILY_VARIANTS,
    RigSpec,
    fold_into_channels,
    standing_file,
    walk_file,
)
from rig.test_mapping import drop_bones

FIX = Path(__file__).resolve().parents[1] / "mocap" / "fixtures"
ZERO: dict = {}
PRESETS = {
    "sad": {Emotion.SADNESS: 1.0},
    "angry": {Emotion.ANGER: 1.0},
    "happy": {Emotion.HAPPINESS: 1.0},
    "fear": {Emotion.FEAR: 1.0},
}
PRESET_RIGS = [RigSpec(), replace(RigSpec(), up_axis="z"), replace(RigSpec(), toes=False)]


@dataclass(frozen=True)
class Walk:
    path: Path


class Walks(dict):
    """``spec -> Walk``, each walk file written on first use in its own directory."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root

    def __missing__(self, spec: RigSpec) -> Walk:
        directory = self.root / spec_id(spec)
        directory.mkdir(parents=True, exist_ok=True)
        # A Froude-similar source: the step scales with the leg (0.5 m on the 0.9 m base leg),
        # so the 0.6 m-leg variant does not walk a 0.83-leg-length stride.
        step = 0.5 * spec.leg_length_m / RigSpec().leg_length_m
        walk = Walk(walk_file(spec, directory, step_length=step))
        self[spec] = walk
        return walk


class Rigs(dict):
    """``spec -> LoadedRig`` on ``Walks``' files, loaded on first use."""

    def __init__(self, walks: Walks) -> None:
        super().__init__()
        self.walks = walks

    def __missing__(self, spec: RigSpec) -> LoadedRig:
        rig = load_rig(self.walks[spec].path, scale=None, up_axis=None, mapping_overrides=None)
        self[spec] = rig
        return rig


@pytest.fixture(scope="module")
def loaded(tmp_path_factory) -> Walks:
    return Walks(tmp_path_factory.mktemp("synthesis"))


@pytest.fixture(scope="module")
def rigs(loaded) -> Rigs:
    return Rigs(loaded)


@pytest.fixture(scope="module")
def style100() -> LoadedRig:
    path = FIX / "style100_neutral_4s.bvh"
    return load_rig(path, scale=0.01, up_axis="y", mapping_overrides=None)


@pytest.fixture(scope="module")
def fixture_artefact_path(tmp_path_factory) -> Path:
    from artefacts import FULL_AGREEING, make_artefact

    from emotive_gait.learn.artefact import write_artefact
    from emotive_gait.learn.fit import Cell

    source = make_artefact(FULL_AGREEING, thin=[("anger", "cadence")])
    cells: dict = {"froude": {}, "ratio": {}}
    for (scaling, _, parameter), c in source.cells.items():
        cells[scaling].setdefault(parameter, []).append(
            Cell(**{f.name: getattr(c, f.name) for f in fields(Cell)})
        )
    path = tmp_path_factory.mktemp("artefact") / "coefficients.yaml"
    write_artefact(path, cells, provenance={"source": "test"})
    return path


def _walk(rig: LoadedRig, intensities, *, model=ANALYTIC, model_name="analytic", **kw):
    return walk(
        rig,
        intensities=intensities,
        intensities_source="flags",
        model=model,
        model_name=model_name,
        **kw,
    )


def _failing(r) -> dict[str, float]:
    ok = within_tolerance(r.round_trip)
    return {k: r.round_trip.residuals[k] for k, v in ok.items() if not v}


@pytest.mark.parametrize("spec", FAMILY_VARIANTS, ids=spec_id)
def test_zero_emotion_round_trip_recovers_p0_on_every_rig(rigs, spec):
    r = _walk(rigs[spec], ZERO)
    assert all(within_tolerance(r.round_trip).values()), _failing(r)
    assert r.p1 == r.rig.p0
    assert r.feasibility.feasible
    assert r.slip.max_travel_m <= 0.002 and r.slip.max_height_error_m <= 0.002, r.slip


@pytest.mark.parametrize("spec", PRESET_RIGS, ids=spec_id)
@pytest.mark.parametrize("preset", list(PRESETS))
def test_emotion_round_trip_recovers_p1(rigs, spec, preset):
    r = _walk(rigs[spec], PRESETS[preset])
    assert r.p1 != r.rig.p0
    ok = within_tolerance(r.round_trip)
    excluded = pitch_innermost_exclusion(r.rig.mapping, r.rig.file, r.p1)
    gated = {k: v for k, v in ok.items() if k not in excluded}
    assert all(gated.values()), {k: r.round_trip.residuals[k] for k, v in gated.items() if not v}
    if r.feasibility.feasible:
        assert r.slip.max_travel_m <= 0.002, r.slip
        assert r.slip.max_height_error_m <= 0.002, r.slip


def test_style100_fixture_round_trips_at_zero_and_sad(style100):
    heading = gait_timing(style100.clip, style100.mapping.landmarks).frame.forward[0, :2]
    heading = heading / np.linalg.norm(heading)
    pelvis = style100.clip.joint_names.index(style100.compiled.pelvis)
    for intensities in (ZERO, PRESETS["sad"]):
        r = _walk(style100, intensities)
        assert all(within_tolerance(r.round_trip).values()), _failing(r)
        assert r.slip.max_travel_m <= 0.002, r.slip
        # The output walks straight along the source's first-frame heading.
        track = r.output.to_clip().positions[:, pelvis, :2]
        travel = track[-1] - track[0]
        assert np.linalg.norm(travel / np.linalg.norm(travel) - heading) < 0.05


def test_emotional_walk_is_load_rig_then_walk(loaded, rigs):
    r = emotional_walk(
        loaded[RigSpec()].path,
        intensities=ZERO,
        intensities_source="flags",
        model=ANALYTIC,
        model_name="analytic",
        duration_s=2.0,
    )
    same = _walk(rigs[RigSpec()], ZERO, duration_s=2.0)
    assert np.array_equal(r.output.motion, same.output.motion) and r.p1 == same.p1


def test_output_keeps_the_source_hierarchy_channels_and_fps(rigs):
    r = _walk(rigs[RigSpec()], ZERO)
    assert [j.name for j in r.output.joints] == [j.name for j in r.rig.file.joints]
    assert r.output.frame_time == r.rig.file.frame_time
    assert all(a.channels == b.channels for a, b in zip(r.output.joints, r.rig.file.joints))
    assert r.output.scale == r.rig.file.scale and r.output.up_axis == r.rig.file.up_axis
    # The default duration is the source's length.
    assert r.output.motion.shape[0] == r.rig.file.motion.shape[0]


def test_output_starts_at_the_source_pelvis_position(rigs):
    r = _walk(rigs[replace(RigSpec(), up_axis="z")], ZERO, duration_s=1.0)
    pelvis = r.rig.clip.joint_names.index(FAMILIES["mixamo"]["pelvis"])
    source = r.rig.clip.positions[0, pelvis]
    out = r.output.to_clip().positions[0, pelvis]
    assert math.dist(source[:2], out[:2]) < 1e-6


def test_root_above_pelvis_output_puts_the_pelvis_where_commanded(rigs):
    r = _walk(rigs[replace(RigSpec(), root_above_pelvis=True)], ZERO, duration_s=1.0)
    clip = r.output.to_clip()
    pelvis = clip.joint_names.index(FAMILIES["mixamo"]["pelvis"])
    # Phase 0: no sway, full bounce; the nominal is the lowered one (load_rig).
    nominal = r.rig.nominal_pelvis_height * r.p1.com_height_scale + r.p1.com_bounce_amp
    assert abs(clip.positions[0, pelvis, 2] - (r.rig.reference.floor_z + nominal)) < 1e-4
    assert abs(r.rig.nominal_pelvis_height - r.rig.source_pelvis_height) < 0.25


def test_learned_model_goes_through_the_rig_profile(rigs, fixture_artefact_path):
    m = LearnedEmotionModel(load_artefact(fixture_artefact_path), scaling="froude")
    r = _walk(rigs[RigSpec()], PRESETS["sad"], model=m, model_name="learned", duration_s=2.0)
    assert r.p1.cadence < r.rig.p0.cadence
    assert r.model_name == "learned" and r.scaling == "froude"


def test_ratio_scaling_is_relative_to_the_rigs_own_neutral(rigs, fixture_artefact_path):
    rig = rigs[RigSpec()]
    inner = LearnedEmotionModel(load_artefact(fixture_artefact_path), scaling="ratio")
    cell_estimate = inner.cell(Emotion.SADNESS, "step_length").estimate
    assert cell_estimate != 0.0
    wrapped = RigScaledModel(inner, scaling="ratio", p0=rig.p0)
    deltas = wrapped.deltas({Emotion.SADNESS: 1.0}, rig.profile)
    assert deltas["step_length"] == pytest.approx(cell_estimate * rig.p0.step_length, rel=1e-6)


def test_the_wrapper_is_bit_identical_on_the_g1_for_the_analytic_model():
    for intensities in ({Emotion.SADNESS: 1.0}, {Emotion.ANGER: 0.7, Emotion.FEAR: 0.4}):
        for scaling in ("froude", "ratio"):
            wrapped = RigScaledModel(ANALYTIC, scaling=scaling, p0=GaitParameters())
            inner = ANALYTIC.deltas(intensities, G1_PROFILE_WITH_NECK)
            assert dict(wrapped.deltas(intensities, G1_PROFILE_WITH_NECK)) == dict(inner)
    presets = {
        "neutral": {},
        "sad": {Emotion.SADNESS: 1.0},
        "angry": {Emotion.ANGER: 1.0},
        "happy": {Emotion.HAPPINESS: 1.0},
    }
    assert set(presets) == set(PRESET_PARAMS)
    for name, intensities in presets.items():
        wrapped = RigScaledModel(ANALYTIC, scaling="froude", p0=GaitParameters())
        p = modulate(GaitParameters(), intensities, G1_PROFILE_WITH_NECK, model=wrapped)
        assert p == PRESET_PARAMS[name]


def test_the_froude_wrapper_re_dimensions_g1_deltas_through_the_rig_leg(rigs):
    rig = rigs[RigSpec()]
    intensities = {Emotion.ANGER: 1.0}
    inner = ANALYTIC.deltas(intensities, rig.profile)
    wrapped = RigScaledModel(ANALYTIC, scaling="froude", p0=rig.p0).deltas(intensities, rig.profile)
    l, h = rig.profile.leg_length, rig.profile.nominal_pelvis_height
    dimensionless = {}
    for name, d in inner.items():
        if name in LENGTH_FIELDS:
            dimensionless[name] = d / 0.69
        elif name == "cadence":
            dimensionless[name] = d * math.sqrt(0.69 / G)
        elif name == "com_height_scale":
            dimensionless[name] = d * 0.68 / 0.69
        elif name != "expansiveness":
            dimensionless[name] = d
    base = GaitParameters()
    moved = to_gait_parameters(
        dimensionless, base, scaling="froude", leg_length=l, nominal_pelvis_height=h
    )
    for name in dimensionless:
        expected = getattr(moved, name) - getattr(base, name)
        assert wrapped[name] == pytest.approx(expected, rel=1e-9, abs=1e-12), name
    assert wrapped["expansiveness"] == inner["expansiveness"]
    assert RigScaledModel(ANALYTIC, scaling="froude", p0=rig.p0).profile_aware is True


def test_unreachable_stride_is_flagged_and_still_written(rigs):
    rig = rigs[RigSpec()]
    long = replace(rig.p0, step_length=1.2)  # 1.2 m steps on 0.9 m legs
    out, traj = resynthesise(rig, long, duration_s=2.0)
    assert not traj.feasibility.feasible and out.motion.shape[0] == 120
    slip = measure_slip(out, rig.mapping.landmarks, long, rig.reference.floor_z)
    assert slip.stances > 0 and math.isfinite(slip.max_travel_m)  # reported, never raised


def test_invert_double_support_is_the_inverse_of_the_rule():
    p = GaitParameters()
    assert invert_double_support(rule_double_support(p), p) == pytest.approx(0.20, abs=1e-3)


def test_p0_regenerates_the_sources_raw_double_support_reading(rigs):
    rig = rigs[RigSpec()]
    assert rig.p0.com_height_scale == 1.0 and rig.p0.expansiveness == 1.0
    assert rig.p0.cadence == pytest.approx(rig.summary.mean.cadence)
    assert rig.unobservable == ()
    if 0.0 < rig.p0.double_support_frac < 0.5:
        assert rule_double_support(rig.p0) == pytest.approx(rig.raw_double_support, abs=2e-3)


def test_a_standing_clip_stops_with_the_discard_reasons(tmp_path):
    with pytest.raises(RigError, match="no usable cycle"):
        load_rig(
            standing_file(RigSpec(), tmp_path), scale=None, up_axis=None, mapping_overrides=None
        )


def test_mapping_overrides_split_into_landmarks_and_roles(loaded, rigs):
    path = loaded[RigSpec()].path
    with pytest.raises(RigError, match="NoSuchBone"):
        load_rig(path, scale=None, up_axis=None, mapping_overrides={"head": "NoSuchBone"})
    with pytest.raises(RigError, match="no_such_key"):
        load_rig(path, scale=None, up_axis=None, mapping_overrides={"no_such_key": "x"})
    # "knee_l" is both a landmark and a role: the "role:" prefix makes it the role override.
    hinge = rigs[RigSpec()].mapping.roles[JointRole.KNEE_L].hinge
    again = load_rig(path, scale=None, up_axis=None, mapping_overrides={"role:knee_l": hinge})
    assert any(d.what == "role:knee_l" and d.source == "user" for d in again.mapping.decisions)


def test_an_unknown_intensities_source_is_refused(rigs):
    with pytest.raises(ValueError, match="intensities_source"):
        walk(
            rigs[RigSpec()],
            intensities=ZERO,
            intensities_source="telepathy",
            model=ANALYTIC,
            model_name="analytic",
        )


def test_the_nominal_pelvis_height_is_lowered_until_p0_is_reachable(rigs):
    rig = rigs[RigSpec()]
    assert rig.source_pelvis_height == rig.reference.pelvis_height
    assert rig.nominal_pelvis_height == rig.profile.nominal_pelvis_height
    factor = rig.nominal_pelvis_height / rig.source_pelvis_height
    assert 0.75 <= factor < 1.0  # the walker's pelvis sits at the straight-leg height
    assert generate(rig.p0, rig.robot, welded=rig.robot).feasibility.feasible
    higher = replace(rig.profile, nominal_pelvis_height=(factor + 0.01) * rig.source_pelvis_height)
    robot = replace(rig.robot, profile=higher)
    assert not generate(rig.p0, robot, welded=robot).feasibility.feasible
    decision = next(d for d in rig.mapping.decisions if d.what == "pelvis_height")
    assert decision.source == "geometry" and decision.chosen.startswith(f"{factor:.2f} x")


def test_the_pitch_innermost_exclusion_names_only_that_rig_and_large_abduction(rigs):
    z_up, y_up = rigs[replace(RigSpec(), up_axis="z")], rigs[RigSpec()]
    wide = replace(z_up.p0, arm_abduction=0.45, expansiveness=1.35)  # angry: 0.61 rad
    narrow = replace(z_up.p0, arm_abduction=0.45, expansiveness=1.0)
    assert pitch_innermost_exclusion(z_up.mapping, z_up.file, wide) == PITCH_INNERMOST_EXCLUDED
    assert pitch_innermost_exclusion(z_up.mapping, z_up.file, narrow) == ()
    assert pitch_innermost_exclusion(y_up.mapping, y_up.file, wide) == ()
    # Either side alone triggers it.
    sides = {JointRole.SHOULDER_PITCH_L, JointRole.SHOULDER_PITCH_R}
    for kept in sides:
        roles = {r: b for r, b in z_up.mapping.roles.items() if r not in sides - {kept}}
        one_sided = replace(z_up.mapping, roles=roles)
        assert pitch_innermost_exclusion(one_sided, z_up.file, wide) == PITCH_INNERMOST_EXCLUDED


def test_a_p0_field_the_extractor_cannot_read_is_an_unobservable_decision(tmp_path):
    spec = replace(RigSpec(), clavicles=False)
    source = read_bvh_file(walk_file(spec, tmp_path), scale=0.01, up_axis="y")
    names = FAMILIES["mixamo"]
    arms = tuple(
        names[part].replace("Left", side)
        for side in ("Left", "Right")
        for part in ("upper_arm", "elbow", "wrist", "finger")
    )
    path = tmp_path / "armless.bvh"
    write_bvh(drop_bones(source, arms), path)
    rig = load_rig(path, scale=None, up_axis=None, mapping_overrides=None)
    arm_fields = ("arm_swing_amp", "arm_abduction", "elbow_flexion", "elbow_swing_ratio")
    assert rig.unobservable == arm_fields
    defaults = GaitParameters()
    for name in arm_fields:
        assert getattr(rig.p0, name) == getattr(defaults, name)
        decision = next(d for d in rig.mapping.decisions if d.what == f"p0:{name}")
        assert decision.source == "geometry" and "default" in decision.chosen
    r = _walk(rig, ZERO, duration_s=4.0)
    assert set(arm_fields) <= set(r.round_trip.unobservable)
    ok = within_tolerance(r.round_trip)
    assert not set(arm_fields) & set(ok)
    assert not set(arm_fields) & set(r.round_trip.residuals)
    assert all(ok.values()), _failing(r)


def test_a_still_arm_leaves_the_elbow_ratio_unobservable_and_a_straight_elbow_renders_none(
    rigs,
):
    rig = rigs[RigSpec()]
    still = replace(rig.p0, arm_swing_amp=0.0, elbow_swing_ratio=0.5)
    out, traj = resynthesise(rig, still, duration_s=4.0)
    assert math.isnan(rendered(traj)["elbow_swing_ratio"])
    rt = round_trip(out, rig.mapping.landmarks, still, trajectory=traj)
    assert rt.unobservable == ("elbow_swing_ratio",)
    assert "elbow_swing_ratio" not in within_tolerance(rt)
    assert "elbow_swing_ratio" not in rt.residuals
    # Commanding less than straight holds the elbow at its straight limit: no swing renders,
    # and the flexion is compared floored at 0.
    bent_back = replace(rig.p0, elbow_flexion=-0.3, elbow_swing_ratio=0.4)
    out, traj = resynthesise(rig, bent_back, duration_s=4.0)
    assert abs(rendered(traj)["elbow_swing_ratio"]) < 1e-6
    rt = round_trip(out, rig.mapping.landmarks, bent_back, trajectory=traj)
    ok = within_tolerance(rt)
    assert ok["elbow_flexion"] and ok["elbow_swing_ratio"], rt.residuals


def test_a_rig_with_constant_bone_translations_round_trips_and_keeps_them(tmp_path):
    spec = RigSpec()
    source = walk_file(spec, tmp_path)
    file = read_bvh_file(source, scale=1.0, up_axis=spec.up_axis)
    names = FAMILIES[spec.family]
    for bone in (names["knee"], names["spine2"]):
        file = fold_into_channels(file, bone, 0.6)
    folded = tmp_path / "folded.bvh"
    write_bvh(file, folded)
    r = _walk(load_rig(folded, scale=None, up_axis=None, mapping_overrides=None), ZERO)
    assert all(within_tolerance(r.round_trip).values()), _failing(r)
    assert r.slip.max_travel_m <= 0.002, r.slip
    for bone in (names["knee"], names["spine2"]):
        columns = r.output.channel_slice(bone)
        kept = r.output.motion[:, columns][:, :3]
        assert np.allclose(kept, file.motion[0, columns][:3])  # the source's constants
