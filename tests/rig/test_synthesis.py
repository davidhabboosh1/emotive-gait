"""End-to-end synthesis on a rig: load, modulate, generate, write back, read back, measure.

Module-scoped: one walk file per spec (``loaded``), written once; each ``emotional_walk`` loads
the rig, generates once, writes the output and reads it back for the two claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from pathlib import Path

import pytest

from emotive_gait.emotion.evidence import Emotion
from emotive_gait.emotion.learned import LearnedEmotionModel, load_artefact
from emotive_gait.emotion.model import ANALYTIC, PRESET_PARAMS, modulate
from emotive_gait.gait.footsteps import rule_double_support
from emotive_gait.gait.generator import generate
from emotive_gait.gait.params import GaitParameters
from emotive_gait.gait.scaling import LENGTH_FIELDS, G, to_gait_parameters
from emotive_gait.rig.errors import RigError
from emotive_gait.rig.scaled import RigScaledModel
from emotive_gait.rig.synthesis import (
    emotional_walk,
    invert_double_support,
    load_rig,
    measure_slip,
    rendered,
    resynthesise,
    round_trip,
    within_tolerance,
)
from emotive_gait.robot.profiles.g1 import G1_PROFILE_WITH_NECK
from emotive_gait.robot.roles import JointRole
from rig.driven import spec_id
from rig.rigs import FAMILIES, FAMILY_VARIANTS, RigSpec, standing_file, walk_file

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


@pytest.fixture(scope="module")
def loaded(tmp_path_factory) -> Walks:
    return Walks(tmp_path_factory.mktemp("synthesis"))


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


# Documented limitation (spec 4.7 rulings, Task 8): a driven hinge reproduces its BVH Euler
# channel, so when the shoulder PITCH channel is the innermost (last) rotation channel it turns
# about the abducted arm's own lateral axis, not the trunk's; at a large abduction the
# extractor's sagittal arm angle then over-reads the swing and the abduction reading drops
# during it (angry on the z-up ZXY rig: +0.087 rad swing, -0.031 rad abduction). The static
# posture map cannot linearise that coupling.
PITCH_INNERMOST_ABDUCTION = 0.5  # rad, commanded abduction above which the coupling shows
PITCH_INNERMOST_EXCLUDED = ("arm_swing_amp", "arm_abduction")


def _pitch_innermost_exclusion(r) -> tuple[str, ...]:
    """The fields excluded from the gate for the pitch-innermost limitation, else ``()``."""
    binding = r.rig.mapping.roles.get(JointRole.SHOULDER_PITCH_L)
    if binding is None:
        return ()
    bone, channel = binding.hinge.rsplit("/", 1)
    joint = next(j for j in r.rig.file.joints if j.name == bone)
    innermost = [c for c in joint.channels if c.endswith("rotation")][-1] == channel
    abduction = r.p1.expansiveness * r.p1.arm_abduction
    if innermost and abduction > PITCH_INNERMOST_ABDUCTION:
        return PITCH_INNERMOST_EXCLUDED
    return ()


def _failing(r) -> dict[str, float]:
    ok = within_tolerance(r.round_trip)
    return {k: r.round_trip.residuals[k] for k, v in ok.items() if not v}


@pytest.mark.parametrize("spec", FAMILY_VARIANTS, ids=spec_id)
def test_zero_emotion_round_trip_recovers_p0_on_every_rig(loaded, spec):
    r = emotional_walk(
        loaded[spec].path,
        intensities=ZERO,
        intensities_source="flags",
        model=ANALYTIC,
        model_name="analytic",
    )
    assert all(within_tolerance(r.round_trip).values()), _failing(r)
    assert r.p1 == r.rig.p0
    assert r.feasibility.feasible
    assert r.slip.max_travel_m <= 0.002 and r.slip.max_height_error_m <= 0.002, r.slip


@pytest.mark.parametrize("spec", PRESET_RIGS, ids=spec_id)
@pytest.mark.parametrize("preset", list(PRESETS))
def test_emotion_round_trip_recovers_p1(loaded, spec, preset):
    r = emotional_walk(
        loaded[spec].path,
        intensities=PRESETS[preset],
        intensities_source="flags",
        model=ANALYTIC,
        model_name="analytic",
    )
    assert r.p1 != r.rig.p0
    ok = within_tolerance(r.round_trip)
    excluded = _pitch_innermost_exclusion(r)
    gated = {k: v for k, v in ok.items() if k not in excluded}
    assert all(gated.values()), {k: r.round_trip.residuals[k] for k, v in gated.items() if not v}
    if r.feasibility.feasible:
        assert r.slip.max_travel_m <= 0.002, r.slip
        assert r.slip.max_height_error_m <= 0.002, r.slip


def test_style100_fixture_round_trips_at_zero_and_sad():
    for intensities in (ZERO, PRESETS["sad"]):
        r = emotional_walk(
            FIX / "style100_neutral_4s.bvh",
            intensities=intensities,
            intensities_source="flags",
            model=ANALYTIC,
            model_name="analytic",
            scale=0.01,
            up_axis="y",
        )
        assert all(within_tolerance(r.round_trip).values()), _failing(r)
        assert r.slip.max_travel_m <= 0.002, r.slip


def test_output_keeps_the_source_hierarchy_channels_and_fps(loaded):
    r = emotional_walk(
        loaded[RigSpec()].path,
        intensities=ZERO,
        intensities_source="flags",
        model=ANALYTIC,
        model_name="analytic",
    )
    assert [j.name for j in r.output.joints] == [j.name for j in r.rig.file.joints]
    assert r.output.frame_time == r.rig.file.frame_time
    assert all(a.channels == b.channels for a, b in zip(r.output.joints, r.rig.file.joints))
    assert r.output.scale == r.rig.file.scale and r.output.up_axis == r.rig.file.up_axis
    # The default duration is the source's length.
    assert r.output.motion.shape[0] == r.rig.file.motion.shape[0]


def test_output_starts_at_the_source_pelvis_position(loaded):
    r = emotional_walk(
        loaded[replace(RigSpec(), up_axis="z")].path,
        intensities=ZERO,
        intensities_source="flags",
        model=ANALYTIC,
        model_name="analytic",
    )
    pelvis = r.rig.clip.joint_names.index(FAMILIES["mixamo"]["pelvis"])
    source = r.rig.clip.positions[0, pelvis]
    out = r.output.to_clip().positions[0, pelvis]
    assert math.dist(source[:2], out[:2]) < 1e-6


def test_root_above_pelvis_output_puts_the_pelvis_where_commanded(loaded):
    spec = replace(RigSpec(), root_above_pelvis=True)
    r = emotional_walk(
        loaded[spec].path,
        intensities=ZERO,
        intensities_source="flags",
        model=ANALYTIC,
        model_name="analytic",
    )
    clip = r.output.to_clip()
    pelvis = clip.joint_names.index(FAMILIES["mixamo"]["pelvis"])
    # Phase 0: no sway, full bounce; the nominal is the lowered one (load_rig).
    nominal = r.rig.nominal_pelvis_height * r.p1.com_height_scale + r.p1.com_bounce_amp
    assert abs(clip.positions[0, pelvis, 2] - (r.rig.reference.floor_z + nominal)) < 1e-4
    assert abs(r.rig.nominal_pelvis_height - r.rig.source_pelvis_height) < 0.25


def test_learned_model_goes_through_the_rig_profile(loaded, fixture_artefact_path):
    m = LearnedEmotionModel(load_artefact(fixture_artefact_path), scaling="froude")
    r = emotional_walk(
        loaded[RigSpec()].path,
        intensities=PRESETS["sad"],
        intensities_source="flags",
        model=m,
        model_name="learned",
    )
    assert r.p1.cadence < r.rig.p0.cadence
    assert r.model_name == "learned" and r.scaling == "froude"


def test_ratio_scaling_is_relative_to_the_rigs_own_neutral(loaded, fixture_artefact_path):
    rig = load_rig(loaded[RigSpec()].path, scale=None, up_axis=None, mapping_overrides=None)
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
    for name, emotion in (("sad", Emotion.SADNESS), ("angry", Emotion.ANGER)):
        wrapped = RigScaledModel(ANALYTIC, scaling="froude", p0=GaitParameters())
        p = modulate(GaitParameters(), {emotion: 1.0}, G1_PROFILE_WITH_NECK, model=wrapped)
        assert p == PRESET_PARAMS[name]


def test_the_froude_wrapper_re_dimensions_g1_deltas_through_the_rig_leg(loaded):
    rig = load_rig(loaded[RigSpec()].path, scale=None, up_axis=None, mapping_overrides=None)
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
    # The profile-aware learned model passes through untouched.
    assert RigScaledModel(ANALYTIC, scaling="froude", p0=rig.p0).profile_aware is True


def test_unreachable_stride_is_flagged_and_still_written(loaded):
    rig = load_rig(loaded[RigSpec()].path, scale=None, up_axis=None, mapping_overrides=None)
    long = replace(rig.p0, step_length=1.2)  # 1.2 m steps on 0.9 m legs
    out, traj = resynthesise(rig, long, duration_s=2.0)
    assert not traj.feasibility.feasible and out.motion.shape[0] == 120
    slip = measure_slip(out, rig.mapping.landmarks, long, rig.reference.floor_z)
    assert slip.stances > 0 and math.isfinite(slip.max_travel_m)  # reported, never raised


def test_invert_double_support_is_the_inverse_of_the_rule():
    p = GaitParameters()
    assert invert_double_support(rule_double_support(p), p) == pytest.approx(0.20, abs=1e-3)


def test_p0_regenerates_the_sources_raw_double_support_reading(loaded):
    rig = load_rig(loaded[RigSpec()].path, scale=None, up_axis=None, mapping_overrides=None)
    assert rig.p0.com_height_scale == 1.0 and rig.p0.expansiveness == 1.0
    assert rig.p0.cadence == pytest.approx(rig.summary.mean.cadence)
    if 0.0 < rig.p0.double_support_frac < 0.5:
        assert rule_double_support(rig.p0) == pytest.approx(rig.raw_double_support, abs=2e-3)


def test_a_standing_clip_stops_with_the_discard_reasons(tmp_path):
    with pytest.raises(RigError, match="no usable cycle"):
        load_rig(
            standing_file(RigSpec(), tmp_path), scale=None, up_axis=None, mapping_overrides=None
        )


def test_mapping_overrides_split_into_landmarks_and_roles(loaded):
    path = loaded[RigSpec()].path
    with pytest.raises(RigError, match="NoSuchBone"):
        load_rig(path, scale=None, up_axis=None, mapping_overrides={"head": "NoSuchBone"})
    with pytest.raises(RigError, match="no_such_key"):
        load_rig(path, scale=None, up_axis=None, mapping_overrides={"no_such_key": "x"})
    # "knee_l" is both a landmark and a role: a "Bone/Channel" value makes it a role override.
    rig = load_rig(path, scale=None, up_axis=None, mapping_overrides=None)
    hinge = rig.mapping.roles[next(r for r in rig.mapping.roles if r.value == "knee_l")].hinge
    again = load_rig(path, scale=None, up_axis=None, mapping_overrides={"knee_l": hinge})
    assert any(d.what == "role:knee_l" and d.source == "user" for d in again.mapping.decisions)


def test_an_unknown_intensities_source_is_refused(loaded):
    with pytest.raises(ValueError, match="intensities_source"):
        emotional_walk(
            loaded[RigSpec()].path,
            intensities=ZERO,
            intensities_source="telepathy",
            model=ANALYTIC,
            model_name="analytic",
        )


def test_the_nominal_pelvis_height_is_lowered_until_p0_is_reachable(loaded):
    rig = load_rig(loaded[RigSpec()].path, scale=None, up_axis=None, mapping_overrides=None)
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


def test_the_pitch_innermost_exclusion_names_only_that_rig_and_large_abduction(loaded):
    kw = {"intensities_source": "flags", "model": ANALYTIC, "model_name": "analytic"}
    z_up = emotional_walk(
        loaded[replace(RigSpec(), up_axis="z")].path, intensities=PRESETS["angry"], **kw
    )
    y_up = emotional_walk(loaded[RigSpec()].path, intensities=PRESETS["angry"], **kw)
    calm = emotional_walk(loaded[replace(RigSpec(), up_axis="z")].path, intensities=ZERO, **kw)
    assert _pitch_innermost_exclusion(z_up) == PITCH_INNERMOST_EXCLUDED
    assert _pitch_innermost_exclusion(y_up) == ()
    assert _pitch_innermost_exclusion(calm) == ()


def test_a_still_arm_leaves_the_elbow_ratio_unobservable_and_a_straight_elbow_renders_none(
    loaded,
):
    rig = load_rig(loaded[RigSpec()].path, scale=None, up_axis=None, mapping_overrides=None)
    still = replace(rig.p0, arm_swing_amp=0.0, elbow_swing_ratio=0.5)
    out, traj = resynthesise(rig, still, duration_s=4.0)
    assert math.isnan(rendered(traj)["elbow_swing_ratio"])
    rt = round_trip(out, rig.mapping.landmarks, still, trajectory=traj)
    assert rt.unobservable == ("elbow_swing_ratio",)
    assert "elbow_swing_ratio" not in within_tolerance(rt)
    # Commanding less than straight holds the elbow at its straight limit: no swing renders,
    # and the flexion is compared floored at 0.
    bent_back = replace(rig.p0, elbow_flexion=-0.3, elbow_swing_ratio=0.4)
    out, traj = resynthesise(rig, bent_back, duration_s=4.0)
    assert abs(rendered(traj)["elbow_swing_ratio"]) < 1e-6
    rt = round_trip(out, rig.mapping.landmarks, bent_back, trajectory=traj)
    ok = within_tolerance(rt)
    assert ok["elbow_flexion"] and ok["elbow_swing_ratio"], rt.residuals
