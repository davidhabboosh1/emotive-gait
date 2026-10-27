"""Emotion for any rig, end to end (spec 4.1, 4.6, 4.7).

``load_rig`` maps the file's landmarks, takes the reference pose, binds the roles, compiles the
``driven`` chain, measures the posture map and reads the source's neutral ``P0`` with the frozen
extractor; the profile's nominal pelvis height is then lowered from the source's mean until
``P0`` is reachable (``reachable_pelvis_factor``, recorded as a ``pelvis_height`` decision), and
``P0.com_height_scale`` stays 1.0 relative to it (a field the extractor cannot read keeps the
default and is recorded as unobservable). ``walk`` modulates ``P0`` through ``RigScaledModel``,
generates on the welded rig (``resynthesise``) and writes the trajectory back into the file's
own channels, then measures the two claims on the written file read back: round-trip fidelity (``round_trip``,
``within_tolerance``) and zero foot slip over every commanded stance (``measure_slip``).
``emotional_walk`` is ``load_rig`` then ``walk``.

The trajectory-to-BVH step is ``rig.writer.write_trajectory``.

One foot point everywhere: the slip is read on the same landmark the extractor and the IK site
use (toe, else ankle), over the stances ``foot_point`` commands, sampled at the output frames.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np

from emotive_gait.emotion.evidence import Emotion
from emotive_gait.emotion.model import EmotionModel, modulate, unexpressible
from emotive_gait.gait.footsteps import Side, foot_point, phase_at, rule_double_support
from emotive_gait.gait.generator import GaitTrajectory, generate
from emotive_gait.gait.ik import FeasibilityReport
from emotive_gait.gait.params import GaitParameters
from emotive_gait.gait.scaling import MEASURED_FIELDS, Scaling
from emotive_gait.mocap.extract import ClipSummary, summarise
from emotive_gait.mocap.readers.bvh import BvhFile, read_bvh_file
from emotive_gait.mocap.skeleton import Landmark, LandmarkMap, SkeletonClip
from emotive_gait.mocap.writers.bvh import write_bvh
from emotive_gait.rig.calibration import (
    ReferencePose,
    landmark_site_names,
    posture_map,
    reference_pose,
)
from emotive_gait.rig.compile import (
    CompiledRig,
    compile_driven,
    compile_full,
)
from emotive_gait.rig.errors import RigError
from emotive_gait.rig.mapping import (
    Decision,
    RigMapping,
    bind_roles,
    infer_scale,
    infer_up,
    map_landmarks,
)
from emotive_gait.rig.profile import rig_profile, rig_robot
from emotive_gait.rig.scaled import RigScaledModel
from emotive_gait.rig.writer import write_trajectory
from emotive_gait.robot.loader import RobotModel
from emotive_gait.robot.roles import JointRole, RobotProfile

L = Landmark
_FEET = ((Side.LEFT, L.ANKLE_L, L.TOE_L), (Side.RIGHT, L.ANKLE_R, L.TOE_R))
_EXPANSIVE = ("step_height", "arm_swing_amp", "arm_abduction")  # scaled by expansiveness
_SOURCES = ("flags", "lexicon", "caller")

# The round-trip claim (spec 4.7, frozen): relative for cadence and step length, absolute
# (metres, radians, the ratio, the fraction) for the rest. Double support is compared with
# ``rule_double_support`` of the commanded parameters at the Phase 4.5 round trip's 0.05.
TOLERANCES: Mapping[str, float] = MappingProxyType(
    {
        "cadence": 0.02,
        "step_length": 0.02,
        "step_height": 0.003,
        "com_bounce_amp": 0.003,
        "lateral_sway_amp": 0.003,
        "trunk_pitch": 0.02,
        "neck_pitch": 0.02,
        "arm_swing_amp": 0.02,
        "arm_abduction": 0.02,
        "elbow_flexion": 0.02,
        "elbow_swing_ratio": 0.05,
        "double_support_frac": 0.05,
    }
)
RELATIVE_TOLERANCES = ("cadence", "step_length")


@dataclass(frozen=True, eq=False)
class LoadedRig:
    """A rig ready to walk: its file, mapping, reference, neutral ``p0`` and compiled chain."""

    file: BvhFile
    clip: SkeletonClip  # the source's FK clip, End Sites included
    mapping: RigMapping
    reference: ReferencePose
    summary: ClipSummary  # the source under the frozen extractor
    p0: GaitParameters  # the rig's neutral (spec 4.1)
    raw_double_support: float  # the extractor's reading, before the inversion
    robot: RobotModel  # the driven chain with the measured posture map
    profile: RobotProfile
    p_ref: dict[str, float]  # the posture reading at the reference pose
    compiled: CompiledRig  # the driven chain's names (bones -> bodies)
    # P0 fields the extractor cannot read on this rig (a landmark is absent): P0 carries the
    # GaitParameters default for each, recorded as a decision, and the round trip skips them
    unobservable: tuple[str, ...] = ()

    @property
    def source_pelvis_height(self) -> float:
        """The source's mean pelvis height above the floor (m)."""
        return float(self.reference.pelvis_height)

    @property
    def nominal_pelvis_height(self) -> float:
        """The profile's nominal pelvis height (m): the source's, lowered until ``p0`` is
        reachable (``reachable_pelvis_factor``)."""
        return float(self.profile.nominal_pelvis_height)


@dataclass(frozen=True)
class SlipReport:
    """Foot-point slip over the commanded stances (worst over both feet and every stance)."""

    max_travel_m: float  # horizontal distance from the stance's first-frame position
    max_height_error_m: float  # |z - floor_z|
    stances: int


@dataclass(frozen=True, eq=False)
class RoundTrip:
    """The extractor's reading of the output against the commanded parameters."""

    commanded: GaitParameters
    measured: ClipSummary
    # measured minus commanded per observable MEASURED_FIELDS (double support against
    # rule_double_support(commanded); step height and abduction against the
    # expansiveness-scaled values, elbow flexion floored at straight, arm swing and elbow ratio
    # against the rendered trajectory (``rendered``); com_height_scale as the pelvis height
    # above the floor over the nominal, NaN without them)
    residuals: Mapping[str, float]
    # fields without a residual, left out of ``within_tolerance``: P0 fields the extractor
    # cannot read on the rig, and fields the commanded motion leaves undefined (the elbow ratio
    # of an arm that does not swing)
    unobservable: tuple[str, ...] = ()


@dataclass(frozen=True, eq=False)
class WalkResult:
    rig: LoadedRig
    intensities: Mapping[Emotion, float]
    intensities_source: Literal["flags", "lexicon", "caller"]
    model_name: str
    scaling: Scaling
    p1: GaitParameters
    unexpressible: tuple[tuple[Emotion, str], ...]
    feasibility: FeasibilityReport
    slip: SlipReport
    round_trip: RoundTrip
    output: BvhFile


# --- loading -------------------------------------------------------------------------------


def invert_double_support(raw: float, params: GaitParameters) -> float:
    """The ``double_support_frac`` in ``[0, 0.5]`` whose ``rule_double_support`` reads ``raw``.

    Bisection to 1e-4 on the monotone rule; a reading outside the rule's range gives the
    nearest end.
    """

    def reading(d: float) -> float:
        return rule_double_support(replace(params, double_support_frac=d)) - raw

    lo, hi = 0.0, 0.5
    if reading(lo) >= 0.0:
        return lo
    if reading(hi) <= 0.0:
        return hi
    while hi - lo > 1e-4:
        mid = 0.5 * (lo + hi)
        if reading(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _split_overrides(
    overrides: Mapping[str, str] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    """``(landmark overrides, role overrides)``. Role keys are namespaced ``role:<role value>``
    (as the decisions are); a bare key is a landmark value."""
    landmarks = {lm.value for lm in Landmark}
    roles = {r.value for r in JointRole}
    lm_over: dict[str, str] = {}
    role_over: dict[str, str] = {}
    for key, value in (overrides or {}).items():
        if key.startswith("role:"):
            if key[5:] not in roles:
                raise RigError(f"mapping override {key!r} names an unknown role")
            role_over[key[5:]] = value
        elif key in landmarks:
            lm_over[key] = value
        else:
            raise RigError(
                f"mapping override {key!r} is not a landmark; put joint roles under the "
                "'roles:' section (of the mapping file or the MCP mapping argument)"
            )
    return lm_over, role_over


def _p0(summary: ClipSummary) -> tuple[GaitParameters, float, tuple[str, ...]]:
    """``(P0, raw double support, unobservable fields)``: the cycle mean as parameters,
    ``com_height_scale`` and ``expansiveness`` 1.0, double support inverted through the rule.
    A field the extractor could not read (NaN: a landmark is absent) keeps the
    ``GaitParameters`` default and is listed as unobservable; ``load_rig`` records it."""
    m = summary.mean
    defaults = GaitParameters()
    values = {}
    unobservable = []
    for f in fields(GaitParameters):
        if f.name in ("com_height_scale", "expansiveness"):
            values[f.name] = 1.0
            continue
        value = float(getattr(m, f.name))
        if not math.isfinite(value):
            unobservable.append(f.name)
            value = float(getattr(defaults, f.name))
        values[f.name] = value
    p0 = GaitParameters(**values)
    raw = p0.double_support_frac
    p0 = replace(p0, double_support_frac=invert_double_support(raw, p0))
    return p0, raw, tuple(unobservable)


PELVIS_FACTOR_MIN = 0.75  # the lowest nominal pelvis height tried, x the source's mean
PELVIS_FACTOR_STEP = 0.01  # bisection resolution of the factor


def reachable_pelvis_factor(p0: GaitParameters, robot: RobotModel, source_height: float) -> float:
    """The largest ``f`` in ``[PELVIS_FACTOR_MIN, 1]`` (to ``PELVIS_FACTOR_STEP``) at which
    ``generate(p0)`` on ``robot`` with the nominal pelvis height ``f * source_height`` is
    feasible; ``PELVIS_FACTOR_MIN`` when none is (the flag then stays false downstream).

    The generator has a flat foot, a pelvis of fixed orientation and no heel-toe roll, so the
    source's own mean pelvis height (often near the straight-leg height) can leave its stride
    unreachable. The G1 does the same: nominal 0.68 m below its 0.79 m straight-leg stand.
    ``f = 1`` is tried first, so a rig that already fits keeps the source height.

    Assumes feasibility is monotone in ``f`` (a lower pelvis only makes the stride easier to
    reach, down to the window's floor); on a rig where it is not, the ``f`` found is feasible
    but not necessarily the largest one.
    """

    def feasible(f: float) -> bool:
        profile = replace(robot.profile, nominal_pelvis_height=f * source_height)
        lowered = replace(robot, profile=profile)
        return generate(p0, lowered, welded=lowered).feasibility.feasible

    if feasible(1.0):
        return 1.0
    lo, hi = PELVIS_FACTOR_MIN, 1.0
    # No probe of ``lo`` itself: if nothing above it is feasible the answer is ``lo`` whether
    # or not ``lo`` is (then the flag stays false downstream), so it would cost a generate.
    while hi - lo > PELVIS_FACTOR_STEP:
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _read(
    source: Path, scale: float | None, up_axis: str | None
) -> tuple[BvhFile, tuple[Decision, ...]]:
    """``(file, inference decisions)``: ``source`` read with ``scale`` and ``up_axis``, each
    inferred from the rest hierarchy only when ``None`` (``infer_scale``, ``infer_up``), and the
    ``geometry`` decision of each inferred value (none for a given one)."""
    decisions: list[Decision] = []
    if scale is None or up_axis is None:
        raw = read_bvh_file(source, scale=1.0, up_axis="z")
        if up_axis is None:
            up_axis, decision = infer_up(raw)
            decisions.append(decision)
        if scale is None:
            scale, decision = infer_scale(raw)
            decisions.append(decision)
    if up_axis not in ("y", "z"):
        raise RigError(f"up axis must be 'y' or 'z', got {up_axis!r}")
    return read_bvh_file(source, scale=float(scale), up_axis=up_axis), tuple(decisions)


def _summarise(clip: SkeletonClip, landmarks: LandmarkMap) -> ClipSummary:
    """``summarise`` the clip; ``RigError`` naming the discard reasons when no cycle is usable."""
    summary = summarise(clip, landmarks)
    if summary.n_cycles == 0:
        raise RigError(f"no usable cycle: {dict(summary.discard_reasons)}")
    return summary


def measure(
    source: Path,
    *,
    scale: float | None,
    up_axis: str | None,
    mapping_overrides: Mapping[str, str] | None,
) -> ClipSummary:
    """The source's ``ClipSummary``: read, map the landmarks, summarise; nothing else.

    No role binding, calibration or compilation, so any walk whose landmarks map can be
    measured. Landmark overrides are honoured; ``role:`` overrides are ignored.
    """
    file, _ = _read(Path(source), scale, up_axis)
    clip = file.to_clip(end_sites=True)
    lm_over, _ = _split_overrides(mapping_overrides)
    landmarks = map_landmarks(file, clip, overrides=lm_over)
    return _summarise(clip, landmarks.landmarks)


def load_rig(
    source: Path,
    *,
    scale: float | None,
    up_axis: str | None,
    mapping_overrides: Mapping[str, str] | None,
) -> LoadedRig:
    """Read, map, calibrate and compile ``source``; ``scale`` and ``up_axis`` are inferred
    from the rest hierarchy when ``None`` (each inference recorded as a ``geometry`` decision,
    first in ``mapping.decisions``). Raises ``RigError`` (unusable mapping, no usable
    cycle, unknown override)."""
    source = Path(source)
    file, units = _read(source, scale, up_axis)
    clip = file.to_clip(end_sites=True)
    lm_over, role_over = _split_overrides(mapping_overrides)
    landmarks = map_landmarks(file, clip, overrides=lm_over)
    reference = reference_pose(file, clip, landmarks.landmarks)
    full = compile_full(file, pelvis=landmarks.landmarks[L.PELVIS], pelvis_height=1.0)
    mapping = bind_roles(full, reference, landmarks, overrides=role_over)
    driven = compile_driven(file, mapping, reference)
    name = f"rig:{source.name}"
    bare = rig_robot(driven, rig_profile(name, mapping, reference, posture_map={}))
    rows, p_ref = posture_map(
        bare, mapping, landmark_sites=landmark_site_names(driven, mapping.landmarks)
    )
    profile = rig_profile(name, mapping, reference, posture_map=rows)
    summary = _summarise(clip, mapping.landmarks)  # reference_pose has already refused n == 0
    p0, raw, unobservable = _p0(summary)
    source_height = float(reference.pelvis_height)
    factor = reachable_pelvis_factor(p0, rig_robot(driven, profile), source_height)
    profile = replace(profile, nominal_pelvis_height=factor * source_height)
    mapping = replace(
        mapping,
        decisions=(
            *units,
            *reference.decisions,
            *mapping.decisions,
            Decision(
                what="pelvis_height",
                chosen=f"{factor:.2f} x source mean ({factor * source_height:.3f} m)",
                source="geometry",
                note="lowered until P0 is reachable by the flat-foot generator",
            ),
            *(
                Decision(
                    what=f"p0:{name}",
                    chosen=f"{getattr(p0, name):g} (GaitParameters default)",
                    source="geometry",
                    note="unobservable: the extractor cannot read it on this rig",
                )
                for name in unobservable
            ),
        ),
        warnings=(*mapping.warnings, *reference.warnings),
    )
    return LoadedRig(
        file=file,
        clip=clip,
        mapping=mapping,
        reference=reference,
        summary=summary,
        p0=p0,
        raw_double_support=raw,
        robot=rig_robot(driven, profile),
        profile=profile,
        p_ref=p_ref,
        compiled=driven,
        unobservable=unobservable,
    )


# --- writing (rig.writer) ------------------------------------------------------------------


def resynthesise(
    rig: LoadedRig, p1: GaitParameters, *, duration_s: float
) -> tuple[BvhFile, GaitTrajectory]:
    """Generate ``p1`` on the rig and write it into the source's channels (``rig.writer``).

    The output has ``round(duration_s * fps)`` frames at the source frame time and keeps the
    hierarchy, channel order, units and up-axis. Infeasible parameters are written as well; the
    trajectory's ``feasibility`` says so.
    """
    traj = generate(p1, rig.robot, welded=rig.robot)
    output = write_trajectory(
        traj,
        p1,
        robot=rig.robot,
        compiled=rig.compiled,
        reference=rig.reference,
        clip=rig.clip,
        landmarks=rig.mapping.landmarks,
        duration_s=duration_s,
    )
    return output, traj


# --- the claims ----------------------------------------------------------------------------


def _foot_tracks(clip: SkeletonClip, landmarks: LandmarkMap) -> list[np.ndarray]:
    tracks = clip.landmark_positions(landmarks)
    return [tracks.get(toe, tracks[ankle]) for _, ankle, toe in _FEET]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``(start, stop)`` of every run of True in ``mask`` (stop exclusive)."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def measure_slip(
    output: BvhFile, landmarks: LandmarkMap, p1: GaitParameters, floor_z: float
) -> SlipReport:
    """Foot-point travel and height error over every stance ``foot_point(p1, ...)`` commands,
    sampled at the output frames (both feet; partial stances at the ends count)."""
    clip = output.to_clip(end_sites=True)
    times = np.arange(clip.frames()) / clip.fps
    travel, height, stances = 0.0, 0.0, 0
    for (side, _, _), foot in zip(_FEET, _foot_tracks(clip, landmarks), strict=True):
        stance = np.array([foot_point(p1, phase_at(p1, float(t)), side).in_stance for t in times])
        for s, e in _runs(stance):
            stances += 1
            run = foot[s:e]
            travel = max(travel, float(np.max(np.linalg.norm(run[:, :2] - run[0, :2], axis=1))))
            height = max(height, float(np.max(np.abs(run[:, 2] - floor_z))))
    return SlipReport(max_travel_m=travel, max_height_error_m=height, stances=stances)


_ARMS = (
    (JointRole.SHOULDER_PITCH_L, JointRole.ELBOW_L),
    (JointRole.SHOULDER_PITCH_R, JointRole.ELBOW_R),
)
_STILL_ARM_RAD = 1e-6  # a rendered shoulder swing below this leaves the elbow ratio undefined


def rendered(trajectory: GaitTrajectory, samples: int = 240) -> dict[str, float]:
    """What the trajectory renders for the fields the generator shapes after the parameters,
    over ``samples`` phases, both arms averaged (NaN: not observable):

    - ``arm_swing_amp``: half peak-to-peak of the shoulder-pitch angles, after the hip-normalised
      drive and the clamp; the Phase 4.5 round trip compares the arm swing the same way
      (``tests/mocap/test_roundtrip.py``);
    - ``elbow_swing_ratio``: ``elbow_swing_sign`` times the slope of the rendered elbow angle
      against the rendered shoulder pitch, so an elbow held at its straight limit renders 0;
      NaN when the shoulder does not swing (the ratio is then undefined).
    """
    arms = [(s, e) for s, e in _ARMS if s in trajectory.roles and e in trajectory.roles]
    out = {"arm_swing_amp": math.nan, "elbow_swing_ratio": math.nan}
    if not arms:
        return out
    angles = [trajectory.angles_at(k / samples) for k in range(samples)]
    swings, ratios = [], []
    for shoulder, elbow in arms:
        pitch = np.array([a[shoulder] for a in angles])
        bend = np.array([a[elbow] for a in angles])
        swings.append(np.ptp(pitch) / 2.0)
        if np.ptp(pitch) >= _STILL_ARM_RAD:
            slope = np.polyfit(pitch - pitch.mean(), bend - bend.mean(), 1)[0]
            ratios.append(trajectory.elbow_swing_sign * float(slope))
    out["arm_swing_amp"] = float(np.mean(swings))
    if ratios:
        out["elbow_swing_ratio"] = float(np.mean(ratios))
    return out


def _commanded(p: GaitParameters, name: str, shaped: Mapping[str, float] | None = None) -> float:
    """What the generator renders for ``name`` (the value the extractor should read).

    Double support is the rule's reading of the commanded path; step height, arm swing and
    abduction are scaled by ``expansiveness``; ``elbow_flexion`` is floored at 0 (the elbow
    range stops at straight, ``RoleBinding.zero_offset``, and the extractor's measure is
    unsigned); ``shaped`` (``rendered``) overrides the arm swing and elbow ratio.
    """
    if name == "double_support_frac":
        return rule_double_support(p)
    if shaped is not None and name in shaped:
        return shaped[name]
    value = float(getattr(p, name))
    if name == "elbow_flexion":
        return max(0.0, value)
    return p.expansiveness * value if name in _EXPANSIVE else value


def round_trip(
    output: BvhFile,
    landmarks: LandmarkMap,
    p1: GaitParameters,
    *,
    floor_z: float | None = None,
    nominal_pelvis_height: float | None = None,
    trajectory: GaitTrajectory | None = None,
    unobservable: Iterable[str] = (),
) -> RoundTrip:
    """``summarise`` the output and subtract the commanded values (``RoundTrip.residuals``).

    With ``trajectory`` the arm swing and elbow ratio are compared with what it renders
    (``rendered``; the arm swing as in the Phase 4.5 round trip), else with the parameters.
    Fields in ``unobservable`` (``LoadedRig.unobservable``: P0 could not read them) and fields
    the trajectory leaves undefined get no residual and are listed in
    ``RoundTrip.unobservable``.
    """
    shaped = None if trajectory is None else rendered(trajectory)
    measured = summarise(output.to_clip(end_sites=True), landmarks)
    m = measured.mean
    residuals: dict[str, float] = {}
    skipped = list(dict.fromkeys(unobservable))
    for name in MEASURED_FIELDS:
        if name in skipped:
            continue
        if name == "com_height_scale":
            if floor_z is None or nominal_pelvis_height is None:
                residuals[name] = math.nan
            else:
                reading = (m.pelvis_height - floor_z) / nominal_pelvis_height
                residuals[name] = float(reading - p1.com_height_scale)
            continue
        commanded = _commanded(p1, name, shaped)
        if not math.isfinite(commanded):
            skipped.append(name)
            continue
        residuals[name] = float(getattr(m, name)) - commanded
    return RoundTrip(
        commanded=p1,
        measured=measured,
        residuals=MappingProxyType(residuals),
        unobservable=tuple(skipped),
    )


def within_tolerance(rt: RoundTrip) -> dict[str, bool]:
    """Per ``TOLERANCES`` field, whether the residual is within it (NaN is not); fields in
    ``rt.unobservable`` are left out."""
    out = {}
    for name, tol in TOLERANCES.items():
        if name in rt.unobservable:
            continue
        r = rt.residuals[name]
        if name in RELATIVE_TOLERANCES:
            tol = tol * abs(_commanded(rt.commanded, name))
        out[name] = bool(math.isfinite(r) and abs(r) <= tol)
    return out


# --- the pipeline --------------------------------------------------------------------------


def walk(
    rig: LoadedRig,
    *,
    intensities: Mapping[Emotion, float],
    intensities_source: str,
    model: EmotionModel,
    model_name: str,
    scaling: Scaling = "froude",
    duration_s: float | None = None,
) -> WalkResult:
    """Move ``rig``'s neutral by ``intensities`` and write the walk back (spec 4.7).

    ``duration_s`` defaults to the source's length. The claims are measured on the output
    written to a temporary BVH and read back; the caller writes ``output`` where it wants it.
    Infeasible parameters are flagged in ``feasibility`` and still written.
    """
    if intensities_source not in _SOURCES:
        raise ValueError(
            f"intensities_source must be one of {_SOURCES}, got {intensities_source!r}"
        )
    scaled = RigScaledModel(model, scaling=scaling, p0=rig.p0)
    p1 = modulate(rig.p0, intensities, rig.profile, model=scaled)
    dropped = unexpressible(intensities, rig.profile, scaled)
    if duration_s is None:
        duration_s = rig.file.motion.shape[0] * rig.file.frame_time
    output, traj = resynthesise(rig, p1, duration_s=duration_s)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.bvh"
        write_bvh(output, path)
        written = read_bvh_file(path, scale=output.scale, up_axis=output.up_axis)
    landmarks = rig.mapping.landmarks
    return WalkResult(
        rig=rig,
        intensities=MappingProxyType(dict(intensities)),
        intensities_source=intensities_source,  # type: ignore[arg-type]
        model_name=model_name,
        scaling=scaling,
        p1=p1,
        unexpressible=dropped,
        feasibility=traj.feasibility,
        slip=measure_slip(written, landmarks, p1, rig.reference.floor_z),
        round_trip=round_trip(
            written,
            landmarks,
            p1,
            floor_z=rig.reference.floor_z,
            nominal_pelvis_height=rig.profile.nominal_pelvis_height,
            trajectory=traj,
            unobservable=rig.unobservable,
        ),
        output=output,
    )


def emotional_walk(
    source: Path,
    *,
    intensities: Mapping[Emotion, float],
    intensities_source: str,
    model: EmotionModel,
    model_name: str,
    scaling: Scaling = "froude",
    duration_s: float | None = None,
    mapping_overrides: Mapping[str, str] | None = None,
    scale: float | None = None,
    up_axis: str | None = None,
) -> WalkResult:
    """``load_rig(source, ...)``, then ``walk`` on it (see both)."""
    if intensities_source not in _SOURCES:
        raise ValueError(
            f"intensities_source must be one of {_SOURCES}, got {intensities_source!r}"
        )
    rig = load_rig(source, scale=scale, up_axis=up_axis, mapping_overrides=mapping_overrides)
    return walk(
        rig,
        intensities=intensities,
        intensities_source=intensities_source,
        model=model,
        model_name=model_name,
        scaling=scaling,
        duration_s=duration_s,
    )
