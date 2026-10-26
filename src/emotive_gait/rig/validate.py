"""Validate one rig: zero emotion and the four analytic presets, tabulated (spec 4.7, tier 3).

``validate_rig`` loads a BVH walk and runs ``neutral`` plus ``sad``, ``angry``, ``happy`` and
``fear`` at intensity 1.0 through ``walk`` (the analytic model), one ``ValidationRow`` each;
``render_report`` turns the tables of several rigs into the Markdown of
``docs/reports/phase-4-6-rig-validation.md``. Nothing is gated here: infeasible presets and
residuals beyond tolerance are reported as measured. The report carries aggregates only (no
clip or frame data).
"""

from __future__ import annotations

import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from emotive_gait.emotion.evidence import Emotion
from emotive_gait.emotion.model import ANALYTIC
from emotive_gait.gait.params import GaitParameters
from emotive_gait.mocap.readers.bvh import BvhFile
from emotive_gait.rig.calibration import POSTURE_GROUPS
from emotive_gait.rig.mapping import RigMapping
from emotive_gait.rig.synthesis import (
    RELATIVE_TOLERANCES,
    TOLERANCES,
    LoadedRig,
    WalkResult,
    load_rig,
    walk,
    within_tolerance,
)
from emotive_gait.robot.roles import JointRole

RUNS: tuple[tuple[str, Mapping[Emotion, float]], ...] = (
    ("neutral", MappingProxyType({})),
    ("sad", MappingProxyType({Emotion.SADNESS: 1.0})),
    ("angry", MappingProxyType({Emotion.ANGER: 1.0})),
    ("happy", MappingProxyType({Emotion.HAPPINESS: 1.0})),
    ("fear", MappingProxyType({Emotion.FEAR: 1.0})),
)

# Documented limitation (spec 4.7 rulings, Task 8): a driven hinge reproduces its BVH Euler
# channel, so when the shoulder PITCH channel is its bone's innermost (last) rotation channel it
# turns about the abducted arm's own lateral axis, not the trunk's; above this commanded
# abduction the extractor over-reads the swing and under-reads the abduction during it, which
# the static posture map cannot linearise (angry on the z-up ZXY synthetic rig: +0.089 rad
# swing, -0.032 rad abduction), so those two fields are not gated.
PITCH_INNERMOST_ABDUCTION = 0.5  # rad
PITCH_INNERMOST_EXCLUDED = ("arm_swing_amp", "arm_abduction")


@dataclass(frozen=True)
class ValidationRow:
    """One run: feasibility, the residual per field, whether all are within tolerance, slip."""

    run: str
    feasible: bool
    max_residual: Mapping[str, float]  # field -> residual (measured - commanded)
    within: bool  # every gated field within its tolerance
    slip_m: float  # worst horizontal stance-foot travel
    height_m: float  # worst stance-foot height error
    failing: tuple[str, ...] = ()  # gated fields beyond tolerance
    excluded: tuple[str, ...] = ()  # fields left ungated by the pitch-innermost exclusion
    worst_field: str = "-"  # the residual furthest from its tolerance (as a ratio)
    worst_residual: float = math.nan
    commanded_abduction: float = math.nan  # expansiveness * arm_abduction of P1 (rad)


@dataclass(frozen=True)
class ValidationTable:
    """One rig's five runs plus the rig facts the report states (numbers and names only)."""

    source: str  # file name
    family: str | None
    scale: float
    scale_given: bool
    up_axis: str
    up_given: bool
    leg_length: float
    cycles: int
    source_pelvis_height: float
    nominal_pelvis_height: float
    skewed: tuple[str, ...]
    unobservable: tuple[str, ...]
    half_bound: tuple[str, ...]  # two-sided posture groups with one side bound
    pitch_innermost: bool
    rows: tuple[ValidationRow, ...]

    @property
    def pelvis_factor(self) -> float:
        """``reachable_pelvis_factor``: nominal over source mean pelvis height."""
        return self.nominal_pelvis_height / self.source_pelvis_height


def pitch_innermost(mapping: RigMapping, file: BvhFile) -> bool:
    """Whether either shoulder pitch channel is its bone's innermost rotation channel."""
    for role in (JointRole.SHOULDER_PITCH_L, JointRole.SHOULDER_PITCH_R):
        binding = mapping.roles.get(role)
        if binding is None:
            continue
        bone, channel = binding.hinge.rsplit("/", 1)
        joint = next(j for j in file.joints if j.name == bone)
        if [c for c in joint.channels if c.endswith("rotation")][-1] == channel:
            return True
    return False


def pitch_innermost_exclusion(
    mapping: RigMapping, file: BvhFile, p1: GaitParameters
) -> tuple[str, ...]:
    """``PITCH_INNERMOST_EXCLUDED`` when a shoulder pitch channel is innermost and the commanded
    abduction exceeds ``PITCH_INNERMOST_ABDUCTION``, else ``()``."""
    over = p1.expansiveness * p1.arm_abduction > PITCH_INNERMOST_ABDUCTION
    return PITCH_INNERMOST_EXCLUDED if over and pitch_innermost(mapping, file) else ()


def worst_field(result: WalkResult) -> tuple[str, float]:
    """The residual furthest from its tolerance (as a ratio): ``(field, residual)``."""
    rt = result.round_trip
    worst, worst_ratio = ("-", math.nan), -1.0
    for name, r in rt.residuals.items():
        tol = TOLERANCES.get(name)
        if tol is None:
            continue
        if name in RELATIVE_TOLERANCES:
            tol *= abs(getattr(rt.commanded, name))
        ratio = abs(r) / tol if math.isfinite(r) and tol > 0 else math.inf
        if ratio > worst_ratio:
            worst, worst_ratio = (name, r), ratio
    return worst


def validation_row(name: str, result: WalkResult) -> ValidationRow:
    """The row of one run's ``WalkResult``."""
    excluded = pitch_innermost_exclusion(result.rig.mapping, result.rig.file, result.p1)
    ok = within_tolerance(result.round_trip)
    failing = tuple(k for k, v in ok.items() if not v and k not in excluded)
    field, residual = worst_field(result)
    return ValidationRow(
        run=name,
        feasible=bool(result.feasibility.feasible),
        max_residual=MappingProxyType(dict(result.round_trip.residuals)),
        within=not failing,
        slip_m=float(result.slip.max_travel_m),
        height_m=float(result.slip.max_height_error_m),
        failing=failing,
        excluded=excluded,
        worst_field=field,
        worst_residual=float(residual),
        commanded_abduction=float(result.p1.expansiveness * result.p1.arm_abduction),
    )


def validate_loaded(
    rig: LoadedRig, *, source: str, scale_given: bool, up_given: bool
) -> ValidationTable:
    """Zero emotion and the four presets on an already loaded ``rig``."""
    rows = tuple(
        validation_row(
            name,
            walk(
                rig,
                intensities=intensities,
                intensities_source="flags",
                model=ANALYTIC,
                model_name="analytic",
            ),
        )
        for name, intensities in RUNS
    )
    m = rig.mapping
    half = tuple(
        k
        for k, roles in POSTURE_GROUPS.items()
        if len(roles) == 2 and sum(r in m.roles for r in roles) == 1
    )
    return ValidationTable(
        source=source,
        family=m.family,
        scale=float(m.scale),
        scale_given=scale_given,
        up_axis=m.up_axis,
        up_given=up_given,
        leg_length=float(m.leg_length),
        cycles=int(rig.summary.n_cycles),
        source_pelvis_height=rig.source_pelvis_height,
        nominal_pelvis_height=rig.nominal_pelvis_height,
        skewed=tuple(r.value for r in m.skewed),
        unobservable=tuple(rig.unobservable),
        half_bound=half,
        pitch_innermost=pitch_innermost(m, rig.file),
        rows=rows,
    )


def validate_rig(
    source: Path,
    *,
    scale: float | None,
    up_axis: str | None,
    mapping_overrides: Mapping[str, str] | None,
) -> ValidationTable:
    """Load ``source`` (``load_rig``) and tabulate zero emotion and the four presets."""
    source = Path(source)
    rig = load_rig(source, scale=scale, up_axis=up_axis, mapping_overrides=mapping_overrides)
    return validate_loaded(
        rig, source=source.name, scale_given=scale is not None, up_given=up_axis is not None
    )


# --- the report ----------------------------------------------------------------------------


def _commit() -> str:
    """``git describe --always --dirty`` of this checkout, or ``unknown``."""
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def _fmt(x: float) -> str:
    return format(x, "+.4f") if math.isfinite(x) else "nan"


def _tolerance_text(name: str) -> str:
    tol = TOLERANCES[name]
    if name in RELATIVE_TOLERANCES:
        return f"{tol * 100:g} % of the commanded value"
    if name in ("step_height", "com_bounce_amp", "lateral_sway_amp"):
        return f"{tol * 1000:g} mm"
    if name == "double_support_frac":
        return f"{tol:g} (against rule_double_support of the commanded parameters)"
    if name == "elbow_swing_ratio":
        return f"{tol:g}"
    return f"{tol:g} rad"


def _given(flag: bool) -> str:
    return "given" if flag else "inferred"


def _rig_section(name: str, t: ValidationTable) -> list[str]:
    lines = [
        f"## {name}",
        "",
        f"- file: `{t.source}`; naming family: {t.family or 'unknown'}",
        (
            f"- scale: {t.scale:g} m per file unit ({_given(t.scale_given)}); "
            f"up axis: {t.up_axis} ({_given(t.up_given)}); leg length {t.leg_length:.3f} m"
        ),
        f"- usable cycles: {t.cycles}",
        (
            f"- pelvis height: source mean {t.source_pelvis_height:.3f} m, nominal "
            f"{t.nominal_pelvis_height:.3f} m, factor {t.pelvis_factor:.3f}"
        ),
        f"- skewed roles: {', '.join(t.skewed) or 'none'}",
        f"- unobservable fields: {', '.join(t.unobservable) or 'none'}",
        f"- half-bound posture groups: {', '.join(t.half_bound) or 'none'}",
        f"- shoulder pitch channel innermost: {'yes' if t.pitch_innermost else 'no'}",
        "",
        (
            "| run | feasible | worst field (residual) | within | failing fields | slip mm "
            "| height mm |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for r in t.rows:
        failing = ", ".join(r.failing) or "none"
        if r.excluded:
            failing += f" (ungated: {', '.join(r.excluded)})"
        lines.append(
            f"| {r.run} | {'yes' if r.feasible else 'NO'} | {r.worst_field} "
            f"({_fmt(r.worst_residual)}) | {'yes' if r.within else 'NO'} | {failing} | "
            f"{r.slip_m * 1000:.1f} | {r.height_m * 1000:.1f} |"
        )
    fields = list(dict.fromkeys(f for r in t.rows for f in r.max_residual))
    lines += [
        "",
        (
            "Residual per field, measured minus commanded (steps/s, metres, radians, the ratio, "
            "the fraction; fields without a tolerance above are informational, not gated):"
        ),
        "",
        "| field | " + " | ".join(r.run for r in t.rows) + " |",
        "|---|" + "---|" * len(t.rows),
    ]
    for f in fields:
        cells = [_fmt(r.max_residual.get(f, math.nan)) for r in t.rows]
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def render_report(
    tables: Mapping[str, ValidationTable],
    *,
    not_run: Mapping[str, str],
    commit: str | None = None,
    commands: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> str:
    """The Markdown validation report for ``tables`` (rig name -> table).

    A header naming ``commit`` (``git describe`` of this checkout when ``None``) and the
    tolerances, the ``commands`` used, one section per rig with its five runs, the pelvis
    factors, the pitch-innermost exclusion, the infeasible presets, the known limitations hit,
    the "not run" list with reasons and how many rigs hit ``skewed``; ``notes`` are listed as
    findings at the end.
    """
    lines = [
        "# Phase 4.6 rig validation",
        "",
        "Generated by `emotive_gait.rig.validate.render_report`; do not edit by hand.",
        f"Commit: `{commit or _commit()}`.",
        "",
        (
            "Each rig is walked at zero emotion (`neutral`) and at the analytic presets' "
            "intensities (`sad`, `angry`, `happy`, `fear`, each 1.0); the output is written in the "
            "rig's own channels, read back and measured by the frozen extractor. Infeasible presets "
            "and residuals beyond tolerance are reported, not gated. Aggregates only: no clip or "
            "frame data."
        ),
        "",
        "Round-trip tolerances (frozen, spec 4.7):",
        "",
        "| field | tolerance |",
        "|---|---|",
        *(f"| {k} | {_tolerance_text(k)} |" for k in TOLERANCES),
        "",
    ]
    if commands:
        lines += ["Command lines used:", "", "```sh", *commands, "```", ""]
    for name, t in tables.items():
        lines += _rig_section(name, t)

    lines += [
        "## Pelvis-lowering factor",
        "",
        (
            "`reachable_pelvis_factor` (the `pelvis_height` decision): the generator has a flat "
            "foot, a pelvis of fixed orientation and no heel-toe roll, so the source's mean pelvis "
            "height can leave its stride unreachable. The nominal pelvis height is lowered from the "
            "source's mean until the rig's neutral is feasible, so the output walks with a lower "
            "pelvis than the source in order to reach the same stride (the G1 does the same: "
            "0.68 m nominal below its 0.79 m straight-leg stand). Earlier measurements (Task 8): "
            "synthetic base rig 0.938, no-toes rig 0.922, 100STYLE excerpt 0.906."
        ),
        "",
        "| rig | source mean m | nominal m | factor |",
        "|---|---|---|---|",
        *(
            f"| {n} | {t.source_pelvis_height:.3f} | {t.nominal_pelvis_height:.3f} | "
            f"{t.pelvis_factor:.3f} |"
            for n, t in tables.items()
        ),
    ]
    innermost = [n for n, t in tables.items() if t.pitch_innermost]
    lines += [
        "",
        "## Pitch-innermost shoulder exclusion",
        "",
        (
            "A driven hinge reproduces its BVH Euler channel, so when the shoulder pitch channel is "
            "its bone's innermost rotation channel it turns about the abducted arm's own lateral "
            "axis. Above a commanded abduction of "
            f"{PITCH_INNERMOST_ABDUCTION:g} rad the extractor over-reads the arm swing and "
            f"under-reads the abduction, so {' and '.join(PITCH_INNERMOST_EXCLUDED)} are not gated "
            "on such a run (marked ungated in the tables)."
        ),
        "",
        f"Applies to: {', '.join(innermost) or 'none of the rigs run'}.",
        "",
        *(
            f"- {n}: largest commanded abduction "
            f"{max(r.commanded_abduction for r in tables[n].rows):.3f} rad; ungated runs: "
            f"{', '.join(r.run for r in tables[n].rows if r.excluded) or 'none'}"
            for n in innermost
        ),
        "",
        "## Infeasible presets",
        "",
        *(
            f"- {n}: {', '.join(r.run for r in t.rows if not r.feasible) or 'none'}"
            for n, t in tables.items()
        ),
        "",
        "## Known limitations hit",
        "",
    ]
    hits = []
    for n, t in tables.items():
        items = []
        if t.unobservable:
            items.append(f"unobservable fields {', '.join(t.unobservable)}")
        if t.skewed:
            items.append(f"skewed roles {', '.join(t.skewed)}")
        if t.half_bound:
            items.append(f"half-bound posture groups {', '.join(t.half_bound)}")
        failing = sorted({f for r in t.rows for f in r.failing})
        if failing:
            items.append(f"fields beyond tolerance on some run: {', '.join(failing)}")
        if items:
            hits.append(f"- {n}: " + "; ".join(items))
    lines += hits or ["- none"]
    lines += ["", "## Not run", ""]
    lines += [f"- {n}: {why}" for n, why in not_run.items()] or ["- none"]
    # a rig is its file: several tables of one file (scale checks) count once
    sources = {t.source for t in tables.values()}
    skewed = {t.source for t in tables.values() if t.skewed}
    lines += [
        "",
        (
            f"{len(skewed)} of {len(sources)} rigs run hit `skewed` (a role whose hinge axis "
            "purity is below the mapper's minimum, left unbound or bound with a warning)"
            + (f": {', '.join(sorted(skewed))}." if skewed else ".")
        ),
    ]
    if notes:
        lines += ["", "## Findings", "", *(f"- {n}" for n in notes)]
    return "\n".join(lines) + "\n"
