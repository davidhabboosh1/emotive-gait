"""Validate one rig: zero emotion and the four analytic presets, tabulated (spec 4.7, tier 3).

``validate_rig`` loads a BVH walk and runs ``neutral`` plus ``sad``, ``angry``, ``happy`` and
``fear`` at intensity 1.0 through ``walk`` (the analytic model), one ``ValidationRow`` each;
``render_report`` turns the tables of several rigs into the Markdown of
``docs/reports/phase-4-6-rig-validation.md``. Nothing is gated here: infeasible presets and
residuals beyond tolerance are reported as measured. Every run is written for at least
``MIN_STEPS`` steps at its own cadence; a run whose output still holds no complete cycle is
reported unmeasured, and a run that raises as an ``error:`` row. The report carries aggregates
only (no clip or frame data).

The committed report is regenerated from its spec (rigs, not-run entries, authored findings),
from the repository root: ``python -m emotive_gait.rig.validate SPEC.yaml --out REPORT.md``
with ``docs/reports/phase-4-6-rig-validation.yaml`` and ``.md``.
"""

from __future__ import annotations

import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from emotive_gait.emotion.evidence import Emotion
from emotive_gait.emotion.model import ANALYTIC, modulate
from emotive_gait.gait.params import GaitParameters
from emotive_gait.mocap.readers.bvh import BvhFile
from emotive_gait.rig.calibration import POSTURE_GROUPS
from emotive_gait.rig.mapping import RigMapping
from emotive_gait.rig.scaled import RigScaledModel
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

# Every run writes at least this many steps (four gait cycles) at its own P1 cadence, so a short
# source clip slowed by a preset still holds a complete cycle for the extractor.
MIN_STEPS = 8
MEASURED = "measured"
UNMEASURED = "unmeasured: no complete cycle"

# The zero-slip claim on feasible runs (the CI bound): worst stance-foot travel and height error.
# Feasibility is a per-knot 2 mm position threshold, while slip is a frame-to-frame measure over
# the spline-interpolated stance, so a feasible run can slip up to about twice the threshold.
SLIP_BOUND_M = 0.002


@dataclass(frozen=True)
class ValidationRow:
    """One run: feasibility, the residual per field, whether all are within tolerance, slip."""

    run: str
    feasible: bool | None  # None: the run raised
    max_residual: Mapping[str, float]  # field -> residual (measured - commanded)
    within: bool | None  # every gated field within its tolerance; None: not measured
    slip_m: float  # worst horizontal stance-foot travel
    height_m: float  # worst stance-foot height error
    failing: tuple[str, ...] = ()  # gated fields beyond tolerance
    excluded: tuple[str, ...] = ()  # fields left ungated by the pitch-innermost exclusion
    worst_field: str = "-"  # the residual furthest from its tolerance (as a ratio)
    worst_residual: float = math.nan
    commanded_abduction: float = math.nan  # expansiveness * arm_abduction of P1 (rad)
    duration_s: float = math.nan  # length of the written output
    status: str = "measured"  # MEASURED, UNMEASURED or "error: <message>"
    # travel and height within SLIP_BOUND_M; None on an infeasible, unmeasured or error run
    slip_within: bool | None = None

    @property
    def measured(self) -> bool:
        return self.status == MEASURED


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
    source_duration_s: float
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


def validation_row(name: str, result: WalkResult, duration_s: float) -> ValidationRow:
    """The row of one run's ``WalkResult`` written over ``duration_s``; a run whose output holds
    no complete cycle is ``UNMEASURED`` (no ``within``, no failing fields). ``slip_within`` is
    gated on feasible measured runs only (``SLIP_BOUND_M`` on travel and height)."""
    common = {
        "run": name,
        "feasible": bool(result.feasibility.feasible),
        "max_residual": MappingProxyType(dict(result.round_trip.residuals)),
        "slip_m": float(result.slip.max_travel_m),
        "height_m": float(result.slip.max_height_error_m),
        "commanded_abduction": float(result.p1.expansiveness * result.p1.arm_abduction),
        "duration_s": float(duration_s),
    }
    if result.round_trip.measured.n_cycles == 0:
        return ValidationRow(**common, within=None, status=UNMEASURED)
    if common["feasible"]:
        common["slip_within"] = (
            common["slip_m"] <= SLIP_BOUND_M and common["height_m"] <= SLIP_BOUND_M
        )
    excluded = pitch_innermost_exclusion(result.rig.mapping, result.rig.file, result.p1)
    ok = within_tolerance(result.round_trip)
    failing = tuple(k for k, v in ok.items() if not v and k not in excluded)
    field, residual = worst_field(result)
    return ValidationRow(
        **common,
        within=not failing,
        failing=failing,
        excluded=excluded,
        worst_field=field,
        worst_residual=float(residual),
    )


def _error_row(name: str, exc: Exception) -> ValidationRow:
    return ValidationRow(
        run=name,
        feasible=None,
        max_residual=MappingProxyType({}),
        within=None,
        slip_m=math.nan,
        height_m=math.nan,
        status=f"error: {exc}",
    )


def run_duration(rig: LoadedRig, p1: GaitParameters) -> float:
    """The source's length, or ``MIN_STEPS`` steps at ``p1``'s cadence when that is longer."""
    source = rig.file.motion.shape[0] * rig.file.frame_time
    return max(source, MIN_STEPS / p1.cadence)


def validate_loaded(
    rig: LoadedRig, *, source: str, scale_given: bool, up_given: bool
) -> ValidationTable:
    """Zero emotion and the four presets on an already loaded ``rig``, each written for
    ``run_duration``. A run that raises becomes an ``error: <message>`` row."""
    rows = []
    for name, intensities in RUNS:
        try:
            # the P1 ``walk`` computes (the analytic model at walk's default froude scaling)
            scaled = RigScaledModel(ANALYTIC, scaling="froude", p0=rig.p0)
            duration = run_duration(rig, modulate(rig.p0, intensities, rig.profile, model=scaled))
            result = walk(
                rig,
                intensities=intensities,
                intensities_source="flags",
                model=ANALYTIC,
                model_name="analytic",
                duration_s=duration,
            )
            rows.append(validation_row(name, result, duration))
        # keep the rig's other runs; the row names the failure (RigError is a ValueError)
        except (ValueError, ArithmeticError, RuntimeError, OSError) as exc:
            rows.append(_error_row(name, exc))
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
        source_duration_s=float(rig.file.motion.shape[0] * rig.file.frame_time),
        skewed=tuple(r.value for r in m.skewed),
        unobservable=tuple(rig.unobservable),
        half_bound=half,
        pitch_innermost=pitch_innermost(m, rig.file),
        rows=tuple(rows),
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


_YES = {True: "yes", False: "NO", None: "n/a"}


def _commit_line(commit: str) -> str:
    if commit.endswith("-dirty"):
        base = commit[: -len("-dirty")]
        return (
            f"Commit: generated on uncommitted changes on top of `{base}`; the tree measured is "
            "that of the commit that adds this report."
        )
    return f"Commit: `{commit}`."


def _fmt(x: float, spec: str = "+.4f") -> str:
    return format(x, spec) if math.isfinite(x) else "nan"


def _largest(values) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return max(finite) if finite else math.nan


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
            f"- source length {t.source_duration_s:.2f} s; written per run (at least "
            f"{MIN_STEPS} steps at the run's cadence): "
            + ", ".join(f"{r.run} {_fmt(r.duration_s, '.2f')} s" for r in t.rows)
        ),
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
            "| height mm | slip within |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in t.rows:
        if r.measured:
            failing = ", ".join(r.failing) or "none"
            if r.excluded:
                failing += f" (ungated: {', '.join(r.excluded)})"
            worst = f"{r.worst_field} ({_fmt(r.worst_residual)})"
        else:
            failing, worst = r.status, "-"
        lines.append(
            f"| {r.run} | {_YES[r.feasible]} | {worst} | {_YES[r.within]} | {failing} | "
            f"{_fmt(r.slip_m * 1000, '.1f')} | {_fmt(r.height_m * 1000, '.1f')} | "
            f"{_YES[r.slip_within]} |"
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
        cells = [_fmt(r.max_residual.get(f, math.nan)) if r.measured else "-" for r in t.rows]
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
        _commit_line(commit or _commit()),
        "",
        (
            "Each rig is walked at zero emotion (`neutral`) and at the analytic presets' "
            "intensities (`sad`, `angry`, `happy`, `fear`, each 1.0); the output is written in the "
            "rig's own channels, read back and measured by the frozen extractor. Infeasible presets "
            "and residuals beyond tolerance are reported, not gated. Aggregates only: no clip or "
            "frame data."
        ),
        "",
        (
            f"Slip within: worst stance-foot travel and height error both at most "
            f"{SLIP_BOUND_M * 1000:g} mm, gated on feasible measured runs only (n/a otherwise)."
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
            f"{_largest(r.commanded_abduction for r in tables[n].rows):.3f} rad; ungated runs: "
            f"{', '.join(r.run for r in tables[n].rows if r.excluded) or 'none'}"
            for n in innermost
        ),
        "",
        "## Infeasible presets",
        "",
        *(
            f"- {n}: {', '.join(r.run for r in t.rows if r.feasible is False) or 'none'}"
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
        slipped = [
            f"{r.run} ({_fmt(r.slip_m * 1000, '.1f')} mm travel, "
            f"{_fmt(r.height_m * 1000, '.1f')} mm height)"
            for r in t.rows
            if r.slip_within is False
        ]
        if slipped:
            items.append(
                f"slip beyond {SLIP_BOUND_M * 1000:g} mm on feasible runs: {', '.join(slipped)}"
            )
        other = [f"{r.run} ({r.status})" for r in t.rows if not r.measured]
        if other:
            items.append(f"runs not measured: {', '.join(other)}")
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


# --- the committed report, regenerated from its spec ---------------------------------------


@dataclass(frozen=True)
class RigRun:
    """One table of the report: a BVH path (relative to the working directory) and flags."""

    name: str
    path: str
    scale: float | None = None
    up_axis: str | None = None

    def command(self) -> str:
        flags = "" if self.scale is None else f" --scale {self.scale:g}"
        flags += "" if self.up_axis is None else f" --up {self.up_axis}"
        return f"emotive-gait rig validate {self.path}{flags}"


@dataclass(frozen=True)
class ReportSpec:
    """``docs/reports/phase-4-6-rig-validation.yaml``: which rigs, what was not run and why,
    the setup commands and the authored findings; the numbers are recomputed."""

    rigs: tuple[RigRun, ...]
    not_run: Mapping[str, str]
    setup_commands: tuple[str, ...] = ()
    findings: tuple[str, ...] = ()


class SpecError(ValueError):
    """The report spec is malformed; the message names the key."""


def load_spec(path: Path) -> ReportSpec:
    """Read a report spec; ``SpecError`` naming the key on a malformed file."""
    import yaml

    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise SpecError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise SpecError(f"{path}: expected a mapping")
    unknown = sorted(set(data) - {"rigs", "not_run", "setup_commands", "findings"})
    if unknown:
        raise SpecError(f"{path}: unknown key(s) {', '.join(unknown)}")
    rigs = []
    for i, raw in enumerate(data.get("rigs") or []):
        if not isinstance(raw, dict) or not {"name", "path"} <= set(raw):
            raise SpecError(f"{path}: rigs[{i}] needs name and path")
        extra = sorted(set(raw) - {"name", "path", "scale", "up_axis"})
        if extra:
            raise SpecError(f"{path}: rigs[{i}]: unknown key(s) {', '.join(extra)}")
        scale = raw.get("scale")
        if scale is not None and (not isinstance(scale, int | float) or scale <= 0):
            raise SpecError(f"{path}: rigs[{i}].scale must be a positive number")
        up = raw.get("up_axis")
        if up not in (None, "y", "z"):
            raise SpecError(f"{path}: rigs[{i}].up_axis must be y or z")
        rigs.append(RigRun(str(raw["name"]), str(raw["path"]), scale, up))
    not_run = {}
    for i, raw in enumerate(data.get("not_run") or []):
        if not isinstance(raw, dict) or set(raw) != {"name", "reason"}:
            raise SpecError(f"{path}: not_run[{i}] needs exactly name and reason")
        not_run[str(raw["name"])] = str(raw["reason"])
    return ReportSpec(
        rigs=tuple(rigs),
        not_run=MappingProxyType(not_run),
        setup_commands=tuple(str(c) for c in data.get("setup_commands") or ()),
        findings=tuple(" ".join(str(f).split()) for f in data.get("findings") or ()),
    )


def report_from_spec(spec: ReportSpec, *, command: str, progress=None) -> str:
    """Validate every rig of ``spec`` and render the report. A rig whose file is missing or
    that ``load_rig`` refuses goes to the not-run list with the measured reason."""
    tables: dict[str, ValidationTable] = {}
    not_run = dict(spec.not_run)
    for rig in spec.rigs:
        source = Path(rig.path)
        if not source.is_file():
            not_run[rig.name] = f"{rig.path} is not on disk (fetch it first)"
            continue
        try:
            tables[rig.name] = validate_rig(
                source, scale=rig.scale, up_axis=rig.up_axis, mapping_overrides=None
            )
        except ValueError as exc:  # RigError, MocapFormatError: the rig cannot be loaded
            not_run[rig.name] = f"refused: {exc}"
        if progress is not None:
            progress(rig.name)
    commands = [*spec.setup_commands, *(r.command() for r in spec.rigs), command]
    return render_report(tables, not_run=not_run, commands=commands, notes=spec.findings)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m emotive_gait.rig.validate SPEC.yaml --out REPORT.md`` (run from the
    repository root). Exit codes: 2 for a malformed spec, 1 for environment failures."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="python -m emotive_gait.rig.validate")
    parser.add_argument("spec", help="the report spec (YAML)")
    parser.add_argument("--out", required=True, help="the Markdown report to write")
    args = parser.parse_args(argv)
    command = f"python -m emotive_gait.rig.validate {args.spec} --out {args.out}"
    try:
        spec = load_spec(Path(args.spec))
        text = report_from_spec(
            spec, command=command, progress=lambda n: print(f"done {n}", file=sys.stderr)
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(text.encode("utf-8"))  # LF on every platform
    except OSError as exc:
        print(f"environment failure: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
