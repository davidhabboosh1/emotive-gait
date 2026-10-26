"""``emotive-gait rig`` (``inspect``, ``walk``, ``validate``) and ``serve`` (spec 4.10).

Each subcommand builds one plain-data dictionary (``--json`` prints it as is) and the text
output prints the same dictionary, so the two cannot drift. Exit codes: 2 for user-input
errors, 1 for environment failures; no traceback escapes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from emotive_gait.cli_common import (
    _add_model_arguments,
    _err,
    _load_learned,
    _parse_emotions,
    _warn_learned_cells,
    warn_infeasible,
    warn_unsourced,
)
from emotive_gait.emotion.evidence import Emotion
from emotive_gait.emotion.model import ANALYTIC
from emotive_gait.emotion.prompt import PromptResult, parse_emotion_prompt
from emotive_gait.mocap.writers.bvh import write_bvh
from emotive_gait.rig.errors import RigError
from emotive_gait.rig.synthesis import (
    RELATIVE_TOLERANCES,
    TOLERANCES,
    LoadedRig,
    WalkResult,
    load_rig,
    walk,
    within_tolerance,
)

# the keys ``inspect --json`` adds under "mapping" for the reader; a mapping file may keep them
_INFORMATIONAL = ("family", "scale", "up_axis", "leg_length", "decisions", "warnings", "skewed")
_VALIDATE_RUNS = (
    ("neutral", {}),
    ("sad", {Emotion.SADNESS: 1.0}),
    ("angry", {Emotion.ANGER: 1.0}),
    ("happy", {Emotion.HAPPINESS: 1.0}),
    ("fear", {Emotion.FEAR: 1.0}),
)
_OK = {True: "ok", False: "NO", None: "-"}


# --- parsers -------------------------------------------------------------------------------


def _check_positive_finite(value: object, name: str) -> float:
    """``value`` as a positive finite float, else ``ValueError`` naming ``name`` and the value."""
    try:
        if isinstance(value, bool):
            raise TypeError
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}") from None
    if not (math.isfinite(number) and number > 0):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return number


def _positive_finite(text: str) -> float:
    try:
        return _check_positive_finite(text, "value")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _rig_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("source", metavar="IN.bvh", help="the rig's BVH walk")
    p.add_argument(
        "--scale", type=_positive_finite, default=None, help="metres per file unit (inferred)"
    )
    p.add_argument("--up", choices=["y", "z"], default=None, help="the file's up axis (inferred)")
    p.add_argument("--mapping", default=None, metavar="MAP.yaml", help="landmark/role overrides")
    p.add_argument("--json", action="store_true", help="print one JSON object instead of text")


def add_rig_parser(sub) -> None:
    """Register the ``rig`` group (``inspect``, ``walk``, ``validate``) on ``sub``."""
    rig = sub.add_parser("rig", help="emotion for any rig: inspect, walk or validate a BVH")
    rsub = rig.add_subparsers(dest="rig_command", required=True)

    i = rsub.add_parser("inspect", help="print the mapping decisions and the rig's neutral gait")
    _rig_arguments(i)

    w = rsub.add_parser("walk", help="write the rig's walk moved by an emotion")
    _rig_arguments(w)
    w.add_argument("--out", required=True, metavar="OUT.bvh", help="output BVH path")
    w.add_argument(
        "--emotion",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="emotion intensity in [0, 1]; repeatable (or use --prompt)",
    )
    w.add_argument("--prompt", default=None, help="free-text emotion, through the lexicon")
    w.add_argument(
        "--duration", type=_positive_finite, default=None, help="seconds (default: the source's)"
    )
    _add_model_arguments(w)

    v = rsub.add_parser("validate", help="walk zero and the four presets and tabulate the claims")
    _rig_arguments(v)


def add_serve_parser(sub) -> None:
    """Register ``serve`` (the MCP server) on ``sub``."""
    sub.add_parser("serve", help="run the MCP server (needs the mcp extra)")


# --- data ----------------------------------------------------------------------------------


def _clean(value: Any) -> Any:
    """JSON-able: enums as values, numpy scalars as Python, NaN and infinity as ``None``."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {_clean(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_mapping(path: str | None) -> tuple[dict[str, str] | None, dict[str, Any]]:
    """``(overrides, informational)`` from a ``landmarks:``/``roles:`` YAML file.

    ``overrides`` is the flat dict ``load_rig`` takes (role keys prefixed ``role:``);
    ``informational`` holds the keys ``inspect --json`` adds (``scale``, ``up_axis``, ...).
    """
    if path is None:
        return None, {}
    import yaml

    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise RigError(f"cannot read the mapping file {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise RigError(f"{path} is not valid YAML: {exc}") from exc
    return mapping_overrides_from(data, path)


def mapping_overrides_from(
    data: Any, where: str = "mapping"
) -> tuple[dict[str, str], dict[str, Any]]:
    """``(overrides, informational)`` from an already parsed ``landmarks:``/``roles:`` mapping.

    Shared by the file reader and the MCP server's ``mapping`` argument; ``where`` names the
    source in error messages. Raises ``RigError`` on a malformed mapping.
    """
    path = where
    if not isinstance(data, dict):
        raise RigError(f"{path}: expected the keys 'landmarks' and 'roles'")
    unknown = sorted(str(k) for k in set(data) - {"landmarks", "roles", *_INFORMATIONAL})
    if unknown:
        raise RigError(f"{path}: unknown key(s) {', '.join(unknown)}; use landmarks and roles")
    merged: dict[str, str] = {}
    for section, prefix in (("landmarks", ""), ("roles", "role:")):
        entries = data.get(section) or {}
        if not isinstance(entries, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in entries.items()
        ):
            raise RigError(f"{path}: '{section}' must map names to bone names")
        merged.update({prefix + k: v for k, v in entries.items()})
    return merged, {k: data[k] for k in _INFORMATIONAL if k in data}


def resolve_scale_up(
    info: Mapping[str, Any], scale: float | None, up_axis: str | None
) -> tuple[float | None, str | None]:
    """``scale`` and ``up_axis`` with a mapping's values as defaults; explicit values win.

    Raises ``RigError`` on an invalid mapping-side ``scale`` or ``up_axis``.
    """
    try:
        file_scale = (
            _check_positive_finite(info["scale"], "mapping file scale") if "scale" in info else None
        )
    except ValueError as exc:
        raise RigError(str(exc)) from None
    file_up = info.get("up_axis")
    if file_up is not None and file_up not in ("y", "z"):
        raise RigError(f"mapping file up_axis must be 'y' or 'z', got {file_up!r}")
    return (
        scale if scale is not None else file_scale,
        up_axis if up_axis is not None else file_up,
    )


def _load(args: argparse.Namespace) -> LoadedRig:
    source = Path(args.source)
    if not source.is_file():
        raise RigError(f"{source}: no such file")
    overrides, info = _read_mapping(args.mapping)
    # the file's values act as if given on the command line; explicit flags win
    scale, up_axis = resolve_scale_up(info, args.scale, args.up)
    ignored = [k for k in info if k not in ("scale", "up_axis")]
    if ignored:
        _err(f"ignoring informational mapping keys: {', '.join(ignored)}")
    return load_rig(source, scale=scale, up_axis=up_axis, mapping_overrides=overrides)


def _params(p) -> dict[str, float]:
    return {f.name: getattr(p, f.name) for f in dataclasses.fields(p)}


def inspect_data(rig: LoadedRig) -> dict:
    """What ``inspect`` reports: the mapping (in the ``--mapping`` file's shape), ``P0``, cycles."""
    m, s = rig.mapping, rig.summary
    return {
        "mapping": {
            "landmarks": {lm.value: bone for lm, bone in m.landmarks.items()},
            "roles": {role.value: b.hinge for role, b in m.roles.items()},
            "family": m.family,
            "scale": m.scale,
            "up_axis": m.up_axis,
            "leg_length": m.leg_length,
            "decisions": [dataclasses.asdict(d) for d in m.decisions],
            "warnings": list(m.warnings),
            "skewed": [r.value for r in m.skewed],
        },
        "source_pelvis_height": rig.source_pelvis_height,
        "nominal_pelvis_height": rig.nominal_pelvis_height,
        "pelvis_factor": rig.nominal_pelvis_height / rig.source_pelvis_height,
        "p0": _params(rig.p0),
        "raw_double_support": rig.raw_double_support,
        "unobservable": list(rig.unobservable),
        "cycles": {
            "used": s.n_cycles,
            "discarded": s.n_discarded,
            "discard_reasons": dict(s.discard_reasons),
        },
    }


def walk_data(result: WalkResult, prompt: tuple[str, PromptResult] | None, out: str) -> dict:
    """What ``walk`` reports: ``inspect_data`` plus the intensities, ``P1`` and the claims."""
    data = inspect_data(result.rig)
    f, s, rt = result.feasibility, result.slip, result.round_trip
    ok = within_tolerance(rt)
    data.update(
        {
            "intensities": {e.value: v for e, v in result.intensities.items()},
            "intensities_source": result.intensities_source,
            "model": result.model_name,
            "scaling": result.scaling,
            "p1": _params(result.p1),
            "unexpressible": [
                {"emotion": e.value, "parameter": p} for e, p in result.unexpressible
            ],
            "feasibility": {
                "feasible": f.feasible,
                "max_foot_position_error_mm": f.max_foot_position_error_m * 1000.0,
                "roles_at_limit": [r.value for r in f.roles_at_limit],
            },
            "slip": {
                "max_travel_mm": s.max_travel_m * 1000.0,
                "max_height_error_mm": s.max_height_error_m * 1000.0,
                "stances": s.stances,
            },
            "round_trip": {
                "residuals": [
                    {"field": n, "residual": r, "ok": ok.get(n)} for n, r in rt.residuals.items()
                ],
                "unobservable": list(rt.unobservable),
            },
            "out": out,
        }
    )
    if prompt is not None:
        text, parsed = prompt
        data["prompt"] = {
            "text": text,
            "matched": [
                {"term": t, "emotion": e.value, "contribution": c} for t, e, c in parsed.matched
            ],
            "unmatched": list(parsed.unmatched),
        }
    return data


def _worst(result: WalkResult) -> tuple[str, float, bool]:
    """The residual furthest from its tolerance: ``(field, residual, all within)``."""
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
    return worst[0], worst[1], all(within_tolerance(rt).values())


def validate_data(rig: LoadedRig) -> dict:
    """One row per run (zero and the four presets) on one loaded rig."""
    runs = []
    for name, intensities in _VALIDATE_RUNS:
        result = walk(
            rig,
            intensities=intensities,
            intensities_source="flags",
            model=ANALYTIC,
            model_name="analytic",
        )
        field, residual, within = _worst(result)
        runs.append(
            {
                "run": name,
                "feasible": result.feasibility.feasible,
                "worst_field": field,
                "worst_residual": residual,
                "within": within,
                "slip_mm": result.slip.max_travel_m * 1000.0,
                "height_mm": result.slip.max_height_error_m * 1000.0,
            }
        )
    return {"runs": runs}


# --- text ----------------------------------------------------------------------------------


def _table(rows: Sequence[Sequence[str]]) -> None:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip())


def _num(x: float | None, spec: str = "+.4f") -> str:
    return "nan" if x is None else format(x, spec)


def _print_header(d: dict) -> None:
    m, c = d["mapping"], d["cycles"]
    print(f"family: {m['family'] or 'unknown'}")
    print(f"scale: {m['scale']:g} m/unit  up-axis: {m['up_axis']}")
    print(f"leg length: {m['leg_length']:.3f} m")
    print(
        f"pelvis height: source {d['source_pelvis_height']:.3f} m, nominal "
        f"{d['nominal_pelvis_height']:.3f} m (x {d['pelvis_factor']:.2f})"
    )
    reasons = f" {c['discard_reasons']}" if c["discard_reasons"] else ""
    print(f"cycles: {c['used']} used, {c['discarded']} discarded{reasons}")
    print("decisions:")
    rows = [("what", "chosen", "source", "note")]
    rows += [(x["what"], x["chosen"], x["source"], x["note"]) for x in m["decisions"]]
    _table(rows)
    for w in m["warnings"]:
        _err(f"warning: {w}")
    if m["skewed"]:
        print(f"skewed roles: {', '.join(m['skewed'])}")


def _print_inspect(d: dict) -> None:
    _print_header(d)
    print("P0:")
    for name, v in d["p0"].items():
        extra = f"  (raw {d['raw_double_support']:.4f})" if name == "double_support_frac" else ""
        extra += "  unobservable" if name in d["unobservable"] else ""
        print(f"  {name:<22}{v: .4f}{extra}")


def _print_walk(d: dict) -> None:
    _print_header(d)
    print(f"model: {d['model']} ({d['scaling']})")
    given = ", ".join(f"{e}={v:g}" for e, v in d["intensities"].items()) or "none"
    print(f"intensities ({d['intensities_source']}): {given}")
    if "prompt" in d:
        p = d["prompt"]
        print("matched: " + ", ".join(f"{x['term']} ({x['emotion']})" for x in p["matched"]))
        if p["unmatched"]:
            print("unmatched: " + " ".join(p["unmatched"]))
    dropped = ", ".join(f"{u['emotion']}.{u['parameter']}" for u in d["unexpressible"])
    print(f"unexpressible: {dropped or 'none'}")
    print(f"{'parameter':<22}{'P0':>10}{'P1':>10}")
    for name, v1 in d["p1"].items():
        raw = f"  (raw {d['raw_double_support']:.4f})" if name == "double_support_frac" else ""
        print(f"{name:<22}{d['p0'][name]:>10.4f}{v1:>10.4f}{raw}")
    s = d["slip"]
    print(
        f"slip: max travel {s['max_travel_mm']:.1f} mm, max height error "
        f"{s['max_height_error_mm']:.1f} mm over {s['stances']} stances"
    )
    print("round trip:")
    rows = [("field", "residual", "ok")]
    rows += [(r["field"], _num(r["residual"]), _OK[r["ok"]]) for r in d["round_trip"]["residuals"]]
    _table(rows)
    if d["round_trip"]["unobservable"]:
        print("unobservable: " + ", ".join(d["round_trip"]["unobservable"]))
    print(f"wrote {d['out']}")


def _print_validate(d: dict) -> None:
    rows = [("run", "feasible", "max|residual| field", "round-trip within", "slip_mm", "height_mm")]
    for r in d["runs"]:
        rows.append(
            (
                r["run"],
                "yes" if r["feasible"] else "NO",
                f"{_num(r['worst_residual'])} {r['worst_field']}",
                "yes" if r["within"] else "NO",
                f"{r['slip_mm']:.1f}",
                f"{r['height_mm']:.1f}",
            )
        )
    _table(rows)


# --- commands ------------------------------------------------------------------------------


def _intensities(args: argparse.Namespace):
    """``(intensities, source, (prompt, PromptResult) | None)``; ``ValueError`` on bad input."""
    if bool(args.emotion) == (args.prompt is not None):
        raise ValueError("give exactly one of --emotion and --prompt")
    if args.prompt is None:
        return _parse_emotions(args.emotion), "flags", None
    parsed = parse_emotion_prompt(args.prompt)
    if not parsed.matched:
        raise ValueError(
            f"no emotion word found in the prompt {args.prompt!r}; unmatched: "
            + (", ".join(parsed.unmatched) or "(empty)")
        )
    return parsed.intensities, "lexicon", (args.prompt, parsed)


def _walk_command(args: argparse.Namespace) -> int:
    intensities, source, prompt = _intensities(args)
    warn_unsourced(intensities)
    model = ANALYTIC
    if args.model == "learned":
        model = _load_learned(args.scaling)
        if model is None:
            return 1
        _warn_learned_cells(model, intensities)
    rig = _load(args)
    result = walk(
        rig,
        intensities=intensities,
        intensities_source=source,
        model=model,
        model_name=args.model,
        scaling=args.scaling,
        duration_s=args.duration,
    )
    warn_infeasible(result.feasibility)
    out = Path(args.out)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        write_bvh(result.output, out)
    except OSError as exc:
        _err(f"environment failure: {exc}")
        return 1
    _emit(args, walk_data(result, prompt, str(out)), _print_walk)
    return 0


def _emit(args: argparse.Namespace, data: dict, printer) -> None:
    if args.json:
        print(json.dumps(_clean(data), indent=2))
    else:
        printer(data)


def run_rig(args: argparse.Namespace) -> int:
    """Run a ``rig`` subcommand. Exit codes: 0, 1 environment failure, 2 user-input error."""
    try:
        if args.rig_command == "walk":
            return _walk_command(args)
        rig = _load(args)
        if args.rig_command == "inspect":
            _emit(args, inspect_data(rig), _print_inspect)
        else:
            _emit(args, validate_data(rig), _print_validate)
    except ValueError as exc:  # RigError, MocapFormatError, IntensityError, bad prompt, ...
        _err(str(exc))
        return 2
    except OSError as exc:
        _err(f"environment failure: {exc}")
        return 1
    return 0


def run_serve(args: argparse.Namespace) -> int:
    """Run the MCP server; 1 with the install hint when the ``mcp`` extra is missing."""
    try:
        import mcp  # noqa: F401
    except ImportError:
        _err("the MCP server needs the mcp extra: pip install -e '.[mcp]'")
        return 1
    from emotive_gait.rig.server import serve

    serve()
    return 0
