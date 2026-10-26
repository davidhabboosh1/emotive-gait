"""Command line: ``demo`` renders a stimulus, ``evidence`` lists claims, ``fit`` and ``fetch``
build the learned emotion model from the data manifest.

``cli`` imports ``learn/`` (and so ``pandas`` and ``statsmodels``) only inside the ``fit`` and
``fetch`` handlers: ``demo --model learned`` needs ``pyyaml`` alone.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from emotive_gait.emotion.evidence import EVIDENCE, UNSOURCED_EMOTIONS, Emotion
from emotive_gait.emotion.model import IntensityError, modulate
from emotive_gait.gait.params import GaitParameters, SpeedConstraintError
from emotive_gait.gait.scaling import MEASURED_FIELDS

_REPORT = "docs/reports/phase-4-5-fit-report.md"
_DERIVED = "data/derived"
_ROOTS = {"free": "FREE", "z-rail": "Z_RAIL", "xz-rail": "XZ_RAIL", "weld": "WELD"}
_DIRECTION = {1: "+", 0: "0", -1: "-"}


class _ModelRevisionError(Exception):
    """The installed robot model does not match the constants this package was derived for."""


def _err(msg: str) -> None:
    print(f"emotive-gait: {msg}", file=sys.stderr)


def _parse_emotions(specs: Sequence[str]) -> dict[Emotion, float]:
    names = ", ".join(e.value for e in Emotion)
    out: dict[Emotion, float] = {}
    for spec in specs:
        name, sep, raw = spec.partition("=")
        name = name.strip().lower()
        if not sep:
            raise ValueError(f"emotion {spec!r} must look like NAME=VALUE (e.g. {name}=0.8)")
        try:
            emotion = Emotion(name)
        except ValueError:
            raise ValueError(f"unknown emotion {name!r}; choose from {names}") from None
        try:
            out[emotion] = float(raw)
        except ValueError:
            raise ValueError(f"intensity for {name} is not a number: {raw!r}") from None
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="emotive-gait", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="modulate a gait by emotion intensities and render it")
    d.add_argument(
        "--emotion",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="emotion intensity in [0, 1]; repeatable (e.g. sadness=0.8)",
    )
    d.add_argument("--duration", type=float, default=6.0, help="seconds (default 6)")
    d.add_argument("--fps", type=int, default=30)
    d.add_argument("--width", type=int, default=960)
    d.add_argument("--height", type=int, default=720)
    d.add_argument("--out", default="demo.mp4", help="output video path (default demo.mp4)")
    d.add_argument(
        "--hold-speed",
        type=float,
        default=None,
        metavar="M_PER_S",
        help="pin walking speed so posture cues survive",
    )
    d.add_argument(
        "--root",
        choices=sorted(_ROOTS),
        default="z-rail",
        help="pelvis constraint for --physics (default z-rail; xz-rail (experimental: falls under stride))",
    )
    d.add_argument(
        "--camera",
        choices=["tracking", "fixed"],
        default="tracking",
        help="video camera: follow the pelvis, or stay fixed (default tracking)",
    )
    mode = d.add_mutually_exclusive_group()
    mode.add_argument(
        "--kinematic",
        dest="physics",
        action="store_false",
        default=False,
        help="pose the robot with no physics (default)",
    )
    mode.add_argument(
        "--physics",
        dest="physics",
        action="store_true",
        help="also run the open-loop physics simulation and print a summary",
    )
    d.add_argument("--no-video", action="store_true", help="skip rendering (with --physics)")
    d.add_argument("--gif", action="store_true", help="also convert the video to a GIF")
    d.add_argument(
        "--dry-run",
        action="store_true",
        help="print the modulated parameters and exit; loads no model",
    )
    _add_model_arguments(d)

    e = sub.add_parser("evidence", help="print every literature claim with its grounding")
    _add_model_arguments(e)

    f = sub.add_parser("fit", help="fit the learned emotion model from the data manifest")
    _add_data_arguments(f)
    f.add_argument("--datasets", nargs="+", metavar="ID", help="fit only these dataset ids")
    f.add_argument(
        "--include-uncommittable",
        action="store_true",
        help="include datasets whose numbers may not be committed (needs --out outside src/)",
    )
    f.add_argument(
        "--out", default=None, metavar="PATH", help="artefact path (default: the packaged one)"
    )
    f.add_argument(
        "--report",
        default=None,
        help=f"report path (default {_REPORT}; with --include-uncommittable, fit-report.md "
        "beside --out)",
    )
    f.add_argument("--derived", default=_DERIVED, help=f"per-cycle tables (default {_DERIVED})")
    f.add_argument(
        "--check",
        action="store_true",
        help="refit to a temporary file and compare every estimate with the artefact",
    )

    g = sub.add_parser("fetch", help="download and verify the manifest's data files")
    _add_data_arguments(g)
    g.add_argument("--datasets", nargs="+", metavar="ID", help="fetch only these dataset ids")
    g.add_argument(
        "--from",
        dest="from_paths",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="use a local copy instead of downloading (an archive, or a directory); repeatable",
    )
    return p


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        choices=["analytic", "learned"],
        default="analytic",
        help="emotion model (default analytic; learned needs the fitted artefact)",
    )
    parser.add_argument(
        "--scaling",
        choices=["froude", "ratio"],
        default="froude",
        help="re-dimensioning of the learned model's effects (default froude)",
    )


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default="data/raw", help="raw data folder (default data/raw)")
    parser.add_argument(
        "--manifest",
        default="data/manifest.yaml",
        help="data manifest (default data/manifest.yaml)",
    )


def _print_params(params: GaitParameters) -> None:
    for f in dataclasses.fields(params):
        print(f"{f.name:<22}{getattr(params, f.name): .4f}")
    print(f"{'speed (m/s)':<22}{params.speed: .4f}")


def _to_gif(video: Path) -> Path:
    from emotive_gait.sim.render import FfmpegUnavailableError

    if shutil.which("ffmpeg") is None:
        raise FfmpegUnavailableError(
            "ffmpeg was not found: install ffmpeg and make sure it is on PATH to write a GIF."
        )
    gif = video.with_suffix(".gif")
    vf = "fps=15,scale=480:-1:flags=lanczos"
    base = ["ffmpeg", "-y", "-loglevel", "error"]
    with tempfile.TemporaryDirectory() as tmp:
        palette = str(Path(tmp) / "palette.png")
        subprocess.run([*base, "-i", str(video), "-vf", f"{vf},palettegen", palette], check=True)
        subprocess.run(
            [
                *base,
                *("-i", str(video), "-i", palette),
                *("-lavfi", f"{vf} [x]; [x][1:v] paletteuse", str(gif)),
            ],
            check=True,
        )
    return gif


def _video_problem(args: argparse.Namespace, model) -> str | None:
    """Return a message for video settings the renderer or encoder cannot honour, else None."""
    max_w, max_h = model.vis.global_.offwidth, model.vis.global_.offheight
    if args.width > max_w:
        return f"--width {args.width} exceeds the offscreen framebuffer width {max_w}"
    if args.height > max_h:
        return f"--height {args.height} exceeds the offscreen framebuffer height {max_h}"
    if args.width % 2 or args.height % 2:
        return (
            f"--width and --height must be even for H.264/yuv420p video "
            f"(got {args.width}x{args.height})"
        )
    if int(args.duration * args.fps) < 1:
        return (
            f"--duration {args.duration} at --fps {args.fps} yields no frame; "
            "need duration * fps >= 1"
        )
    return None


def _demo(args: argparse.Namespace) -> int:
    try:
        intensities = _parse_emotions(args.emotion)
    except ValueError as exc:
        _err(str(exc))
        return 2
    for emotion, value in intensities.items():
        if emotion in UNSOURCED_EMOTIONS and math.isfinite(value) and value > 0:
            _err(
                f"warning: {emotion.value} is unsourced: no source reports its gait kinematics, "
                "so it is a flagged arousal/valence placement, not a measured effect"
            )

    from emotive_gait.robot.profiles.g1 import G1_PROFILE_WITH_NECK

    model = None
    if args.model == "learned":
        model = _load_learned(args.scaling)
        if model is None:
            return 1
    try:
        params = modulate(
            GaitParameters(),
            intensities,
            G1_PROFILE_WITH_NECK,
            model=model,
            hold_speed=args.hold_speed,
        )
    except (IntensityError, SpeedConstraintError) as exc:
        _err(str(exc))
        return 2
    if model is None:
        print("model: analytic")
    else:
        _warn_learned_cells(model, intensities)
        print(f"model: learned ({args.scaling})")
    if args.dry_run:
        _print_params(params)
        return 0
    if min(args.duration, args.fps, args.width, args.height) <= 0:
        _err("--duration, --fps, --width and --height must be positive")
        return 2

    from emotive_gait.gait.generator import generate
    from emotive_gait.robot.loader import ModelUnavailableError
    from emotive_gait.robot.loader import load_robot as _load_robot
    from emotive_gait.robot.root import RootMode
    from emotive_gait.sim.kinematic import play_kinematic
    from emotive_gait.sim.physical import simulate, summarize
    from emotive_gait.sim.render import (
        CameraSpec,
        FfmpegUnavailableError,
        render_frames,
        write_video,
    )

    def load_robot(**kwargs):
        try:
            return _load_robot(**kwargs)
        except (ValueError, KeyError) as exc:
            raise _ModelRevisionError(
                f"the installed robot model revision does not match this package ({exc}); "
                "install the pinned mujoco-menagerie==2026.10.4"
            ) from exc

    warned = False

    def make_traj(rob):
        nonlocal warned
        traj = generate(params, rob)
        if not warned:
            warned = True
            report = traj.feasibility
            if not report.feasible:
                roles = ", ".join(r.name.lower() for r in report.roles_at_limit) or "none"
                _err(
                    "warning: parameters exceed the robot's reach: max foot error "
                    f"{report.max_foot_position_error_m * 1000:.1f} mm; roles at limit: {roles}"
                )
        return traj

    try:
        video = not (args.physics and args.no_video)
        if video:
            robot = load_robot()  # the kinematic video needs the free-root model
            problem = _video_problem(args, robot.model)
            if problem is not None:
                _err(problem)
                return 2
        if args.physics:
            sim_robot = load_robot(root=RootMode[_ROOTS[args.root]])
            res = simulate(sim_robot, make_traj(sim_robot), duration=args.duration)
            print(f"root: {args.root}")
            print(f"fell: {res.fell}")
            print(f"fell_at: {res.fell_at if res.fell else 'n/a'}")
            if res.fell:
                print(f"fail_reason: {res.fail_reason}")
            rep = summarize(res)
            held = " (pelvis held)" if rep.pelvis_held else ""
            print(f"zmp_outside_frac: {rep.zmp_outside_frac:.4f}{held}")
            print(f"zmp_mean_margin_m: {rep.zmp_mean_margin_m:.4f}{held}")
            left, right = rep.foot_slip_rms_m_s
            print(f"foot_slip_rms_m_s: {left:.4f} {right:.4f}")
            print(f"pelvis_bob_m: {rep.pelvis_bob_m:.4f}")
            print(f"pelvis_tracking_ptp_m: {rep.pelvis_tracking_ptp_m:.4f}")
            print(f"pelvis_held: {rep.pelvis_held}")
            changed = [
                f"{f.name}={getattr(rep.played_params, f.name)}"
                for f in dataclasses.fields(params)
                if getattr(rep.played_params, f.name) != getattr(params, f.name)
            ]
            print(f"played_params: {' '.join(changed) if changed else 'unchanged'}")
            if args.no_video:
                return 0
        frames = render_frames(
            robot,
            play_kinematic(robot, make_traj(robot), duration=args.duration, fps=args.fps),
            width=args.width,
            height=args.height,
            camera=CameraSpec(tracking=args.camera == "tracking"),
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_video(frames, out, fps=args.fps, width=args.width, height=args.height)
        print(f"wrote {out}")
        if args.gif:
            print(f"wrote {_to_gif(out)}")
    except (ModelUnavailableError, FfmpegUnavailableError) as exc:
        _err(str(exc))
        return 1
    except _ModelRevisionError as exc:
        _err(str(exc))
        return 1
    except (subprocess.CalledProcessError, RuntimeError, OSError) as exc:
        _err(f"environment failure: {exc}")
        return 1
    return 0


def _load_learned(scaling: str):
    """The learned model on the packaged artefact, or ``None`` after printing why not."""
    from emotive_gait.emotion.learned import (
        ArtefactMissingError,
        ArtefactSchemaError,
        LearnedEmotionModel,
    )

    try:
        return LearnedEmotionModel.default(scaling=scaling)
    except (ArtefactMissingError, ArtefactSchemaError, ValueError) as exc:
        _err(str(exc))
        return None


def _warn_learned_cells(model, intensities: dict[Emotion, float]) -> None:
    """One warning for the thin cells and one for the delegated cells of the requested emotions."""
    asked = {e.value for e, v in intensities.items() if v > 0}
    thin = [
        f"{e}.{p} ({model.cell(e, p).n_actors} actors)" for e, p in model.thin_cells if e in asked
    ]
    if thin:
        _err(
            "warning: thin cells (one or two actors, no interval) in the learned model: "
            + ", ".join(thin)
        )
    delegated: dict[str, list[str]] = {}
    for e, p in model.analytic_cells:
        if e in asked:
            delegated.setdefault(e, []).append(p)
    if delegated:
        parts = [
            f"{e} (all)" if len(ps) == len(MEASURED_FIELDS) else f"{e}.{', '.join(ps)}"
            for e, ps in sorted(delegated.items())
        ]
        _err("warning: cells taken from the analytic model, not measured: " + "; ".join(parts))


def _learned_columns(cell) -> list[str]:
    if cell is None:
        return ["-", "-", "-"]
    if cell.grounding == "missing":
        return ["-", "-", cell.direction_check]
    if cell.grounding == "fitted":
        interval = f"[{cell.ci_low:.4f}, {cell.ci_high:.4f}]"
    else:
        interval = f"{cell.n_actors} actors"
    return [f"{cell.estimate:+.4f}", interval, cell.direction_check]


def _evidence(args: argparse.Namespace) -> int:
    artefact = None
    if args.model == "learned":
        from emotive_gait.emotion import learned

        try:
            artefact = learned.load_artefact(learned.default_artefact_path())
        except (learned.ArtefactMissingError, learned.ArtefactSchemaError) as exc:
            _err(str(exc))
            return 1
    head = ["EMOTION", "PARAMETER", "DIR", "GROUNDING"]
    if artefact is not None:
        head += ["ESTIMATE", "CI", "CHECK"]
    rows = [(*head, "CITATION")]
    last = None
    for c in EVIDENCE:
        label = c.emotion.value if c.emotion is not last else ""
        last = c.emotion
        row = [label, c.parameter, _DIRECTION[c.direction], c.grounding.value.upper()]
        if artefact is not None:
            row += _learned_columns(
                artefact.cells.get((args.scaling, c.emotion.value, c.parameter))
            )
        rows.append((*row, c.citation))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]) - 1)]
    for r in rows:
        print("  ".join(r[i].ljust(widths[i]) for i in range(len(widths))) + "  " + r[-1])
    return 0


def _fit_import():
    """``learn.pipeline`` or ``None`` after printing the install hint (fit extra missing)."""
    try:
        from emotive_gait.learn import pipeline
    except ImportError as exc:
        _err(
            f"the fit pipeline needs {exc.name or 'a fit dependency'}; "
            'install the fit extra with: pip install -e ".[fit]"'
        )
        return None
    return pipeline


def _fit(args: argparse.Namespace) -> int:
    from emotive_gait.emotion.learned import (
        ArtefactMissingError,
        ArtefactSchemaError,
        default_artefact_path,
    )

    pipeline = _fit_import()
    if pipeline is None:
        return 1
    if args.include_uncommittable and args.out is None:
        _err("--include-uncommittable needs --out PATH outside src/ (its numbers stay local)")
        return 2
    target = Path(args.out) if args.out else default_artefact_path()
    if args.report:
        report = Path(args.report)
    elif args.include_uncommittable:
        report = target.parent / "fit-report.md"
    else:
        report = Path(_REPORT)
    common = {"datasets": args.datasets, "include_uncommittable": args.include_uncommittable}
    try:
        if args.check:
            return _fit_check(pipeline, args, target, common)
        summary = pipeline.run_fit(
            Path(args.manifest),
            Path(args.root),
            out_artefact=target,
            out_report=report,
            derived=Path(args.derived),
            **common,
        )
        from emotive_gait.learn.console import print_summary

        print_summary(summary)
        _report_feasibility(target, report)
    except pipeline.FitInputError as exc:
        _err(str(exc))
        return 2
    except (pipeline.FitEnvironmentError, ArtefactMissingError, ArtefactSchemaError) as exc:
        _err(str(exc))
        return 1
    except (OSError, ValueError) as exc:
        _err(f"environment failure: {exc}")
        return 1
    return 0


def _feasibility_row(preset: str, scaling: str, params, artefact, robot) -> dict:
    """One row for ``learn.report.append_feasibility``: the IK verdict and the driving cells."""
    from emotive_gait.emotion.evidence import Emotion
    from emotive_gait.gait.generator import generate
    from emotive_gait.gait.params import GaitParameters
    from emotive_gait.gait.scaling import MEASURED_FIELDS

    rep = generate(params, robot).feasibility
    emotion = {"sad": Emotion.SADNESS, "angry": Emotion.ANGER, "happy": Emotion.HAPPINESS}.get(
        preset
    )
    driving = []
    if not rep.feasible and emotion is not None:
        neutral = GaitParameters()
        for name in MEASURED_FIELDS:
            if getattr(params, name) == getattr(neutral, name):
                continue
            trial = dataclasses.replace(params, **{name: getattr(neutral, name)})
            if generate(trial, robot).feasibility.feasible:
                cell = artefact.cells.get((scaling, emotion.value, name))
                driving.append(
                    {
                        "parameter": name,
                        "value": getattr(params, name),
                        "estimate": None if cell is None else cell.estimate,
                        "per_dataset": {} if cell is None else dict(cell.per_dataset),
                    }
                )
    return {
        "preset": preset,
        "scaling": scaling,
        "feasible": rep.feasible,
        "position_error_mm": rep.max_foot_position_error_m * 1000.0,
        "orientation_error_rad": rep.max_foot_orientation_error_rad,
        "roles_at_limit": [r.value for r in rep.roles_at_limit],
        "stride": {
            "step_length": params.step_length,
            "cadence": params.cadence,
            "double_support_frac": params.double_support_frac,
        },
        "driving": driving,
    }


def _report_feasibility(artefact_path: Path, report: Path) -> None:
    """Run the G1 IK on the learned presets and append the result to the report (spec 4.5).

    Learned presets are flagged and rendered, not gated (decision 2026-10-26). Skipped with a
    note when the robot model cannot be loaded; the fit's exit code does not change.
    """
    from emotive_gait.emotion.learned import SCALINGS, LearnedEmotionModel, load_artefact
    from emotive_gait.emotion.model import learned_presets
    from emotive_gait.learn.report import append_feasibility

    try:
        from emotive_gait.robot.loader import ModelUnavailableError, load_robot

        robot = load_robot()
    except (ImportError, OSError, ModelUnavailableError) as exc:
        print(f"feasibility of the learned presets skipped (robot model unavailable: {exc})")
        return
    artefact = load_artefact(artefact_path)
    scalings = [s for s in SCALINGS if any(k[0] == s for k in artefact.cells)]
    rows = []
    for scaling in scalings:
        presets = learned_presets(LearnedEmotionModel(artefact, scaling=scaling))
        for preset, params in presets.items():
            rows.append(_feasibility_row(preset, scaling, params, artefact, robot))
    append_feasibility(report, rows)
    bad = [f"{r['preset']} ({r['scaling']})" for r in rows if not r["feasible"]]
    if bad:
        print(
            "learned presets the G1 cannot reach at the IK thresholds (flagged, still rendered): "
            + ", ".join(bad)
        )


def _fit_check(pipeline, args: argparse.Namespace, target: Path, common: dict) -> int:
    from emotive_gait.emotion.learned import load_artefact
    from emotive_gait.learn.artefact import first_difference

    committed = load_artefact(target)
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        pipeline.run_fit(
            Path(args.manifest),
            Path(args.root),
            out_artefact=scratch / "refit.yaml",
            out_report=scratch / "report.md",
            derived=scratch / "derived",
            **common,
        )
        refit = load_artefact(scratch / "refit.yaml")
    difference = first_difference(committed, refit)
    if difference is not None:
        _err(f"--check: the refit differs from {target}: {difference}")
        return 2
    print(f"--check: the refit matches {target} (every estimate within 1e-6)")
    return 0


def _source_url(manifest, shown_path: str) -> str:
    """Where ``<dataset>/<path>`` comes from: the archive URL, or the base URL joined with path."""
    dataset_id, _, rest = shown_path.partition("/")
    dataset = manifest.dataset(dataset_id)
    if dataset.archive is not None:
        return dataset.url
    return dataset.url.rstrip("/") + "/" + rest


def _fetch(args: argparse.Namespace) -> int:
    import zipfile

    from emotive_gait.learn.fetch import fetch
    from emotive_gait.learn.manifest import ManifestError, describe_problem, load_manifest

    from_paths: dict[str, Path] = {}
    for spec in args.from_paths:
        dataset_id, sep, path = spec.partition("=")
        if not sep or not path:
            _err(f"--from {spec!r} must look like ID=PATH")
            return 2
        from_paths[dataset_id] = Path(path)
    try:
        manifest = load_manifest(Path(args.manifest))
        problems = fetch(manifest, Path(args.root), from_paths=from_paths, datasets=args.datasets)
    except (ManifestError, ValueError, zipfile.BadZipFile) as exc:
        _err(str(exc) if isinstance(exc, ManifestError) else f"{args.manifest}: {exc}")
        return 2
    except OSError as exc:
        _err(f"environment failure: {exc}")
        return 1
    for problem in problems:
        print(describe_problem(problem), file=sys.stderr)
        if problem.error is not None or problem.actual is None:
            print(f"  fetch by hand: {_source_url(manifest, problem.path)}", file=sys.stderr)
    if problems:
        _err(f"{len(problems)} file(s) missing or with a different hash")
        return 2
    print(f"fetched and verified every file under {args.root}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Exit codes: 0 success, 1 environment failure, 2 user-input error."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    if args.command == "evidence":
        return _evidence(args)
    if args.command == "fit":
        return _fit(args)
    if args.command == "fetch":
        return _fetch(args)
    return _demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
