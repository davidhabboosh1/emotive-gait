"""Command line: ``emotive-gait demo`` renders a stimulus, ``emotive-gait evidence`` lists claims."""

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

    sub.add_parser("evidence", help="print every literature claim with its grounding")
    return p


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

    try:
        params = modulate(
            GaitParameters(), intensities, G1_PROFILE_WITH_NECK, hold_speed=args.hold_speed
        )
    except (IntensityError, SpeedConstraintError) as exc:
        _err(str(exc))
        return 2
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


def _evidence() -> int:
    rows = [("EMOTION", "PARAMETER", "DIR", "GROUNDING", "CITATION")]
    last = None
    for c in EVIDENCE:
        label = c.emotion.value if c.emotion is not last else ""
        last = c.emotion
        rows.append(
            (label, c.parameter, _DIRECTION[c.direction], c.grounding.value.upper(), c.citation)
        )
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    for r in rows:
        print("  ".join(r[i].ljust(widths[i]) for i in range(4)) + "  " + r[4])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Exit codes: 0 success, 1 environment failure, 2 user-input error."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    if args.command == "evidence":
        return _evidence()
    return _demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
