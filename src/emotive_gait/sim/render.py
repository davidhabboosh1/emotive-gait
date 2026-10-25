"""Offscreen rendering of posed frames and ffmpeg video export."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from emotive_gait.robot.loader import RobotModel


class FfmpegUnavailableError(RuntimeError):
    """Raised when video export is requested but ``ffmpeg`` is not on PATH."""


@dataclass(frozen=True)
class CameraSpec:
    """Camera placement, framed on a standing robot by default.

    ``tracking`` follows the pelvis body (it looks at that body, so ``lookat_z`` applies to the
    fixed camera only).
    """

    distance: float = 3.2
    azimuth: float = 135.0
    elevation: float = -8.0
    lookat_z: float = 0.9
    tracking: bool = True


def make_camera(model: mujoco.MjModel, spec: CameraSpec) -> mujoco.MjvCamera:
    """Build the camera for ``spec``: tracking the pelvis, or a free camera at a fixed point."""
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    if spec.tracking:
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    else:
        cam.lookat[2] = spec.lookat_z
    cam.distance = spec.distance
    cam.azimuth = spec.azimuth
    cam.elevation = spec.elevation
    return cam


def render_frames(
    robot: RobotModel,
    frames: Iterable[mujoco.MjData],
    *,
    width: int = 960,
    height: int = 720,
    camera: CameraSpec = CameraSpec(),  # noqa: B008 - frozen, safe as a default
) -> Iterator[np.ndarray]:
    """Render each already-forwarded ``MjData`` to a ``(height, width, 3)`` uint8 array.

    One renderer serves the whole iteration and is closed when it ends. Width and height must
    not exceed the model's offscreen framebuffer. Physics is never advanced here.
    """
    model = robot.model
    cam = make_camera(model, camera)

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        for data in frames:
            renderer.update_scene(data, cam)
            yield renderer.render()  # allocates a fresh array on every call
    finally:
        renderer.close()


def write_video(
    frames: Iterable[np.ndarray],
    path: str | Path,
    *,
    fps: int = 30,
    width: int = 960,
    height: int = 720,
) -> Path:
    """Encode ``frames`` (uint8 RGB, ``(height, width, 3)``) to an H.264 mp4 at ``path``."""
    if shutil.which("ffmpeg") is None:
        raise FfmpegUnavailableError(
            "ffmpeg was not found: install ffmpeg and make sure it is on PATH to write video."
        )
    path = Path(path)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path),
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i, frame in enumerate(frames):
            if i == 0 and frame.shape != (height, width, 3):
                raise ValueError(
                    f"first frame has shape {frame.shape}, expected {(height, width, 3)} "
                    f"from width={width}, height={height}"
                )
            proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with code {rc}")
    return path
