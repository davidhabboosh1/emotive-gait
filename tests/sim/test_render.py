import numpy as np
import pytest

from emotive_gait.gait.generator import generate
from emotive_gait.gait.params import GaitParameters
from emotive_gait.robot.loader import load_robot
from emotive_gait.sim.kinematic import play_kinematic
from emotive_gait.sim.render import FfmpegUnavailableError, render_frames, write_video


def test_frames_have_the_requested_shape():
    r = load_robot()
    t = generate(GaitParameters(), r)
    f = next(
        iter(render_frames(r, play_kinematic(r, t, duration=0.1, fps=10), width=320, height=240))
    )
    assert f.shape == (240, 320, 3) and f.dtype == np.uint8


def test_render_above_the_default_framebuffer_works():
    """Default offscreen buffer is 640x480; the model must raise offwidth/offheight."""
    r = load_robot()
    t = generate(GaitParameters(), r)
    f = next(
        iter(render_frames(r, play_kinematic(r, t, duration=0.1, fps=10), width=960, height=720))
    )
    assert f.shape == (720, 960, 3)


def test_write_video_produces_a_playable_file(tmp_path):
    r = load_robot()
    t = generate(GaitParameters(), r)
    out = write_video(
        render_frames(r, play_kinematic(r, t, duration=0.5, fps=10), width=320, height=240),
        tmp_path / "v.mp4",
        fps=10,
        width=320,
        height=240,
    )
    assert out.is_file() and out.stat().st_size > 1000


def test_missing_ffmpeg_raises_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(FfmpegUnavailableError, match="ffmpeg"):
        write_video(iter([]), tmp_path / "v.mp4")


def test_frame_size_mismatch_is_rejected(tmp_path):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="240"):
        write_video(iter([frame]), tmp_path / "v.mp4", width=960, height=720)
