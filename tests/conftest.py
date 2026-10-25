"""Skip ``requires_gl`` tests when offscreen rendering or ffmpeg is unavailable."""

from __future__ import annotations

import functools
import shutil

import pytest

from emotive_gait.gait.params import GaitParameters


@functools.cache
def _gl_skip_reason() -> str | None:
    if shutil.which("ffmpeg") is None:
        return "ffmpeg is not on PATH"
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        renderer = mujoco.Renderer(model, height=32, width=32)
        renderer.close()
    except Exception as exc:  # noqa: BLE001 - any failure means no usable GL context
        return f"mujoco.Renderer cannot be constructed ({type(exc).__name__}: {exc}); set MUJOCO_GL"
    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    gl_items = [item for item in items if "requires_gl" in item.keywords]
    if not gl_items:
        return
    reason = _gl_skip_reason()
    if reason is None:
        return
    marker = pytest.mark.skip(reason=f"requires_gl: {reason}")
    for item in gl_items:
        item.add_marker(marker)


# Phase 0 spike presets (crude, minus the retired knee bias), shared by the playback and
# feasibility tests.
PHASE0_PRESETS = {
    "neutral": GaitParameters(
        cadence=1.8,
        arm_swing_amp=0.30,
        elbow_flexion=1.28,
        trunk_pitch=0.00,
        neck_pitch=0.00,
    ),
    "sad": GaitParameters(
        cadence=1.0,
        arm_swing_amp=0.08,
        elbow_flexion=1.05,
        trunk_pitch=0.32,
        neck_pitch=0.70,
    ),
    "angry": GaitParameters(
        cadence=2.7,
        arm_swing_amp=0.60,
        elbow_flexion=1.85,
        trunk_pitch=0.18,
        neck_pitch=-0.10,
    ),
    "happy": GaitParameters(
        cadence=2.3,
        arm_swing_amp=0.62,
        elbow_flexion=0.95,
        trunk_pitch=-0.12,
        neck_pitch=-0.35,
    ),
}
