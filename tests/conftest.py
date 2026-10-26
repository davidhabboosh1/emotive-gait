"""Skip ``requires_gl`` tests when offscreen rendering or ffmpeg is unavailable."""

from __future__ import annotations

import functools
import shutil
import sys
from pathlib import Path

import pytest

from emotive_gait.gait.params import GaitParameters

# tests/emotion has no __init__.py, so its helpers (artefacts.py) are importable by bare name
# from any test module once its directory is on the path.
sys.path.insert(0, str(Path(__file__).parent / "emotion"))


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


@pytest.fixture(scope="session")
def _toy_fit_files(tmp_path_factory):
    from learn.ondisk import write_fit_dataset

    return write_fit_dataset(tmp_path_factory.mktemp("toy_fit"))


@pytest.fixture
def synthetic_dataset_on_disk(_toy_fit_files, monkeypatch):
    """Three-actor toy dataset plus a one-actor ``committable: false`` one, on disk.

    The files are written once per session; the dataset ids are registered in ``LANDMARKS``
    for the duration of the test only.
    """
    from learn.ondisk import register_landmarks

    register_landmarks(monkeypatch)
    return _toy_fit_files


@pytest.fixture(scope="session")
def fitted_toy(_toy_fit_files, tmp_path_factory):
    """One ``run_fit`` on the toy dataset (committable data only): ``(summary, directory)``.

    The directory holds ``coef.yaml``, ``report.md`` and ``derived/``. Shared because a fit
    takes about half a minute.
    """
    from learn.ondisk import register_landmarks

    from emotive_gait.learn.pipeline import run_fit

    tmp = tmp_path_factory.mktemp("fit")
    patch = pytest.MonkeyPatch()
    register_landmarks(patch)
    try:
        summary = run_fit(
            _toy_fit_files.manifest,
            _toy_fit_files.root,
            out_artefact=tmp / "coef.yaml",
            out_report=tmp / "report.md",
            derived=tmp / "derived",
            datasets=None,
            include_uncommittable=False,
            log=lambda _: None,
        )
    finally:
        patch.undo()
    return summary, tmp
