# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Emotion-modulated humanoid gait for the Unitree G1 in MuJoCo. Six emotion intensities move the
*parameters* of a walking gait through three affective axes (activation, valence, tension). Every
literature claim is cited or flagged. The package is `src/emotive_gait/`.

## Commands

```sh
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m mujoco_menagerie prefetch unitree_g1        # first model download; needs Python >= 3.11.4
emotive-gait demo --emotion sadness=0.8               # kinematic video, needs ffmpeg
emotive-gait demo --emotion fear=0.5 --physics --no-video
emotive-gait evidence                                 # claim table with grounding
pytest                                                # whole suite
pytest tests/gait/test_generator.py::test_name -v     # one test
ruff check .
ruff format --check .
```

CI runs `ruff check .`, `ruff format --check .` and `pytest -q`. `docs/archive` is excluded from ruff.

## Architecture

Four layers, depending downward only: `robot/` -> `gait/` -> `emotion/` -> `sim/`, with `cli.py` on top.

- `robot/`: semantic `JointRole`s, the G1 profile (role -> actuator name), the model loader, neck
  injection, root modes (`FREE`, `Z_RAIL`, `WELD`).
- `gait/`: `GaitParameters`, periodic C2 curves (`spline.py`), the in-place `generate()`.
- `emotion/`: `evidence.py` (claims), `axes.py` (intensities -> axes), `model.py` (axes -> parameters).
- `sim/`: `kinematic.py`, `physical.py`, `metrics.py` (ZMP/CoP/uprightness), `render.py`.

The one sanctioned crossing: `emotion/model.py` imports `robot.roles` and `robot.profiles.g1`, to
drop cues the profile cannot show. Nothing else in `emotion/` or `gait/` imports `robot/`, and
nothing below `sim/` imports it.

## Facts that bite

- Load the menagerie `scene`, never `g1`: the bare model has no floor.
- Neck injection deletes the model's keyframe, so the standing pose is rebuilt by joint name
  (`standing_qpos`), not by index.
- Mass split: the head takes 1.8025 kg out of the torso's explicit 7.818 kg. Total robot mass
  must stay 33.3411.
- `nq != nv` on the free root (quaternion); `nq == nv` on the Z-rail. Do not index `qvel` with
  `qpos` addresses on the free root.
- Roll joints are mirrored left/right, so symmetric modulation negates them.
- Positive `neck_pitch` is gaze DOWN.
- Seven `GaitParameters` fields are inert in the in-place generator and must not be read there:
  `step_length`, `step_height`, `double_support_frac`, `feet_spacing`, `com_height_scale`,
  `com_bounce_amp`, `lateral_sway_amp`. They exist for a future stepping generator.
- The leg constants in `generator.py` are calibrated against the Z-rail survival test
  (`test_z_rail_survives_six_seconds_on_every_preset`). Changing one is a physics decision.
- Emotion is applied to PARAMETERS, never to joint angles. The generator clamps to joint range last.
- Fear needs two category offsets (cadence, neck_pitch) because it breaks a pure three-axis mapping.
- No claim may state a magnitude in degrees; claims are signed directions with a grounding level.
- Disgust and surprise are unsourced. Keep them flagged and keep the CLI warning.
- Physics is open loop with position servos. ZMP/CoP are instrumentation, never control. A free
  floating base falls in 0.74-2.25 s, which is why the supported physics tier is the Z-rail.
- Forward stride is kinematic-only; the in-place generator does not step forward.

## Conventions

- `emotive-gait demo` returns 2 with a stderr message on user-input errors and never lets a
  traceback escape `main`.
- Design and plan documents live under `docs/superpowers/`; do not edit the design doc.
- `docs/archive/` is unmaintained history.
- Generated media: `docs/media/*.gif` are committed, `*.mp4` are ignored.
