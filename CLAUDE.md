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

CI runs `ruff check .`, `ruff format --check .` and `pytest -q`. `docs/archive` (archived transcript) and `docs/superpowers` (fenced Python in the design/plan docs) are excluded from ruff.

## Architecture

Four layers plus `cli.py` on top:

- `robot/`: semantic `JointRole`s, the G1 profile (role -> actuator name), the model loader, neck
  injection, root modes (`FREE`, `Z_RAIL`, `WELD`).
- `gait/`: `GaitParameters`, periodic C2 curves (`spline.py`), the in-place `generate()`.
- `emotion/`: `evidence.py` (claims), `axes.py` (intensities -> axes), `model.py` (axes -> parameters).
- `sim/`: `kinematic.py`, `physical.py`, `metrics.py` (ZMP/CoP/uprightness), `render.py`.

Dependency rules:

- `robot/` is the base layer and imports nothing from the other three.
- `gait/` depends on `robot/` (the generator needs `RobotModel` and `JointRole`).
- `emotion/` depends on `gait.params`; ONLY `emotion/model.py` additionally imports `robot.roles`
  and `robot.profiles.g1` (to drop cues the profile cannot show). Nothing in `emotion/` imports
  `gait.generator` or `sim/`.
- `sim/` depends on all three; nothing imports `sim/` except `cli.py`.

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
- The leg constants in `generator.py` are calibrated against the Z-rail bob and contact bounds
  (pelvis bob < 6 cm, no-foot-contact steps < 25%) in
  `test_z_rail_survives_six_seconds_on_every_preset`. Changing one is a physics decision.
- On `Z_RAIL`/`WELD` the pelvis cannot rotate, so uprightness is floored at cos(0.52)^2 = 0.753
  by the waist limits and can never signal a fall. `simulate` uses uprightness < 0.6 only on
  `FREE`; constrained roots fall on rail-stop contact, lost foot contact or a 15 cm pelvis drop
  (`SimResult.fail_reason` says which).
- Emotion is applied to PARAMETERS, never to joint angles. The generator clamps to joint range last.
- Fear has four category offsets (`knee_flexion_bias`, `elbow_swing_ratio`, `cadence`, `neck_pitch`), two of which (cadence, neck_pitch) override the axes, because it breaks a pure three-axis mapping. Anger has one offset, `neck_pitch = -0.135`, that
  exactly cancels its valence-driven head drop (the evidence says no head change); it must track
  `AXIS_GAINS["neck_pitch"]` and anger's valence weight.
- No claim may state a magnitude in degrees; claims are signed directions with a grounding level.
- Disgust and surprise are unsourced. Keep them flagged and keep the CLI warning.
- Physics is open loop with position servos. ZMP/CoP are instrumentation, never control. A free
  floating base falls in 0.74-2.25 s, which is why the supported physics tier is the Z-rail.
- Stride is rendered nowhere yet (both tiers are in-place): `step_length`/`speed` are modelled but not shown until the task-space IK phase.

## Conventions

- `emotive-gait demo` returns 2 on user-input errors and 1 on environment failures (ffmpeg, disk), always with a stderr message, and never lets a
  traceback escape `main`.
- Design and plan documents live under `docs/superpowers/`; do not edit the design doc.
- `docs/archive/` is unmaintained history.
- Generated media: `docs/media/*.gif` are committed, `*.mp4` are ignored.
