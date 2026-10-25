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
emotive-gait demo --emotion anger=0.8 --physics --root xz-rail --no-video   # experimental, falls
emotive-gait demo --emotion sadness=1.0 --camera fixed   # default camera is tracking
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
  injection, root modes (`FREE`, `Z_RAIL`, `XZ_RAIL` (experimental), `WELD`).
- `gait/`: `GaitParameters`, periodic C2 curves (`spline.py`), `footsteps.py` (pure foot and
  pelvis paths in the pelvis frame), `ik.py` (mink leg solver on a welded root), and
  `generate()`, which combines the IK legs with spline-driven arms, trunk and neck and clamps
  to joint range last.
- `emotion/`: `evidence.py` (claims), `axes.py` (intensities -> axes), `model.py` (axes -> parameters).
- `sim/`: `kinematic.py`, `physical.py`, `metrics.py` (ZMP/CoP/uprightness), `render.py`.

Dependency rules:

- `robot/` is the base layer and imports nothing from the other three.
- `gait/` depends on `robot/` (the generator needs `RobotModel` and `JointRole`).
  `gait/footsteps.py` imports nothing from `robot/` (pure math, no MuJoCo); `gait/ik.py`
  imports `mink`, `mujoco` and `robot/`.
- `emotion/` depends on `gait.params`; ONLY `emotion/model.py` additionally imports `robot.roles`
  and `robot.profiles.g1` (to drop cues the profile cannot show). Nothing in `emotion/` imports
  `gait.generator` or `sim/`.
- `sim/` depends on `gait/` and `robot/` only, never `emotion/`; nothing imports `sim/` except
  `cli.py`.

## Facts that bite

- **MuJoCo is pinned to `>=3.17,<3.18`, and the pin is load-bearing.** `robot/neck.py` and
  `robot/root.py` use the 3.17 `MjSpec` API: `spec.body(name)` (was `find_body`) and
  `spec.delete(obj)` (was `geom/joint/key.delete()`). Raising the upper bound is a deliberate
  re-verify: port any further `MjSpec` changes and use `tests/robot/test_neck.py` (total mass
  33.3411, torso remainder, CoM) as the mass/inertia gate.

- Load the menagerie `scene`, never `g1`: the bare model has no floor.
- Neck injection deletes the model's keyframe, so the standing pose is rebuilt by joint name
  (`standing_qpos`), not by index.
- Mass split: the head takes 1.8025 kg out of the torso's explicit 7.818 kg. Total robot mass
  must stay 33.3411.
- `nq != nv` on the free root (quaternion); `nq == nv` on the Z-rail. Do not index `qvel` with
  `qpos` addresses on the free root.
- Roll joints are mirrored left/right, so symmetric modulation negates them.
- Positive `neck_pitch` is gaze DOWN.
- Gait geometry is "treadmill style": foot targets live in the pelvis frame, the stance foot
  slides back at walking speed while the world pelvis advances at that speed, so the stance foot
  is stationary in the world (< 1 mm). Phase 0 starts the left swing; sway goes toward the
  stance side (G1 +y is left). Pelvis-frame swing x overshoots +-D/2 by about 11%.
- Nominal pelvis height is 0.68 m (`com_height_scale = 1.0`), below the straight-leg 0.79 m
  stand. 0.70 leaves anger's 0.47 m steps unreachable (132.8 mm foot error); 0.66 trips the
  ankle pitch limit (13.6 mm); 0.68 solves every preset to <= 0.065 mm. The preset feasibility
  test is the arbiter.
- IK is seeded from a bent-knee pose (hip -0.2, knee 0.4, ankle -0.2): the keyframe's straight
  knee is a Jacobian singularity. `generate()` costs about 0.6 s (the IK solve), so tests share
  trajectories instead of regenerating.
- Sole orientation is a hard task (level) in stance and a soft preference (cost 0.1) in swing;
  the swing foot may pitch (worst ~0.034 rad on neutral). No heel-toe roll is modelled.
- Feasibility: `feasible` is the flag (thresholds 2 mm position everywhere, 0.01 rad stance
  orientation). `roles_at_limit` is informational and non-empty even on neutral (the ankle
  touches its limit on a few swing samples). The CLI warns once on infeasible parameters and
  still renders.
- `expansiveness` scales `arm_swing_amp`, `arm_abduction` and `step_height`, never
  `step_length` (speed = step_length * cadence is canonical). Arms follow the opposite hip's
  normalised pitch and fade out below 0.1 rad hip excursion, so a static gait has no arm swing.
- `feet_spacing` defaults to 0.20 m.
- Physics plays a recorded projection of the parameters (`played_params`, printed by the CLI):
  `Z_RAIL` zeroes `step_length` and `lateral_sway_amp`; `WELD` also zeroes `com_bounce_amp` and
  forces `com_height_scale` 1.0 (pelvis welded at 0.68 m); `FREE` plays everything and falls.
- `XZ_RAIL` is EXPERIMENTAL and a measured negative result: a prescribed forward slide (zero
  root lag) still loses foot contact on neutral (first gap 0.074 s) and falls at about 0.54 s,
  and a servoed rail (kp 5000-100000, ramped) failed the same way at 0.52-1.12 s. Forward
  walking is a kinematic-tier result; the physics tier is in place. Sway is lost on every rail.
- The Z-rail survival gate is the deviation of pelvis z from the commanded path (peak-to-peak
  < 6 cm, sag < 5 cm, no-contact steps < 25%), measured post-ramp in
  `test_z_rail_survives_six_seconds_on_every_preset`: neutral 2.46 / sad 2.80 / angry 5.01 /
  happy 3.60 cm. Angry and happy exceed 3 cm from real servo-compliant leg bounce at 2.3-2.7
  steps/s. Changing the leg tuning is a physics decision.
- Posture rule: amplitudes are relative to the rest pose, postures are absolute.
- On `Z_RAIL`/`WELD` the pelvis cannot rotate, so uprightness is floored at cos(0.52)^2 = 0.753
  by the waist limits and can never signal a fall. `simulate` uses uprightness < 0.6 only on
  `FREE`; constrained roots fall on rail-stop contact, lost foot contact or a 15 cm pelvis drop
  (`SimResult.fail_reason` says which).
- Emotion is applied to PARAMETERS, never to joint angles. The generator clamps to joint range last.
- `knee_flexion_bias` is retired. Fear has four category offsets (`com_height_scale`,
  `elbow_swing_ratio`, `cadence`, `neck_pitch`), two of which (cadence, neck_pitch) override the
  axes, because it breaks a pure three-axis mapping. Its crouch is `com_height_scale` -0.02
  (about 1.4 cm): -0.04 and -0.03 fail the preset feasibility test (ankles at limit), a G1
  ankle-range limitation, not an evidence claim. Anger has one offset, `neck_pitch = -0.135`, that
  exactly cancels its valence-driven head drop (the evidence says no head change); it must track
  `AXIS_GAINS["neck_pitch"]` and anger's valence weight.
- No claim may state a magnitude in degrees; claims are signed directions with a grounding level.
- Disgust and surprise are unsourced. Keep them flagged and keep the CLI warning.
- Physics is open loop with position servos. ZMP/CoP are instrumentation, never control. A free
  floating base falls in 0.74-2.25 s, which is why the supported physics tier is the Z-rail.
- The kinematic tier renders real forward travel (the pelvis advances 2 * `step_length` per
  cycle; tracking camera by default, `--camera fixed` to stay put). The physics tier is in place.

## Conventions

- `emotive-gait demo` returns 2 on user-input errors and 1 on environment failures (ffmpeg,
  disk), always with a stderr message, and never lets a traceback escape `main`.
- Design and plan documents live under `docs/superpowers/`; do not edit the design doc.
- `docs/archive/` is unmaintained history.
- Generated media: `docs/media/*.gif` are committed, `*.mp4` are ignored.
