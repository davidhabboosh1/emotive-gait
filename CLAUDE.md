# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Emotion-modulated humanoid gait for the Unitree G1 in MuJoCo. Six emotion intensities move the
*parameters* of a walking gait through three affective axes (activation, valence, tension). Every
literature claim is cited or flagged. The package is `src/emotive_gait/`.

## Commands

```sh
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                               # dev includes statsmodels and pandas (the `fit` extra)
python -m mujoco_menagerie prefetch unitree_g1        # first model download; needs Python >= 3.11.4
emotive-gait demo --emotion sadness=0.8               # kinematic video, needs ffmpeg
emotive-gait demo --emotion fear=0.5 --physics --no-video
emotive-gait demo --emotion anger=0.8 --physics --root xz-rail --no-video   # experimental, falls
emotive-gait demo --emotion sadness=1.0 --camera fixed   # default camera is tracking
emotive-gait evidence                                 # claim table with grounding
emotive-gait demo --emotion sadness=0.8 --model learned [--scaling ratio]   # fitted model, froude default
emotive-gait evidence --model learned                 # claim table plus estimate, interval, direction check
emotive-gait fetch                                    # download/verify data/raw against data/manifest.yaml
emotive-gait fit                                      # refit; rewrites the artefact and docs/reports/phase-4-5-fit-report.md
emotive-gait fit --check                              # refit to a temp file, compare every estimate with the artefact
emotive-gait fit --include-uncommittable --out data/derived/with_bml/coef.yaml   # local fit with BML
emotive-gait rig inspect in.bvh [--scale S] [--up y|z] [--mapping map.yaml] [--json]   # bindings, units, decisions
emotive-gait rig walk in.bvh --out out.bvh --emotion sadness=0.8   # or --prompt "very gloomy"; --model, --scaling, --duration
emotive-gait rig validate in.bvh [--report path.md]   # neutral and the four presets, round-trip table
emotive-gait serve                                    # MCP server over stdio; needs the `mcp` extra
python -m emotive_gait.rig.validate docs/reports/phase-4-6-rig-validation.yaml --out docs/reports/phase-4-6-rig-validation.md
pytest                                                # whole suite
pytest tests/gait/test_generator.py::test_name -v     # one test
ruff check .
ruff format --check .
```

CI runs `ruff check .`, `ruff format --check .` and `pytest -q`. `docs/archive` (archived transcript) and `docs/superpowers` (fenced Python in the design/plan docs) are excluded from ruff.

## Architecture

Seven packages (`robot/`, `gait/`, `emotion/`, `sim/`, `mocap/`, `learn/`, `rig/`) plus `cli.py` on top:

- `robot/`: semantic `JointRole`s, the G1 profile (role -> actuator name), the model loader, neck
  injection, root modes (`FREE`, `Z_RAIL`, `XZ_RAIL` (experimental), `WELD`).
- `gait/`: `GaitParameters`, periodic C2 curves (`spline.py`), `footsteps.py` (pure foot and
  pelvis paths in the pelvis frame), `ik.py` (mink leg solver on a welded root), and
  `generate()`, which combines the IK legs with spline-driven arms, trunk and neck and clamps
  to joint range last.
- `emotion/`: `evidence.py` (claims), `axes.py` (intensities -> axes), `model.py` (axes -> parameters).
- `sim/`: `kinematic.py`, `physical.py`, `metrics.py` (ZMP/CoP/uprightness), `render.py`.
- `mocap/`: readers, `events.py` (foot events), extractor (per-cycle gait parameters), normalisation,
  landmark maps. NumPy, SciPy and `gait.params` only; imports nothing from `robot/`, `emotion/`,
  `sim/`, `learn/`.
- `learn/`: manifest, fetch, table, fit, structure, artefact, report, pipeline, console. May import
  `mocap/`, `gait.params`, `emotion.evidence`, `emotion.model`, `emotion.learned`; `fit.py` and
  `artefact.py` also import `robot.profiles.g1` for the hand-tuned baseline (allowed). Needs
  statsmodels and pandas; ONLY `cli.py` imports `learn/`.
- `gait/scaling.py` (`Scaling`, `G`, `to_gait_parameters`) is imported by `mocap/`, `emotion/`
  and `learn/`. `emotion/learned.py` reads the packaged artefact with pyyaml, imports
  `robot.roles` under `TYPE_CHECKING` only, and never imports `learn/`, `mocap/` or `sim/`;
  `emotion/model.py` holds the `EmotionModel` protocol. Only `cli.py` runs the learned-preset IK. `tests/mocap/` may import `sim/`; the package may not.
- `rig/`: any BVH walk, read, measured, moved by an emotion and written back on its own rig
  (mapping, compile to MuJoCo, calibration, synthesis, writer, validate, MCP server). May import
  everything except `learn/` and `sim/`; ONLY `cli.py` imports it (lazily, for `rig` and `serve`).
  `rig/server.py` is the only module that uses `mcp` (inside functions); `rig/cli.py` only probes
  whether it is installed. `emotion/prompt.py` (the
  English lexicon) imports only `emotion.evidence` and pyyaml. `mocap/writers/` holds the BVH writer.

Dependency rules:

- `robot/` is the base layer and imports nothing from the other packages.
- `gait/` depends on `robot/` (the generator needs `RobotModel` and `JointRole`).
  `gait/footsteps.py` imports nothing from `robot/` (pure math, no MuJoCo); `gait/ik.py`
  imports `mink`, `mujoco` and `robot/`.
- `emotion/` depends on `gait.params`; ONLY `emotion/model.py` additionally imports `robot.roles`
  and `robot.profiles.g1` (to drop cues the profile cannot show; `emotion/learned.py` imports
  `robot.roles` for type hints only). Nothing in `emotion/` imports
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
  is stationary in the world (< 1 mm). Gait phase 0 (phi = 0) starts the left swing; sway goes toward the
  stance side (G1 +y is left). Pelvis-frame swing x overshoots +-D/2 by about 11%.
- Nominal pelvis height is 0.68 m (`com_height_scale = 1.0`), below the straight-leg 0.79 m
  stand. 0.70 leaves anger's 0.47 m steps unreachable (132.8 mm foot error); 0.66 trips the
  ankle pitch limit (13.6 mm); 0.68 solves every preset to <= 0.065 mm. The preset feasibility
  test is the arbiter.
- IK is seeded from a bent-knee pose (hip -0.2, knee 0.4, ankle -0.2): the keyframe's straight
  knee is a Jacobian singularity. `generate()` costs a few tenths of a second (two warm-started IK laps; design doc
  section 9), so tests share
  trajectories instead of regenerating.
- Sole orientation is a hard task (level) in stance and a soft preference (cost 0.1) in swing;
  the swing foot may pitch (worst ~0.034 rad on neutral). No heel-toe roll is modelled.
- Feasibility: `feasible` is the flag (thresholds 2 mm position everywhere, 0.01 rad stance
  orientation). `roles_at_limit` is informational and non-empty even on neutral (the ankle
  touches its limit on a few swing samples). The CLI warns once on infeasible parameters and
  still renders.
- `expansiveness` scales `arm_swing_amp`, `arm_abduction` and `step_height`, never
  `step_length` (speed = step_length * cadence is canonical). Arms follow the opposite hip's
  normalised pitch and fade out below 0.02 rad hip excursion (every preset exceeds it, even in its in-place projection),
  so a static gait has no arm swing.
- `feet_spacing` defaults to 0.20 m.
- Physics plays a recorded projection of the parameters (`played_params`, printed by the CLI):
  `Z_RAIL` zeroes `step_length` and `lateral_sway_amp`; `WELD` also zeroes `com_bounce_amp` and
  forces `com_height_scale` 1.0 (pelvis welded at 0.68 m); `XZ_RAIL` (experimental) zeroes only
  `lateral_sway_amp`; `FREE` plays everything and falls.
- `XZ_RAIL` is EXPERIMENTAL and a measured negative result: a prescribed forward slide (no
  servo lag) still loses foot contact on neutral (first gap 0.074 s with the default 0.5 s joint ramp; with no ramp the first gap is at 0.33 s and the fall at 0.552 s) and falls at about 0.54 s,
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
- The learned artefact `src/emotive_gait/emotion/data/learned_coefficients.yaml` is generated by
  `emotive-gait fit`, never hand-edited. The fit is Phase 4.5; the analytic model is bit-identical
  and stays the default.
- Grounding is per cell: `fitted` (>= 3 actors, 95 % Wald interval), `thin` (1-2 actors, no
  interval, excluded from headline numbers, CLI warns), `missing` (delegated to the analytic model
  cell by cell; disgust and surprise stay unsourced).
- `NO_CHANGE_CLAIMS` in `evidence.py` are the only direction-0 claims checked, and the interval
  must span zero. `evidence.py` and the data are never edited to resolve a contradiction.
- `expansiveness` is fixed at 1.0 in the learned model for the four fitted emotions.
- `RobotProfile.leg_length` (G1 0.69 m) feeds Hof scaling (lengths / l, cadence x sqrt(l/g),
  `com_height_scale` x l / 0.68).
- The posture round trip (`tests/mocap/test_roundtrip.py`) compares deltas against neutral. Two G1
  rig constants make absolute values differ (strict xfails there, a Phase 4 rendering fix, not made
  here): the generator's `elbow_flexion` joint straightens the arm when positive (geometric bend =
  1.385 - joint, so anger renders straighter elbows than neutral), and the pelvis-to-neck vector
  pitches 0.88 of the commanded waist pitch (pelvis origin 4.4 cm below the waist pivot).
- Foot events are the position-extremum rule (Zeni 2008), not a velocity threshold; the frozen
  constants are in the header of `mocap/events.py`. The G1 double-support rule-vs-parameter offset
  is +0.21. A turning cycle (heading change > 0.35 rad across the cycle) is dropped; one complete
  cycle is enough for a clip summary.
- Aggregates only enter git: `data/raw/` and `data/derived/` are ignored. BML stays
  `committable: false` in `data/manifest.yaml` until written permission arrives (requested from
  the Glasgow PACO lab 2026-10-25); then set `committable: true` with the permission text and
  re-run `emotive-gait fit`.
- `.gitattributes` pins the manifest, the artefact YAML and `docs/reports/*.md` to LF, because the
  provenance records the manifest's SHA-256.
- The maximal mixed model falls back to a random intercept on non-convergence, terminal warnings,
  non-finite SEs or a singular `cov_re` (relative eigenvalue 1e-6, |r| > 0.999) and records
  `fallback_reason`; `maximal_raised` when the maximal fit raised and the fallback succeeded,
  `fit_error` only when the fallback raised too (no result). On the 4-actor committable tier every
  cell falls back with `maximal_raised` (`LinAlgError`: 4 groups, 5x5 random-effect covariance),
  so the Wald intervals carry only cycle-level residual variance and the fixed effects are
  cycle-weighted.
- Learned presets are flagged and rendered, not gated (user decision 2026-10-26): the feasibility
  test gates the analytic presets only; learned residuals (froude angry 6.37 mm, happy 5.62 mm) are
  documented in the fit report.

- **Rig calibration (Phase 4.6).** The calibrated zero is the actor's mean walking pose baked into
  body quaternions (chordal means; pelvis WORLD rotation heading-removed; ankle at the stance
  WORLD orientation). Posture is an affine `posture_map` measured by probing the compiled rig:
  trunk gain about 0.82 on the synthetic base rig and 0.57 with one spine bone (posture-map probes;
  the design estimated 0.77 on 100STYLE proportions). The G1's map is
  empty and its output is bit-identical to before.
- One foot point everywhere: the toe if present, else the foot's End Site, else the ankle.
  `foot_site_floor_height = 0.0` on every rig, because the floor IS the median stance height of
  that point.
- Two compiles: `full` (every joint, for FK) and `driven` (the chain the generator drives). The FK
  gate is 1e-6 m (measured 1e-15) against the reader's forward kinematics.
- `ROLE_CONVENTIONS` in `robot/conventions.py` fixes each role's sign. The G1 elbow is a strict
  xfail there; the ELBOW row is wrist-forward.
- `generate(..., welded=)` takes a prebuilt welded robot. `RobotProfile.elbow_swing_sign` is -1.0
  on rigs: the generator's swing term was tuned on the G1's inverted elbow.
- A rig needs at least twelve leg roles bound. Knee and elbow ranges are about the STRAIGHT
  configuration (`RoleBinding.zero_offset`).
- The double-support inversion goes through `rule_double_support`, now in `gait/footsteps.py`.
- The pelvis-lowering factor (`reachable_pelvis_factor`, the `pelvis_height` decision) lowers the
  nominal pelvis from the source's mean until the neutral stride is reachable by the flat-foot
  generator; the output walks lower than the source (measured 0.836 to 0.992 on the real-rig runs).
- Hinge names are `Bone/Channel`; End Sites are named `<bone>_End` (points and sites, not hinges). The pelvis bone's hinges sit on the
  carrier body `pelvis_hinges` under the joint-less root `pelvis`. A mapping file's `roles:`
  section maps a role to a `Bone/Channel` value; internally the flat overrides dict that
  `load_rig` takes namespaces them `role:<role>`.
- Translation channels below the pelvis: constant columns are folded into the compiled offsets,
  varying ones are refused. A BVH whose position values duplicate its OFFSETs (Bandai) is read by
  the reader at twice its bone length; `--scale 0.005` halves the root travel too.
- Units inference knows only (1, 0.01, 0.0254, 0.1, 0.001) m per unit; CMU's unit is read as
  inches unless `--scale 0.0564444`.
- On a rig whose shoulder PITCH channel is the innermost rotation channel, `arm_swing_amp` and
  `arm_abduction` are not gated above 0.5 rad commanded abduction (Bandai; no run exceeded 0.483).
- A half-bound posture group (one skewed side) moves one arm by the full delta and the extractor's
  mean reads half. The extractor's elbow measure is unsigned.
- No LLM inside the package: the MCP host's model maps words to intensities when present, the
  shipped lexicon otherwise.
- Never commit Bandai (CC BY-NC), the cgspeed CMU conversion or Mixamo files, in any form. The
  validation datasets live in `data/rig-manifest.yaml`, not the frozen `data/manifest.yaml`.
- `docs/reports/phase-4-6-rig-validation.md` is regenerated from `phase-4-6-rig-validation.yaml`
  with the command under Commands, never hand-edited. `validate` runs at least four cycles per
  preset.
- `mcp>=2,<3` is pinned (`mcp` and `dev` extras). The server must stay silent on stdout: stdout is
  the protocol channel.
- `tests/rig/` is a package (`from rig.rigs import ...`).

## Conventions

- `emotive-gait demo` returns 2 on user-input errors and 1 on environment failures (ffmpeg,
  disk), always with a stderr message, and never lets a traceback escape `main`.
- Design and plan documents live under `docs/superpowers/`; the design doc is not edited except
  at the user's request.
- `emotive-gait fit` and `fetch` use the same exit codes: 2 for manifest, hash or user-input
  errors (naming the file, expected and actual hash), 1 for environment failures (including
  statsmodels missing); no traceback escapes `main`.
- `docs/reports/phase-4-5-fit-report.md` is generated by `fit`; do not edit it by hand.
- `docs/archive/` is unmaintained history.
- Generated media: `docs/media/*.gif` are committed, `*.mp4` are ignored.
