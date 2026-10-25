# Phase 4 — Forward Locomotion: Design

**Date:** 2026-10-25
**Status:** awaiting approval
**Parent:** `2026-10-25-emotive-gait-professionalization-design.md` (the professionalization design),
section 6, "Phase 4 — forward locomotion". This document refines that phase; where the two differ,
this one is the later decision and says so explicitly.

---

## 1. Intent and why

Phases 1 to 3 shipped a robot that steps in place. The sad clip and the angry clip differ in
posture and arm swing, but both cover zero metres, and walking speed is the strongest perceptual
cue in the literature this project cites. Seven of the sixteen `GaitParameters` fields
(`step_length`, `step_height`, `double_support_frac`, `feet_spacing`, `com_height_scale`,
`com_bounce_amp`, `lateral_sway_amp`) are modelled by the emotion layer and rendered nowhere.

After Phase 4:

- **The robot goes somewhere.** With today's tuned gains, sadness at intensity 1.0 walks at
  about 0.33 m/s (0.25 m steps at 1.32 steps/s) and covers about 2 m in six seconds; anger at
  1.0 walks at about 1.13 m/s (0.47 m steps at 2.4 steps/s) and covers almost 7 m. Those are
  the model's current numbers, not acceptance targets; the literature's absolute speeds were
  never the tuning target (parent Appendix A.2). Forward travel is real world-frame
  translation, not a scrolling floor.
- **The seven inert parameters become visible.** Feet land where `step_length` and
  `feet_spacing` say, the swing foot clears the floor by `step_height`, the pelvis bobs and sways
  by the two amplitude fields, and `com_height_scale` lowers the body into a real crouch.
- **The camera follows and the floor stays.** A tracking camera over the scene's textured floor,
  so an observer can read "slow" without a second walker to compare against.
- **The physics tier reports, not just survives.** The pelvis rail gains a driven forward axis,
  the legs do the same stride, and each emotion gets numbers: foot slip, and how often the
  zero-moment point leaves the support polygon.
- **Nothing in the emotion layer's meaning changes.** Same six intensities, same cited claims.
  One parameter is retired because inverse kinematics makes it inexpressible (section 4.5), and
  its claims are re-homed on the parameters that can actually show them.

Underneath, this phase pays a debt: `mujoco` is pinned to `>=3.2.0,<3.3` because the `MjSpec`
API changed by 3.17, and every future dependency (the IK library, the viewer work in Phase 6)
sits on the far side of that pin. The port is done first, on its own, with the Phase 0 mass and
inertia checks as the regression gate.

---

## 2. Goals, non-goals, anti-goals

### Goals

- `emotive-gait demo --emotion sadness=0.8` produces a video of forward emotional walking in the
  kinematic tier, with exact foot placement and zero foot slip by construction.
- All sixteen remaining `GaitParameters` fields are rendered; none is inert.
- The `mujoco` pin is raised to the 3.17 line and `mink` is the IK solver, on all four CI jobs.
- Every sample of every generated trajectory carries a feasibility verdict; parameter extremes
  are flagged, never silently distorted.
- The physics tier can walk forward on a driven rail and report per-emotion foot slip and
  zero-moment-point (ZMP) margin, with a failure definition per root mode.
- The three design-level gaps carried out of Phases 1 to 3 are closed: the crouch parameter,
  fear's two leg cues, and the absolute-versus-relative posture rule.

### Non-goals

- **Balance or feedback control of any kind.** The physics tier stays open loop with position
  servos. ZMP and centre of pressure (CoP) remain instrumentation.
- **Free-base walking under physics.** Measured impossible open loop in Phase 0; not revisited.
- **Heel-toe foot roll.** Feet stay level through swing and stance (flat-foot walking). Design
  risk 9.3 stands and is reported, not solved.
- **Turning, uneven ground, or any path other than a straight line along world +X.**
- **A learned or optimized gait.** The foot and pelvis paths are closed-form curves of the
  parameters, exactly as the joint curves were.
- **GUI, evaluation harness, README polish** (Phases 5 to 7).

### Anti-goals

- **Feasibility that quietly edits the emotion.** If a parameter extreme cannot be reached, the
  trajectory is flagged and the number is reported. The solver never rescales `step_length`
  behind the user's back; that would make the emotion-to-parameter map lie.
- **A "walk" that is the old in-place gait with a translating root.** The stance foot must be
  stationary in the world, which is only true if the legs are solved against foot targets. A
  root that slides while the feet paddle is the failure this phase exists to avoid.
- **A pin raised without a port.** Phase 1's first CI run showed what an unbounded `mujoco`
  resolves to. The new pin is as tight as the old one, for the same reason.
- **Fabricated precision.** Nominal pelvis height, swing profiles, and the fear offsets are
  tuned by eye and labelled as tuned, exactly like the Phase 3 gains.

---

## 3. Constraints

- Windows 10, Python 3.11.3 in `.venv`, CI on ubuntu/windows x py3.11/3.12. NumPy is already 2.x
  in the project venv (2.4.6), so the NumPy 2 requirement of newer MuJoCo wheels costs nothing.
- `mujoco==3.17.0` imports and runs on Python 3.11.3 (verified in the throwaway venv).
- `mink==1.3.0` installs on Windows/py3.11 with `qpsolvers==4.13.0` and the `daqp` QP solver
  (binary wheel, no compiler). It requires `mujoco`, `qpsolvers`, `typing_extensions`.
- The `MjSpec` API on 3.17 (verified): `spec.find_body(name)` is `spec.body(name)`;
  `MjsGeom.delete()`, `MjsJoint.delete()` and `MjsKey.delete()` do not exist and are
  `spec.delete(obj)`; `first_geom`/`next_geom`, `.geoms`, `.joints`, `add_geom`, `add_joint`,
  `spec.keys`, `meshname`, and `getattr(spec.visual, "global")` are unchanged. The port is three
  substitutions in `robot/neck.py` and `robot/root.py`.
- The G1 model exposes `left_foot` and `right_foot` sites. At the `stand` keyframe they sit
  0.0331 m above the floor with the sole level (site z-axis = world z). Foot targets are posed
  on these sites.
- **The G1 `stand` keyframe has straight legs.** Pelvis z 0.790, hip pitch 0.687, knee 0.351,
  ankle 0.033; hip-to-ankle is 0.654 m, the full thigh-plus-shank length. Two consequences,
  both measured in the spike (section 9): no foot can move at that pelvis height, and a straight
  knee is a Jacobian singularity the QP cannot leave. The nominal walking height is therefore
  *below* the standing height, and the IK is seeded from a bent-knee pose, never the keyframe.
- Feasible envelope for G1 (spike, section 9; "separation" below means the front-to-back
  distance between the two feet at double support, which is `step_length * (1 +
  double_support_frac)`, so a 0.50 m separation is `step_length` 0.42 at the default 0.20
  double support): at pelvis height 0.70 to 0.74 m, separations up to 0.50 m, swing apex
  0.08 m, 0.03 m sway and 0.02 m bounce all solve to 0 mm residual (0.74 m leaves 1.2 mm at
  0.50). At 0.76 m a 0.50 m separation fails (back leg cannot reach). At 0.66 m every case
  carries an 8 to 10 mm residual because the rear ankle pitch sits on its -0.873 rad limit.
  **Ankle pitch range is the binding constraint at deep crouch**, as design risk 9.2 predicted,
  and **leg reach is the binding constraint on stride at the top of the height range.** Anger
  at intensity 1.0 needs a 0.564 m separation (0.47 m steps), i.e. feet at ±0.282 m, which by
  leg geometry (hip 0.103 m below the pelvis, ankle 0.033 m above the floor, leg 0.654 m) is
  unreachable at 0.74 m, marginal at 0.72 m and has 2 cm of reach to spare at 0.70 m. That
  sets the nominal height (4.3).
- Cost: a 120-sample cycle at 8 warm-started QP iterations per sample takes about 0.3 s per
  lap on the author's machine. Two laps per `generate()` is affordable; tests that generate
  many trajectories should share fixtures.
- Repository conventions in `CLAUDE.md` hold: emotion applies to parameters, roll joints are
  mirrored, positive `neck_pitch` is gaze down, load `scene` never `g1`, no magnitudes in degrees.

---

## 4. Decisions

### 4.1 Port first, as a checkpoint with its own green CI

The port lands before any IK code, on the same branch, as a commit series that ends with CI
green on all four jobs. Order inside the port:

1. `robot/neck.py` and `robot/root.py` rewritten against the 3.17 `MjSpec` API (the three
   substitutions in section 3). No other behaviour changes.
2. `pyproject.toml`: `mujoco>=3.17,<3.18`. **Same shape of pin as before** (one minor line),
   because the lesson of Phase 1 was that an open upper bound resolves to whatever is newest.
   Raising it later is a deliberate re-verify, recorded in `CLAUDE.md`.
3. The existing `tests/robot/test_neck.py` suite **is** the Phase 0 mass/CoM/inertia check
   (total mass 33.3411, torso remainder 6.0155, head 1.8025, whole-body CoM conserved,
   remainder inertia positive). It must pass **unchanged** on 3.17. If any number moves, the
   port stops and the difference is investigated; it is not re-baselined.
4. The full suite (198 passed, 1 xfailed) must pass on 3.17 before the next section starts.
   Any 3.17 behavioural change in physics (contact, integrator defaults) that moves the Z-rail
   bob or contact-fraction numbers outside their calibrated bounds is reported, not absorbed by
   loosening the bounds.

The author decides at that checkpoint whether the port merges on its own or rides with the
rest of the phase. This document does not decide it.

Dependencies added with the port: none. `mink` and `qpsolvers[daqp]` are added in 4.2, so the
port's CI run proves the MuJoCo line alone.

### 4.2 IK library: `mink`, with `daqp`, on a welded-root IK model

`mink` is adopted as the design intended. The design's hand-rolled damped-least-squares fallback
(section 8 of the parent) is **not** taken, because the QP's hard joint-limit handling is exactly
what makes the feasibility residual meaningful: an unconstrained solver reaches an infeasible
target by leaving the joint range, and the clamp afterwards hides the failure.

Dependencies: `mink>=1.3,<2`, `qpsolvers[daqp]>=4,<5`. The solver name `"daqp"` is a constant
in one place. (`mink` pulled `daqp` in transitively on the author's machine; the explicit extra
is there so a resolver change cannot leave the project without a solver.)

**The IK model has a welded root.** IK is solved on the robot compiled with `RootMode.WELD`
(pelvis fixed at the origin), in the **pelvis frame**, treadmill style:

- The stance foot moves *backward* under the pelvis at the walking speed; the swing foot moves
  forward along its swing curve. Both are expressed relative to the pelvis.
- Pelvis height variation (crouch, bounce) and lateral sway are realised by moving the *foot
  targets* the opposite way, since the pelvis itself cannot move in this model.
- At playback, the world-frame root pose is added back (section 4.6). Because the stance foot
  moves backward relative to the pelvis at exactly the speed the pelvis moves forward in the
  world, the stance foot is stationary in the world **by construction**, with no contact model
  involved. This is the "exact placement, zero slip" the parent design promised for the
  kinematic tier, and it is testable to numerical precision.

This choice means one IK solution per phase sample regardless of how far the robot has walked,
so the leg cycle is periodic and cacheable exactly like the joint curves were.

Tasks, per sample: two `FrameTask`s on the `left_foot` and `right_foot` sites (position and
orientation, level sole, zero yaw), one low-cost `PostureTask` toward the bent-knee seed for
regularity, `ConfigurationLimit` for joint ranges. Upper-body joints are not in any task; they
are set procedurally after the solve (4.3) and do not affect foot pose.

### 4.3 Gait architecture: task-space legs, procedural upper body

`gait/generator.py` splits into two sources that compose into one `GaitTrajectory`:

- **Legs (new, `gait/footsteps.py` + `gait/ik.py`):** closed-form foot and pelvis paths as
  functions of phase and `GaitParameters`, solved to leg joint angles by IK, sampled at N
  points per cycle (N = 120 is the starting value; builder's call), two laps warm-started with
  the second lap kept, then fitted per leg role with the existing `PeriodicCurve` (C2 periodic
  cubic through the samples). `angles_at(phi)` evaluates the curve, so the interface and the
  continuity tests at the wrap and at the stance-to-swing handoff carry over.
- **Upper body (existing code, unchanged in form):** waist pitch, neck pitch, shoulder pitch
  swing contralateral to the legs, shoulder roll, elbow flexion with `elbow_swing_ratio`,
  `expansiveness` on the amplitude group. The knee is **removed** from the expansiveness group
  because the knee is no longer an amplitude; `expansiveness` now scales `arm_swing_amp`,
  `arm_abduction` and (new) `step_height`, which keeps "overall movement size, legs included"
  (Phase 3 ruling R37) as the intent. **`step_length` is deliberately not in the group**:
  `speed = step_length * cadence` is canonical, `with_speed` and `hold_speed` pin it, and the
  pelvis path and the X-rail command are driven by it. A stride scaled by expansiveness while
  the pelvis moved at `speed` would slide the stance foot and make the pinned speed a lie. The
  activation axis already drives `step_length` directly, so the leg amplitude cue is not lost.

**The old joint-space leg generator is deleted, not kept beside the new one.** The in-place
case is `step_length == 0` in the new generator: feet lift and set in place, pelvis bobs,
nothing travels. Two leg generators would be two truths about what the parameters mean. The
calibrated leg constants (`_HIP_PITCH_OFFSET` etc.) and the `CLAUDE.md` fact about them go away
with it; the Z-rail survival test is recalibrated against the new gait (4.7).

Composition order is unchanged from the parent design 4.3.1: procedural base (now: IK legs +
procedural upper body), hand-authored per-role offsets or overrides in joint space, joint-range
clamp last. Offsets on leg roles break foot placement; that is the author's explicit intent, as
before.

**Foot and pelvis paths** (closed form; the builder picks the exact easing but these are the
constraints):

- Cycle = two steps. Each foot is in stance for `(1 + double_support_frac) / 2` of the cycle and
  in swing for the rest; the two feet are offset by half a cycle; both are in stance during the
  double-support windows.
- Stance: foot x (pelvis frame) decreases linearly at `speed`. Over a stance it travels
  `step_length * (1 + double_support_frac)`, so the foot's x range is symmetric about zero.
- Swing: x returns from the back of the range to the front with a C2 easing whose velocity is
  zero at both ends **in the world frame**; in the pelvis frame the swing therefore inherits the
  stance drift (pelvis-frame x = world easing minus the pelvis travel during the swing), so its
  end slope equals the stance slope and the handoff is C2. (Corrected during execution, ruling
  R5: an earlier wording put the zero velocity in the pelvis frame, which leaves a velocity jump
  equal to the walking speed at every handoff.) z rises to `step_height` at mid-swing and
  returns to the floor, C2 at both ends. The sole stays level.
- Feet y = `± feet_spacing / 2`, constant.
- Pelvis (world): x = `speed * t`; z = `nominal_height * com_height_scale +
  com_bounce_amp * cos(2 * pi * 2 * phi)` (one bounce period per step, lowest at mid-stance of
  each foot is a tuning choice, not a constraint); y = `-lateral_sway_amp * sin(2 * pi * phi)`,
  one period per cycle, **toward the stance side**: with `phi = 0` starting the left swing the
  right foot is in stance for the first half-cycle, and G1's +y is the robot's left (the
  `left_foot` site sits at y = +0.1185), so the pelvis moves to -y then. The sign is the
  constraint; the formula follows from the phase convention below.
- `nominal_height` is a **robot-profile constant**, tuned, labelled tuned: for G1 the starting
  value is **0.70 m** (0.886 of the 0.79 m straight-leg stand), the height at which anger at
  intensity 1.0 (feet ±0.282 m) still has 2 cm of leg reach to spare (section 3). Lower is
  safer for reach and worse for the ankle; 0.70 sits between the two limits.
  `com_height_scale = 1.0` means this height, not the standing height; the README says so.

**Phase convention:** `phi = 0` is the instant the left foot begins its swing (first moment of
left single-support after double support), so left swing occupies the start of the cycle and
right swing the start of the second half. The existing "legs antiphase, arms contralateral"
test keeps its meaning under this convention.

### 4.4 Feasibility monitoring

Built here, not retrofitted, as the parent design insists. `GaitTrajectory` gains a
`feasibility: FeasibilityReport` field (frozen dataclass, part of equality like the rest):

- `max_foot_position_error_m`, `max_foot_orientation_error_rad`, taken over all samples of the
  kept lap, per foot and combined.
- `roles_at_limit`: leg roles that sit within a small margin of a joint-range limit on any
  sample.
- `feasible: bool` = position error ≤ 2 mm and orientation error ≤ 0.01 rad. Thresholds are
  constants in one place, tuned, and the spike shows the honest cases land at 0.00 mm while the
  infeasible ones land at 8 mm and up, so there is a decade of separation.
- `worst_phase`: the phase of the worst sample, for diagnostics.

`generate()` **never raises** on infeasibility and never alters parameters. The CLI prints one
warning line to stderr when `feasible` is false (naming the error and the roles at limit) and
still renders, because seeing the grotesque pose is the point of flagging it. Tests assert the
six single-emotion presets at intensity 1.0, the four Phase 0 presets, and the all-zero neutral
are feasible; and that an absurd stride (0.9 m) is flagged with the ankle at its limit.

**If a preset is infeasible, the sanctioned levers are, in order:** the nominal height
(within the ankle envelope), then the offending category offset (fear's crouch), then the
activation gain on `step_length` in `emotion/model.py`. The preset test is never relaxed to
"flagged is acceptable", and the solver never rescales. By section 3's arithmetic the only
preset near a limit today is anger (reach) and the only offset near one is fear's crouch
(ankle), and both are inside the envelope at 0.70 m with the offset in 4.5.

The joint-range clamp after offsets stays. With the QP enforcing limits it is a no-op on the
procedural path and only acts on hand-authored offsets, which is the right division.

### 4.5 Parameter semantics: one retirement, three activations, one rule

**`knee_flexion_bias` is retired.** Under IK the six leg joints are fully determined by the
6-DoF foot pose (no nullspace), so a knee bias is not an independent variable; keeping the
field would make it "modelled but not shown", the exact state this phase exists to end. This
**revises the parent design's parameter table** (section 4.3 of the parent), and the author
can overrule it. Consequences, each of which the plan must carry:

- `GaitParameters` loses the field; `from_yaml` rejects it as unknown with the existing error.
- Evidence claims re-home (`emotion/evidence.py`), keeping their citations and grounding:
  - fear, "increased hip and knee flexion, widespread postural tension" (Roether) →
    `com_height_scale`, direction **-1** (lower pelvis = more flexion).
  - fear's amplitude cue ("smaller and slower than speed-matched neutral", Roether) is already
    the `expansiveness` -1 claim; no new claim is added. Because `expansiveness` now scales
    `step_height` (4.3) and the activation axis drives `step_length`, that claim is rendered
    on the legs for the first time. This is the untangling of fear's two leg cues the Phase 3
    record asked for: posture goes to the crouch, amplitude goes to expansiveness and stride.
  - anger, "hip/knee flexion not significant" (Roether) → `com_height_scale`, direction 0, and
    it joins the explicit "no change" allowlist in the direction test, as anger's head did.
- `CATEGORY_OFFSETS[FEAR]`: `knee_flexion_bias: +0.08` becomes `com_height_scale: -0.04`
  (tuned starting value: 0.70 m x 0.96 = 0.672 m, above the 0.66 m where the ankle limit
  starts to bite in the spike; the direction test is the arbiter and the preset feasibility
  test keeps it inside the envelope, so the builder picks the largest crouch that passes).
  `com_height_scale` sits on no axis; sadness gets no crouch unless the evidence module has a
  sadness claim for it (it does not today).
- `model.py`'s `_PARAMETER_ROLES` maps `com_height_scale`, `step_length`, `step_height`,
  `feet_spacing`, `double_support_frac`, `com_bounce_amp`, `lateral_sway_amp` to the leg roles,
  and `modulate`'s non-negative floor gains `step_length` and `step_height`.
- `tests/sim/test_playback.py`'s strict xfail crouch test is **replaced** by a passing test on
  `com_height_scale` (0.90 vs 1.00 lowers the post-ramp mean pelvis z by at least 3 cm on the
  Z-rail). The `knee_flexion_bias` ramp and no-fall tests move to `com_height_scale`.

**Activated:** `step_length`, `step_height`, `double_support_frac`, `feet_spacing`,
`com_height_scale`, `com_bounce_amp`, `lateral_sway_amp` are consumed by `footsteps.py`. The
`CLAUDE.md` fact listing them as inert is deleted. Default `feet_spacing` moves from 0.15 to
0.20 (G1's natural stance is 0.237; 0.15 is reachable but pigeon-toed; tuned, builder may
adjust by eye).

**The posture rule, stated once:** *amplitudes are relative, postures are absolute.*
Oscillation amplitudes (`arm_swing_amp`, and the elbow swing it drives) are centred on the
profile's rest pose; posture parameters (`elbow_flexion`, `arm_abduction`, `trunk_pitch`,
`neck_pitch`, `com_height_scale`) are absolute targets and the rest pose contributes nothing to
them. This is what the code already does; it is now the documented rule rather than an
accident, and the Phase 3 record's inconsistency note is closed by naming it.

### 4.6 Execution tiers and root modes

**Kinematic tier (`sim/kinematic.py`, `RootMode.FREE`):** per frame, root position from the
pelvis world path (4.3), root quaternion identity, joints from `angles_at(phi)`. `mj_forward`
only, as before. Forward travel is real: the robot's pelvis x grows at `speed`. Tests: the
stance foot's world position is constant to 1 mm through its stance; pelvis x advances exactly
`2 * step_length` per cycle; `data.time` stays 0.

**Physical tier:**

- **Constrained roots play a projection of the parameters**, zeroing exactly what the root
  cannot realise, because a pelvis that cannot move in some direction would otherwise drag the
  stance foot across the floor in that direction. The projected parameters are recorded in
  `SimResult.played_params` so no one mistakes them for the emotion's values:

  | Root | Zeroed | Kept |
  |---|---|---|
  | FREE | nothing (full stride; it falls, as the existing test asserts) | all |
  | Z_RAIL | `step_length`, `lateral_sway_amp` | bounce, crouch (the rail is free in z) |
  | XZ_RAIL | `lateral_sway_amp` | stride, bounce, crouch |
  | WELD | `step_length`, `lateral_sway_amp`, `com_bounce_amp`; `com_height_scale` forced to 1.0 | nothing the pelvis would have to move for |

  `WELD` welds the pelvis at the profile's **nominal height**, not at the straight-leg keyframe
  height, so the feet reach the floor. Sway is lost on every rail; a passive Y slide was
  considered and rejected because it reopens the stability question Phase 0 settled for
  pitch, and sway is a kinematic-tier cue anyway. The foot-slip number on `XZ_RAIL` therefore
  measures servo tracking error only, not a sway artefact.
- `RootMode.Z_RAIL` (existing) thus plays in place, as today.
- `RootMode.XZ_RAIL` (**new**): slide X plus slide Z on the pelvis, no pitch (the planar
  root's pitch is what face-planted the Atlas-era robot). The X slide carries a position
  actuator with high gain whose command is `speed * t` (ramped like the joints), so the pelvis
  is **driven** forward at the derived speed while Z stays free for the bob measurement. The
  legs play the full stride. The X range is long (0 to 20 m) so the rail stop is a real fault,
  not a horizon. This is the parent design's "prescribed-root forward variant", decided as a
  driven rail rather than a mocap weld so the Z instrumentation survives.
- `RootMode.WELD`: pelvis welded at the nominal height; projection per the table.

**Fall criteria, per mode** (closing the gap carried out of Phase 3):

| Mode | Criteria (`fail_reason` names which) |
|---|---|
| FREE | uprightness < 0.6 (unchanged; never height) |
| Z_RAIL | rail stop, foot contact lost > 25 % of a 0.5 s window, pelvis drop > 15 cm (unchanged) |
| XZ_RAIL | all of Z_RAIL's, plus **root lag**: X slide more than 10 cm behind its command after the ramp |
| WELD | contact lost, pelvis drop (unchanged) |

**ZMP versus support polygon is a reported margin, never a fall criterion.** Per step after
the ramp: the signed distance from the ZMP to the nearest support-polygon edge (negative =
outside). `SimResult` gains `zmp_margin` (per step), and a `GaitReport` summary gives the
fraction of post-ramp time outside, the mean margin, and, on `XZ_RAIL`, the **foot slip**:
RMS world-frame horizontal speed of each foot's site while that foot has floor contact. The CLI
prints the summary after a `--physics` run. On `WELD` the numbers are computed but labelled as
"pelvis held" in the printout, since the support is artificial there.

### 4.7 Calibration and the Z-rail survival test

The Z-rail survival test (`test_z_rail_survives_six_seconds_on_every_preset`) keeps its
purpose and its two instruments (bounded pelvis bob, bounded no-contact fraction) and is
recalibrated against the IK in-place gait. **Gate:** before the old leg generator is deleted,
the plan runs the survival test against the new gait at `step_length = 0` on all four Phase 0
presets. If it does not survive, the tunables are `step_height`, `double_support_frac` and the
nominal height, in that order; the leg constants are gone and there is nothing else to tune. If
it still fails, the old generator is kept behind the deletion commit and the author is told;
the phase does not quietly ship two generators.

### 4.8 Rendering

`CameraSpec` gains a tracking mode: `mjCAMERA_TRACKING` on the pelvis body with the existing
distance/azimuth/elevation, so the framing is identical to today's but follows the walker. The
scene's checker floor supplies the motion reference. Demo videos default to tracking;
`--camera fixed` keeps the old free camera for the in-place physics clips. `docs/media/*.gif`
are regenerated for the README once the gait is tuned.

### 4.9 CLI

- `demo` renders forward walking by default (kinematic) and prints the feasibility warning if
  any. `--physics` accepts `--root z_rail|xz_rail|weld|free` (default `z_rail`, as now) and
  prints the `GaitReport` summary.
- Exit codes unchanged: 2 user input, 1 environment, no traceback escapes `main`.

### 4.10 Documentation and repo changes

- `pyproject.toml`: pins per 4.1 and 4.2; version bumps to 0.2.0.
- `CLAUDE.md` "facts that bite": the pin paragraph is rewritten for 3.17 (what was ported, why
  the upper bound stays tight); the inert-fields fact and the leg-constants fact are deleted;
  the knee bias fact becomes the `com_height_scale` and nominal-height fact; the straight-leg
  keyframe and bent-knee seed become a fact; `XZ_RAIL` joins the root-mode facts; the
  dependency rule gains `gait/ik.py` (imports `mink`, `robot/`) and `gait/footsteps.py`
  (pure functions of `GaitParameters`, imports nothing from `robot/`).
- `README.md`: the Limitations entry saying stride is not rendered is replaced by the honest
  statement of what each tier shows; the nominal-height meaning of `com_height_scale` is stated.
- Phase 3's record note about posture absolute/relative is closed by the rule in 4.5.

---

## 5. Order of work

1. Port (4.1) to a green-CI checkpoint. Author decides merge-or-continue.
2. Dependencies (`mink`, `qpsolvers[daqp]`), `footsteps.py` (pure, tested without MuJoCo),
   `ik.py` (welded-root solve, seed, two laps, feasibility report).
3. New `generate()` composing IK legs with the procedural upper body; feasibility on the
   trajectory; old leg generator still present.
4. Kinematic tier with the world root path and tracking camera; zero-slip and stride-distance
   tests; the demo video.
5. Parameter changes (4.5): retire `knee_flexion_bias`, re-home the claims, fear offsets,
   crouch test replaces the xfail.
6. Z-rail gate (4.7), then delete the old leg generator and its constants.
7. `XZ_RAIL`, root-lag criterion, `zmp_margin`, foot slip, `GaitReport`, CLI printout.
8. Docs (4.10), media regeneration, whole-branch review.

**Drop order if the schedule slips:** step 7's `XZ_RAIL` and `GaitReport` first (the physics
tier then stays in-place and the parent design's "honest limitation" paragraph stays in the
README), then the tracking camera (a fixed wide camera still shows the stride). Nothing in
steps 1 to 6 is droppable; they are the phase.

---

## 6. Left to the builder

- Exact easing functions for swing x and z (C2 at both ends is the constraint).
- N samples per cycle and QP iterations per sample (120 and 8 are the spike's values).
- The bent-knee seed pose (spike used hip -0.2, knee 0.4, ankle -0.2) and the posture-task cost.
- Tuned defaults: nominal height (0.70 m start), `feet_spacing`, `step_height`, fear's crouch
  offset (-0.04 start), the X-rail actuator gain, and the feasibility thresholds. All labelled
  tuned, and all bounded by the preset feasibility test rather than by taste alone.
- Whether `FeasibilityReport` is computed from the solver's own task errors or re-measured with
  `mj_forward` on the fitted curves (the latter is the honest one, since the spline fit can add
  error; it costs one forward pass per sample).
- How `GaitTrajectory` exposes the root path (a method taking `(phase, cycle_index)` or
  absolute time; either is fine as long as the kinematic and physical tiers share it).
- Whether `XZ_RAIL`'s rail actuator is added in `root.py` beside the slide joint or in
  `physical.py` as a control-time constraint.

---

## 7. Risks

1. **`daqp` wheels on the CI matrix.** Verified on Windows/py3.11 only. If a job lacks a wheel,
   `qpsolvers[clarabel]` or `[osqp]` is the fallback; the solver name is one constant.
2. **3.17 physics differences.** Contact or integrator defaults may have moved since 3.2.5; the
   port checkpoint (4.1) is where that shows, before any gait change confounds it.
3. **The in-place IK gait on the Z-rail** may bob or lose contact differently from the
   hand-tuned legs. Gate in 4.7; the tunables are named.
4. **Ankle pitch range** binds at deep crouch (measured at 0.66 m). Fear's crouch offset must
   stay inside the feasible envelope at intensity 1.0, which the preset feasibility test
   enforces.
5. **Flat feet read as robotic** (parent risk 9.3). Accepted for this phase; a heel-toe foot
   roll is a later parameter, not a patch.
6. **Spline fit error.** Fitting a C2 periodic cubic through 120 samples of a curve with a
   stance-to-swing corner can overshoot near the corner; the honest feasibility measurement in
   section 6 catches it, and N is the knob.

---

## 8. What changes against the parent design

- Parameter table: `knee_flexion_bias` retired (4.5). Everything else in the table stands.
- Section 4.5 "prescribed-root forward variant": decided as a driven X slide plus free Z slide
  (`XZ_RAIL`), not a mocap weld.
- Section 8 "IK fallback": not taken; `mink` is committed, with `daqp`.
- Section 4.3.1 "feasibility clamping is strictly last": kept as the joint clamp; feasibility
  itself is a flag, not a clamp (4.4).
- `com_height_scale = 1.0` means the tuned nominal walking height, not the standing height.
- `expansiveness` scales `step_height` but not `step_length` (4.3), so the amplitude group is
  arm swing, abduction and swing clearance; the knee leaves the group with the old generator.
- On constrained roots the physics tier plays a recorded projection of the parameters (4.6);
  the parent's "feet-planted and Z-rail physics cannot express stride" paragraph becomes the
  projection table.

---

## 9. Spike record (throwaway, not product code)

Run 2026-10-25 in a scratch venv with `mujoco==3.17.0`, `mink==1.3.0`, `qpsolvers==4.13.0`,
`daqp==0.10.3`, Python 3.11.3, against the cached menagerie G1 `scene.xml`. Script and full
output are kept outside the repo in the session scratchpad (`phase4-spikes/README.md`).

"Separation d" below means foot targets at x = ±d/2 relative to the pelvis, both on the floor
(the double-support instant), i.e. `step_length * (1 + double_support_frac) = d`.

| Case (pelvis height h, feet level, y ±0.1185) | Max residual |
|---|---|
| h 0.76, separation 0.35 | 0.00 mm |
| h 0.76, separation 0.50 | 130 mm (back leg cannot reach) |
| h 0.74, separation 0.35 / 0.50 / apex 0.08 with other foot 0.175 back | 0.00 / 1.2 / 0.00 mm |
| h 0.72 and 0.70, all three cases | 0.00 mm |
| h 0.66, all three cases | 8 to 10 mm, rear ankle pitch on its -0.873 limit |
| h 0.74, feet 0.15 apart; sway 0.03 + bounce 0.02 | 0.00 mm |
| h 0.74, separation 0.90 (absurd) | 68 mm, ankle on its +0.524 limit |
| Seeded from the straight-leg keyframe, any case | 35 to 130 mm (singular knee) |

Not measured in the spike and left to the preset feasibility test: anger's 0.564 m separation
at 0.70 m (geometry says 2 cm of reach to spare) and fear's 0.672 m height.

Timing: 10 to 14 ms per 40-iteration solve; a 120-sample lap at 8 iterations ≈ 0.28 s. The
second warm-started lap had 0.00 mm residual on every sample and its wrap step (0.09 rad) was
no larger than its largest intra-cycle step.
