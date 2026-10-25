# Emotive Gait — Design for Professionalization

**Date:** 2026-10-25
**Status:** awaiting approval

---

## 1. Intent and why

The project's goal is that a simulated humanoid walks in a way a human observer reads as a
specific emotion, and that the emotional modulation is a controllable, continuous knob.

The repository currently cannot reach that goal, for a reason deeper than bugs. Three facts, each
verified during this review:

1. **There is no walk cycle.** `clipped_spline_points.npy`, the artifact the whole pipeline hands
   off, contains motion on exactly one joint (`back_bky`, torso pitch, 0.64 rad peak-to-peak).
   Every hip, knee, and ankle channel is exactly zero. No gait was ever authored.
2. **The robot cannot express the emotions.** The model was hand-reduced to 11 DoF with no arms and
   no head yaw. The affective-gait literature finds that upper-body movement carries more
   expressive information than lower-body for every emotion except fear, and that head inclination
   is the single strongest posture cue. The chosen robot structurally excluded most of the signal.
3. **The gait format cannot hold the parameters the emotion model needs.** Every quantity the
   literature names — step length, step height, trunk lean, arm-swing amplitude, bounce — is a
   task-space or postural quantity. In a per-joint curve representation none of them is a variable;
   they are emergent consequences of joint amplitudes and phase offsets. The old `emotion_from`
   reached for quantities its own data structure could not express, which is why it was reduced to
   adding scalar offsets to joint angles.

So this is not a cleanup that happens to touch the gait code. The gait and emotion layers are being
rebuilt on a representation where the literature's parameters are literal variables, on a robot
that has the joints to show them. Repo hygiene rides along.

**Why now:** the work stalled mid-port from Webots/NAO to MuJoCo/Atlas, leaving dead code paths
that make the project unrunnable. It needs to become something a stranger can clone and run, and
that the author can build on.

---

## 2. Goals, non-goals, anti-goals

### Goals

- `git clone`, install, one command, and a humanoid moves with visibly emotional gait. Stated
  precisely, because the physics measurements in section 4.5 constrain this: the one command
  produces **forward emotional walking in the kinematic tier** and **emotional in-place stepping
  under physics**. It does not produce dynamically-balanced forward walking, which is impossible
  open-loop and is a non-goal. Phases 1 to 3 are a shippable milestone on the second of those;
  Phase 4 delivers the first.
- Six continuous emotion intensities, mixable, neutral at all-zero.
- Every emotion-to-kinematics claim traceable to a citation, or explicitly marked as unsourced.
- A rewritten authoring GUI in real units with live emotion preview.
- **A measured result, not only a capability.** An evaluation harness that renders randomized
  stimuli (rendered robot and point-light), collects forced-choice ratings, and reports recognition
  rates — plus a small, transparently caveated pilot run. See section 10.3.
- Tests, linting, CI, a README with embedded demo video.
- Runs on the author's current environment: Windows 10, Python 3.11.3, MuJoCo 3.2.5.

### Non-goals

- **Locomotion research.** No novel walking controller. Not a contribution of this project.
- **Machine learning.** No classifier, no training, no datasets acquired. A documented seam is left
  where a learned emotion model could later replace the analytic one.
- **Hardware.** Simulation only.
- **Webots support.** The `.motion` import path is dropped, not ported.

### Anti-goals (would make this a failure even if it technically works)

- **A gait that is stable and emotionally flat.** This is the likely failure mode, more likely than
  falling over. Any balance machinery with authority over the trajectory will fight exactly the
  features that carry emotion — the slump, the shuffling steps, the exaggerated bounce. The more
  effectively such a controller works, the more completely it cancels the signal. Stability must
  never be allowed to override expression.
- **Fabricated precision.** The literature contains no joint angles in degrees for these effects.
  Presenting tuned numbers as literature-derived would be worse than presenting them as tuned.
- **A README that overclaims.** Related 2026 work exists (section 7). The repo must position
  honestly against it.

---

## 3. Constraints

- Windows 10, Python 3.11.3, MuJoCo 3.2.5, SciPy 1.10.1, NumPy 1.24.3 installed. `pytest` and
  `ffmpeg` present. `ruff`, `uv`, `pre-commit` absent (ordinary pip installs).
- **The project gets its own virtual environment.** The author's packages are currently global, and
  Phase 4 raises MuJoCo from 3.2.5 to 3.17, which would otherwise reach into their unrelated work.
  A venv makes that upgrade reversible and contained, and is required for honest CI anyway. The
  declared floor is `mujoco>=3.2.0` for Phases 1 to 3, raised only if `mink` is adopted in Phase 4.
  The global 3.2.5 install is left untouched.
- `placo` is unusable here: no Windows wheels.
- `mink` (task-space IK) requires `mujoco>=3.10.0`; the environment has 3.2.5. This is the only
  dependency that forces a MuJoCo upgrade, and it is confined to Phase 4.
- `mujoco-menagerie` requires only `mujoco>=3.2.0`, so Phases 1 to 3 need no upgrade.
- Public repo: every vendored or downloaded asset needs a license check.

---

## 4. Decisions made (and why)

### 4.1 Robot: Unitree G1, 29-DoF, with an added neck joint

`mujoco_menagerie/unitree_g1/g1.xml`. `nq=36 nv=35 nu=29 nkey=1` (`stand`), 33.34 kg,
BSD-3-Clause, ~36 MB of meshes, `compiler angle="radian"`. All 29 actuators are
`<position kp="500" dampratio="1">`, and **actuator name == joint name** throughout.

Chosen on measured evidence, not documentation. Candidates were compiled under the project's own
MuJoCo 3.2.5 and tested. G1 tracks commanded angles to 1 to 2 degrees on the channels that carry
emotion, and survived a 6-second scripted expressive in-place gait at plus/minus 0.45, 0.90 and
1.80 rad amplitude with pelvis height flat to 2 mm.

The decisive requirement: commanding joint angles is only meaningful if the commanded angle is
actually *reached*. Apptronik Apollo is the only anatomically complete candidate and the only one
satisfying every requirement on paper, but its stock gains gravity-sag the neck and shoulder — the
two highest-value channels — by about 15 degrees, and it fell when forward lean and leg flex
combined. Booster T1 tracks beautifully and has a real head, but its `Waist` joint is yaw-only,
failing the torso-pitch requirement, and it collapses from its own keyframe at stock gains.

Rejected: `unitree_h1` (no torso pitch, torque actuators, `nkey=0`), `robotis_op3` (no torso joint
at all), `pal_talos` (opaque numeric joint names defeat semantic role mapping), `berkeley_humanoid`
and `agility_cassie` (no arms), MuJoCo's built-in `humanoid.xml` (`angle="degree"`, a trap for a
radian contract, and absent from the pip wheel). The 23-DoF G1 variant **does not exist in
Menagerie**; the one in `unitree_ros` has waist yaw only and `nkey=0`.

**G1's one real gap: it has no head or neck joint.** This matters more than its position in the
requirement list suggested, because head inclination is the strongest single posture cue in the
literature and is what separates sadness from everything else. The fix: G1's `head_link` is already
a standalone mesh geom with explicit pos/quat, rigidly parented to `torso_link`. Adding a neck is
roughly 15 lines of `MjSpec` — new child body of `torso_link`, move the head geoms onto it, add a
`hinge` on axis `0 1 0` plus a `position` actuator. No mesh authoring. This was verified feasible
but **not implemented or tested**; the head body's mass and inertia must be split out of
`torso_link` sensibly, and that is real work, not a copy-paste.

### 4.2 Semantic joint roles

The original `emotion_from` hardcoded NAO joint names (`LHipPitch`, `RShoulderPitch`) against an
Atlas model, which is why its call site was commented out. The permanent fix is an indirection: a
`JointRole` enum of semantic roles, and a `RobotProfile` mapping roles to actuator names for a
given model. The profile reports which roles are **absent** so the emotion model degrades
explicitly rather than raising deep inside a loop.

The profile also owns **sign conventions**. G1's `left_shoulder_roll` and `right_shoulder_roll`
have mirrored ranges, as do the hip rolls. A left/right-symmetric modulation must negate the roll
channels, not copy them. Encoding this once in the profile prevents a whole class of bug.

**Named pitfall:** the old GUI built curves from `for i in range(nq)` with `model.joint(i)`. On G1
that yields 30 joints including the freejoint (range `0 0`) and is off-by-one against `qpos`.
Correct form is to iterate `range(model.nu)` and resolve `model.actuator_trnid[i,0]` to
`model.jnt_qposadr[j]`.

### 4.3 Gait representation: parameters as the source of truth

A `GaitParameters` dataclass, serialized to YAML, is canonical. Phase is normalized `phi` in
`[0,1)` over a full two-step cycle. Sampled arrays are a cache keyed by a parameter hash, never the
source of truth. The old `(nq, 600, 2)` contract — which welded time to array index and angle to
canvas pixels — is retired. Nothing of value is lost; the saved file holds no gait.

Hand-authored per-joint offset curves, where a human drags a control point, are stored as
**periodic cubic spline knots** in phi via `scipy.interpolate.CubicSpline(..., bc_type='periodic')`.
This is a strict improvement on the old code: the GUI hand-shared the first and last anchor's Y to
make the cycle loop, which bought only C0 continuity; `bc_type='periodic'` matches first and second
derivatives at the wrap, giving C2 for free with less code.

The parameter set to expose, grouped, with the literature-facing ones first:

| Parameter | Unit | Note |
|---|---|---|
| `cadence` | steps/s | anger up, sadness down |
| `step_length` | m | anger/happiness up, sadness/fear down |
| `step_height` | m | swing-foot apex clearance |
| `double_support_frac` | [0,1) | sadness up (shuffling) |
| `feet_spacing` | m | |
| `com_height_scale` | [0.8,1.0] | crouch; fear/sadness down |
| `com_bounce_amp` | m | one period per step |
| `lateral_sway_amp` | m | sadness up |
| `trunk_pitch` | rad | positive is forward slump |
| `neck_pitch` | rad | positive is gaze down; sadness up |
| `knee_flexion_bias` | rad | fear up |
| `arm_swing_amp` | rad | shoulder pitch, contralateral |
| `arm_abduction` | rad | shoulder roll; expansiveness |
| `elbow_flexion` | rad | base flexion; anger/fear up |
| `elbow_swing_ratio` | [0,1] | fear moves mainly the lower arm, sadness the whole arm |
| `expansiveness` | [0.5,1.5] | post-multiplier on the amplitude group |

`speed` is **derived** (`step_length * cadence`) and read-only. `expansiveness` is implemented as a
post-multiplier over the amplitude group, not as a term inside each formula, so its semantics live
in one place.

**`hold_speed_constant` mode is required, from day one.** Speed is the canonical confound in this
literature. The ability to vary slump, bounce and arm swing with speed pinned is what permits the
claim that an effect is not merely speed.

Pinning speed over-determines `cadence` and `step_length`, so something must absorb the constraint,
and the three ways of doing it are perceptually different (short quick shuffling steps are not the
same stimulus as long slow strides at the same m/s). **Default: proportional renormalization.**
Let the emotion model set `cadence` and `step_length` as usual, then scale both by
`k = sqrt(target_speed / (cadence * step_length))`. This preserves the emotion's cadence-to-stride
*ratio*, which is itself a cue, while pinning their product. The two alternatives — hold cadence and
solve for stride, or hold stride and solve for cadence — are exposed as an enum rather than hidden,
because which quantity gives way is part of what the methodological claim controls for. This makes
`GaitParameters` a free/derived/pinned system rather than a flat struct; it needs a small
constraint resolution step, not just a boolean.

### 4.3.1 Composition order of the three motion sources

Three things produce joint motion and the order matters, so it is fixed here rather than left to
discovery:

1. **Procedural base.** `GaitParameters` evaluated by the generator at phase `phi`.
2. **Emotion.** Applied to the *parameters*, never to the generator's output. This is what makes
   the emotion model interpretable and citable — a modulation is a statement about stride length or
   trunk pitch, not about an array of numbers.
3. **Hand-authored spline offsets.** Applied last, in joint space, per channel. Default mode is
   **additive correction** on top of the procedural result. A per-channel **override** mode is also
   supported, replacing that channel's procedural curve outright, because hand-authoring is a
   first-class workflow and a user must be able to say "no, this joint does exactly this".
4. **Feasibility clamping** (Phase 4) is strictly last, after everything above.

Consequences, stated so the interfaces are not guessed at:

- `expansiveness` and the other amplitude multipliers scale the **procedural** amplitudes only.
  Hand-authored offsets are the author's explicit intent and are never silently rescaled.
- Spline knots live in the **same canonical YAML** as the parameters, as a per-channel optional
  field. One artifact, one hash, one thing to version.
- "Neutral is the identity" therefore means: with all six intensities at zero, the output equals the
  procedural base plus any hand-authored offsets, unchanged. The test is well-posed against that
  definition, not against the bare procedural base.

### 4.4 Emotion model: six sliders over three affective axes

The six intensities stay as the user-facing control, continuous in [0,1], neutral at all-zero.
Underneath, they map onto three axes rather than being averaged independently:

- **activation** — speed, cadence, stride, overall amplitude
- **valence** — neck pitch, trunk erectness
- **tension** — elbow and limb flexion

Then a small number of category-specific offsets are added on top (fear's knee and hip flexion;
fear's low `elbow_swing_ratio`).

This is a deliberate correction. Averaging six independent per-emotion joint offsets is defensible
for the arousal-linked parameters, which vary monotonically. It is *not* defensible for posture
cues: averaging happy and sad cancels the head and elbow cues toward neutral rather than producing
anything mixed. The literature's own confusion structure is organized by activation (anger confuses
with happiness, fear with sadness), with posture giving finer discrimination inside an activation
level.

**Evidence handling is a first-class requirement.** An `evidence` module carries, per emotion and
per parameter, the direction, the citation, and a grounded/ungrounded flag.

- **Grounded** (direction, and speed in m/s): anger, fear, happiness, sadness.
- **Unsourced, must be flagged in code, docs and UI:** **disgust and surprise have no gait
  kinematics in the literature at all.** Not thin evidence — none. They are placed by
  arousal/valence reasoning, which is a modelling choice and must never be presented as a finding.
- **No parameter may claim a magnitude in degrees from a citation.** The literature yields walking
  speeds in m/s and swing-phase percentages; everything else is directional only. Joint offsets are
  relative changes against neutral, with the literature constraining sign and rank ordering.

Encouragingly, this minimal approach has support: Roether et al. built artificial happy and sad
walkers by adding only the two largest posture and movement changes to a neutral gait, and obtained
perceptual adaptation effects statistically indistinguishable from natural emotional gaits.

### 4.5 Execution: two tiers, and no free-base walking

Generation and stabilization are separate modules. The generator emits a reference trajectory; how
much physics is applied to it is a configuration choice.

**Kinematic tier** — set `qpos`, call `mj_forward`, never `mj_step`. No gravity, no contacts. Exact
foot placement, zero slip by construction. This drives the GUI preview and produces perceptual
stimuli. This is not a shortcut but the methodological norm: the entire body of psychophysics
establishing which cues read as which emotion used point-light displays with no dynamics
whatsoever.

**Physical tier** — `mj_step` with G1's stock position actuators as the per-joint tracker. No
additional control code. Measured to work today.

Root options for the physical tier, all measured with an identical stepping trajectory:

| Root | nq/nv | Result |
|---|---|---|
| free (6-DoF floating) | 36/35 | fell at t=1.20 s |
| planar: slide X + slide Z + hinge pitch (the old Atlas trick) | 32/32 | fell at t=1.65 s |
| **slide Z only (height rail)** | 30/30 | **survived 8 s**, 3.5 cm pelvis bob |
| weld (pelvis fixed) | 29/29 | survived 8 s, no bob |

**Corrected by Phase 0 (see section 11).** An earlier version of this section implied that a
free-root physical tier was viable for in-place stepping so long as the leg motion was symmetric.
Phase 0 measured that directly, with the neck present and with the four emotion presets, and it is
false at expressive amplitudes:

| Configuration | Result across all four presets |
|---|---|
| A — antiphase legs, free root | fell at 0.74 to 2.21 s |
| B — symmetric leg bob, free root | fell at 1.02 to 2.25 s |
| **C — antiphase legs, Z-rail root** | **survived 6 s, all four presets**, nq == nv == 31 |

Symmetry is not what saved the original measurement; small amplitude was. The configuration that
survived 6 s in the robot survey used only small flex around the standing pose, which is not an
expressive gait. **Therefore the Z-rail is the physical tier, not one option among several.** A free
root is a stretch goal, viable only at amplitudes too small to carry emotion, and must not be
promised.

**Open-loop free-base walking is impossible and must not be attempted.** It falls within roughly
one gait cycle at every amplitude tested (0.3, 0.6, 1.0 rad). This is not a gain-tuning problem:
walking is an unstable equilibrium requiring foot placement and ankle torque modulated against
measured state, and a scripted trajectory has no feedback path. The Z-rail is the supported
constrained option; the old planar root is specifically the wrong constraint, since it removes
lateral tipping while still permitting pitch, so the robot face-plants instead of toppling.

**Honest limitation, to be stated in the README:** feet-planted and Z-rail physics cannot express
`step_length` or forward `speed` — and speed is the strongest perceptual cue. True stride comes
from the kinematic tier, or from a prescribed root path in Phase 4. The physical tier demonstrates
that the upper-body and posture modulation is dynamically plausible; it is not the stimulus source.

### 4.6 Retire the LQR controllers

`new.py` and `new2.py` are deleted. This **reverses the guidance in the current `CLAUDE.md`**,
which recommended keeping both as a deliberate A/B.

The reason is not that they are low quality. It is that their control law,
`ctrl = ctrl0 - K @ [differentiatePos(qpos0, qpos); qvel]`, exists to drive the robot *back to*
`qpos0`. A standing balance controller and a trajectory player are opposite objectives: feed it a
gait and it fights the gait. This is precisely the anti-goal in section 2. Position actuators
already are a per-joint PD tracker, which is why the measured in-place gait simply works.

`compute_zmp` is **kept and finally wired to MuJoCo** — as instrumentation, not control. Plotting
ZMP against the support polygon quantifies how dynamically infeasible each emotional gait is, which
is a genuine secondary result. `another.py` (abandoned TVLQR sketch) is deleted.

**Correction to an earlier version of this document, which said to port both helpers.**
`compute_zmp` is the standard formula and ports cleanly. `calculate_moments` must be **replaced,
not ported**. As written (`emotive-gait.py` lines 271 to 283) it sums per-contact-point moments
against a single scalar vertical force derived as `mass * acceleration[2]`, ignoring horizontal
reaction forces and the actual per-contact normals. MuJoCo exposes the centre of pressure directly
via `mj_contactForce` over the foot contacts, which is both simpler and correct. Since section 4.6
promotes ZMP-versus-support-polygon to a reportable result, it has to be trustworthy; porting the
original would quietly make it not.

### 4.7 Dependency on the robot model

`pip install mujoco-menagerie` — verified on PyPI at 2026.10.4, `requires-python >=3.10.12`, single
dependency `mujoco>=3.2.0`, so **no MuJoCo upgrade needed for this**. The wheel is a ~30 KB thin
downloader; models are fetched from a GitHub release on first use and cached under
`%LOCALAPPDATA%`.

Trade-off accepted: tiny install, but first run needs network. The quickstart includes
`python -m mujoco_menagerie prefetch unitree_g1`. The pinned version and the model OID are recorded
for provenance.

**Risk, stated:** this package is about one month old (10 releases, all September to October 2026)
and its API may still shift. The hedge is a documented sparse-checkout fallback:

```sh
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git _menagerie
cd _menagerie && git sparse-checkout set unitree_g1 && git checkout
```

A git submodule is rejected: it clones all ~70 robots for one model. G1 is BSD-3-Clause, so
vendoring is permitted if that becomes preferable; the Unitree copyright notice and disclaimer must
travel with the files.

---

## 5. Repository changes

**Delete** (history preserves them): `nao.urdf`, `nao (1).urdf`, `NAOH25V60.urdf`, `robot.urdf`,
`atlas_v4_with_multisense.urdf`, `atlas_minimal_contact.urdf`, `atlas_minimal_contact.xml`,
`test.xml`, `scene.xml`, `naoV50_generated_urdf/`, `meshes/`, `MUJOCO_LOG.TXT`, `__pycache__/`,
`clipped_spline_points.npy`, `TaiChi.motion`, `another.py`, `new.py`, `new2.py`,
`emotive-gait-pybullet.py`, `emotive-gait-mujoco.py`.

**Preserve by porting:** `compute_zmp` to `sim/metrics.py`. `calculate_moments` is replaced by a
`mj_contactForce`-based centre-of-pressure computation, per section 4.6.
`emotionmodel.py`'s dataset shortlist to `docs/future-learned-model.md`.
`emotive-gait-mujoco.ipynb` to `docs/archive/` with a note that it is a working transcript.
`emotion_from`'s intent to the new emotion model; its NAO joint names do not survive.

**Layout:**

```
src/emotive_gait/
  robot/     roles.py  profiles/g1.py  neck.py  loader.py
  gait/      params.py  spline.py  generator.py  footstep.py
  emotion/   axes.py  model.py  evidence.py
  sim/       kinematic.py  physical.py  metrics.py  render.py
  eval/      stimuli.py  rating.py  report.py  pointlight.py
  gui/       app.py
  cli.py
tests/   docs/   .github/workflows/   LICENSE
```

`LICENSE` is MIT, per section 10.1. `unitree_g1`'s own BSD-3-Clause notice travels with the model
if it is ever vendored rather than fetched.

**Tooling:** `pyproject.toml` (setuptools, src layout), `ruff` (lint and format), `pytest`,
`pre-commit`, GitHub Actions on Windows and Linux, `.gitignore` covering `MUJOCO_LOG.TXT`,
`__pycache__`, `*.npy`, rendered video. Python 3.11 floor.

---

## 6. Phases

**Phase 0 — visual confirmation (do first, about 30 to 60 minutes).** Install Menagerie, render G1
under a large scripted expressive motion, produce a video. This closes the one unverified assumption
behind the robot choice: amplitudes were *measured*, never *seen*.

Stock G1 cannot exercise head pitch, which this document calls the strongest single posture cue, so
Phase 0 includes a **provisional neck**: the hinge and actuator only, with inertia left at whatever
the compiler infers. That is legitimate here precisely because Phase 0 renders kinematically — no
`mj_step`, so mass and inertia do not affect the picture. The careful mass/inertia split and gain
verification stay in Phase 1. Without this, Phase 0 would pass or fail on the wrong channels.

**Decision rule, so a builder can act on the outcome without waiting on anyone:**

- Arm swing, torso pitch, cadence and knee flexion all read clearly, and the provisional neck moves
  the head visibly: **proceed on G1.**
- The expressive channels are present but underwhelming: **proceed on G1 and record the risk.**
  Amplitude is a parameter; a merely weak-looking default is a tuning problem, not a robot problem.
- A channel is structurally unusable — a joint whose range is too small to see, or geometry that
  self-collides at useful amplitude: **fall back in this order.** (1) Booster T1: a real head with
  sub-degree tracking, Apache-2.0, 8.4 MB, accepting the loss of torso pitch and applying the
  verified `kp x 2` keyframe fix. (2) Apptronik Apollo with retuned neck and shoulder gains, plus
  gravity compensation, accepting its 81 kg stability margin. Record which cue was traded away
  either way.

**Phase 1 — foundation.** Packaging, tooling, CI, deletions. Robot layer: roles, G1 profile, neck
injection, loader. Tests: model compiles, every role resolves, neck exists and actuates, roll
mirroring correct.

**Phase 2 — gait core.** `GaitParameters` plus YAML round-trip. Periodic spline primitive.
Joint-space generator (in-place, no IK). Physical runner with the root variants. ZMP/CoM metrics,
fall detection. Tests: continuity at the phi wrap **and** at the stance-to-swing handoff — the
latter is the easier one to get wrong; determinism; parameter-to-trajectory shape.

**Phase 3 — emotion model.** Axes, category offsets, evidence module with grounded/ungrounded
flags, `hold_speed_constant`. Tests: neutral is the identity; monotonicity along each axis;
documented mixing behaviour; graceful degradation when a role is absent.

**Phase 4 — forward locomotion.** MuJoCo upgrade to 3.17, `mink` task-space IK, foot and CoM
trajectories, exact placement, kinematic stimulus tier, prescribed-root forward variant.
**Parameter feasibility monitoring is built here, not retrofitted** — IK will fail or produce
grotesque poses at parameter extremes, and per-tick task residual must be monitored so a sample is
clamped or flagged. **The 3.17 upgrade is a port, not a re-verify** (learned from PR #1's first CI
run, which resolved an unbounded `mujoco>=3.2.0` to 3.17.0): `MjSpec.find_body` became
`MjSpec.body`, and `MjsGeom.delete()` was removed, so `robot/neck.py`'s head-geom move and
`robot/root.py` must be rewritten against the 3.17 `MjSpec` API before the pin
(`mujoco>=3.2.0,<3.3`, added in that fix) can be raised. Re-run the Phase 0 mass, CoM and inertia
checks (section 11) after the port.

**Phase 5 — evaluation harness.** Randomized stimulus generation, forced-choice rating UI,
recognition-rate reporting, point-light render mode via geom-group toggling. Then a small pilot
with transparent reporting of n and recruitment. See section 10.3 for scope and drop order.

**Phase 6 — GUI rewrite.** Parameter editor in real units, emotion sliders, live kinematic preview.

**Phase 7 — presentation.** README with embedded demo video and the measured recognition rates,
docs, honest related-work positioning, `CLAUDE.md` rewritten for the new architecture.

**Note on ordering:** evaluation precedes the GUI deliberately, because the README is the portfolio
artifact and it is built around the measured result. If the schedule slips, a rougher GUI costs less
than an unmeasured project. This inverts the natural "tools first" instinct and is worth flipping if
the author disagrees.

Phases 1 to 3 need no new dependencies beyond `mujoco-menagerie`. Only Phase 4 touches MuJoCo's
version, which is deliberate: the risky upgrade lands after there is something working to regress
against.

---

## 7. Related work, to be positioned against honestly

- **EMoG** ([arXiv 2609.14432](https://arxiv.org/abs/2609.14432), September 2026) —
  emotion-modulated gait generation. Continuous-intensity emotional style code to MLP to periodic
  gait trajectories to an RL tracking policy, plus an LLM parser, on a custom emotion-annotated
  dataset from professional performers. No code released. Independently arrives at the same
  generation/execution split this design uses. The distinction to draw: EMoG is learned from a
  proprietary dataset; this project is analytic, interpretable, literature-cited and open. That is
  a real difference, not a better one.
- **PAMoR** ([arXiv 2608.28213](https://arxiv.org/pdf/2608.28213)) — parameterized affective motion
  generation in real time. Closest to this design's approach; **must be read before the README is
  written**, as it may overlap substantially.
- **EmoPose** ([arXiv 2609.23414](https://arxiv.org/pdf/2609.23414)) — emotion-aware gesture, not
  gait.
- **Walk Like This** — stylized locomotion via example-driven diffusion.

Verified: the EMoG arXiv ID and title. Not verified: which robot EMoG used; a subagent's claim of
Unitree G1 traced to a different paper's reference list and is excluded.

---

## 8. Left to the builder

- Neck joint mass and inertia split from `torso_link`, and the neck's range (subject to the mesh).
- PD gain verification once the neck actuator is added.
- Concrete numeric defaults for every gait parameter — the literature constrains sign and ordering
  only, so these are tuned by eye and must be labelled as tuned.
- The precise six-intensities to three-axes mixing function.
- GUI toolkit: Tkinter is the incumbent and has no dependency cost; a swap is the builder's call if
  live 3D preview demands it.
- Whether the kinematic tier renders through MuJoCo's offscreen renderer or the passive viewer.
- IK fallback: if the MuJoCo 3.17 upgrade proves disruptive in Phase 4, hand-rolled damped least
  squares on `mj_jac` is about 40 lines and works on 3.2.5, at the cost of hard joint-limit
  constraints.

---

## 9. Known risks

1. **Arm swing may look stiff.** G1's shoulder roll and elbow ranges must be checked before
   `arm_abduction` is committed to as the expansiveness carrier.
2. **Ankle pitch range is usually the binding constraint on step length.** G1's is -0.8727 to
   0.5236 rad. Check before promising stride amplitudes.
3. **Flat-foot walking reads as robotic.** Foot pitch during swing is where ankle range runs out
   first; "heavy-footed" as an anger cue may not be reachable.
4. **The emotion-to-parameter map is the least-specified link in the chain.** Cross-study agreement
   is decent for sadness and for anger/happiness, weaker elsewhere, absent for disgust and surprise.
   Expect to tune by eye; do not overstate.
5. **Stability tests behind the robot choice were 3 to 8 seconds.** They establish "does not
   collapse immediately", not long-run or perturbation behaviour.

---

## Appendix A — Source set for the `evidence` module

This is the input Phase 3 was otherwise missing. It makes populating `evidence.py` a transcription
job rather than an open-ended literature review.

**Verification status, stated plainly.** This appendix was assembled by a research pass during the
design conversation, not by the author reading each paper. Full text was read for Roether's 2011
Tübingen thesis (whose Chapter 3 is the 2009 *Journal of Vision* study — the JoV page itself
returned HTTP 403), Barliya et al. 2013, and Randhavane et al. Everything else is from abstracts or
from the secondary review at [arXiv 2003.11461](https://arxiv.org/pdf/2003.11461), Table III, and is
marked second-hand below. **Every number in this appendix must be spot-checked against its primary
source before it appears in a published README or paper.** The ones most worth re-checking are the
walking speeds, since the thesis is internally inconsistent on happiness.

### A.1 Sources

| Source | Access | Role |
|---|---|---|
| Roether, Omlor, Christensen & Giese 2009, *J. Vision* 9(6) — via [Roether 2011 thesis](https://www.compsens.uni-tuebingen.de/compsens//files/publications/thesisroether2011.pdf) ch. 3–4, 6 | full text | **primary source.** Critical-feature identification, production and perception |
| [Barliya et al. 2013](https://www.compsens.uni-tuebingen.de/compsens//files/publications/Barliya2012.pdf), *Exp Brain Res* | full text | gait kinematics, swing-phase fractions, thigh elevation |
| [Randhavane et al. 2019](https://arxiv.org/abs/1906.11884) (EWalk) | full text | affective features, valence/arousal linear fit |
| Montepare, Goldstein & Clausen 1987 | abstract | stride length, arm swing, heavy-footedness |
| Gross, Crane & Fredrickson | second-hand | effort-shape: expanded vs contracted |
| Halovic & Kroos 2018 | abstract | cadence, stride, per-emotion arm segment |
| Michalak 2009 | second-hand | slumped posture, lateral sway, reduced vertical head movement |
| de Meijer 1989 | second-hand | general body-movement features; **not gait** |
| [Bhattacharya et al., STEP](https://arxiv.org/abs/1910.12906v2) | abstract | motivating cue list; no new kinematics |
| Karg et al., survey | abstract | no per-emotion gait kinematics retrieved |
| Crenn et al. | second-hand | body expansion, head tilt |
| [2020 *Scientific Data* kinematic dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC7478954/) | abstract | only source touching disgust/surprise; **daily-event scenarios, not walking** |

### A.2 The only published magnitudes

Everything else in this literature is directional. These are the numbers:

- **Walking speed**, acted, Roether: anger 1.82 +/- 0.22 m/s; happiness 1.31 +/- 0.36 m/s (the
  thesis also states 1.41 +/- 0.19 elsewhere — flag the inconsistency); fear 0.83 +/- 0.31 m/s;
  sadness 0.68 +/- 0.21 m/s.
- **Swing phase as fraction of gait cycle**, Barliya: anger 38%, fear 34%, sadness 33%.
- **Perception rates**, Roether: real emotional gaits classified correctly 78%; speed-matched
  neutral gaits classified as the matching emotion up to 54% (chance 25%); neutral read as neutral
  ~71%. Fear is the exception — fear-speed neutral was called fear only 28% and sad about 60%.
- **Speed-to-expressiveness correlation**, Roether: anger r = +0.76; happiness r = +0.36;
  sadness r = -0.59; fear r = -0.19 (not significant).
- **Head inclination discriminant loading**, Roether: 0.56 — the strongest posture feature.
- **Speed and movement size** explained ~90% of variance in Roether's dynamic discriminant function.

### A.3 Direction table

Grounded for anger, fear, happiness, sadness. Blank means not reported.

| Parameter | Anger | Fear | Happiness | Sadness |
|---|---|---|---|---|
| speed | highest | low | high | lowest |
| cadence | fastest | fast, short steps | slower than anger | slowest |
| stride length | long | short | long (shorter than proud/angry) | short |
| head flexion | no consistent change | more erect | more erect | **strongly down** |
| trunk flexion | inclined | inclined | unusually erect | inclined / slumped |
| lateral sway | | | | larger |
| arm swing amplitude | large | reduced | large (larger effect than anger) | reduced |
| elbow flexion | **pronounced increase** | **pronounced increase** | reduced | reduced |
| arm segment moved | | mainly **lower** arm | | **whole** arm |
| hip / knee flexion | not significant | **increased** | | |
| limb amplitude | larger | reduced | larger | smaller |
| vertical bounce | smoother than sadness | | "bouncing" | reduced |
| jerk | higher | | higher | lower |

### A.4 Ranked critical features

If only a few parameters can be controlled, control these, in this order:

1. **Speed and overall movement size** — dominant; sets the activation axis. But *not sufficient*:
   real emotional gaits beat speed-matched neutral by 78% to 54%, so expression exaggerates past
   what speed predicts.
2. **Head inclination** — strongest posture feature; separates sadness from everything else.
3. **Elbow flexion** — best predictor of anger and fear; separates them from happiness and sadness,
   and separates fear from sadness at similar slow speeds.
4. **Arm swing amplitude** — upper body carried more expressive information than lower body for
   every emotion except fear (all t > 6.53).
5. **Spine erectness**, plus knee and hip flexion for fear specifically.

**Supporting result:** Roether's Experiment 3 built artificial happy and sad walkers by adding only
the two largest posture and movement changes to a neutral gait. Adaptation aftereffects were
statistically indistinguishable from natural emotional gaits (AP shift 0.48 artificial vs 0.42
natural for happy; 0.62 vs 0.65 for sad; baseline 0.57). Caveats: only happy and sad were tested,
and via adaptation rather than direct classification.

### A.5 What is lost without arms or a head

- **Needs a neck:** head inclination (the top valence cue), head bounce, head jerk.
- **Needs arms:** arm swing amplitude, elbow flexion, upper-arm retraction (fear), and the
  lower-arm-versus-whole-arm distinction that separates fear from sadness.
- **Legs and torso alone suffice for:** speed, cadence, stride, trunk erectness, knee and hip
  flexion, swing-phase fraction, bounce, overall movement size.
- **Net:** without arms and head you keep the activation axis and lose most within-pair
  discrimination. Fear survives best (its signature includes leg flexion); sadness suffers most.
  This is the evidence behind requiring arms and adding a neck in section 4.1.

### A.6 Evidence on blending — why three axes, not six averages

**For linearity:** Roether represents all gait movement as linear mixtures of three source
functions, with mixing-weight differences tracking joint amplitude at r = 0.86; perceptual morphs
along a neutral-to-emotion continuum behave as a graded dimension; observers combined cues across
body parts near-optimally, consistent with linear weighted averaging.

**For low dimensionality:** perceptual confusions follow activation, not category — anger confuses
with happiness, fear with sadness. Roether's discussion proposes a third axis, limb flexion or
tension, separating anger and fear (flexed) from happiness and sadness (extended). That is the
origin of this design's activation/valence/tension triple. Note it is Roether's *discussion*, not a
tested model. Randhavane fits perceived affect linearly from class probabilities
(`valence = 0.67*p(happy) - 0.04*p(angry) - 0.74*p(sad)`,
`arousal = -0.35*p(happy) + 0.86*p(angry) - 0.37*p(sad)`), descriptive of labels rather than causal.

**Against naive averaging:** no study was found on perception of *blended* emotional gaits.
Averaging opposing emotions drives speed and amplitude toward neutral and cancels the head and
elbow cues, yielding neutral rather than "mixed". Fear's cues do not follow speed at all, so a
speed-led blend cannot produce fear. Confusions are asymmetric (anger gaits are called happy more
often than the reverse).

### A.7 Disgust and surprise

No gait kinematics exist for either, in any source checked. The only dataset touching them uses
daily-event scenarios rather than walking and reports no kinematic results (its pilot recognition
accuracies, 0.906 disgust and 0.871 surprise, are recognition rates, not kinematics). Any
placement is a modelling choice via arousal and valence and must be flagged as such everywhere it
surfaces. See the open question in section 10 on whether they ship at all.

### A.8 Caution on provenance

Most of this data is acted gait by non-professional actors. Roether found no recognition difference
between lay-theatre and non-theatre actors, but actor gender affected fear recognition
substantially (about 90% for female actors versus 60.5% for male). Treat these as *portrayals*, not
spontaneous emotional response — which is the right frame anyway, since the project cares about
perception.

---

## Section 11 — Phase 0 results (executed 2026-10-25)

**Verdict: GO on Unitree G1.** Four in-place stepping clips were rendered kinematically with a
provisional neck. Sad and angry are unmistakably distinct in a single still frame: sad shows head
down, trunk slumped, arms held close; angry shows erect head, open posture, large arm swing. The
neck injection is the difference between those reading as two emotions and as one posture.

Facts established, each of which the implementation depends on:

1. **Python >= 3.12 is required.** `mujoco-menagerie` 2026.10.4 calls
   `tarfile.TarFile.extractall(filter='data')`, which needs PEP 706: Python >= 3.11.4, or >= 3.10.12
   on the 3.10 line. The author's machine has 3.11.3, which satisfies the package's declared
   `requires-python >=3.10.12` numerically while lacking the feature. This is a packaging bug in the
   dependency, not in this project. **The only manual install this project needs.**
2. **Load `scene.xml`, never `g1.xml`.** `g1.xml` is the robot alone with no ground plane; under
   physics the robot free-falls through the void at 1 g, which is easily mistaken for a control
   failure. `scene.xml` adds the floor, lights and skybox.
3. **The neck mass split, with exact numbers.** `torso_link` has `explicitinertial=1` and
   `mass=7.818`, which already includes the head, so the head collision geom's `density=1000` is
   ignored. The `head_link` mesh at density 1000 is **1.8025 kg**, diaginertia
   **[0.008112, 0.007909, 0.002908]**, centroid **[0.003687, 0.0000248, 0.429132]** from the mesh
   origin. The new head body must carry explicit inertia and the torso's mass must drop to
   **6.0155**, or total mass silently grows. Verified: total stays 33.3411 kg.
4. **There are two head geoms, not one** — a visual (`group=2, contype=0, density=0`) and a
   collision (`group=3, contype=1, density=1000`). Both move to the neck body.
5. **Adding the neck joint invalidates the `stand` keyframe.** `nq` goes 36 to 37 and the stored
   keyframe is length 36, so compilation fails. Delete the keyframe and rebuild the standing pose by
   **joint name**, which is index-independent and matches the semantic-role approach in section 4.2.
6. **Rendering above 640x480 requires raising the offscreen framebuffer** via
   `getattr(spec.visual, "global").offwidth/offheight` before compiling. The attribute is literally
   named `global`, a Python keyword, so it needs `getattr`.
7. **MuJoCo 3.2.5 MjSpec API specifics:** `spec.find_body(name)` (not `spec.body`); geoms iterate
   via `body.first_geom()` / `body.next_geom(g)`; `geom.delete()` exists; actuator `gainprm` and
   `biasprm` must be length 10 (a `position` actuator is `gaintype=fixed` with `gainprm[0]=kp`,
   `biastype=affine` with `biasprm=[0, -kp, -kv]`).
8. **Section 4.5's root table was wrong about the free root.** See the correction there.

Open risk carried forward: elbow flexion is hard to read at this camera angle even at 1.85 rad.
Design risk 9.1 (arm swing may look stiff) stands and should be checked early in tuning.

---

## Section 10 — Resolved decisions

1. **License: MIT.** Author's choice. Compatible with the BSD-3-Clause G1 model and the Apache-2.0
   dependencies. A `LICENSE` file is added in Phase 1.

2. **Disgust and surprise ship, clearly flagged.** Author's choice. All six intensities remain
   user-facing. Disgust and surprise are placed by arousal/valence reasoning and marked as
   unsourced extrapolations in three places: the `evidence` module's per-parameter flag, the
   generated documentation, and the GUI itself, visibly separated from the four grounded emotions.
   Appendix A.7 is the authority for why.

3. **Acceptance criterion: build the evaluation harness, including a point-light mode.** The author
   delegated this decision with the criterion "interesting and professional enough to be a standout
   portfolio item". This is **my call, not theirs**, and the reasoning is recorded so it can be
   overruled.

   A portfolio repository is distinguished far more by having a *result* than by having a
   capability. "Observers identified the intended emotion N% of the time" is a different class of
   claim from "here is a robot walking sadly", and it is the thing that makes this a project rather
   than a demo. It also supplies the terminating condition that tuning otherwise lacks, and makes
   the "stable but emotionally flat" anti-goal detectable instead of a matter of opinion.

   Point-light is included because it is much cheaper than the builder check assumed. It does not
   need a second renderer: MuJoCo can hide geoms by group and render small spheres at body sites,
   so it is a visual-option toggle. The payoff is real for this audience — it matches the stimulus
   format of the literature being cited, demonstrates understanding of *why* that format is used
   (it strips form cues and isolates kinematics), and yields a striking side-by-side figure.

   **Honest scoping, because over-scoping is the real threat to a portfolio piece.** A rating
   harness needs raters, and that is the hidden cost. The author rating their own stimuli is worth
   nothing. So the deliverable is split: the **harness is the engineering artifact** and ships so
   that any reader can run the study themselves; a **small informal pilot** supplies the numbers.
   The pilot's n and recruitment are reported transparently, and the README states them rather than
   implying a controlled study. A pilot with a handful of raters, honestly caveated, is far more
   impressive than no measurement — and far better than an overclaimed one.

   **Droppable if the schedule slips**, in this order: point-light mode first, then the pilot
   (shipping the harness unrun), then the harness itself. The core emotional gait is never traded
   for the evaluation.
