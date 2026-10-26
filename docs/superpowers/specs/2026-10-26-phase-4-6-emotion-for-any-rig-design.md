# Phase 4.6 — emotion for any rig: design (2026-10-26)

Status: draft for the user's approval. Branch `phase-4-6-emotion-for-any-rig`, cut from the
Phase 4.5 head (149776e). Parent decisions: the Phase 4.5 brainstorm record
(`docs/superpowers/research/2026-10-25-phase-4-5-brainstorm-record.md`, decision B, binding) and
the Phase 4.5 design (`2026-10-25-phase-4-5-learned-emotion-model-design.md`, sections 4.2 and
4.4, the extractor interface this phase reuses). The user opted out of the brainstorm questions
on 2026-10-26; the four points the hand-off brief left open (the LLM's role, which rigs to
validate on, how foot contact is defined on an arbitrary rig, what the MCP surface exposes) are
settled here as my calls and marked as such.

---

## 1. Intent and why

Phase 4.5 made the emotion map measurable but left it bolted to one robot: the only thing that
can walk with an emotion is the G1 in MuJoCo. The extractor (`mocap/`) already measures any
skeleton; the learned model already re-dimensions through a profile's leg length; the generator
already solves feet by task-space IK on whatever `RobotModel` it is handed. What is missing is
the ring that closes those three on a skeleton the project has never seen: read a character's
walk, measure it, move its parameters by an emotion, re-synthesise the walk on that character
with its feet planted, and hand the result back in the format it came in.

After this phase:

- **Any BVH character walks with an emotion.** `emotive-gait rig walk in.bvh --emotion
  sadness=0.8 --out out.bvh` reads a neutral walk, measures its gait parameters on that rig,
  applies the analytic or the learned emotion deltas re-dimensioned to that rig, re-synthesises
  the walk through the existing generator on a MuJoCo kinematic chain compiled from the BVH
  hierarchy, and writes a BVH with the same hierarchy, channel order, units and up-axis.
- **The claim is measured, not shown.** Round-trip fidelity: the parameters the extractor reads
  off the output equal the commanded ones within the Phase 4.5 round-trip tolerances, at zero
  emotion and at each emotion, on rigs the project did not design for. Zero foot slip: the
  stance foot does not move while it is planted. Both are tests in CI on a family of synthetic
  rigs and on a 100STYLE rig excerpt, and a committed report on real rigs validated locally.
- **An AI assistant or a DCC workflow can drive it.** The package exposes the same operations as
  an MCP server over stdio (`emotive-gait serve`), with a free-text emotion prompt mapped to the
  six intensities, so Claude Desktop, Claude Code or any MCP host can take "make her walk like
  she just lost the match" and a BVH path and return a BVH. Blender and the Mixamo workflow are
  served by their BVH export and import (documented), not by an add-on.
- **The bone-name problem is solved by a heuristic with a visible audit.** Bone names from the
  common families (Mixamo, MotionBuilder/CMU, 100STYLE, Rokoko/Xsens, SMPL, Rigify) plus the
  rig's rest-pose geometry give a `LandmarkMap`, a rig profile with its own `leg_length`, and
  the hinge-and-sign for every `JointRole` the rig can carry. Every mapping decision is printed
  with its source (name, geometry, user) and can be overridden from a YAML file or from the MCP
  caller, which is where an LLM contributes when one is present.

What this phase is not: a retargeter between two rigs, a Blender add-on, a physics tier for
arbitrary characters, a motion generator learned from data, a classifier, or a perception
claim (Phase 5). The emotion models themselves do not change.

---

## 2. Goals, non-goals, anti-goals

### Goals

- BVH in, BVH out, with the source hierarchy, channel order, rotation orders, units, up-axis and
  frame rate preserved, so the file imports back where it came from without settings.
- The same extractor, unchanged, measures the source walk and the output walk.
- Emotion deltas from either `EmotionModel`, re-dimensioned to the rig through its own
  `leg_length` and nominal pelvis height, applied to the rig's own measured neutral parameters.
- A MuJoCo kinematic chain compiled from the BVH hierarchy whose forward kinematics reproduce
  the reader's positions exactly, with one hinge per rotation channel.
- Leg IK through the existing `gait.ik` and `gait.generator`, foot targets in the pelvis frame,
  stance foot stationary in the world by construction, feasibility reported as on the G1.
- Bone-name to landmark and role mapping: name lexicon plus rest-pose geometry, every decision
  auditable and overridable; works offline, deterministically, with no LLM.
- Free-text emotion prompt to six intensities: a shipped lexicon (CLI and MCP fallback) and an
  MCP prompt that asks the host model to do it when a model is present.
- An MCP server (`mcp` SDK, stdio) exposing inspect, measure, walk, parse-prompt and the
  evidence table; registration snippets for Claude Desktop and Claude Code; a documented Blender
  round trip.
- Measurable claim in CI: round-trip fidelity and zero foot slip on a synthetic rig family and on
  a 100STYLE excerpt; a committed validation report on real rigs run locally (the CMU BVH
  conversion, Bandai Namco, and any rig the user exports from Mixamo through Blender).
- Version 0.4.0; README section; CLAUDE.md facts.

### Non-goals

- Retargeting a clip from one rig to another, or preserving the source's style beyond the 14
  measured parameters: the output is a parametric gait with the source's parameters, not the
  source's frames with an emotion layered on.
- FBX, glTF, Collada or any format other than BVH. ASF/AMC and PTD stay read-only.
- A Blender add-on, Unity or Unreal plugin, or any DCC-side code. The DCC path is export BVH,
  run the tool, import BVH.
- Physics on a compiled rig (no masses, no contacts, no servos). The rig tier is kinematic.
- An LLM called from inside the package, an `anthropic` dependency or an API key.
- Hands, fingers, facial or eye channels: held at the source's mean, never driven.
- Turning, stairs, running, in-place walking, or the pelvis rotating: straight-line walking,
  as the generator produces it.
- Changing the emotion models, the extractor's thresholds, `evidence.py`, or the artefact.

### Anti-goals

- A mapping that silently guesses. Every landmark and role the tool decides is printed with how
  it decided; an unmapped leg landmark is an error, not a default.
- Foot slip hidden by smoothing or by a floor snap after the fact. Slip is measured on the
  output as written, over the stance the generator commanded, and the number is printed.
- A prompt mapper that invents intensities for words it does not know. Unknown words are
  listed; disgust and surprise stay flagged as unsourced in the output report, as the CLI does.
- Re-implementing the generator or the IK for rigs. One generator, one IK, one extractor.
- Committing any non-redistributable clip. Bandai Namco (CC BY-NC 4.0) and anything from
  Mixamo stay local; only residual tables enter git.

---

## 3. Constraints

- Everything in the hand-off brief: no RL, no learned balance; emotion applies to parameters,
  never joint angles; no magnitude in degrees from a citation; disgust and surprise unsourced;
  the artefact generated, never hand-edited; aggregates only enter git; no Claude attribution
  trailer; merging and pushing are the user's; work in place on this branch; `mujoco>=3.17,<3.18`.
- The dependency rules in CLAUDE.md stand; the new package sits on top of all six (section 5).
- The extractor's frozen thresholds (`mocap/events.py` header, `extract.py` constants) are not
  retuned for rigs. If a rig defeats them, that is a finding in the report, not a parameter.
- The G1 path is bit-identical: every existing test passes unchanged, `generate(params, robot)`
  keeps its signature with one new optional keyword and an identity posture map, and the G1's
  two rendering xfails stay.
- Python 3.11+ (the `mcp` SDK needs it; the project already does). The MCP server is an extra
  (`pip install -e ".[mcp]"`); the core never imports `mcp`.
- The compiled chain carries no masses that matter and is never stepped: `mj_forward` only.

---

## 4. Decisions

### 4.1 The pipeline, end to end

```
in.bvh ──read_bvh_file──> BvhFile (hierarchy, channels, motion, scale, up-axis)
        │
        ├──> SkeletonClip (positions; today's reader, unchanged semantics)
        │        └──map_rig──> RigMapping: LandmarkMap + role -> (hinge, sign), audit
        │                 └──summarise──> ClipSummary (the rig's NEUTRAL parameters, P0)
        │
        ├──compile_rig──> RobotModel (MuJoCo chain, welded root, calibrated zero pose)
        │                 + RigProfile (RobotProfile with the rig's leg_length, nominal
        │                   pelvis height, foot sites, floor height)
        │
   intensities (CLI flags | lexicon(prompt) | MCP caller)
        │
        └──modulate(P0, intensities, rig_profile, model=analytic|learned)──> P1
                 └──generate(P1, robot, welded=robot)──> GaitTrajectory (+ feasibility)
                          └──write_bvh(BvhFile, trajectory, duration)──> out.bvh
                                   └──summarise(read_bvh(out.bvh))──> P1' (the round trip)
```

The source clip is treated as the rig's neutral walk and its cycle-mean measurement `P0` is the
base `GaitParameters` for that rig (`expansiveness` 1.0). The emotion models then move `P0`
exactly as they move the G1's defaults: `modulate` is called with the rig profile, so the
learned model re-dimensions through the rig's `leg_length` and the profile's cue-dropping
removes what the rig cannot carry (`unexpressible` is printed). If the extractor finds no usable
cycle in the source, the tool stops with exit 2 and the discard reasons; it does not fall back
to the G1 defaults.

**The analytic model on a foreign rig (my call).** `AnalyticEmotionModel.deltas` returns deltas
in the G1's units and ignores the profile. For a rig they are first made dimensionless as if
measured on the G1 (lengths / 0.69, cadence x sqrt(0.69 / g), the `com_height_scale` delta x
0.68 / 0.69, angles and fractions unchanged) and then re-dimensioned to the rig with
`to_gait_parameters(..., scaling="froude")`. This happens in `rig/`, not in `emotion/`: a thin
`RigScaledModel(inner: EmotionModel)` wrapper that is the identity for a model whose deltas
already depend on the profile (the learned model) and the Froude rescaling for one that does
not. The wrapper asks the inner model whether it is `profile_aware` (a new boolean attribute on
both models, `False` for the analytic one); the protocol in `emotion/model.py` gains that one
optional attribute and nothing else. On the G1 profile the wrapper is bit-identical.

**`ratio` scaling on a rig (my call, after the builder check).** The learned model's `ratio`
deltas are fractions of the G1's default `GaitParameters()` (`learned.py`), and the analytic
deltas are G1 absolutes. On a rig a fraction must be of the rig's own neutral, so under
`scaling="ratio"` the wrapper re-bases every length, `cadence` and `com_height_scale` delta:
`delta_rig = delta_g1 / default_g1 * P0_value` (`P0.com_height_scale` is 1.0 by definition,
section 4.6); angles and fractions pass through. `froude` needs no re-basing: the learned
model already re-dimensions through the rig profile and the analytic model goes through the
Froude step above.

**What "neutral" means (my call).** `P0` is the cycle mean of the source clip under the frozen
extractor, including its posture (trunk pitch, elbow flexion, abduction). A rig whose character
walks slouched gets a slouched neutral, and sadness slouches it further. The tool never
subtracts the actor's posture to reach some canonical upright. One field is inverted rather
than copied: the extractor reads `double_support_frac` with the position-extremum rule, which
on the generator's own swing shape reads about +0.21 above the parameter (the G1 finding,
`rule_double_support` in `tests/mocap/test_roundtrip.py`, rig-independent because it comes
from the foot path's shape). `P0.double_support_frac` is therefore the parameter `d` for which
`rule_double_support(replace(P0, double_support_frac=d))` equals the raw reading (a monotone
one-dimensional root find, `rule_double_support` moves into `gait/footsteps.py`), so that `P0`
regenerates the rig's stance timing as the rule reads it; `inspect` prints the raw reading
beside it. Emotion deltas then apply to the parameter, as on the G1.

### 4.2 Reading and writing BVH without losing anything

`mocap/readers/bvh.py` today keeps only positions. It gains a `BvhFile` dataclass and
`read_bvh_file(path, *, scale, up_axis) -> BvhFile`:

- joints in file order with name, parent index, offset (file units), channel names in file
  order, and end-site offsets (needed to write the file back and to place a toe when the only
  toe is an End Site);
- the motion as an array (frames, channels) in file units and degrees, the frame time, and the
  header's exact `Frames:` / `Frame Time:` lines' formatting (so a byte-level round trip of an
  unmodified file is possible and tested);
- `scale` and `up_axis` as given or as inferred (section 4.4), recorded on the object.

`read_bvh` becomes `read_bvh_file(...).to_clip()` (same output, bit-identical, the 100STYLE
reader tests prove it). A new `mocap/writers/bvh.py` has `write_bvh(file: BvhFile, path)`; a test
reads a fixture, writes it, reads it again and compares hierarchy, channels and motion exactly,
and FK positions to 1e-9. `BvhFile.with_motion(frames)` returns a copy with new motion rows.
Rotation order is whatever the file's channel order is; the writer never reorders channels.

Up-axis and units go back through the inverse of the reader: positions the pipeline produces
in metres and +z up are converted to file units and the file's up-axis on write. `mocap/`
still imports only NumPy and SciPy.

### 4.3 The compiled chain and its calibrated zero

`rig/compile.py` turns a `BvhFile` into a MuJoCo model through `MjSpec`:

- one body per BVH joint at its offset, in the rig's own frame converted to metres and +z up;
  one hinge joint per rotation channel, in channel order, about the parent-frame axis the
  channel names, so that MuJoCo's composition of the hinges reproduces the BVH composition
  `R = R_c1 @ R_c2 @ R_c3` (the builder check verified on MuJoCo 3.17 that several hinges on
  one body compose intrinsically in definition order, matching the reader, so no per-body
  reversal is expected; the gate below still decides);
- the chain is rooted at the `PELVIS` landmark's bone, whose compiled body is named `pelvis`
  (what `gait.ik` looks for) and is welded at the rig's nominal pelvis height, as the G1 is
  for IK. A rig whose BVH root is a floor-level reference bone with the pelvis as a child is
  supported: the bones from the root down to the pelvis are not compiled, they are held at
  the reference pose and written back from it, and the written root translation is the pelvis
  path minus the reference root-to-pelvis offset;
- two compile modes from one spec builder (my call, after the builder check): `full` has
  every rotation channel as a hinge and serves the FK gate and the axis probes; `driven` has a
  hinge only for channels bound to a `JointRole`, with every other channel's reference
  rotation baked into its body frame, and serves the IK and synthesis. The two agree at the
  reference pose to 1e-9 m (a test). MuJoCo 3.17 refuses a `[0, 0]` range and treats it as
  unlimited under `limited="auto"`, so "locked" channels are simply not hinges;
- one site per foot at the foot point the extractor uses, the toe landmark when the rig has
  one (a toe bone, else the foot's End Site) and the ankle otherwise (`foot_sites`), plus
  landmark sites at every mapped landmark so the output clip can be recorded straight from the
  chain for the in-process round trip;
- one position actuator per hinge, named like the hinge, so `RobotModel.ctrl_index`,
  `qpos_adr` and `joint_range` work unchanged (my call: cheaper than teaching `RobotModel` a
  second lookup path);
- no geoms, no masses that matter, no contacts; the model is only ever `mj_forward`-ed.

**Gate: FK equivalence.** For every validation rig and every synthetic rig, posing the `full`
chain with the source motion's channel values (converted to radians) reproduces the reader's
positions for every joint and frame to 1e-6 m. This test is the arbiter of the hinge-axis and
composition convention and of the reference-pose baking (Euler composition is not additive, so
a reference cannot be subtracted per hinge; it is baked as the body's quaternion with the later
hinges' axes re-expressed through it, and a naive subtraction fails this gate).

**Calibrated zero pose (my call, the heart of the posture round trip).** The generator writes
posture roles as absolutes: `WAIST_PITCH = trunk_pitch`, `NECK_PITCH = neck_pitch`,
`SHOULDER_ROLL = sign * arm_abduction`, `ELBOW = elbow_flexion + ...`, and swing relative to a
stand pose (`SHOULDER_PITCH = stand + swing`). On the G1 the joint zeros happen to be
trunk-vertical, head-level, arms-hanging, elbow-straight (except the elbow sign, the known
xfail). A BVH rig's channel zero is its rest pose, often a T-pose with the arms horizontal, so
absolutes would be nonsense. The compiled chain therefore gets a *calibrated zero*:

1. the reference pose is the actor's own mean walking posture: for every joint, the chordal
   mean (quaternion average) of its local rotation over the source's usable cycles, never a
   per-channel Euler median (Euler channels wrap and do not add). For the root the mean is
   taken after removing each frame's path-frame heading, so the reference carries the pelvis
   tilt and no yaw; the *reference heading* is the source's mean heading over the usable
   cycles. The foot bodies are the one exception: their reference rotation is the chordal
   mean over the source's *stance* frames only, so "level" means the foot as it rests on the
   floor, not its cycle average (a toe 19 cm from the ankle would otherwise sit centimetres
   off the floor for the whole stance). The reference is baked into the bodies' frames so
   that every hinge reads 0 there and undriven channels stay there;
2. for each posture role the rig carries, the hinge zero is shifted by the extractor's own
   measurement of that posture on the source (trunk pitch, neck pitch, elbow flexion, arm
   abduction from `P0`), so that hinge 0 is the semantic zero (trunk vertical, head on the
   trunk line, elbow straight, arm along the trunk) and `P0`'s posture value reproduces the
   reference pose;
3. the stand pose (`RobotModel.stand_qpos`) is all zeros, which is the reference pose.

**Posture gain (my call, after the builder check).** The zero shift fixes the offset but not
the gain: the extractor measures `trunk_pitch` on the pelvis-to-neck vector from the pelvis
origin, while the waist hinge pivots at the lowest spine bone, above that origin (100STYLE:
13 cm up a 55 cm trunk, so a hinge angle reads as about 0.77 of itself), and `neck_pitch` is
head angle minus trunk angle, so the same shortfall leaks into the neck. This is the G1's 0.88
finding, and it would reappear on every rig. The remedy is a measured linear posture map: with
the `full` chain at the reference pose, each posture hinge (waist pitch, neck pitch, each
elbow, each shoulder roll) is probed by a small angle and the extractor's four static posture
measures are read, giving the Jacobian `J` of measures with respect to hinges; the rig profile
stores `K = J^-1` as `posture_map` and the generator writes `hinges = K @ (trunk_pitch,
neck_pitch, elbow_flexion, arm_abduction)` for the posture roles instead of copying them.
`RobotProfile` gains an optional `posture_map` (identity by default) and `generate` applies it
in `_procedural`; with the identity map the G1 is bit-identical, keeps its 0.88 xfail, and the
same hook is available to the Phase 4 rendering fix later (not done here).

The extractor is thereby the arbiter of posture semantics at both ends, and the posture round
trip closes by construction up to the extractor's own repeatability and the linearity of the
map over the deltas in play (the test tolerances absorb the rest). A test poses the chain at
zero, builds a path frame from the reference heading, and checks that the extractor's static
posture angles read 0 within 1e-3 rad for every calibrated role, and that a commanded trunk
pitch of 0.3 rad through `posture_map` reads back within 0.02 rad. The G1 keeps its two xfails
(elbow sign, trunk 0.88 gain) untouched: it is not calibrated, rigs are.

**Joint ranges (my call).** BVH joints have no limits. Driven hinges get rig-independent,
human-like ranges from a committed table in `rig/ranges.py` (knee pitch `[0, 2.6]` rad in the
flexion direction, no hyperextension; hip pitch `[-1.2, 2.0]`; hip roll and yaw `+-0.8`; ankle
pitch `+-1.0`; ankle roll `+-0.6`; waist pitch `+-0.8`; neck pitch `+-0.8`; shoulder pitch
`+-3.0`; shoulder roll `[-0.3, 2.5]` abduction; elbow `[0, 2.6]` flexion), all about the
calibrated zero and expressed through the role's sign. Undriven channels are not hinges in the
`driven` chain (above). The generator's clamp-last rule and the IK's `ConfigurationLimit` then
behave as on the G1. These are reachability guards, not evidence claims.

**Seed.** The IK seed is the reference pose with the leg roles bent as on the G1
(`SEED_LEG_POSE`: hip -0.2, knee 0.4, ankle -0.2 in the role sign convention), because a rig's
mean walking pose also has a near-straight knee at some phases and the straight knee is the
same Jacobian singularity.

### 4.4 Mapping bones to landmarks and roles

`rig/mapping.py` produces a `RigMapping`:

```
RigMapping
  landmarks: LandmarkMap                        # Landmark -> joint name(s)
  roles: Mapping[JointRole, RoleBinding]        # hinge joint name, sign (+1/-1), axis_purity
  leg_length: float                             # metres, the extractor's rule
  scale: float; up_axis: "y" | "z"              # as inferred or given
  decisions: tuple[Decision, ...]               # (what, chosen, source: name|geometry|user, note)
  warnings: tuple[str, ...]
```

**Names first.** A committed lexicon (`rig/data/bone_names.yaml`) lists, per landmark, the
bone names of the common families and the tokens that identify side (`Left`/`Right`, `L_`/`R_`,
`.L`/`.R`, `_l`/`_r`, `mixamorig:Left...`) and part (`UpLeg`/`Thigh`/`Hip`/`femur`, `Leg`/
`Shin`/`Knee`/`tibia`, `Foot`/`Ankle`, `ToeBase`/`Toe`, `Spine`/`Chest`, `Neck`, `Head`,
`Shoulder`/`Clavicle`/`Collar`, `Arm`/`UpperArm`/`humerus`, `ForeArm`/`Elbow`/`radius`,
`Hand`/`Wrist`). Matching is case-insensitive on tokens after stripping a namespace prefix.
Families named in the lexicon: Mixamo, MotionBuilder (the CMU BVH conversion), 100STYLE,
Rokoko, Xsens MVN, SMPL, Blender Rigify, Unity Humanoid names. The lexicon is data; adding a
family is a YAML edit.

**Geometry second, and as the referee.** With the chain at its reference pose:

- side: the sign of a candidate's lateral coordinate relative to the pelvis in the path frame
  of the source clip decides left/right, and overrides a name that disagrees (a warning);
- chain structure: hip, knee and ankle must be parent, child and grandchild along a chain from
  the pelvis; when the names give several candidates (`Spine`, `Spine1`, `Spine2`), the lowest
  spine bone is `WAIST_PITCH`'s bone and the trunk landmark `NECK` is the neck bone, else the
  highest spine bone; `HEAD` is the head bone, else its End Site;
- toe: the toe bone if named, else the foot's End Site (as a landmark only);
- hinge axis per role: each candidate hinge of the role's bone is probed by +0.05 rad and the
  child landmark's displacement is projected on the path frame's (forward, left, up): the hinge
  whose displacement is most sagittal is pitch, most lateral is roll, and the remaining one is
  yaw; `axis_purity` is the fraction of the displacement along the winning direction;
- sign: chosen so the probe moves the child landmark in the direction the role convention
  table (section 4.5) prescribes for a positive angle.

A role whose `axis_purity` is below 0.8 is bound anyway for the IK (the IK does not care which
hinge is which, the leg roles are all solved together) but is reported `skewed`; a procedural
upper-body role with a skewed axis is dropped from the profile (its cue then appears in
`unexpressible`), because the generator writes it directly. Minimum for a usable rig (my call,
after the builder check): `PELVIS`, both `HIP`, `KNEE` and `ANKLE` landmarks, and every leg
role `gait.ik.LEG_ROLES` names (hip pitch, roll and yaw, knee pitch, ankle pitch and roll, both
sides), because `solve_leg_cycle` addresses all twelve and stays unchanged. Every common BVH
rig has three rotation channels on the hips and ankles, so this is the ordinary case; a rig
with one- or two-channel hips or ankles is refused at `inspect` with exit 2 naming the missing
roles. The knee's non-pitch channels and the ankle's yaw are undriven.

**Overrides.** `--mapping map.yaml` (and the MCP `mapping` argument) sets any landmark or role
binding by bone name and axis letter; those entries are marked `user` in the decisions. The
`inspect` output is itself valid as such a file, so the workflow is inspect, edit, walk.

**Where an LLM fits (my call).** The package never calls one. In the MCP setting the host model
reads `inspect_rig`'s decisions and warnings and, if it disagrees, passes a `mapping` override
to `emotional_walk`; the server's `emotional_walk_request` prompt tells it to do exactly that.
In the CLI the heuristic is the whole answer. No `[llm]` extra is built in this phase.

**Units and up-axis inference (my call).** `scale` and `up_axis` may be given; when not, the
up-axis is the rest-hierarchy axis with the largest pelvis-to-head extent and the scale is the
one of (1, 0.01, 0.0254, 0.1, 0.001) that puts the hip-to-ankle rest distance nearest 0.8 m,
accepted only if the result lies in 0.3 to 1.3 m, otherwise exit 2 asking for `--scale`. Both
inferences are printed as decisions.

### 4.5 Role sign conventions, made explicit

`robot/roles.py` gains a committed table `ROLE_CONVENTIONS: Mapping[JointRole, str]`, the
positive direction of each role as the generator and the emotion model intend it:

| Role | Positive direction |
|---|---|
| `HIP_PITCH_*` | thigh backward (extension) |
| `KNEE_*` | shank backward (flexion) |
| `ANKLE_PITCH_*` | toes down (plantarflexion) |
| `HIP_ROLL_*`, `ANKLE_ROLL_*`, `SHOULDER_ROLL_*` | away from the body (abduction), both sides |
| `HIP_YAW_*`, `WAIST_YAW` | toes/trunk turning left |
| `WAIST_ROLL` | trunk leaning left |
| `WAIST_PITCH` | trunk inclined forward |
| `NECK_PITCH` | gaze down |
| `SHOULDER_PITCH_*` | upper arm backward |
| `ELBOW_*` | forearm flexing (bend increases) |

A test probes the G1 (with `sign_for` applied) against this table; the elbow row is a strict
xfail there, consistent with the existing rendering finding, and every other row must pass
(the builder confirms them on the G1 before anything else; a G1 row that fails is a table
correction, because the G1 is the reference the generator was tuned on). Every compiled rig
must pass all rows (its signs are chosen to). The roll rows explain the G1's `sign_for` and
make the "symmetric modulation negates mirrored rolls" fact a stated convention.

### 4.6 Foot contact, floor and the zero-slip claim

**Contact on the source (my call).** Contact is defined on landmarks, never on bones: the foot
point is the toe landmark when present, else the ankle, as the extractor already does. The
source's stance intervals come from the frozen position-extremum rule. The rig's *floor* is the
median height of the foot points over their stance frames (both feet pooled); every height the
pipeline uses is relative to it: `nominal_pelvis_height` is the source's mean pelvis height
above the floor, `foot_site_floor_height` is the foot site's stance height above it (zero on a
toe rig by construction, the ankle's stance height otherwise), and the output root translation
adds the source's floor height back so the character stands where it did.

**One foot point everywhere (my call, after the builder check).** The IK site, the slip
measurement, the extractor's events and its `step_height` and placement readings all use the
same point: the toe when the rig has one, else the ankle. On a toe rig the IK therefore
commands the toe's clearance and placement directly and the ankle's swing pitch is the
emergent part, so `step_height` and `step_length` round-trip on the point the extractor reads;
holding the foot at its stance-mean orientation (section 4.3) keeps the whole rigid foot on the
floor during stance.

**Contact on the output (by construction).** The generator's treadmill convention places the
stance foot stationary in the world; the IK holds the foot site at the commanded position and
the foot's stance-mean orientation during stance (hard) and lets the swing pitch be emergent
(soft), as on the G1. The commanded stance interval of each foot is known exactly
(`footsteps.foot_point(...).in_stance`), so slip is measured on it, not on a detected one.

**Zero foot slip, the claim.** On the written output read back with `read_bvh`: for every
stance interval the generator commanded, the horizontal travel of the foot point (toe, else
ankle) is at most 2 mm (the IK position threshold) and its height stays within 2 mm of the
floor, on every rig whose parameters the IK reports feasible. On an infeasible rig the numbers
are reported, as for the learned presets (flagged and rendered, not gated). A second, independent
number is printed beside it: the extractor's own double-support reading on the output against
`rule_double_support` of the commanded path, as in `tests/mocap/test_roundtrip.py`.

### 4.7 Synthesis and the round-trip claim

`rig/synthesis.py` exposes one function and one result:

```
emotional_walk(source: Path, *, intensities, model: EmotionModel, scaling, duration_s,
               mapping_overrides=None, scale=None, up_axis=None) -> WalkResult
WalkResult: mapping, p0 (ClipSummary), p1 (GaitParameters), unexpressible, feasibility,
            slip (per stance max travel, max height error), round_trip (P1' vs P1 residuals),
            output: BvhFile
```

The CLI and the MCP server both call it; the output file is written by the caller. The
`duration_s` default is the source clip's usable length. Output frame rate is the source's.
The output starts at the source's first-frame pelvis position and heading and walks straight.

**Round-trip fidelity, the claim.** `summarise(read_bvh(out))` against the commanded `P1`
within the Phase 4.5 round-trip tolerances (cadence and step length 2 %, step height, bounce
and sway 3 mm, angles 0.02 rad, `elbow_swing_ratio` 0.05, double support against
`rule_double_support`), at zero emotion (`P1 = P0`) and at the four analytic presets'
intensities, on every rig in the test family. Posture is compared as absolutes (the calibrated
zero makes that meaningful); the G1 is not part of this test.

**Test rigs (my call on which rigs).** Three tiers:

1. *Synthetic family, in CI.* A test-side builder (`tests/rig/rigs.py`) writes BVH files for a
   parametrised humanoid and a source walk for it. The source walk is written by the test
   itself in joint space (sinusoidal hip pitch, half-wave knee flexion, counter-swinging arms,
   a translating root; the feet may slide, the extremum rule only needs their fore-aft
   oscillation), not by the pipeline's generator: the calibrated zero needs a clip to exist
   first, and a second, clip-free calibration route is not built (my call, after the builder
   check). The family is one-factor-at-a-time from a base rig (Mixamo names, `Zrotation
   Xrotation Yrotation`, y-up, centimetres, toe bones, three spine bones, fingers, clavicles,
   0.9 m legs), one variant per factor: each other naming family, `XYZ` and `ZYX` channel
   orders, z-up, metres, inches and decimetres, no toe bones (End Sites only), one spine bone,
   a floor-level root above the pelvis, a left/right swap in the rest pose, 0.6 m and 1.1 m
   legs. About fifteen rigs; zero emotion on all of them, the four presets on the base rig and
   on the z-up and no-toe variants, so the suite stays within a minute or two of CI time
   (each round trip is one `generate`, a few tenths of a second, plus a write, a read and a
   `summarise`). Tolerances as above.
2. *A real rig, in CI.* A four-second excerpt of 100STYLE `Neutral_FW.bvh` (CC BY 4.0, already
   fetched and hashed in the manifest, attribution in a sidecar), committed under
   `tests/mocap/fixtures/`. It is a 23-joint rig in centimetres and y-up whose names
   (`LeftHip`, `LeftKnee`, `LeftAnkle`, `Chest..Chest4`) are a family of their own; the rig
   mapper must reproduce `mocap/landmarks.py`'s `STYLE100` map from the names and geometry
   alone (a test), and it must never read that map. The cgspeed CMU conversion was the first
   choice, but its hosting and terms are unverified (my call, after the builder check: it
   moves to tier 3).
3. *Real rigs, local, aggregates in a committed report.* The cgspeed MotionBuilder-named CMU
   conversion (a neutral walk; fetched through the manifest if reachable, its terms recorded;
   31 joints, different names, inches), Bandai Namco Research Motiondataset 2's normal walk
   (CC BY-NC 4.0: fetched through the manifest, `committable: false`, licence recorded; a third
   naming family), 100STYLE's other neutral clips, and any rig the user exports from Blender
   (Mixamo characters: never committed in any form). `emotive-gait rig validate <bvh> --report`
   runs the round trip at zero emotion and at the four presets and writes a residual and slip
   table; the tables for the rigs above are committed as
   `docs/reports/phase-4-6-rig-validation.md`, generated, not hand-edited. A rig that cannot
   be fetched is listed in the report as not run.

### 4.8 Free-text emotion prompt

`emotion/prompt.py`, `parse_emotion_prompt(text) -> PromptResult` with a shipped lexicon
`emotion/data/affect_lexicon.yaml`: a hand-written table of roughly 150 English terms, each
mapping to one emotion with a weight in (0, 1] (`furious` anger 1.0, `annoyed` anger 0.4,
`gloomy` sadness 0.6, `elated` happiness 1.0, `nervous` fear 0.5, `grossed out` disgust 0.7,
`startled` surprise 0.8, ...), intensifiers (`very`, `extremely`, `slightly`, `a bit`) as
multipliers clamped to [0, 1], and negation (`not`, `no longer`) zeroing the next term. The
result carries the intensities, the matched terms with their contributions, the unmatched
content words, and `grounded: bool` (false when any matched emotion is disgust or surprise, the
CLI's existing warning). Several terms for one emotion take the maximum, not the sum. The
lexicon is tuned, not cited, and says so in its header; it imports only `evidence.py` and
`pyyaml`.

The CLI accepts `--prompt "..."` as an alternative to `--emotion`. The MCP server exposes
`parse_emotion_prompt` as a tool and, in its `emotional_walk_request` prompt, tells the host
model to produce the six intensities itself from the user's words and to pass them to
`emotional_walk`, falling back to the tool only when it prefers to. `WalkResult` records
`intensities_source` (`flags`, `lexicon`, `caller`).

### 4.9 The MCP surface (my call)

`rig/server.py`, run by `emotive-gait serve` (stdio only; Streamable HTTP is not offered). It
requires the `mcp` extra and fails with exit 1 and an install hint without it. Paths are local
file paths; no file content crosses the wire, because the server is local by construction and
BVH files are megabytes.

Tools:

- `inspect_rig(bvh_path, scale?, up_axis?, mapping?)`: the `RigMapping` decisions, warnings,
  leg length, and the source's `P0` with cycle counts and discard reasons.
- `measure_gait(bvh_path, scale?, up_axis?, mapping?)`: `ClipSummary` only (any BVH, any walk).
- `emotional_walk(bvh_path, out_path, intensities?, prompt?, model='analytic'|'learned',
  scaling='froude'|'ratio', duration_s?, mapping?, scale?, up_axis?)`: writes `out_path` and
  returns the `WalkResult` report (without the motion).
- `parse_emotion_prompt(text)`: the lexicon result.
- `list_emotions()`: the six emotions, which are grounded, and the learned model's thin cells.

Resources: `emotive-gait://evidence` (the claim table as `evidence` prints it) and
`emotive-gait://rig-mapping-schema` (the override YAML's schema with an example).

Prompt: `emotional_walk_request(bvh_path, description)`, the instructions above (inspect first,
map intensities from the description, call `emotional_walk`, report feasibility, slip and
unexpressible cues to the user in words, never present disgust or surprise as grounded).

Registration snippets for Claude Desktop (`claude_desktop_config.json`) and Claude Code
(`claude mcp add`) go in the README; a smoke test starts the server in a subprocess, lists its
tools over stdio with the SDK's client, and calls `parse_emotion_prompt`. The server has no
state between calls.

### 4.10 CLI

A `rig` command group in `rig/cli.py`, registered by `cli.py` (which stays the only module that
imports `rig/` and `learn/`):

```
emotive-gait rig inspect in.bvh [--scale S] [--up y|z] [--mapping map.yaml]
emotive-gait rig walk in.bvh --out out.bvh (--emotion E=I ... | --prompt "...")
                     [--model analytic|learned] [--scaling froude|ratio] [--duration S]
                     [--scale S] [--up y|z] [--mapping map.yaml]
emotive-gait rig validate in.bvh [--report path.md] [--scale S] [--up y|z] [--mapping map.yaml]
emotive-gait serve
```

Exit codes as the project's: 2 for user-input errors (unreadable BVH with path and line, no
usable cycle, unmapped leg, bad intensity, bad prompt with nothing matched, inference failure),
1 for environment failures (missing `mcp`, disk), no traceback escapes `main`. `walk` prints
the decisions, `P0` and `P1` side by side, `unexpressible`, the feasibility line (the existing
one-line warning when infeasible, still rendered), the slip numbers and the round-trip
residuals; `--json` on every subcommand prints the same as JSON for scripts and DCC glue.

### 4.11 Documentation and version

README gains "Emotion for any rig" (what goes in, what comes out, the Blender round trip with the
exporter's Z-up/Y-forward and scale settings and the importer's settings for the file that
comes back, the Mixamo path through Blender, the MCP registration, the measured claim with the
report's headline numbers). CLAUDE.md gains the commands, the `rig/` layering, and the facts that
bite from this phase (calibrated zero and the posture map, one foot point everywhere, the
floor definition, the two compile modes and the FK-equivalence gate, the convention table,
`generate(..., welded=)`, the twelve-leg-role minimum, the double-support inversion, no LLM
inside, which clips may never be committed). Version
0.4.0 (`pyproject.toml`, `tests/test_packaging.py`).

---

## 5. Layering

New package `src/emotive_gait/rig/` on top of all six: `compile.py` (MuJoCo, `mocap.readers.bvh`,
`robot.loader.RobotModel`), `mapping.py` (`mocap.skeleton`, `robot.roles`, NumPy, pyyaml),
`profile.py`, `ranges.py`, `synthesis.py` (`gait.generator`, `gait.ik`, `emotion.model`,
`emotion.learned`, `emotion.prompt`, `mocap.*`), `validate.py`, `cli.py`, `server.py` (the only
module that imports `mcp`, and only inside the function that starts it), `data/bone_names.yaml`.
Only `cli.py` imports `rig/`. `rig/` never imports `learn/` and never imports `sim/` (the rig
tier records its own clips from the compiled chain).

Changes outside `rig/`:

- `mocap/readers/bvh.py`: `BvhFile`, `read_bvh_file`; `read_bvh` unchanged in behaviour.
- `mocap/writers/bvh.py`: new, pure.
- `gait/generator.py`: `generate(params, robot, *, welded: RobotModel | None = None, ...)`;
  `None` keeps today's `weld_robot_for(robot.profile)` path; `_procedural` writes the posture
  roles through `profile.posture_map` (identity on the G1). The G1 is bit-identical.
- `gait/footsteps.py`: `rule_double_support(params)` moves here from
  `tests/mocap/test_roundtrip.py` (pure math on the foot path; the test imports it).
- `gait/ik.py`: unchanged; `_is_welded` already accepts any model whose body named `pelvis`
  has no joint, which the compiled chain satisfies by naming its root body `pelvis`.
- `robot/roles.py`: `ROLE_CONVENTIONS` (data) and `RobotProfile.posture_map` (an optional
  4 x 4 linear map from the four posture parameters to the four posture roles' hinge angles,
  per side for the arms; identity when absent).
- `emotion/model.py`: `profile_aware: bool` on the protocol (documented, optional) and on the
  two models.
- `emotion/prompt.py`, `emotion/data/affect_lexicon.yaml`: new.
- `cli.py`: registers the `rig` group and `serve`.
- `pyproject.toml`: `[project.optional-dependencies] mcp = ["mcp>=<the installed major>,<next>"]`
  (the builder pins to the major it verified; the SDK is at 2.x on 2026-10-26); `dev` includes
  it so the smoke test runs in CI.

---

## 6. Honest limitations (printed in the README section)

1. The output is a parametric gait, not the source's motion with an emotion on top: the 14
   parameters survive the round trip, the actor's idiosyncrasies do not.
2. Froude re-dimensioning is first-order (Donelan & Kram 1997); a child rig or a giant gets a
   rescaled human effect, not a measured one.
3. Posture semantics are the extractor's operational definitions; a rig's shoulder or spine
   rigging may render "trunk pitch" differently from a human.
4. No hands, fingers, head yaw, or pelvis rotation; the swing-foot pitch is emergent.
5. The name lexicon covers the common families; an unusual rig needs an override file.
6. The emotion prompt lexicon is hand-written and English; the MCP host's model does better.
7. Skewed hinge axes drop procedural upper-body cues; the legs are always solved.
8. Zero slip holds for feasible parameters; an emotion that asks a rig for an unreachable stride
   is reported, not clipped.

---

## 7. What does not change

`evidence.py`, `axes.py`, `AXIS_GAINS`, `CATEGORY_OFFSETS`, the learned artefact, the fit
pipeline, the extractor's thresholds and definitions, `footsteps.py`, the IK's costs and
thresholds, the G1 profile and its two xfails, the physics tier, the kinematic tier, the data
manifest's committed entries (new entries are additive), every existing test.

---

## 8. Left to the builder

- The exact `MjSpec` construction for per-channel hinges and for baking the reference rotation
  (body quaternion plus re-expressed hinge axes); the FK-equivalence test decides.
- The probe angle for the posture Jacobian and whether `posture_map` is stored per side or
  pooled (the arms are symmetric by construction; the static-posture test decides).
- The chordal-mean implementation and the heading removal for the root.
- The probe magnitude and the `axis_purity` threshold (0.8 is the starting value; set once
  against the synthetic family, then frozen).
- The bone-name lexicon's token lists and the exact families' spellings (verified against the
  files in hand: 100STYLE, the CMU conversion, Bandai, and Mixamo's documented `mixamorig:` names).
- The affect lexicon's word list and weights.
- The validation report's layout.
- `mcp` version pin and the smoke test's client code.
- The inference thresholds for units (the candidate scales and the 0.3 to 1.3 m window).

---

## 9. Order of work and drop order

**Order.** (1) `ROLE_CONVENTIONS` and the G1 probe test (elbow xfail), so every later sign has a
reference. (2) `BvhFile`, `read_bvh_file`, `write_bvh`, the inverse test; `read_bvh` bit-identical.
(3) the synthetic rig builder and its test-side walker, then `rig/compile.py` (`full` and
`driven`) with the FK-equivalence gate on the 100STYLE fixture and the synthetic family. (4)
`rig/mapping.py`: lexicon, geometry, probes, the reference pose, the calibrated zero and
`posture_map`, with their tests. (5) `RigProfile`, `generate(welded=, posture_map)`,
`rule_double_support` in `footsteps.py`, `rig/synthesis.py`, the round-trip and slip tests on
the synthetic family and the 100STYLE fixture. (6) `emotion/prompt.py` and its lexicon. (7)
`rig/cli.py`. (8) `rig/server.py` and the smoke test. (9) `rig validate --report`, run on the
real rigs, commit the report. (10) README, CLAUDE.md, version 0.4.0.

**Drop order if the schedule slips** (first dropped first): the Bandai Namco and CMU
validation rows; the `emotive-gait://rig-mapping-schema` resource; `--json`; negation in the
affect lexicon; `rig validate --report` (the numbers then go into the report by hand from the
test output, and the report says so). **Never dropped:** the FK-equivalence gate, the
calibrated zero with `posture_map` and their tests, the round-trip and slip tests on the
synthetic family and the 100STYLE fixture, the MCP server with its five tools and the smoke
test, the `inspect` audit output.

---

## 10. Risks

- **MuJoCo hinge composition versus BVH Euler composition, and the baked reference.**
  Mitigated by the FK gate (1e-6 m) on real files with three channel orders; the composition
  order was verified during the builder check, the baking is what the gate now guards.
- **The posture map is linear about the reference.** Emotion deltas are a few tenths of a
  radian; the static-posture test checks a 0.3 rad trunk command reads back within 0.02 rad.
  If a rig's spine geometry breaks that, the residual appears in the validation report.
- **Rigs whose rest pose is far from any walking pose (T-pose with the legs apart).** The
  reference pose is the walking median, not the rest pose, so the IK seed is near a walk.
- **Rigs with skewed local axes** (bones whose Euler axes are not anatomical). The legs are
  IK-solved over all hinges and do not care; upper-body cues are dropped with a warning; the
  report lists how many rigs in the validation set hit this.
- **Extractor thresholds set on humans and the G1 meeting a stylised character** (very short
  legs, exaggerated bounce). The thresholds are frozen; a failure is reported as a discard
  reason and a finding, never retuned here.
- **The real-rig validation set is thin if the CMU conversion and Bandai cannot be fetched.**
  The 100STYLE fixture and the synthetic family carry the CI claim regardless; the report lists
  what was not run.
- **`mcp` SDK churn (2.x).** Pinned to the verified major; the server uses only tools, one
  resource type and one prompt; the smoke test catches API drift in CI.
- **Scope creep toward retargeting or a Blender add-on.** Both are non-goals; the DCC path is
  the file round trip, documented.
