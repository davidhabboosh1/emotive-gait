# Phase 4.5 — Learned Emotion Model: Design

**Date:** 2026-10-25
**Status:** approved 2026-10-25
**Parent:** `2026-10-25-emotive-gait-professionalization-design.md` (the professionalization design),
sections 2, 4.4, 7 and Appendix A. This document refines the phase the user inserted between
Phase 4 (forward locomotion) and Phase 5 (evaluation harness). Where it differs from the parent,
this is the later decision and says so (section 11).
**Inputs:** `docs/superpowers/research/2026-10-25-phase-4-5-brainstorm-record.md` (binding
decisions) and the three research reports beside it (datasets, ML approaches, related work).
Facts taken from those reports are not re-derived here; where this document relies on one that
the reports mark UNVERIFIED, it says so and names the builder's check.

---

## 1. Intent and why

The emotion map is the least-specified link in the chain. Six intensities become three axes
(`axes.py`), the axes become parameter deltas through `AXIS_GAINS`, and a few category offsets
sit on top (`model.py`). The *directions* are cited (`evidence.py`); every *number* is tuned by
eye and labelled as such. The project's honesty rule has been "never present a tuned number as
literature-derived". This phase replaces that rule's weakest consequence, a model whose
magnitudes nobody can check, with magnitudes that come from measured human walking, carry
confidence intervals, and are validated on actors the fit never saw.

After this phase:

- **A second emotion model exists beside the analytic one.** `LearnedEmotionModel` maps the same
  six intensities onto the same `GaitParameters`, through per-emotion, per-parameter effects
  fitted from public motion capture of acted emotional walking. It is selectable at the seam the
  Phase 1 plan reserved (`emotion.model.modulate`) and from the CLI. The analytic model stays the
  default until the user decides otherwise.
- **Every fitted effect is a measurement with provenance.** The committed artefact says, for each
  emotion and parameter: the effect in leg-length-normalised units, its 95 % interval, how much
  actors differ, how many actors and cycles it rests on, which datasets, and whether it agrees
  with the cited direction. Nothing in `evidence.py` changes: claims stay signed directions.
- **The project makes a measurable claim without a rater study.** The headline is held-out-actor
  prediction error: how well the fitted map predicts an unseen actor's emotional gait changes,
  against how well the current hand-tuned gains predict them. A second, falsifiable claim tests
  the three-axis structure that `axes.py` assumes.
- **The extractor is a reusable tool.** It measures the gait parameters of any skeleton
  trajectory, from any reader, in any proportions, with no reference to the G1. Phase 4.6
  ("emotion for any rig") is built on it; this phase only uses it.
- **The README positions the work correctly** against EMoG and PAMoR and drops a citation nobody
  can find.

What this phase is not: a motion generator, a classifier, a controller, or a claim about what
observers perceive. Those are Phase 5's business or nobody's (section 2).

---

## 2. Goals, non-goals, anti-goals

### Goals

- A gait-parameter extractor that takes world joint positions of an arbitrary skeleton and
  returns per-cycle measurements of the 14 measurable `GaitParameters` fields, plus speed and
  leg length, with gait events detected from the feet.
- Readers for the three source formats (BVH, ASF/AMC, PTD) with no dependency beyond NumPy.
- A mixed-effects fit of emotion effects on leg-length-normalised, within-actor parameter deltas,
  with confidence intervals and between-actor spread, validated leave-one-actor-out.
- The held-out-actor error of the fitted map and of the hand-tuned gains, side by side, as the
  phase's headline number.
- A model-comparison test of the three-axis structure (constrained vs reduced-rank vs full
  per-emotion), reported with its low-power caveat.
- `LearnedEmotionModel`, selectable beside the analytic model, under the same cited-direction
  tests. A fitted sign that contradicts a cited direction is a reported finding, never a silent
  override and never a silent correction.
- Reproducibility: a `fit` command, a data manifest with hashes, fixed seeds, and a committed
  artefact that records everything the fit depended on.
- README and docs corrected and positioned against EMoG and PAMoR; CI green on all four jobs.

### Non-goals

- **Reinforcement learning, learned balance, or any learned controller.** Unchanged from the
  parent's anti-goal: stability never gets authority over expression.
- **Motion generation** (diffusion, VAE, style transfer, retargeted playback). Rejected in the
  brainstorm: no data at the needed scale, no citable magnitudes, outputs joint angles.
- **A classifier**, even as a CI proxy. Ranked fourth in the approaches report; circular on
  parameters and not a perception measure. Reconsidered only if Phase 5 wants a fidelity check.
- **Perception.** Nothing here says a G1 walking with these numbers reads as the emotion. That is
  Phase 5. The README must keep saying no recognition rate has been measured.
- **Disgust and surprise.** No walking data exist. They stay analytic and `UNSOURCED`, and the
  CLI keeps warning.
- **Intensity calibration.** Every dataset records one intensity per emotion, so the scaling law
  is not identifiable (section 6.3). It stays an assumption, stated.
- **Phase 4.6** (BVH out, MuJoCo chain from a BVH skeleton, LLM role mapping, MCP server),
  except where the extractor's interface must anticipate it (section 4.4).
- **Hardware, physics tuning, new root modes.** The physics tier is untouched. Learned presets
  are not added to the Z-rail survival gate (section 7.4).

### Anti-goals

- **A number presented as data-derived that was not fitted from data.** This includes a tuned
  fallback inside the learned artefact, a prior that was not labelled, or a magnitude "cleaned
  up" after the fit. The artefact is generated, never hand-edited.
- **Silently keeping or silently fixing a contradiction.** If the data say sad arms swing more
  and Montepare says less, the model ships the data and the README says so.
- **Committing raw motion capture, or any per-trial table derived from it.** Only aggregates
  (coefficients, intervals, counts) enter the repository, and Body Movement Library aggregates
  only once the Glasgow PACO lab has given written permission.
- **Comparing recognition numbers with EMoG or PAMoR**, or claiming "first". The related-work
  report lists what must not be said; section 8 repeats it.
- **The analytic model quietly changing.** `AXIS_GAINS`, `AXIS_WEIGHTS`, `CATEGORY_OFFSETS` and
  every existing test keep their values. The fit informs a later retune; it does not perform one.

---

## 3. Constraints

- **Project rules that stand.** Emotion is applied to parameters, never joint angles; the
  generator clamps to joint range last. No claim in `evidence.py` states a magnitude in degrees.
  `expansiveness` is a multiplier on arm swing, abduction and step height and is fixed at 1.0 in
  any fitted model. `speed = step_length * cadence`. `knee_flexion_bias` is retired. Positive
  `neck_pitch` is gaze down; positive `trunk_pitch` is forward inclination. Amplitudes are
  relative to the rest pose, postures are absolute. CLI exit codes 2 (user) and 1 (environment),
  no traceback escapes `main`. No Claude attribution trailer in commits; merging and pushing are
  the user's.
- **Dependencies.** `mujoco>=3.17,<3.18` stays pinned and untouched. New runtime dependencies for
  the *installed package*: none. The fit needs `statsmodels` (which brings `pandas`); it lives in
  a `fit` extra, included in `dev` so CI exercises the fit code on a synthetic fixture. No
  PyTorch, JAX, scikit-learn or any framework. Format readers are pure Python on NumPy.
- **Data stay local.** `data/raw/` and `data/derived/` are git-ignored. CI downloads nothing;
  every CI test runs on the committed artefact and on synthetic fixtures.
- **Licences.** 100STYLE is CC BY 4.0 (attribution in README and artefact). CMU is "free for all
  uses", no resale of the data, citation requested. The Body Movement Library states no licence;
  its derived aggregates are committed only after written permission (section 4.1). Bandai Namco
  (CC BY-NC) and E-Gait (no licence, unknown frame rate) are excluded.
- **G1 facts the transfer depends on.** Leg length `l_G1 = 0.69 m` (hip-pitch joint height at
  `qpos0`, computed in the approaches report, section 3); nominal walking pelvis height 0.68 m
  (`com_height_scale = 1.0`); the preset feasibility test is the arbiter of what the legs can
  reach.
- **Environment.** Windows 10 and Ubuntu CI, Python 3.11 and 3.12. The fit must run on the
  author's machine in minutes, CPU only.

---

## 4. Decisions

### 4.1 Data: three sources, two licence tiers, one email

| Source | Actors | Project emotions it feeds | Format used | Licence tier |
|---|---|---|---|---|
| Body Movement Library (Ma, Paterson & Pollick 2006, Glasgow PACO) | 30 | anger, happiness, sadness (+ neutral) | PTD point-light files: 15 joints, 3-D positions, 60 Hz | **pending**: no licence stated; the user emails the PACO lab; aggregates committed only on written permission |
| 100STYLE (Mason, Starke & Komura 2022) | 1 | anger (Angry), sadness (Depressed), happiness (Elated) (+ Neutral) | BVH, 28 bones, 60 fps, forward-walk files only, trimmed by `Frame_Cuts.csv` | CC BY 4.0: commit freely with attribution |
| CMU Graphics Lab MoCap | 5 subjects (82, 91, 104, 105, 142) | sadness, happiness, **fear** (+ each subject's normal walk) | ASF/AMC, 120 Hz, inches to metres | free for all uses, cite: commit freely |

**Label mapping is a declared assumption, per row.** The project's categories are the six basic
emotions; the datasets' labels are style names. A committed table (`data/manifest.yaml`, section
4.6) maps each clip to a project emotion and marks the row `exact` (angry, happy, sad, scared,
neutral) or `mapped` (Depressed -> sadness, Elated -> happiness, Joy and HappyGoLucky ->
happiness). The artefact records, per emotion, how many of its actors come from `mapped` rows.
Styles that are not basic emotions (Proud, Shy, Macho, TooCool, Stern, Confident, Excited,
Rushed, Cool) are listed as excluded with the reason. **My call:** Depressed maps to sadness and
Elated to happiness; a sensible builder would do the same, and the artefact's `mapped` count
lets a reader discount it.

**Per-actor neutral is mandatory.** An actor enters the fit only with a neutral (or "normal")
walk from the same dataset and session, because every effect is estimated relative to that
actor's own neutral (section 4.3). A CMU subject without a normal walk is excluded and listed.
The datasets report says each CMU emotional-walk subject has one; **the builder verifies this on
the subject pages before the manifest is frozen.**

**Actor identity is a manifest field, separate from subject number.** CMU subjects 91 and 105
carry identical style lists (SadWalk, DepressedWalk, ShyWalk, ScaredWalk, MachoWalk,
TooCoolWalk), which in CMU often means one performer in two sessions. **My call, conservative:**
two subjects are the same `actor` unless the CMU subject descriptions say otherwise; the builder
checks the pages and records the decision in the manifest. Actor counts, the `fitted` threshold
and the held-out folds all use `actor`, never `subject`, so a merged performer cannot leak
between folds.

**Coverage, stated plainly.** Anger rests on BML (30 actors) plus one 100STYLE actor; without
BML it rests on one actor and is `thin` (section 4.5). Happiness and sadness have BML plus
100STYLE plus three to four CMU subjects. Fear has only CMU single trials, and after checking the
subject pages (2026-10-25) fewer than first thought: subject 142 has no normal or neutral walk,
so it is excluded under the per-actor-neutral rule, and subjects 91 and 105 carry identical trial
lists and count as one performer. Fear is therefore one or two actors, labelled `thin` whatever
happens, stored and used with its actor count printed in place of an interval, and enters no
headline number (section 4.3). Disgust and surprise have nothing and stay analytic (section 4.5).

**The two licence tiers are a switch, not a fork.** `emotive-gait fit` fits on every dataset
present under `data/raw/` whose manifest entry is marked `committable: true`. BML's entry ships
as `committable: false` with a `permission:` field. When the PACO lab's written permission
arrives, the user sets `committable: true`, pastes the permission text or reference into the
manifest, and the next `fit` includes BML. Until then the committed artefact is the 100STYLE +
CMU fit, and its provenance block says so. The pipeline is developed and validated against BML
locally regardless, because that is where the multi-actor variance lives; only *committing*
BML-derived numbers waits.

**Decision rule for execution** (so the build need not wait on the email): if permission has not
arrived when the fit task runs, ship the permissive-tier artefact with anger marked `thin`, and
leave BML inclusion as a one-command follow-up. If permission is refused, BML is removed from the
manifest and the local pipeline; the design is unchanged.

**Excluded, with reasons recorded in the manifest:** E-Gait (frame rate unstated, so cadence is
unrecoverable; no dataset licence), EWalk (no download found), Bandai Namco (CC BY-NC against an
MIT repository), Emilya, Roether and Barliya (no public route), KIT's two emotional walks
(subsumed by sad/neutral elsewhere; one subject), MPI EBMDB and KDAE (no walking).

### 4.2 The extractor: landmarks in, parameters out, no robot anywhere

A new base-adjacent package `src/emotive_gait/mocap/` with four parts. It imports NumPy and
SciPy, and `gait.params` for the parameter names; nothing from `robot/`, `emotion/` or `sim/`.

**Readers** (`mocap/readers/`): `bvh.py`, `asf_amc.py`, `ptd.py`. Each returns a `SkeletonClip`:
joint names, parent index per joint (or none for marker-style data), world positions as an array
of shape (frames, joints, 3) in metres with +z up, the frame rate, and the source path. BVH and
ASF/AMC readers parse the hierarchy and channels and run forward kinematics to positions; the
PTD reader just reshapes. Units and up-axis are normalised in the reader (BVH is usually y-up and
in centimetres or arbitrary units; CMU ASF lengths are inches scaled by 0.45; `Frame_Cuts.csv`
trims 100STYLE). The BVH parser keeps the hierarchy and channel order as data so Phase 4.6 can add
a writer without re-parsing. A CSM reader for BML's marker files is a droppable fallback (section
10), needed only if the PTD joint list lacks a head or neck point (**builder verifies the PTD
column order from the data page before deciding**).

**Landmarks** (`mocap/skeleton.py`): a `Landmark` enum of the semantic points the extractor needs
and nothing else: `PELVIS`, `HIP_L/R`, `KNEE_L/R`, `ANKLE_L/R`, `TOE_L/R` (optional),
`SHOULDER_L/R`, `ELBOW_L/R`, `WRIST_L/R`, `NECK` (top of trunk), `HEAD`. A `LandmarkMap` is a
mapping from `Landmark` to a joint name or to a list of joint names to average (a pelvis from two
hip markers, a neck from two shoulder markers). One `LandmarkMap` per dataset ships in
`mocap/landmarks.py` (100STYLE, CMU, BML PTD). This is deliberately not `JointRole`: roles name
actuators the robot has; landmarks name points any skeleton has. Phase 4.6's bone-name mapping
produces a `LandmarkMap`.

**Gait events** (`mocap/events.py`): heel strike and toe-off per foot from the ankle landmark
(toe when present), in a path frame. The path frame is rebuilt every frame from the smoothed
horizontal pelvis velocity (forward = velocity direction, left = up x forward), so curved paths
(BML walks a triangle) and arbitrary world headings are handled; frames whose yaw rate exceeds a
threshold are marked turning and the cycles that overlap them are discarded. A foot is in stance
while its path-frame horizontal speed is below a fraction of the mean pelvis speed and its height
is within a band of its own stance minimum; strike is stance onset, toe-off is stance end. The
thresholds and smoothing are the builder's (section 9) and are set once against the synthetic
round-trip test, not per dataset.

**Measurements** (`mocap/extract.py`): a `GaitMeasurement` per gait cycle (two consecutive
strikes of the same foot), with every field defined to match what the generator renders:

| Field | Operational definition (path frame, per cycle) | Matches generator because |
|---|---|---|
| `cadence` | steps per second: 2 / cycle time | `phase = t * cadence / 2` |
| `step_length` | forward distance between successive contralateral foot placements at strike, mean of the two steps | pelvis advances `2 * step_length` per cycle |
| `step_height` | peak swing-foot clearance above that foot's own stance height, mean of two swings | `step_height * clearance(u)` |
| `double_support_frac` | fraction of each step during which both feet are in stance, mean of two | `footsteps.swing_fraction` |
| `feet_spacing` | mean lateral distance between the ankle landmarks over the cycle | foot `y = +-feet_spacing / 2` |
| `pelvis_height` (maps to `com_height_scale`) | mean pelvis landmark height over the cycle, metres; normalised by leg length; `to_gait_parameters` turns its delta into a `com_height_scale` delta through the target's nominal height | `nominal_height * com_height_scale` |
| `com_bounce_amp` | half peak-to-peak of detrended pelvis height | `com_bounce_amp * cos(4 pi phi)` |
| `lateral_sway_amp` | half peak-to-peak of detrended pelvis lateral position | `-lateral_sway_amp * sin(2 pi phi)` |
| `trunk_pitch` | mean sagittal angle of the pelvis-to-neck vector from vertical, positive forward | `WAIST_PITCH = trunk_pitch`, positive = inclined |
| `neck_pitch` | mean sagittal angle of the neck-to-head vector relative to the trunk vector, positive = head forward and down | injected neck is on the torso; positive is gaze down |
| `arm_swing_amp` | half peak-to-peak of the upper-arm (shoulder-to-elbow) sagittal angle relative to the trunk, mean of both arms | shoulder pitch amplitude about the rest pose |
| `arm_abduction` | mean frontal-plane angle of the upper arm from the trunk, mean of both arms | `SHOULDER_ROLL = sign * arm_abduction` |
| `elbow_flexion` | mean of (pi minus the shoulder-elbow-wrist angle): 0 is straight | `ELBOW = elbow_flexion + ...` (**builder confirms the G1 elbow joint zero is the straight arm**) |
| `elbow_swing_ratio` | least-squares slope of elbow-flexion deviation on upper-arm sagittal deviation over the cycle (signed) | `ELBOW = elbow_flexion + elbow_swing_ratio * (shoulder - rest)` |
| `speed` (derived) | mean path-frame pelvis speed | consistency check against `step_length * cadence` |
| `leg_length` (per actor) | hip-to-knee plus knee-to-ankle landmark distance (medians over the neutral clip) plus the ankle's stance height | Hof normalisation (section 4.3) |
| `swing_fraction` (extra) | swing time over cycle time | cross-check against Barliya's 33-38 % |

`expansiveness` is not measured and is fixed at 1.0 (brainstorm record). The extractor sees
rendered motion, so on a G1 trajectory it measures the *folded* arm swing, abduction and step
height (`expansiveness * arm_swing_amp` and so on); wherever analytic parameters are compared
with measurements (the round trip, the hand-tuned baseline of section 4.3) they are folded first.
Angles are in radians, lengths in metres, before normalisation. A clip summary averages cycles and reports the cycle
count, discarded cycles and the reason (turning, event detection failure, missing landmarks). A
missing landmark makes the dependent fields `NaN`, never zero: a dataset without a head point
yields no `neck_pitch`, and the fit then excludes that dataset from that parameter.

**Normalisation and re-dimensioning** (`mocap/normalise.py`): Hof scaling with the actor's
`leg_length` `l` and `g = 9.81`: lengths divided by `l`; cadence multiplied by `sqrt(l / g)`;
speed divided by `sqrt(g * l)`; angles, `double_support_frac` and `elbow_swing_ratio`
unchanged. The inverse maps a normalised delta onto a target rig with its own leg length, which
is how the learned model reaches the G1 (section 4.5) and how Phase 4.6 reaches any rig; the
normalised pelvis-height delta becomes a `com_height_scale` delta by multiplying by the target's
leg length and dividing by its nominal walking height (G1: 0.69 / 0.68). Because Donelan & Kram found Froude matching only approximate, a second,
`ratio` scaling (every length as a fraction of the actor's own neutral value, cadence likewise)
is computed alongside; both are stored and both are tested against the cited directions (section
4.5). `froude` is the default.

**Round-trip test with known truth.** The project's own kinematic tier is the synthetic fixture:
`sim.kinematic` poses the G1 along a generated trajectory, the test records the robot's joint
positions as a `SkeletonClip` with a G1 `LandmarkMap`, and the extractor must recover the
`GaitParameters` that produced it within stated tolerances (cadence and step length within 2 %,
step height, bounce and sway within 3 mm, angles within 0.02 rad, `elbow_swing_ratio` within
0.05; the amplitude fields compared after folding `expansiveness`). This runs in CI, covers the
four analytic presets walking forward, and is the gate for every event-detection threshold.
**My call:** the in-place (zero-translation) case is excluded: the path frame is built from
pelvis velocity and is undefined without travel, no dataset contains in-place walking, and
Phase 4.6 can add a hip-to-hip frame if it needs one. The test lives in `tests/mocap/` and may import
`sim/`; the package `mocap/` may not.

### 4.3 The model: a mixed-effects fit on within-actor deltas

**Data table.** One row per gait cycle: dataset, actor, emotion (neutral included as a level),
cycle index, and the 14 normalised parameters as measured, not pre-centred.

**Model, one fit per parameter, same design throughout:**

```
x[a, e, c] = mu + beta[e] + u[a] + v[a, e] + eps[a, e, c]        beta[neutral] = 0
```

`beta[e]` is the fixed emotion effect for each of anger, fear, happiness, sadness relative to
neutral: the within-actor delta, estimated *with* the uncertainty of each actor's neutral mean
(a CMU subject's single neutral trial carries sampling error that pre-centring would hide, so
the neutral is a level, not a subtraction). `u[a]` is a random intercept per actor (their
neutral gait); `v[a, e]` is a random slope per actor and emotion, the between-actor spread of
the effect and the thing a held-out actor cannot borrow; `eps` is the cycle-to-cycle residual.
Fitted with `statsmodels` `MixedLM` by REML, groups = actor. If the maximal random structure
fails to converge for a parameter, the fit falls back to the random intercept alone and records
the fallback in the artefact; nothing is tuned to make it converge.

**No dataset term.** The approaches report suggested one, but an emotion-independent dataset
shift is unidentifiable from within-actor effects (it would be "this dataset's mean emotional
delta") and would quietly correct a single-dataset emotion by one actor's behaviour on *other*
emotions. **My call:** datasets enter only through their actors; the report prints each cell's
estimate per dataset as a diagnostic so a reader can see disagreement, and the artefact records
`n_datasets` per cell.

**What the artefact stores per (emotion, parameter) cell:** `estimate`, `ci_low`, `ci_high`
(95 % Wald, present only when `n_actors >= 3`), `between_actor_sd`, `residual_sd`, `n_actors`,
`n_cycles`, `n_datasets`, `n_mapped_actors`, `grounding` (`fitted`, `thin` or `missing`,
section 4.5), `direction_check` (section 4.5), for both scalings. Plus a provenance block
(section 4.6). **My call on thin cells:** with one or two actors a Wald interval measures one
person's repeatability, not an emotion, so no interval is stored or printed; the actor count
stands in its place everywhere the interval would appear.

**Held-out-actor validation (the headline).** Leave-one-actor-out over every actor. For each
held-out actor, refit on the rest and predict `beta[e]` for the emotions that actor performed;
the target is the actor's **observed mean delta** (emotion-clip mean minus neutral-clip mean, one
value per actor and emotion, so a 30-second BML clip and a three-cycle CMU trial count once
each). Only `fitted` cells (three or more actors in the full set) enter: a one-actor emotion has
no refit to predict it. Error is the root-mean-square over held-out actor-emotion pairs, per
parameter, **standardised by the between-actor standard deviation of the observed deltas for
that parameter** (the spread of actor-emotion means around their emotion means, pooled over
emotions), so that 1.0 is the irreducible floor of predicting an unseen person from the mean,
and larger numbers read as "this many actors' worth of spread away"; the value in natural
normalised units is printed beside it. The overall row is the unweighted mean over the fitted
parameters. Three predictors share the table: **E_fit** (the refit); **E_hand**, the
hand-tuned gains, meaning the analytic model's delta at intensity 1.0 for that emotion
(`modulate` on `GaitParameters()`, `expansiveness` folded into arm swing, abduction and step
height, then converted to normalised units with `l_G1` and the G1 nominal height); and
**E_zero**, predicting no change, the sanity floor both must beat. The claim reads: "On N
held-out actors from D datasets, the fitted map predicts per-emotion gait-parameter changes
with standardised error E_fit, against E_hand for the hand-tuned gains and E_zero for no
change." The fairness caveat is printed with it: the gains were tuned for a G1 rendering and
never targeted human magnitudes, so E_hand measures how far the hand tuning sits from measured
human effects, not how badly it was done.

**Speed confound.** The sources that separated emotion from speed (Roether, Barliya) are not
available, and none of ours has speed-matched neutrals. The primary fit therefore estimates the
emotion effect *as acted*, speed included, which is also what the robot plays. A secondary fit
adds the actor's normalised speed delta as a covariate and reports how much of each posture
effect survives; it is a row in the fit report, feeds nothing, and is droppable (section 10).

**The three-axis structure test.** On the actor-by-emotion mean deltas (one row per actor and
emotion, equal weight, so the long 100STYLE clips do not dominate a least-squares fit the way
they would at cycle level), four structures of the emotion-by-parameter effect matrix `B`
(4 x 14) are compared by held-out-actor error, same protocol and standardisation as above:

1. **S1, the analytic structure:** `B = W G` with `W` the 4 x 3 `AXIS_WEIGHTS` from `axes.py`
   held fixed and `G` sparse exactly as `AXIS_GAINS` assigns one axis per parameter (parameters
   with no axis get zero); only the 9 gains are fitted, by least squares.
2. **S2, fixed axes, free loadings:** `W` fixed, `G` a dense 3 x 14 matrix.
3. **S3, reduced rank 3:** `W` and `G` both free (rank-3 approximation of the full `B`).
4. **S4, full per-emotion:** `B` unconstrained (rank 4, the saturated model).

Plus **S1 + fear offsets**: S1 with fear's four `CATEGORY_OFFSETS` parameters free, which asks
whether fear's documented overrides are needed. In-sample fit (R^2) and held-out standardised
error are reported for each. **The caveat is part of the result:** five group means always lie
in a four-dimensional affine space, so rank 3 removes one dimension and the test is low-powered
with four sourced emotions; the report says "S1 loses X against S4 on held-out actors", never
"the data discovered three axes". Implemented with NumPy least squares and SVD; no new
dependency.

**Determinism.** Extraction and the REML fit are deterministic. If any step is randomised
(bootstrap intervals, if Wald intervals prove unreliable for `thin` emotions), the seed is a
constant in `fit.py` and recorded in the artefact.

### 4.4 Interface shape, anticipating Phase 4.6

The extractor's public surface is small and rig-free, so the artist tool can reuse it unchanged:

- `read_bvh(path) / read_asf_amc(asf, amc) / read_ptd(path) -> SkeletonClip`
- `extract_cycles(clip, landmarks: LandmarkMap) -> list[GaitMeasurement]` and
  `summarise(cycles) -> ClipSummary`
- `normalise(measurement, leg_length) -> NormalisedMeasurement` and
  `to_gait_parameters(normalised_delta, base: GaitParameters, leg_length) -> GaitParameters`

Nothing in the signatures names a dataset, a frame rate, an up-axis or a joint count. `SkeletonClip`
keeps the hierarchy when the format has one, so a writer can be added. `RobotProfile` gains a
`leg_length` field (G1: 0.69 m) so the learned model re-dimensions through the profile it is
already handed, and a Phase 4.6 rig profile supplies its own. That is the only change to
`robot/`.

### 4.5 The seam: two models, one contract, one set of direction tests

**Shape.** `emotion/model.py` gains a small protocol: an emotion model is anything with
`deltas(intensities: Mapping[Emotion, float], profile: RobotProfile) -> Mapping[str, float]`
returning parameter deltas in the robot's units. The existing axis-plus-offset code becomes
`AnalyticEmotionModel` (behaviour bit-identical; `AXIS_GAINS` and friends stay where they are).
`modulate(params, intensities, profile, *, model=None, hold_speed=None, speed_mode=...)` takes an
optional model and defaults to the analytic one, so every existing call site is unchanged.
`unexpressible` and the profile-based cue dropping apply to both models unchanged.
`PRESET_PARAMS` stays analytic; `learned_presets()` returns the same four presets under the
learned model for the feasibility test.

**`LearnedEmotionModel`** lives in `emotion/learned.py`. It loads the committed artefact
(package data, `src/emotive_gait/emotion/data/learned_coefficients.yaml`, read with the
already-present `pyyaml`; no `statsmodels` at runtime), takes a `scaling` of `froude` (default)
or `ratio`, and re-dimensions each normalised effect with the profile's `leg_length`. Its deltas
are:

```
delta[p] = sum over emotions e of  intensity[e] * effect[e, p]
```

**Intensity and mixing rule, stated as an assumption.** Effects scale linearly from zero at
intensity 0 to the fitted endpoint at 1.0, and emotions mix additively. Neither is identifiable
from single-intensity, single-emotion data; both match the analytic model's behaviour (its axes
are also a linear, additive sum), so the two models differ only in their endpoints. Opposed
emotions therefore still cancel toward neutral under either model; Appendix A.6's argument about
posture cues is unaffected because it concerns *which* cues cancel, which the fit now measures.

**Grounding is per cell, and so is delegation.** Each (emotion, parameter) cell is `fitted`
(three or more actors contribute both a neutral and that emotion *with that parameter
measurable*), `thin` (one or two actors) or `missing` (no data: a dataset without a head point
yields no `neck_pitch`, and disgust and surprise have nothing). **My call:** the learned model
uses `fitted` and `thin` cells and delegates `missing` cells to the analytic model, cell by cell,
so one emotion may mix a fitted cadence with an analytic neck pitch; the artefact and the CLI say
exactly which cells are analytic. A `thin` cell is used because a measured number with its actor
count printed is more honest than a tuned one; the CLI prints one warning listing the thin cells
and their actor counts, as it warns for unsourced emotions. Falling back to the analytic delta
for `thin` cells is a one-line policy change if the user prefers it. An emotion all of whose
cells are `missing` is simply analytic (disgust, surprise).

**Cited-direction tests run on both.** `test_modulation_direction_agrees_with_every_grounded_claim`
is parametrised over the analytic and learned models and over both scalings. For the learned
model the rule is: for every grounded claim with a non-zero direction on a `fitted` or `thin`
cell, either the estimate has that sign, or the pair is listed in the artefact's
`contradictions` block (with its interval when it has one). For the two no-change claims (anger
`neck_pitch`, anger `com_height_scale`), "agrees" means the interval spans zero; on a thin cell
with no interval the check is `thin_unchecked`. The four `expansiveness` claims have no estimate
(the quantity is fixed at 1.0 and its evidence is carried by the arm-swing, abduction and
step-height cells) and are marked `no_estimate`; claims on `missing` cells are checked against
the analytic delta as today. The test fails only on an *unreported* contradiction.
`direction_check` per cell takes one of `agrees`, `agrees_ci_spans_zero`, `contradicts`,
`contradicts_ci_spans_zero`, `thin_unchecked`, `no_estimate`, `no_claim`. A contradiction whose interval excludes zero is a finding: it goes in the fit report
and in the README's results table with the citation it contradicts. The data are not edited and
`evidence.py` is not edited; a later evidence revision is a separate, cited decision.

**Feasibility.** The four learned presets join the preset feasibility test (IK must solve each
to the existing thresholds). If a fitted magnitude is unreachable on the G1 (a sad crouch past
the ankle limit is the likely case), the artefact is not edited: the test fails, the report says
which parameter and by how much, and the user decides between the existing flag-and-render
behaviour and a documented, labelled clip in the learned model's re-dimensioning. Fear's crouch
has already shown this is a G1 ankle-range limitation, not an evidence question.

**CLI.** `emotive-gait demo --model {analytic,learned} [--scaling {froude,ratio}]`, printing
which model produced the parameters. `emotive-gait evidence --model learned` prints the claim
table with a column for the fitted estimate, interval and `direction_check`. `emotive-gait fit`
is section 4.6.

### 4.6 Committed versus local, and reproducibility

**Committed:**

- `src/emotive_gait/emotion/data/learned_coefficients.yaml`: the per-cell records of section 4.3
  under both scalings, the `contradictions` block, and a provenance block: fit date, git commit
  of the fit code, `statsmodels` and NumPy versions, the manifest's hash, the datasets included
  and excluded (with the BML permission state), actor and cycle counts per emotion, model
  formula, convergence fallbacks, seed, and the hand-tuned comparison numbers. Generated only by
  `fit`; a test asserts the file validates against its schema and its provenance hash matches
  the committed manifest.
- `data/manifest.yaml`: for each dataset, URL, archive size and SHA-256, licence and
  `committable` flag with `permission` text, the files used with their SHA-256 and the label
  mapping per clip (subject, `actor` identity, emotion, `exact`/`mapped`, frame cuts), and the
  exclusions with reasons. The pipeline refuses to extract a file whose hash does not match.
- `docs/reports/phase-4-5-fit-report.md`: generated by `fit`: the headline table (E_fit vs
  E_hand), the per-cell effects, the structure test, the speed-adjusted secondary fit, the
  contradictions, and a plain-language limitations section (section 6).
- The extractor, readers, landmark maps, fit code, `LearnedEmotionModel`, tests and synthetic
  fixtures.

**Local, git-ignored:** `data/raw/<dataset>/` (archives and extracted files) and
`data/derived/` (per-cycle CSV tables, held-out predictions, plots). **My call:** per-cycle
tables are not committed even for the permissively licensed sets, so the repository carries one
licence story: aggregates only.

**`emotive-gait fit`:** verifies hashes against the manifest; extracts every clip to
`data/derived/cycles.csv`; fits; runs the held-out validation, the hand-tuned comparison and the
structure test; writes the artefact and the report; exits 2 on a manifest or data problem (named
file, expected and actual hash) and 1 on an environment problem (`statsmodels` missing, with the
`pip install -e ".[fit]"` hint). `fit --check` refits and diffs against the committed artefact
within a tolerance, for anyone verifying the release. **`emotive-gait fetch`** downloads the
100STYLE and CMU files named in the manifest (and BML's PTD archive, whose download is open) into
`data/raw/` and verifies them; droppable (section 10), in which case the manifest's URLs and
hashes are the instructions.

**Reproducibility claim:** anyone with the manifest's files reproduces the committed artefact
bit-for-bit up to floating-point noise, with one command, on CPU, in minutes.

**Who downloads, and when (user decision, 2026-10-25).** The build session downloads
everything: the CMU files and BML's PTD archive (264 MB) through `fetch`, which also verifies
hashes. BML is downloaded before the PACO reply because its download is open and the licence
question concerns committing derived numbers, not local use; raw data stay git-ignored. The user
has already downloaded `100STYLE.zip` (1.47 GB) to their Downloads folder, so `fetch` must accept
an existing local archive (`fetch --from <path>` copies or moves it into `data/raw/` and hashes
it) rather than downloading it again. The fit (step 7 of section 10) therefore runs inside the
build session. The only manual steps left to the user are the PACO email and, if a download
fails for reasons the session cannot fix (a blocked host, a changed URL), fetching that one file
by hand; `fetch` prints the exact URL and expected hash so that is a copy-paste task.

### 4.7 Documentation

- README: a "Two emotion models" section (what the learned model is, how to select it, that the
  analytic one is the default); a results table with the headline E_fit vs E_hand, the actor and
  dataset counts, any contradictions, and the structure-test line, each with its caveat; the
  related-work fixes of section 8; dataset attributions (100STYLE CC BY 4.0 credit line, CMU
  acknowledgement, BML citation if included); the "no recognition rate has been measured"
  sentence stays.
- `CLAUDE.md`: the two new packages and their dependency rules (section 5), the `fit` command,
  the artefact-is-generated rule, the licence tiers, and the facts that bite (thin emotions,
  delegation, the no-change rule, `leg_length`).
- `docs/future-learned-model.md`: replaced by a pointer to this design and the report; its
  dataset shortlist is superseded by the datasets report.
- `evidence.py` docstring: one sentence that fitted magnitudes live in the artefact, not here.

---

## 5. Layering

```
robot/   base layer (gains leg_length on RobotProfile; nothing else)
gait/    depends on robot/
mocap/   NEW: numpy, scipy, gait.params only; imports nothing from robot/, emotion/, sim/, learn/
emotion/ depends on gait.params; model.py also robot.roles and robot.profiles.g1 (as today);
         learned.py additionally reads its own package data; nothing in emotion/ imports mocap/,
         learn/ or sim/
learn/   NEW: fit pipeline; depends on mocap/, gait.params, emotion.evidence (direction checks),
         emotion.model (hand-tuned comparison), emotion.learned (artefact schema); imports
         statsmodels; nothing imports learn/ except cli.py
sim/     depends on gait/ and robot/ only, as today
cli.py   on top of everything
tests/mocap/ may import sim/ for the synthetic round trip; the package may not
```

The runtime package therefore gains no dependency: `mocap/` is NumPy and SciPy, `emotion/learned.py`
is `pyyaml`. `statsmodels` and `pandas` are reached only through `learn/`, behind the `fit` extra.

---

## 6. Honest limitations, to be printed in the report and the README

1. **Human effects, not robot perception.** The fit says how acted emotion changes human gait
   parameters. Whether a G1 showing those changes reads as the emotion is Phase 5's question.
2. **Intensity and mixing are assumptions** (section 4.5), not identified from these data.
3. **Acted, single-intensity portrayals by mostly non-professional actors**, pooled across
   capture protocols. The dataset term absorbs level shifts, not protocol differences in kind.
4. **Speed is confounded with emotion** in every available set. The primary effect is "as
   acted"; the secondary fit shows what survives a speed covariate, and that is all that can be
   said.
5. **Fear is thin** (two or three actors, single trials), **anger is thin without BML**, and
   **disgust and surprise are absent**. Thin cells carry an actor count instead of an interval
   and enter no headline number; the CLI warns.
6. **Transfer is approximate.** Froude similarity is a first-order rule (Donelan & Kram 1997);
   the G1 walks crouched with a bent knee and its joint limits clip some effects. The robot plays
   a rescaled, clipped version of a human effect; data value and played value are reported
   separately.
7. **The structure test is low-powered**: four emotions give five group means, so rank 3 is one
   dimension short of saturated. It is a model comparison, not a discovery.
8. **Label mapping is a declared assumption** for 100STYLE's Depressed and Elated and for CMU's
   style names; the artefact counts `mapped` actors per emotion.
9. **Operational definitions are the project's.** `elbow_swing_ratio`, `arm_swing_amp` and
   `neck_pitch` (head on trunk) are defined to match the generator, not to match any one paper's
   measure, so fitted values are not directly comparable with published numbers.

---

## 7. What does not change

1. `AXIS_GAINS`, `AXIS_WEIGHTS`, `CATEGORY_OFFSETS`, `evidence.py`'s claims and every existing
   test keep their values; the analytic model is bit-identical through the refactor.
2. The generator, IK, footsteps, kinematic and physical tiers, root modes and metrics.
3. The MuJoCo pin, the Menagerie pin, `mink` and `daqp`.
4. The Z-rail survival gate runs on the analytic presets only. A learned-preset run is reported
   in the fit report but is not a gate: changing leg tuning is a physics decision, and this phase
   makes none.
5. The CLI's exit-code contract and its unsourced-emotion warning.

---

## 8. Positioning against EMoG and PAMoR, and the README corrections

**What to say.** EMoG (arXiv 2609.14432) learns an emotion-conditioned periodic gait generator
from a proprietary eight-performer dataset and executes it on a physical G1 through an RL
tracker, with human recognition measured at up to 141 raters; its own direct retargets were
recognised better than its learned gaits, and it states that intensity is not perceptually
calibrated. PAMoR (arXiv 2608.28213) is a latent-diffusion generator with a valence-arousal
control whose definition is a stated convention, not a calibration. emotive-gait is the opposite
mechanism: an interpretable map from emotion to named gait parameters, with effects fitted from
*public* mocap, intervals, held-out-actor validation, no RL and no learned balance, released
open. What is new, if built and measured, is the open, data-derived emotion-to-parameter table
with intervals; nothing reviewed publishes one.

**What not to say** (from the related-work report, binding): not "first emotional humanoid
gait" or "first on G1"; no recognition claim and no comparison with EMoG's 61-65 % or PAMoR's
0.38 (different classes, chance levels, raters and renderings); not "physically validated"
(forward walking is kinematic-tier); not "data-derived" for anything still tuned, which after
this phase means the analytic model's gains and the disgust and surprise placements; no
generalisation beyond the fitted actors.

**README corrections (three; the third is my call):**

1. PAMoR's bullet: replace "the closest in approach" with a sentence saying it is a learned
   latent-diffusion generator with a continuous valence-arousal control, close in the *knob*
   and opposite in *mechanism*.
2. Remove the "Walk Like This" bullet: two researchers could not find a paper by that title.
3. EMoG's bullet: the related-work report reads the robot as a Unitree G1 from the paper's full
   text, where the parent design had excluded an earlier, mistraced claim. **The builder opens
   the arXiv HTML and quotes the sentence** before changing "which robot it used is unverified";
   if the sentence is not there, the bullet stays as it is.

Additions the report suggests (SONIC, SMooDi, Bionic style transfer, Emilya, Animated Cassie) are
the builder's call and droppable.

---

## 9. Left to the builder

- Event-detection thresholds (stance speed fraction, height band, yaw-rate turning threshold),
  smoothing windows and the detrending method for bounce and sway; all set against the synthetic
  round trip, then frozen.
- The BVH and ASF/AMC forward-kinematics conventions (Euler orders, the ASF axis block, the AMC
  root channels) and the exact 100STYLE and CMU unit scales; verified by a reader test that
  reconstructs a known bone length.
- The PTD column order and whether BML has head and neck points (decides the CSM fallback).
- Which CMU trial numbers constitute each subject's neutral and emotional walks, whether
  subjects 91 and 105 are one performer (section 4.1), and which 100STYLE frame cuts to apply;
  all recorded in the manifest.
- `MixedLM` fitting details: optimiser, convergence criteria, how the fallback is detected.
- Report layout and plots (local only; the committed report is tables and text).
- The artefact schema's exact field names, provided the section 4.3 and 4.6 contents are all
  present.

---

## 10. Order of work and drop order

**Order.** (1) Readers and `SkeletonClip`, with reader tests on bone lengths. (2) Events,
extractor, normalisation, and the synthetic round-trip test; the thresholds are frozen here.
(3) Manifest, landmark maps and label mapping for the three datasets; `fetch`. (4) `learn/`:
table building, `MixedLM` fit, held-out validation, hand-tuned comparison, structure test,
synthetic recovery test (known effects plus random actor spread, recovered within interval).
(5) `AnalyticEmotionModel` refactor (bit-identical), `LearnedEmotionModel`, `leg_length`, the
parametrised direction tests, learned presets in the feasibility test. (6) CLI (`fit`, `fetch`,
`--model`, `evidence --model`). (7) Run the fit on local data; commit the artefact and report;
BML in or out per section 4.1. (8) README, `CLAUDE.md`, docs, version 0.3.0.

**Drop order if the schedule slips** (first dropped first): `fetch`; the CSM reader; the
speed-adjusted secondary fit; the `S1 + fear offsets` row; the `ratio` scaling; the structure
test as a whole. **Never dropped:** the extractor with its round-trip test, the mixed fit with
held-out validation and the hand-tuned comparison, `LearnedEmotionModel` under the direction
tests, the manifest and provenance, the README corrections.

---

## 11. What changes against the parent design

- Parent section 2, non-goal "Machine learning: no classifier, no training, no datasets
  acquired": withdrawn by the user (brainstorm record). The RL and learned-balance exclusions
  stand and are restated in section 2.
- Parent section 4.4, "no parameter may claim a magnitude in degrees from a citation": kept for
  `evidence.py`. Fitted magnitudes are a different object (a measurement with dataset, N and
  interval, in radians and normalised units), live only in the generated artefact, and never
  enter a claim.
- Parent section 7 (related work): superseded by section 8 and the related-work report.
- Parent Appendix A.6 (why three axes): now a hypothesis under test (section 4.3), not only a
  design rationale.
- The Phase 1 plan's seam note (`docs/future-learned-model.md`): realised as section 4.5; its
  retraining-a-classifier plan is not followed.

---

## 12. Risks

1. **The PACO lab does not answer.** Mitigated by the licence-tier switch and the decision rule
   in 4.1: the phase ships on 100STYLE + CMU with anger `thin`.
2. **Event detection on 15-point, 60 Hz position data** (BML) may be noisier than on BVH; the
   round-trip test uses clean data. Mitigation: a second synthetic test adds Gaussian position
   noise and 60 Hz resampling and requires the same tolerances within a stated margin.
3. **A fitted magnitude is infeasible on the G1.** Handled in 4.5: the artefact is not edited;
   the user decides between flag-and-render and a labelled clip.
4. **`MixedLM` convergence** on parameters with tiny effects and few actors. The fallback is
   defined and recorded; no tuning for convergence.
5. **CMU trials turn out not to be walks with usable neutrals** for some subjects. Fear could
   drop below three actors and become analytic-delegated; the design already handles it, and
   the README says so.
6. **Contradictions with the literature.** Likely for at least one posture cue on at least one
   emotion. This is a finding, handled in 4.5; the risk is only to anyone hoping for agreement.
7. **CI time.** `statsmodels` and `pandas` installation plus the synthetic fits add about a
   minute per job; the round trip reuses existing trajectories. If the matrix slows past
   tolerance, the synthetic fit test is marked slow and run on one job.
