# Phase 4.7: rig reader fidelity

Date 2026-10-26. Follows Phase 4.6 (emotion for any rig, PR #4, master 087d2ef, version 0.4.0).
Closes the three findings the Phase 4.6 validation report names as unresolved, and the report's
own commit line. Approved scope: the user chose "the first one" (fix the reading, not warn and
let the user correct) and set a standing rule: when the correct approach costs more than a partial
one, take the correct one.

## 1. Intent and why

`emotive-gait rig walk` writes a walk on the rig's own channels. A tool that writes a walk must
write it at the right size. Today two of the four real validation rigs are read at the wrong
size, and the report carries "(check)" tables at hand-given scales to show what the right size
would have given:

- **Bandai Namco dataset-2.** Every joint has six channels and the bones below the pelvis carry
  constant position channels that repeat their OFFSETs; the Hips (the pelvis, a child of
  `joint_Root`, whose six channels are all identically zero over the clip) carry the actor's
  travel in their position channels with an OFFSET of about the rest pelvis height. The reader adds a position channel to the OFFSET, so every
  bone is twice its length and the pelvis sits at 1.71 m (leg length 1.848 m). `--scale 0.005`
  restores bone length but halves the root's travel (the "bones at true size, root travel at
  half" check table).
- **CMU 07_01 (cgspeed MotionBuilder conversion).** The file keeps the CMU ASF length unit,
  1/0.45 inch = 0.0564444 m, which the units inference does not know; the nearest known scale
  (inches) reads the rig at 0.45 of its size (leg 0.387 m). The check table re-runs it at
  `--scale 0.0564444`.
- **Bones with two rotation channels.** The reference pose bakes a bone's chordal-mean rotation
  into its body quaternion. The chordal mean of rotations that two channels produced is in
  general not a rotation two channels can produce, so `channel_angles` drops the third angle
  silently: the compiled chain and the written BVH then disagree for such a bone. No rig in hand
  has one, the synthetic family does not cover it, and nothing warns.

And the generated report's header says "generated on uncommitted changes on top of `6dfc461`;
the tree measured is that of the commit that adds this report", a claim about a future commit
that nothing can check.

## 2. Goals, non-goals, anti-goals

Goals:

1. The BVH reader positions a joint by its position channels when it has them (the OFFSET is the
   rest value), so Bandai-style exports are read at true size with true travel, with no flag.
2. The CMU ASF unit is an inference candidate, so the cgspeed file is read at its true size with
   no flag, and `inspect` names a runner-up scale when the choice is ambiguous.
3. A bone with two rotation channels gets a reference rotation its channels can write back
   exactly, the dropped component is recorded, and the synthetic family covers such a bone so the
   FK gate and the round trip guard it in CI.
4. The report header states the commit hash and the actual uncommitted tracked files at
   generation, nothing more.
5. The validation report is regenerated with the two check tables gone and the Bandai and CMU
   headline rows at true size; README and CLAUDE.md say what the reader now does; version 0.4.1.

Non-goals: varying translation channels below the pelvis (still refused; sliding joints are not
modelled); representing a bone's missing rotation axis (the fit projects onto what the channels
have; the residual is reported); any change to inference's leg window or target; the deferred
minors of the Phase 4.6 note not named here; Phase 5 (perception).

Anti-goals (a failure even if the tests pass): a change in what the committed fit reads
(100STYLE BVH through `read_bvh`; `emotive-gait fit --check` must pass unchanged); a change to
the G1 path or its two xfails; a change to the extractor's thresholds or definitions; a new
flag as the way to get the right size; a report that still asserts anything about the tree it
cannot measure.

## 3. Constraints

- CLAUDE.md's dependency rules hold: `mocap/` imports NumPy, SciPy and `gait.params` only; `rig/`
  may import everything except `learn/` and `sim/`; only `cli.py` imports `rig/`.
- `data/manifest.yaml` is frozen (the committed artefact pins its SHA-256). The two validation
  datasets stay in `data/rig-manifest.yaml`, `committable: false`; Bandai and the cgspeed
  conversion never enter git in any form. Aggregates only.
- `docs/reports/phase-4-6-rig-validation.md` is regenerated from its `.yaml` spec with
  `python -m emotive_gait.rig.validate docs/reports/phase-4-6-rig-validation.yaml --out
  docs/reports/phase-4-6-rig-validation.md`, never hand-edited. The spec's prose is edited.
- The Phase 4.6 design document is not edited. This document supersedes its sentence "the
  reader adds a position channel to the OFFSET" and its "Facts that bite" entries on Bandai and
  CMU units in CLAUDE.md, which are rewritten.
- `ruff check . && ruff format --check . && pytest -q` green; the G1 preset feasibility test, the
  Z-rail test, the artefact test and the round-trip xfails untouched.
- Process: subagent-driven, one implementer and one reviewer per task, a whole-branch review at
  the end; design and plan approved by the user before the build; merge and push are the
  user's call.

## 4. Decisions

### 4.1 Position channels replace the OFFSET (`mocap/readers/bvh.py`)

The convention, for EVERY joint including the root, axis by axis: an axis with no position
channel keeps its OFFSET component; an axis whose position column is identically zero over the
clip keeps its OFFSET component; otherwise the column is that axis of the translation and the
OFFSET component is ignored. This is the reading of Blender's importer and of MotionBuilder's
exporter, which write child position channels equal to the OFFSET at rest and the root's as its
absolute position. The zero-column rule is per axis, decided here (the reviewer check showed it
is not a rare case: the synthetic family's root carries its travel in X and zeros in Y and Z
with its height in the OFFSET, and a per-joint rule would drop every synthetic rig to the floor;
per axis is never worse on a real file, since no exporter means "height zero" by a zero column
under a non-zero OFFSET). It is the only concession to the additive convention, and the module
docstring says so.

Where it applies: `BvhFile._kinematics` (so `to_clip`, `world_rotations` and the mocap
`read_bvh` all follow). For a root with OFFSET zero the two conventions agree, so 100STYLE
(fixture and full clip) and the CMU conversion read bit-identically and the committed fit is
untouched. The hand-written fixture `tests/mocap/fixtures/two_bone.bvh` (ROOT OFFSET `0 100 0`,
Yposition 10 in its third frame) is rewritten under the new convention (ROOT OFFSET `0 0 0`,
Yposition `100 100 110`), so every assertion on it in `tests/mocap/test_readers.py` and
`tests/mocap/test_bvh_file.py` keeps its numbers. `tests/mocap/test_bvh_file.py` gains a test
per branch (duplicate constant column: bone at OFFSET length; constant column different from the
OFFSET: bone at the column; varying child column: FK follows it frame by frame; zero column under
a non-zero OFFSET: the OFFSET stands; a root with a non-zero OFFSET and absolute channels: the
channels win). The inverse test (write, read back) stays bit-identical.

Identically zero: every value within `1e-9` file units of zero, the same tolerance `compile.py`
uses for "constant".

The writer changes by one line. `rig/writer.py` derives the root's position channels as
`targets - pelvis_held`, where `pelvis_held` is the forward kinematics of a held row whose root
position channels are zero; under the new convention a zero column leaves the root OFFSET in
place, so the written channels must carry it: root channels `= targets - pelvis_held + root
OFFSET` (file axes and units). On every real rig in hand the root OFFSET is zero and the output
is bit-identical; on the synthetic family (root OFFSET at leg height) the round trip would
otherwise read the pelvis a leg length low. The docstring's bullet on the root is updated.

### 4.2 The compile step takes the column (`rig/compile.py`)

`bone_offset` becomes the bone VECTOR under 4.1: OFFSET when the position columns are absent or
all zero; otherwise each present column must be constant (within `CONSTANT_TRANSLATION_TOL`, else
`RigError` as today: only the pelvis may translate) and its value replaces that axis of the
OFFSET. The docstring and the module header's bullet are rewritten. Non-root position channels
keep being written at the source's first-frame values (the constants), which is right under both
conventions.

`tests/rig/rigs.py::fold_into_channels` encodes the additive convention and is replaced by two
helpers: `duplicate_offsets(file, bone)` (position channels prepended, equal to the OFFSET: the
Bandai case; reader positions unchanged) and `move_offset_into_channels(file, bone, vector)`
(position channels set to `vector`, OFFSET left as the stale rest value; the reader positions the
bone at `vector`). The FK gate (1e-6 m against the reader's forward kinematics) runs on both in
`test_compile.py`, and the synthesis round trip keeps its folded-rig case on `duplicate_offsets`.
`RigSpec` gains `root_height_in_channels: bool = False`: when true, `build_rig` and `write_walk`
put the root's height into its Yposition (Zposition on a z-up spec) column with the OFFSET left
at the same value (the Blender export shape); the variant joins `FAMILY_VARIANTS`, so the round
trip exercises the writer's root rule of 4.1.

### 4.3 The CMU ASF unit is an inference candidate (`rig/mapping.py`)

`SCALES = (1.0, 0.01, 0.0254, 0.0564444, 0.1, 0.001)`; the rule (nearest to `LEG_TARGET_M`,
inside `LEG_WINDOW_M`) is unchanged. On the cgspeed file the hip-to-ankle REST distance (from
the OFFSETs, which is what `_rest_positions` measures) is 14.33 units: 0.809 m at the CMU unit
against 0.364 m at inches, so the rule picks the CMU unit (the report's 15.2-unit, 0.861 m figure
is the clip-based leg length, a different measure). The other rigs' picks are unchanged and have
no runner-up inside the window (Bandai 80.07 units at 0.01, 100STYLE 86.96 units at 0.01).

Ambiguity is named, not hidden: when the runner-up scale also lands the leg inside the window,
the `scale` Decision's note appends "runner-up 0.0254 (0.387 m); pass --scale to override"
(both scales and both metre values). `inspect` already prints every decision, so the user sees
it. The constant's comment names the unit: "CMU ASF length unit, 1/0.45 inch (the cgspeed BVH
conversion keeps it)". `tests/rig/test_mapping.py` gains the pick on a 15.2-unit leg and the
runner-up note on a leg that fits two scales.

### 4.4 Two-channel bones get a representable reference (`rig/calibration.py`, `rig/compile.py`)

After the reference rotation of every bone is formed by whatever rule applies to it (plain
chordal mean; the pelvis's heading-removed world mean; the ankles' stance world orientation), a
bone with exactly two rotation channels has its reference replaced by the nearest rotation its
channels can express: minimise the Frobenius distance between `R_c1(a) @ R_c2(b)` and the mean
over `(a, b)`, started from the two angles `channel_angles` returns today (the three-axis
decomposition with the missing axis appended, third angle dropped). `scipy.optimize.least_squares`
on the nine matrix residuals; two parameters, a handful of iterations. The residual is the
rotation angle of `mean^T @ fitted`.

Recording: a Decision `reference:<bone>` of source `geometry`, chosen `fitted`, note
"two rotation channels (`Zrotation Xrotation`): reference fitted, residual 0.0123 rad", for every
such bone; additionally a warning when the residual exceeds `TWO_CHANNEL_WARN_RAD = 0.01`
naming the bone and the residual. Plumbing: `ReferencePose` gains `decisions: tuple[Decision,
...] = ()` and `warnings: tuple[str, ...] = ()` (defaults, so every existing construction stands);
`reference_pose` fills them; `load_rig` merges them into `RigMapping.decisions` and
`RigMapping.warnings` next to the unit decisions it already prepends, so `inspect`, `walk`,
`validate` and the MCP `inspect_rig` show them through the paths that print mapping decisions
and warnings today. `calibration.py` imports `Decision` from `rig/mapping.py` (it already
imports from it). Bones with one rotation channel need nothing: the chordal
mean of rotations about one axis is about that axis, and `channel_angles` is exact for one axis
(a comment says so). Bones with no rotation channel are unchanged.

Why the fit and not a warning alone: the compiled chain bakes the reference into the body
quaternion and the writer decomposes the final local rotation into the bone's channels. With a
representable reference, `R_c1(a + q1) @ R_c2(b + q2)` is what the two hinges produce and what
the two channels write, so the written BVH reproduces the trajectory exactly; with an
unrepresentable reference nothing downstream can be exact.

Coverage: `tests/rig/rigs.py` gains a `RigSpec` variant in which the knees and elbows carry two
rotation channels (the two axes of the three-channel order that are not the bone's own long
axis, in the spec's order) and the walker writes their motion through `channel_angles` with the
long-axis component absent, so the reader and the walker agree. The variant joins `FAMILIES`,
so the FK gate, the reference-pose test and the round-trip test run on it; one test makes a
two-channel bone's chordal mean unrepresentable (a shoulder given two channels and a swing
about the missing axis) and asserts the `reference:<bone>` decision, the warning above the
threshold, and that writing and re-reading the reference pose gives the fitted rotation to
1e-9.

### 4.5 The report's commit line (`rig/validate.py`)

`_commit()` returns `git rev-parse --short HEAD` and the list of modified tracked files from
`git status --porcelain --untracked-files=no`, both from the repository root. `_commit_line`
writes "Commit: `<hash>`." when the list is empty or holds only the report being written, and
otherwise "Commit: `<hash>`; uncommitted changes in N tracked files at generation: a, b, c." No
sentence about a future commit. `render_report(commit=...)` keeps accepting a string for tests.
`git describe` is not used (the repository has no tags; `--always` gave a bare hash anyway).

### 4.6 The report spec and the docs

`docs/reports/phase-4-6-rig-validation.yaml`: the Bandai and CMU check rows are removed (the
headline rows now read both at true size); the `Bandai units` and `CMU units` findings are
rewritten to say what the reader and the inference do now and what the earlier reading was (one
sentence each, with the old leg lengths as the record); the command-lines block loses the two
`--scale` lines. Every number in the regenerated report comes from the run; prose quotes none
that could change, except in a sentence that says it is the earlier reading.

README: the "Emotion for any rig" section's claim table rows for CMU and Bandai update to the
regenerated numbers; limitation 8 (slip) names whatever feasible runs still exceed 2 mm after
regeneration (the CMU neutral at the true unit measured 0.3 mm in the check table, so it is
expected to drop out and only the 100STYLE full clip's 2.9 mm to remain, but the run decides);
the two bullets on CMU inches and Bandai doubling are replaced by one bullet per convention
(position channels replace the OFFSET; the known scales, now six). CLAUDE.md: the same two
"Facts that bite" entries rewritten, the two-channel rule added, the commit-line rule added
under the report's regeneration fact. The comment block of `data/rig-manifest.yaml` (its hash is
pinned by nothing) is rewritten where it states the old reading: the CMU entry says the inference
now knows the CMU unit, the Bandai entry says position channels replace the OFFSET and the clip
reads at true size; the dataset entries themselves do not change. `pyproject.toml` version 0.4.1
(a reader behaviour change that alters no committed number; the committed fit is bit-identical).

## 5. Layering

No new module and no new dependency. `mocap/readers/bvh.py` keeps importing only NumPy and
`mocap.skeleton`. `rig/calibration.py` gains a `scipy.optimize` import (SciPy is already a core
dependency through `mocap/`). `rig/validate.py` already imports `subprocess`. The G1 path does
not touch any changed function.

## 6. What does not change

`evidence.py`, `axes.py`, the learned artefact, the fit pipeline and its manifest, the
extractor's thresholds and definitions, `footsteps.py`, the IK, the G1 profile and its two
xfails, the physics and kinematic tiers, the writer apart from the root rule of 4.1, the
lexicon, the MCP server's tools and schema, the inference's leg target and window,
`LEG_WINDOW_M`, `axis_purity`, every existing test's assertion except the ones 4.2 names (the
`two_bone.bvh` fixture is rewritten so its assertions keep their numbers).

## 7. Left to the builder

- The initial guess and tolerances of the two-channel fit; the residual must be below 1e-9 rad
  on a representable mean (the test). Fit before the ankle stance rotations are composed into
  `reference_world`, so the ankles' baked rotations are the fitted ones.
- Which two axes the two-channel synthetic variant drops and how the walker generates a motion
  with no component about the missing axis.
- The exact wording of the rewritten findings, limitation bullets, CLAUDE.md facts and manifest
  comments.
- Threading the report's output path into `_commit()` so "only the report is modified" can be
  recognised; regenerating the report after the code and spec are committed, so its line reads
  "Commit: `<hash>`." with no file list.
- Extending the explicit parametrize lists in `tests/rig/test_compile.py` (the FK gate and the
  reference-pose test do not iterate `FAMILY_VARIANTS`) to the two new variants.
- The validation data: this worktree's `data/raw` is a directory junction to the main checkout's
  `C:\emotive-gait\data\raw` (git-ignored), holding all five datasets; never a commit. If a clip
  is missing, `emotive-gait fetch --manifest data/rig-manifest.yaml --datasets cmu_bvh bandai2`
  with the recorded sources.

## 8. Order of work

1. Reader convention (4.1) with its tests and the bit-identity check (`fit --check`, the fixture
   read). 2. Compile bone vector and the two test helpers (4.2), FK gate. 3. Inference candidate
and the runner-up note (4.3). 4. Two-channel synthetic variant, the fit, the decision and
warning (4.4). 5. Commit line (4.5). 6. Report spec, regeneration on the four real rigs, README,
CLAUDE.md, version (4.6). 7. Whole-branch review.

## 9. Risks

- **The FK gate on Bandai under the new convention.** The compile folds and the reader must
  agree on the Hips (varying, the pelvis, parent `joint_Root` uncompiled) and on the duplicated
  children. The gate decides; if Bandai fails for a reason outside this scope the report says
  "not run" with the measured reason, as Phase 4.6 provided.
- **A rig with additive child deltas.** Would now be read wrongly (positioned by the deltas).
  No such file is in hand or known from the common exporters; the docstring states the
  convention and the all-zero concession so a later reader change is a one-line decision.
- **Scale ambiguity grows by one candidate.** Inches and the CMU unit differ by 2.2x (the
  existing 0.01 and 0.0254 differ by 2.5x); a true-inch rig with a 0.5 m leg would be read at the
  CMU unit. The runner-up note makes that visible on `inspect`.
- **The two-channel fit on a far-from-representable mean.** Least squares on two angles can
  find a local minimum; the warning names the residual, and the decision records the fit, so a
  bad rig is visible rather than silent. The synthetic test uses a moderate (0.3 rad) off-axis
  component.
- **Regenerated numbers moving elsewhere.** 100STYLE rows must stay identical (no position
  channels below the root); the implementer diffs the regenerated report against the committed
  one and the review confirms only the Bandai and CMU rows changed.
