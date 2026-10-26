# Phase 4.6 — hand-off brief for a fresh session (2026-10-26)

Paste this as the first message of a new Claude Code session in `C:\emotive-gait`.

## Background

emotive-gait is an MIT-licensed Python package: emotion-modulated humanoid gait for the Unitree
G1 in MuJoCo. Six emotion intensities map onto a 15-field `GaitParameters` dataclass through one
of two emotion models behind the `EmotionModel` protocol in `src/emotive_gait/emotion/model.py`:
the analytic model (cited directions, hand-tuned magnitudes; the default) and, since Phase 4.5,
`LearnedEmotionModel` (`emotion/learned.py`), whose per-emotion, per-parameter effects were
fitted from public motion capture and live in the generated artefact
`src/emotive_gait/emotion/data/learned_coefficients.yaml` (written only by `emotive-gait fit`).

Phase 4.5 is COMPLETE on branch `phase-4-5-learned-emotion-model` (version 0.3.0; the
whole-branch review's findings were applied in a final fix commit; the user decides whether it
merges into master — never make that decision). What it
built, and what Phase 4.6 inherits:

- `src/emotive_gait/mocap/`: rig-free readers (BVH, ASF/AMC, PTD) into `SkeletonClip`; a
  `Landmark` enum and `LandmarkMap` (one per dataset in `mocap/landmarks.py`); a path frame and
  position-extremum foot events (`events.py`, thresholds frozen in its header); the per-cycle
  extractor `extract.py` (`summarise(clip, landmarks) -> ClipSummary`) measuring the 14
  measurable `GaitParameters` fields; Hof normalisation (`normalise.py`) and re-dimensioning
  (`gait/scaling.py`, `to_gait_parameters`, froude and ratio scalings) through
  `RobotProfile.leg_length` (G1 0.69 m). The G1 round trip (`tests/mocap/test_roundtrip.py`)
  recovers the generator's timing, length and amplitude parameters within the spec's tolerances;
  double support is checked against the same extremum rule on the commanded foot path (a +0.21
  G1 rule-vs-parameter offset), and the elbow and trunk/neck posture deltas are strict xfails
  (the two rig constants below).
- `src/emotive_gait/learn/`: hashed data manifest and fetch, per-cycle table, mixed-effects fit
  per parameter with leave-one-actor-out validation against the hand-tuned gains, the three-axis
  structure test, artefact writer, fit report, CLI (`fetch`, `fit`, `fit --check`,
  `demo --model learned`, `evidence --model learned`).
- Committed results (`docs/reports/phase-4-5-fit-report.md`; README "Two emotion models"):
  100STYLE + CMU, 4 actors, held-out error E_fit 1.19 against E_hand 1.70 and E_zero 1.41
  (froude); happiness and sadness fitted, anger and fear thin (one actor each). The Body Movement
  Library (30 actors) is fetched and used locally only, `committable: false` in
  `data/manifest.yaml` until the Glasgow PACO lab's written permission arrives (requested
  2026-10-25).

## Findings Phase 4.6 must know (all documented in CLAUDE.md "Facts that bite")

- Two G1 rig constants found by the round trip and left as strict xfails: the generator writes
  `elbow_flexion` to a G1 elbow joint whose positive direction straightens the arm (geometric
  bend = 1.385 - joint), so anger renders straighter elbows; and the pelvis-to-neck vector
  pitches 0.88 of the commanded waist pitch. A fix is a Phase 4 rendering change (user decision).
- Learned presets are flagged and rendered, not gated (user decision 2026-10-26): on the G1 the
  learned angry and happy presets exceed the 2 mm IK threshold (6.4 mm and 5.6 mm, froude). A
  per-profile playable range in the re-dimensioning was the rejected alternative.
- Foot events use the position-extremum rule (Zeni 2008); the velocity rule in the spec could
  not measure human double support. The G1's double support reads +0.21 above its parameter
  under that rule (rest-start, overshooting swing).
- The 100STYLE actor walks back and forth in a small room and BML walks a triangle: cycles whose
  heading changes by more than 0.35 rad are discarded; some tight-triangle BML clips yield no
  cycles (a follow-up for the turn rule).

## Task: Phase 4.6 "emotion for any rig"

Per the brainstorm record (`docs/superpowers/research/2026-10-25-phase-4-5-brainstorm-record.md`,
decision B): BVH in, the same extractor, emotion deltas from either model, a MuJoCo kinematic
chain compiled from the BVH skeleton, IK re-synthesis with exact foot placement, BVH out; a
free-text emotion prompt mapped to the six intensities; bone-name to `JointRole` mapping (LLM or
heuristic) producing a `LandmarkMap` and a `RobotProfile`-like rig profile with its own
`leg_length`; exposed as an MCP server / integration so an AI assistant or a DCC tool can drive
it. Measurable claim: round-trip fidelity and zero foot slip on rigs not designed for;
recognition via the Phase 5 harness later.

Process (project rule): research-first brainstorm if the record leaves questions, then a written
design (`docs/superpowers/specs/2026-10-26-phase-4-6-...-design.md`), the superpowers:brainstorming
builder check, user approval, superpowers:writing-plans, user approval of the plan, then
subagent-driven execution in another new session. Rules that stand: no RL, no learned balance;
emotion applies to parameters, never joint angles; no magnitude in degrees from a citation;
disgust and surprise stay unsourced; the artefact is generated, never hand-edited; aggregates
only enter git; no Claude attribution trailer in commits; merging and pushing are the user's;
work in place on a branch from the Phase 4.5 head (the `.venv` editable install resolves to this
checkout); `mujoco>=3.17,<3.18` stays pinned.

Read first: `CLAUDE.md`; the brainstorm record; the Phase 4.5 design
(`docs/superpowers/specs/2026-10-25-phase-4-5-learned-emotion-model-design.md`, sections 4.2 and
4.4 define the extractor interface Phase 4.6 reuses); `src/emotive_gait/mocap/skeleton.py`,
`mocap/extract.py`, `gait/scaling.py`, `emotion/model.py`, `emotion/learned.py`,
`robot/roles.py`; the fit report's limitations.

## Follow-ups left by Phase 4.5 (list, do not do unasked)

- BML inclusion on permission: set `committable: true` with the permission text in
  `data/manifest.yaml`, re-run `emotive-gait fit`, commit the artefact and report (anger and fear
  then become fitted and the maximal random structure becomes estimable; no BML-derived number
  may be written anywhere in the repository before that).
- The G1 elbow sign and trunk 0.88 gain (Phase 4 rendering fix, then drop the xfails).
- The turn rule on tight triangles (24 BML clips yield no cycles); a CSM reader is NOT needed
  (BML PTD has head and neck points).
- The 0.1.0 / 0.2.0 version drift fixed in passing (now pinned by `tests/test_packaging.py`).
- Deferred minors the whole-branch review judged "can wait" (none changes a committed number):
  readers: BVH/ASF rotation-order tests beyond single channels, `read_text()` without encoding,
  AMC fps hard-coded, `SkeletonClip` shape validation; events/extractor: NaN pelvis window,
  clip-wide `mean_speed`, standing period reads as swing, unpaired-extremum discard reason,
  turn-edge bias, half-peak-to-peak noise bias (spec-mandated); manifest/fetch: unknown keys
  ignored (per-file hashes still guard every clip), archives hashed twice, `.part` leftovers;
  fit: LOO standardiser pools over every emotion, per-cycle speed covariate, `actor_deltas`
  recomputed per refit; structure test: per-emotion vs row weighting, `sd` from the whole
  sample; learned model: `default_artefact_path` not zip-safe; pipeline: the self-check runs ten
  synthetic fits per `fit`, `--scaling` ignored under `--model analytic`, `cli.py` length.
- Residuals of the final fix wave's re-review (cosmetic, parked): the `_rejection` docstring in
  `learn/fit.py` has broken indentation and `_mixed_fit`'s docstring has a ~150-character line;
  one CLAUDE.md bullet (`gait/scaling.py`) runs past 100 columns; the README's "pooled
  between-actor SD of 0.23" beside the happiness elbow finding is the random-intercept (baseline)
  SD, not the slope spread, so call it the baseline SD or drop it; the per-actor CMU elbow deltas
  quoted in the README (+0.61, +0.02) are not individually traceable to the report, which shows
  only the CMU mean 0.32.
