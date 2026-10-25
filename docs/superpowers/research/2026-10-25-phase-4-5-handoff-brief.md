# Phase 4.5 — hand-off brief for a fresh session (2026-10-25)

Paste this as the first message of a new Claude Code session in `C:\emotive-gait`, or ask the
Phase 4 session to start it (it must be asked from the desktop app, not Remote Control).

## Background

emotive-gait is an MIT-licensed Python package: emotion-modulated humanoid gait for the Unitree
G1 in MuJoCo. Six emotion intensities map through three affective axes plus category offsets onto
a 15-field `GaitParameters` dataclass, rendered by a mink task-space IK generator in a kinematic
tier (forward walking) and a physics tier (in-place stepping on a rail). The emotion map is
analytic: directions cited (`src/emotive_gait/emotion/evidence.py`), magnitudes hand-tuned and
labelled as such. Phase 4 is COMPLETE on branch `phase-4-forward-locomotion` (259 tests pass,
whole-branch review clean); whether it merges into master is the user's open decision — never
make it. Phase 4.5 "learned emotion model" sits between Phase 4 and Phase 5; after a
research-first brainstorm the user decided to build (A) a data-derived emotion model first and
(B) an "emotion for any rig" artist tool with MCP integration afterwards.

Project rules: the user approves the design and the plan before any build; subagent-driven
execution with models per task complexity; no Claude attribution trailer in commits; merging and
pushing are the user's; no magnitude in degrees presented as literature-derived; disgust and
surprise stay unsourced; work in place on a branch (the `.venv` editable install resolves to this
checkout); each finished big task hands the next to a new session.

## Task

Write the Phase 4.5 design and get it approved, then plan it. Branch
`phase-4-5-learned-emotion-model` from the current head of `phase-4-forward-locomotion`.

Read first: `CLAUDE.md`; `docs/superpowers/research/2026-10-25-phase-4-5-brainstorm-record.md`
(binding decisions); the three research reports beside it; the professionalization design
sections 2, 4.4, Appendix A; `src/emotive_gait/emotion/{model,axes,evidence}.py`,
`src/emotive_gait/gait/params.py`.

Process: the brainstorm is done (the record is its outcome; do not re-ask the direction). Write
`docs/superpowers/specs/2026-10-25-phase-4-5-learned-emotion-model-design.md`, run the
superpowers:brainstorming builder check, resolve its questions (answer from the record, settle
minors, bring only user-only ones), hand the design to the user for approval; after approval run
superpowers:writing-plans for `docs/superpowers/plans/2026-10-25-phase-4-5-learned-emotion-model.md`
and ask the user to review it. STOP there: execution starts in another new session after plan
approval.

The design must decide: (1) data and licensing — Body Movement Library (30 actors; no licence
stated: the user emails the Glasgow PACO lab; aggregates only; raw data local and git-ignored),
100STYLE (CC BY 4.0, one actor), CMU single trials; which emotions each gives; what is missing
(fear thin; disgust/surprise none). (2) A gait-parameter extractor from skeleton trajectories,
built for arbitrary skeletons from day one (Phase 4.6 reuses it): gait events, cadence, step
length/height, double-support fraction, feet spacing, pelvis height/bounce/sway, trunk and neck
pitch, arm swing, abduction, elbow flexion, elbow swing ratio; per-actor deltas from that actor's
neutral; leg-length/Froude normalisation (approaches report section 3; G1 leg about 0.69 m);
`expansiveness` fixed at 1.0. (3) The model: mixed-effects linear model, emotion indicators to
normalised parameters, actor random effects, per-parameter CIs, held-out-actor validation; the
headline claim is held-out-actor prediction error vs the hand-tuned gains; the three-axis
structure test (constrained vs reduced-rank vs full per-emotion) with its low-power caveat.
(4) Plugging into the seam in `emotion/model.py`: a `LearnedEmotionModel` beside the analytic
one, selectable; cited-direction tests run against both; a fitted magnitude contradicting a cited
direction is reported, not silently kept; a stated rule for intensity scaling and mixing (not
identifiable from single-intensity data). (5) Committed vs local: coefficients with CIs,
provenance, extractor, tests in; raw mocap out; a `fit` command, fixed seeds, a data manifest
with hashes. (6) Positioning against EMoG and PAMoR per the related-work report; README fixes
(PAMoR is a latent-diffusion generator, not "closest in approach"; "Walk Like This" is
unverifiable). (7) Constraints: no RL, no learned balance, evidence tests stay, data-derived
means from data, measurable claim. (8) Dependencies (statsmodels or similar; small pure-Python
BVH/ASF/AMC/CSM/PTD readers; no heavy ML frameworks), CI impact (fits offline; tests on committed
coefficients plus a synthetic fixture), drop order. Phase 4.6 is out of scope except where the
extractor's interface must anticipate it.

Done: design committed and approved; plan committed and presented; a short message on what the
user must do next (approve the plan; send the PACO email; the Phase 4 merge is still theirs). Do
not build, install dependencies, or download datasets.
