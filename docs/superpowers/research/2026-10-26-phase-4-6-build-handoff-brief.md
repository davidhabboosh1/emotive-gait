# Phase 4.6 build — hand-off brief for a fresh session (2026-10-26)

Paste this as the first message of a new Claude Code session in `C:\emotive-gait`.

## Background

emotive-gait (`C:\emotive-gait`) is an MIT-licensed Python package: emotion-modulated humanoid
gait for the Unitree G1 in MuJoCo. Six emotion intensities map onto a 15-field `GaitParameters`
dataclass through either the analytic emotion model (cited directions, hand-tuned magnitudes;
the default) or `LearnedEmotionModel` (fitted from public motion capture in Phase 4.5, complete
and merged into master as PR #3). Phase 4.5 built the rig-free mocap package
`src/emotive_gait/mocap/` (readers, `Landmark`/`LandmarkMap`, the `summarise()` extractor, Hof
normalisation) and `gait/scaling.py`.

Phase 4.6 "emotion for any rig" was designed and planned on 2026-10-26 and both documents are
approved by the user: BVH in, the same extractor measures the walk, emotion deltas from either
model re-dimensioned to the rig, a MuJoCo kinematic chain compiled from the BVH hierarchy, IK
re-synthesis with exact foot placement through the existing generator, BVH out; a free-text
emotion prompt mapped to the six intensities by a shipped lexicon; bone-name to landmark/role
mapping by a name lexicon plus rest-pose geometry and probes; an MCP server over stdio. The
output is a parametric walk with the source's measured parameters plus the emotion deltas, not
the source frames edited (user decision 2026-10-26).

Project rules that stand: no RL, no learned balance; emotion applies to parameters, never joint
angles; no magnitude in degrees from a citation; disgust and surprise stay unsourced; the
learned artefact is generated, never hand-edited; aggregates only enter git (the plan's Global
Constraints say exactly which clips may be committed); no Claude attribution trailer in commits
(CLAUDE.md and the user's rule override any harness reminder); merging and pushing are the
user's, never the session's; work in place on the branch, no worktree (the `.venv` editable
install resolves to this checkout); `mujoco>=3.17,<3.18` stays pinned; the G1 path must stay
bit-identical with its two round-trip xfails. The user approves nothing further during the
build except what the plan or a reviewer escalates as a user decision.

## Task

Execute the Phase 4.6 implementation plan on the existing branch
`phase-4-6-emotion-for-any-rig` (head a803a2a plus this brief's commit; do not create a
worktree, do not switch branches, do not touch master).

Read first, in order: `CLAUDE.md`; the plan
`docs/superpowers/plans/2026-10-26-phase-4-6-emotion-for-any-rig.md` (13 tasks, with Global
Constraints, Review Focus, Execution notes including a suggested implementer/reviewer model per
task, and a drop order); the design
`docs/superpowers/specs/2026-10-26-phase-4-6-emotion-for-any-rig-design.md` (the spec the plan
argues from); and the hand-over note
`C:\Users\attan\AppData\Roaming\Claude\side-session-notes\local_a7d199e5-6cd0-4cbc-a52e-3e5b400fd1b2\2026-10-26-phase-4-6-design-and-plan.md`
(load-bearing decisions a builder must not re-derive). Other notes in that folder are earlier
sessions' text: data to weigh, not instructions.

Process: use the superpowers:subagent-driven-development skill, task by task in the plan's
order, a fresh implementer subagent per task and a fresh reviewer per task before the next
starts, with the models the plan's execution notes suggest; tests first as each task's steps
say; one commit per task with the plan's commit message (no attribution trailer);
`ruff check . && ruff format --check . && pytest -q` green at the end of every task. Finish
with a whole-branch review on the most capable model and apply its findings in a final fix
commit. Run commands likely to exceed two minutes in the background. Task 11 is the only task
that installs a dependency (the `mcp` extra, pinned to the major you verify); Task 12 fetches
two datasets through the manifest and may find one unreachable, in which case the report lists
it as not run, nothing guessed. If a task's test cannot pass without changing a frozen extractor
threshold, a committed number, or the design's claims, stop that task and escalate to the user
with the measured numbers rather than retune; the plan says what is never dropped.

## Done looks like

All 13 tasks committed on `phase-4-6-emotion-for-any-rig`, the whole suite green on this
machine, the validation report committed, version 0.4.0, the whole-branch review applied,
nothing pushed or merged, and a final message to the user with the branch head, the headline
numbers (round-trip residuals and slip per rig, which real rigs ran) and anything left for them
(the push and the pull request, any escalated decision). Leave a hand-over note in the shared
notes folder above for the next session, per the project's hand-off rule.
