# Phase 4.5 — brainstorm record (2026-10-25)

Research-first brainstorm for the "learned emotion model" phase inserted between Phase 4 (forward
locomotion, branch `phase-4-forward-locomotion`) and Phase 5 (evaluation harness). The three
cited research reports beside this file are the evidence. The design document is still to be
written; this record carries the decisions already made so a fresh session can write it.

## Scope change (user)

The professionalization design's non-goal "Machine learning: no classifier, no training, no
datasets acquired" is withdrawn. Constraints that stand: no reinforcement learning and no learned
balance control (the "stable but emotionally flat" anti-goal); the `evidence` module and the
cited-direction tests stay as sanity checks on anything learned; any magnitude presented as
data-derived must actually come from data; the result must be a measurable claim, not only a
capability. The user wants the project to show BOTH "ML core / non-LLM AI" skills and
"incorporate AI / MCP / connections with the software" skills, for a software/AI engineer's
portfolio.

## Findings that bind the design

- **Data.** The only downloadable multi-actor emotional walking set is the Glasgow Body Movement
  Library (30 actors; angry/happy/sad/neutral; 15-joint 3-D positions at 60 Hz; no licence
  stated, so the lab must be asked before derived numbers enter the MIT repo; aggregates only).
  100STYLE (CC BY 4.0) adds Angry/Depressed/Elated/Proud walks from one actor. CMU adds single
  trials (free for all uses). Roether, Emilya, Barliya, EMoG data: no public route. No usable
  walking data for disgust or surprise; fear is single trials only.
- **Related work.** EMoG (arXiv 2609.14432, Sep 2026) generates emotional periodic gaits with an
  MLP on a physical G1 via an RL tracker, rated by up to 141 people (61-65 % six-way); its direct
  retargets were recognised better than its learned gaits; intensity not perceptually calibrated.
  PAMoR (arXiv 2608.28213) is a latent-diffusion generator with a valence-arousal knob, NOT an
  analytic parameter model: the README's "closest in approach" sentence must be reworded, and the
  README's "Walk Like This" citation could not be found by two researchers (fix both).
- **Approach ranking.** (1) mixed-effects model, emotion -> leg-length-normalised gait
  parameters, held-out-actor validation, CIs; (2) test of the three-axis structure on the same
  fit; (3) Phase-5 perception regression later; (4) classifier as a CI check only; (5)
  diffusion/VAE generation rejected (no data, no citable magnitudes, re-implements EMoG).

## Decision (user, after the brainstorm)

Build both, in this order:

- **A — data-derived emotion model (Phase 4.5).** Parameter extractor from mocap (designed for
  arbitrary skeletons from day one), per-actor deltas from that actor's neutral, leg-length /
  Froude normalisation, mixed-effects fit behind the documented seam in `emotion/model.py` with
  confidence intervals, the three-axis structure test, held-out-actor error vs the hand-tuned
  gains as the headline claim. Cited-direction tests remain as guards. Disgust and surprise stay
  unsourced.
- **B — "emotion for any rig" (Phase 4.6).** BVH in, the same extractor, emotion deltas, a
  MuJoCo kinematic chain compiled from the BVH skeleton, IK re-synthesis with exact foot
  placement, BVH out; a free-text emotion prompt mapped to the six intensities; bone-name to
  `JointRole` mapping (LLM or heuristic); exposed as an MCP server / integration so an AI
  assistant or a DCC tool (Blender, Mixamo workflow) can drive it. Measurable claim: round-trip
  fidelity and zero foot slip on rigs not designed for; recognition via the Phase 5 harness.

Process for each: written design, builder check, writing-plans, subagent-driven execution; the
user approves the design and the plan before any build. Each finished phase hands off to a new
session (project rule, 2026-10-25).

## Facts a Phase 4.5 builder inherits from Phase 4

Nominal walking height 0.68 m (`com_height_scale = 1.0`); `knee_flexion_bias` retired;
`expansiveness` is a multiplier on arm swing, abduction and step height, not an independent
quantity (fix it at 1.0 in any fitted model); `speed = step_length * cadence`; G1 leg length
about 0.69 m; sway is lost on every rail; forward walking is a kinematic-tier result (`XZ_RAIL`
is a measured negative result).
