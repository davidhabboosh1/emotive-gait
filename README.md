# emotive-gait

Emotion-modulated humanoid gait in MuJoCo. Six continuous emotion intensities (anger, disgust,
fear, happiness, sadness, surprise) are mapped through three affective axes onto the parameters
of a walking gait for the Unitree G1. Every claim the model relies on is either cited to a
source or explicitly flagged as unsourced.

| Sadness = 1.0 | Anger = 1.0 |
| --- | --- |
| ![sad gait](docs/media/sad.gif) | ![angry gait](docs/media/angry.gif) |

## Quickstart

```sh
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m mujoco_menagerie prefetch unitree_g1
emotive-gait demo --emotion sadness=0.8     # writes demo.mp4 (needs ffmpeg on PATH)
emotive-gait demo --emotion sadness=0.8 --model learned   # the fitted model, see below
emotive-gait fit                            # refit the learned model from data/raw (see below)
emotive-gait evidence                       # every claim with its grounding and citation
pytest
```

`demo` also takes `--physics`, `--root {free,z-rail,xz-rail,weld}` (`xz-rail` is experimental,
see Limitations), `--camera {tracking,fixed}` (the video camera follows the pelvis by default),
`--hold-speed`, `--gif`, `--dry-run`, `--model {analytic,learned}` and `--scaling {froude,ratio}`
(the learned model's re-dimensioning, `froude` by default); see `emotive-gait demo --help`.

With `--physics` the CLI prints a report after the run: `root`, `fell`, `zmp_outside_frac`,
`zmp_mean_margin_m`, `foot_slip_rms_m_s`, `pelvis_bob_m`, `pelvis_tracking_ptp_m`,
`pelvis_held` and a `played_params:` line listing every parameter the chosen root changed
before playing it. If the parameters are beyond what the robot's legs can reach (the IK
feasibility check), the CLI prints one `warning: parameters exceed the robot's reach` line and
still renders.

## Requirements

Python 3.11 or newer, 3.12 recommended. `mujoco-menagerie` downloads the robot model on first
use and extracts it with `tarfile.extractall(filter=...)` (PEP 706), which needs Python 3.11.4
or newer. If that step fails the loader raises an error naming the prefetch command above and
the sparse-checkout fallback. On Python below 3.11.4 the `prefetch` command itself fails with a
raw `TypeError` traceback (a `mujoco-menagerie` bug); run `emotive-gait demo` instead, which
reports the same failure as an actionable message. MuJoCo 3.17.x (pinned `>=3.17,<3.18`);
`mink` and `qpsolvers[daqp]` are installed with the package.

## Architecture

**`robot/`** maps semantic joint roles (hip pitch, elbow, neck pitch, ...) to actuator names, so
nothing above it hard-codes a model. The G1 has no neck, yet head inclination is the strongest
posture cue in the literature, so the loader injects a pitch neck joint and splits the head's
mass out of the torso.

**`gait/`** defines `GaitParameters`, which are the same quantities the literature reports
(cadence, step length, trunk and neck pitch, arm swing, ...). In-place projections (constrained roots) keep the full arm swing. Joint trajectories are periodic
C2 curves. `footsteps.py` computes foot and pelvis paths as pure functions of the parameters
and the gait phase; `ik.py` solves those foot targets for leg joint angles with `mink` on a
welded-root copy of the robot; the generator combines the IK legs with the spline-driven arms,
trunk and neck into per-role angles over a gait cycle.

**`emotion/`** has two models behind one protocol. The analytic model (the default) runs
evidence, then axes, then the model: the evidence module records signed directions (more, same,
less) with a citation and a grounding level for each claim, emotion intensities become
activation, valence and tension, and those shift the parameters. It encodes directions, not
magnitudes; the gains are tuned, not cited. The learned model reads per-emotion effects from a
committed, generated artefact (`emotion/data/learned_coefficients.yaml`) and delegates any cell
it lacks to the analytic model.

**`mocap/`** turns motion-capture clips into the same gait parameters the generator uses
(foot events, per-cycle measurements, Hof re-dimensioning); it needs only NumPy and SciPy.
**`learn/`** is the offline fit pipeline (manifest, fetch, mixed-effects fit, held-out validation,
artefact and report writers); it needs the `fit` extra and only `cli.py` imports it.

**`sim/`** has a kinematic tier that poses the robot with no physics and walks it forward with
exact foot placement (the source of stimuli for video), and a physical tier that steps position
servos open loop with the pelvis on a vertical rail. ZMP and center-of-pressure are measured
for instrumentation and never used for control.

## Two emotion models

`emotive-gait demo --model analytic` (the default) uses the hand-authored map described above.
`--model learned` uses per-emotion effects fitted to public motion capture of acted emotional
walking: a mixed-effects model with actors nested in datasets, re-dimensioned to the G1 by Froude
scaling (`--scaling froude`, default) or by a plain ratio (`--scaling ratio`). The fitted numbers
live in the committed artefact, written only by `emotive-gait fit` and never edited by hand;
`emotive-gait evidence --model learned` prints the claim table with the fitted estimate, its
interval and whether the data agree with the cited direction. The fit report is
[`docs/reports/phase-4-5-fit-report.md`](docs/reports/phase-4-5-fit-report.md).

**What the committed fit contains.** 100STYLE (1 actor, 4 clips, 63 cycles) and CMU (3 actors, 12
clips, 51 cycles): 4 actors, 114 cycles, 2 datasets. Happiness and sadness are fitted (3 actors,
with 95 % intervals). Anger and fear are thin (1 actor each): they carry an actor count instead
of an interval, enter no headline number, and the CLI warns. Disgust and surprise have no data
and stay with the analytic model, unsourced. Every happiness actor rests on a `mapped` label (3 of
3: 100STYLE Elated, CMU 82 "happy or fast walk forward", CMU 104 HappyGoLuckWalk), and 2 of the 3
sadness actors do (100STYLE Depressed, CMU 91 DepressedWalk beside its exact SadWalk); the report's
`n_mapped_actors` column counts them per cell.

**Held-out validation.** Leaving each actor out in turn (6 held-out actor-emotion pairs), the
standardised error of the fitted map (E_fit) against the hand-tuned gains (E_hand) and against
predicting no change (E_zero); 1.0 is the floor of predicting an unseen person from the mean.

| scaling | E_fit | E_hand | E_zero |
| --- | --- | --- | --- |
| froude | 1.189 | 1.702 | 1.413 |
| ratio | 1.175 | 1.756 | 1.407 |

The hand-tuned gains were tuned for a G1 rendering and never targeted human magnitudes, so
E_hand measures how far they sit from measured human effects, not how badly they were tuned.

**Where the data contradict a cited direction.** The data are not edited and `evidence.py` is not
edited. One contradiction has an interval that excludes zero: happiness `elbow_flexion`, +0.47 rad
[0.42, 0.52], against Roether et al. 2011 (both scalings). Four more sit on fitted cells whose
interval spans zero, so they are not findings: happiness `neck_pitch`, sadness `com_bounce_amp`,
sadness `elbow_flexion` and sadness `elbow_swing_ratio` (and, under `ratio`, sadness
`lateral_sway_amp`). Five lie on thin one-actor cells: anger `arm_swing_amp` (Roether 2011), fear
`cadence`, `elbow_swing_ratio` and `step_length` (Halovic and Kroos 2018) and fear `neck_pitch`
(Roether 2011).

Every committed cell is the random-intercept fallback (`maximal_raised`: with four actor groups
and a 5x5 random-effect covariance the maximal fit raises `LinAlgError`), so the Wald intervals
carry only cycle-level residual variance, not the between-actor variance of the emotion slopes,
and the fixed effects are cycle-weighted. The sign of the happiness `elbow_flexion` finding
survives this: the observed per-actor deltas are +0.56 rad (100STYLE), +0.61 (CMU 104) and +0.02
(CMU 82), against a pooled between-actor SD of 0.23.

On the G1 the learned posture effects pass through two rig constants. The elbow's geometric bend
is 1.385 rad minus the joint angle, so the +0.47 rad happiness `elbow_flexion` finding (more
flexed human elbows) renders as straighter G1 elbows; and the pelvis-to-neck vector pitches 0.88
of the commanded waist pitch, so learned trunk effects show at that gain and the head-on-trunk
angle inherits the difference.

**Three-axis structure test.** On held-out actors the analytic three-axis structure loses 0.072
against a full free-form structure (`froude`; in-sample R^2 0.047 against 0.750; held-out error
1.444 against 1.372). Caveat: five group means always lie in a four-dimensional affine space, so
the test is low-powered with four sourced emotions. Read it as "S1 loses 0.072 against S4 on
held-out actors", never as "the data discovered three axes".

**G1 feasibility is a documented limitation.** Learned presets are flagged and rendered, not
gated (user decision 2026-10-26); only the analytic presets must solve the leg IK. Under `froude`
the learned anger preset misses its foot target by 6.37 mm (driven by `step_length` 0.497 m and
`com_height_scale` 0.9795) and the learned happy preset by
5.62 mm (`step_length` 0.422 m and `com_bounce_amp` 0.054 m), against the 2 mm threshold. Under
`ratio` anger misses by 9.07 mm and happy by 141.6 mm. Neutral and sad are feasible. The CLI
warns once and still renders.

**Body Movement Library.** BML (30 actors) is downloaded and used locally, but its numbers are
not in the committed artefact: it states no licence, and written permission was requested from
the Glasgow PACO lab on 2026-10-25 and is pending. When it arrives, the manifest entry becomes
`committable: true` with the permission text, and `emotive-gait fit` is re-run.

**Limits that matter here.** The fit describes human effects of acted emotion, not how a G1
showing them is perceived. Intensity scaling and the mixing of emotions are assumptions, not
identified from these data. Fear and anger are thin and disgust and surprise are absent. The
structure test is a model comparison, not a discovery. Forward walking is a kinematic-tier
result, and nothing here generalises beyond the fitted actors. No recognition rate has been
measured.

## Limitations

- The kinematic tier walks forward: the stance foot stays put in the world (under 1 mm) and
  the pelvis advances 2 x `step_length` per gait cycle. With today's tuned gains sadness = 1.0
  covers roughly 2 m in 6 s (about 0.33 m/s) and anger = 1.0 about 7 m (about 1.13 m/s); these
  are outputs of the tuning, not acceptance targets.
- The physics tier steps in place on the vertical (Z) rail with position servos, open loop, and
  reports instead of controlling. Its report gives the ZMP margin, foot slip and pelvis
  tracking against the commanded path. For example, the neutral gait for 3 s on the Z rail:
  `zmp_outside_frac` 0.0042, `zmp_mean_margin_m` 0.0379, `foot_slip_rms_m_s` 0.024 / 0.029,
  `pelvis_bob_m` 0.0545 (includes the commanded bounce), `pelvis_tracking_ptp_m` 0.0235
  (uncommanded). A constrained root plays a projection of the parameters (`played_params:`
  lists what changed): the Z rail zeroes `step_length` and `lateral_sway_amp`; the weld also
  zeroes `com_bounce_amp` and fixes `com_height_scale` at 1.0; the experimental `xz-rail` zeroes only `lateral_sway_amp`; the free root plays everything
  and falls.
- `--root xz-rail`, a prescribed forward slide, is experimental and falls under stride: the
  neutral gait loses foot contact (first gap at 0.074 s with the default 0.5 s joint ramp; with no ramp the first gap is at 0.33 s and the fall at 0.552 s) and falls at about 0.54 s. A servoed
  rail (kp 5000 to 100000, ramped) failed the same way, at 0.52 to 1.12 s. Forward walking is
  therefore a kinematic-tier result; the physics tier is in place. It is kept as a measured
  negative result.
- Open-loop stepping on a free floating base falls within 0.74-2.25 s at every expressive
  amplitude we measured, and there is no balance controller.
- Lateral sway is lost on every rail. Soles are level in stance and may pitch in swing (worst
  about 0.034 rad on neutral); there is no heel-to-toe roll.
- `com_height_scale = 1.0` is a tuned 0.68 m pelvis walking height, below the straight-leg
  0.79 m stand. Higher leaves anger's long steps out of reach and lower runs the ankle into its
  pitch limit. Fear's crouch is faint (`com_height_scale` -0.02, about 1.4 cm) because the
  G1's ankle range binds; deeper crouches fail the feasibility test. That is a robot
  limitation, not an evidence claim.
- Disgust and surprise have no gait kinematics in any source we found. They ship as placements
  on the arousal/valence plane, flagged `UNSOURCED`, and the CLI warns when you use them.
- No recognition rate has been measured: nobody has yet checked that people read these gaits as
  the intended emotions.

## Grounded versus unsourced

| Emotion | Grounding | Primary source |
| --- | --- | --- |
| anger | measured speed, directional posture | Roether et al. 2011 |
| fear | measured speed, directional posture | Roether et al. 2011 |
| happiness | measured speed, directional posture | Roether et al. 2011 |
| sadness | measured speed, directional posture | Roether et al. 2011 |
| disgust | unsourced | none |
| surprise | unsourced | none |

Other cited sources include Montepare 1987, Michalak 2009, Gross 2012, Barliya 2013,
Halovic and Kroos 2018, and Randhavane 2019. `emotive-gait evidence` prints the full table.

## Related work

- **EMoG** ([arXiv 2609.14432](https://arxiv.org/abs/2609.14432)): emotion-modulated gait
  generation from a continuous style code through an MLP and an RL tracking policy, trained on
  a proprietary performer dataset and run on a Unitree G1. It measured human recognition (141
  raters for its retargeted references, 106 for executed motions); its own direct retargets were
  recognised better than its learned gaits, and it states that its classifier-derived intensity
  is not perceptually calibrated.
- **PAMoR** ([arXiv 2608.28213](https://arxiv.org/pdf/2608.28213)): a learned latent-diffusion
  generator with a continuous valence-arousal control: close in the knob, opposite in mechanism.
- **EmoPose** ([arXiv 2609.23414](https://arxiv.org/pdf/2609.23414)): emotion-aware gesture,
  not gait.

This project differs in being an interpretable map from emotion to named gait parameters, with
effects fitted from public motion capture, intervals, held-out-actor validation, no RL and no
learned balance, released open, where EMoG is learned from a proprietary dataset. That is a
difference, not an advantage, and we have no recognition result to set against theirs.

## Attributions

- The 100STYLE Dataset - Ian Mason, CC BY 4.0
  ([Zenodo record 8127870](https://zenodo.org/records/8127870)). Reduced here to aggregate gait
  statistics.
- The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with
  funding from NSF EIA-0196217.
- Body Movement Library: Ma, Paterson and Pollick 2006. Used locally, pending permission to
  commit derived numbers; none appear in this repository.

## License

MIT. The Unitree G1 model is BSD-3-Clause, from MuJoCo Menagerie.
