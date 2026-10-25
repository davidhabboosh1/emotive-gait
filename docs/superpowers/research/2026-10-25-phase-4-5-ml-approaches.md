# Machine-learning approaches for emotive-gait: options, evidence and a recommendation

Date: 2026-10-25. Scope: which ML approach should sit behind the `emotion.model.modulate` seam,
under the user's constraints (no RL, no learned balance; the evidence module and cited-direction
tests stay; data-derived magnitudes must really come from data; the result must be a measurable
claim; emotional-gait datasets are small).

Convention: every factual claim carries a URL. **UNVERIFIED** marks anything I could not confirm
from a primary or near-primary source. Numbers marked *(computed)* I computed myself from the
repository or the MuJoCo Menagerie G1 model; they are not literature values. Effort estimates
are my own judgement and are labelled as such.

---

## 0. Bottom line

| Rank | Approach | Claim strength | Small-data feasibility | Fit with cited parametric pipeline |
| --- | --- | --- | --- | --- |
| 1 | **(1) Data-derived parameter statistics**: hierarchical (mixed-effects) linear model, emotion to 15 normalised parameters, fitted on human mocap, with CIs and held-out-actor validation | High | High (it is how the source literature itself works with 13-30 actors) | Drop-in at the seam; the evidence tests still apply |
| 2 | **(4) Test the three-axis structure** on the same data (constrained vs unconstrained reduced-rank fits, held-out-actor likelihood) | Medium-high, falsifiable | Medium (only 4 sourced emotions, so low power; see section 4) | Directly tests `axes.py`; nearly free once (1) exists |
| 3 | **(5a) Perception model from Phase 5 ratings**: regress forced-choice responses on the robot's own parameters (lens-model / discriminant-axis approach) | Highest *about the robot*, but only after Phase 5 | Medium (needs a rating study designed for it) | Parametric; reuses the same regression machinery |
| 4 | **(2) Learned classifier as an automatic rater** | Low-medium (a proxy, and partly circular) | Medium (inter-subject accuracy drops) | Useful as a regression test, not as evidence |
| 5 | **(3) Style-transfer / diffusion / VAE generation** retargeted to the G1 | Capability, not a citable claim | Low (needs far more data than exists) | Poor: outputs joint angles, not parameters |

Recommendation: build (1) and run (4) on the same fitted data as one Phase 4.5 package; design
Phase 5 so that its ratings also feed (5a); keep (2) only as a cheap CI regression check; do not
adopt (3) except as a source of comparison stimuli.

---

## 1. What the system needs from ML (from the repository, read-only)

- The map is intensities (6) -> axes (activation, valence, tension) -> parameter deltas via
  `AXIS_GAINS` (one axis per parameter) plus per-emotion `CATEGORY_OFFSETS`
  (`src/emotive_gait/emotion/axes.py`, `src/emotive_gait/emotion/model.py`). The docstrings state
  that the gains and axis weights are "tuned, not cited"; only signs and groupings are grounded.
- The design records "No classifier, no training, no datasets acquired" and a seam
  (`docs/superpowers/specs/2026-10-25-emotive-gait-professionalization-design.md`, line 62;
  `docs/superpowers/plans/2026-10-25-emotive-gait-core.md`, line 106, naming
  `emotion.model.modulate` as the function to swap).
- `expansiveness` is not an independent kinematic quantity: the generator uses it as a
  multiplier on arm swing, arm abduction and swing-foot clearance (`gait/generator.py` lines 12,
  94, 190). It cannot be measured from mocap separately from those three parameters and should be
  fixed at 1.0 (or dropped) in any fitted model. *(from the repository)*
- `speed = step_length * cadence` (`gait/params.py`, `speed` property). Default neutral:
  cadence 1.8 steps/s, step_length 0.35 m, so 0.63 m/s *(computed)*.
- Aside, not part of the task: `CLAUDE.md` says MuJoCo is pinned `<3.3` and seven parameters are
  inert in an in-place generator, whereas `README.md` says MuJoCo `>=3.17,<3.18` and describes a
  forward-walking IK generator. One of them is stale.

---

## 2. Data landscape (summary only; a separate thread is cataloguing datasets)

| Source | Actors | Emotions incl. walking | Access | Note |
| --- | --- | --- | --- | --- |
| Roether et al. 2009, J. Vision 9(6) / Roether PhD thesis 2011 | 25 lay actors; 3 repetitions per emotion; 11 also walked neutral at 4 extra speeds | anger, fear, happiness, sadness, neutral | Public release **UNVERIFIED** | Thesis: [compsens.uni-tuebingen.de](https://www.compsens.uni-tuebingen.de/compsens//files/publications/thesisroether2011.pdf) |
| Barliya et al. 2013, Exp. Brain Res. 225:159-176 | 13 naive + 8 professional actors, Vicon 120 Hz, ~5 m walks | neutral, anger, fear, joy, sadness, plus speed-controlled neutral | Public release **UNVERIFIED** | [preprint PDF](https://www.compsens.uni-tuebingen.de/compsens//files/publications/Barliya2012.pdf) |
| Ma, Paterson & Pollick 2006, Behav. Res. Methods 38(1) | 30 non-professional (15 female) | angry, happy, neutral, sad (walk, knock, lift, throw); 4,080 movements | Glasgow library; terms **UNVERIFIED** | [UEA record](https://ueaeprints.uea.ac.uk/id/eprint/90416) |
| Halovic & Kroos 2018, Hum. Mov. Sci. 57:478-488 | count **UNVERIFIED** | basic emotions (happiness, sadness, anger, fear reported) | Mendeley Data, CC BY 4.0 | [data](https://data.mendeley.com/datasets/x8zr2b2gmt/1) |
| Emilya (Fourati & Pelachaud, LREC 2014) | 11 | 8 incl. joy, anger, panic, fear, anxiety, sadness, shame, neutral; walking is one of 7 actions | access terms **UNVERIFIED** | [ACL Anthology](https://aclanthology.org/L14-1295/) |
| Zhang et al. 2020, Sci. Data | 22 | all six incl. disgust and surprise | PhysioNet restricted licence | Scenario-based, **not gait-specific** ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7478954/)) |
| Bandai Namco Research Motiondataset | 3 professional actors | happy, sad, angry and other styles; walk included | CC BY-NC-ND 4.0 per secondary sources (check the repo) | [GitHub](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset/blob/master/README.md) |
| E-Gait / EWalk (Bhattacharya 2020; Randhavane 2019) | video-derived 3D poses, *perceived* labels | happy, sad, angry, neutral | public per papers | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/5490), [arXiv 1906.11884](https://arxiv.org/pdf/1906.11884) |
| EMoG (Lu et al., arXiv 2609.14432) | 8 professional performers; 12,243 clips, ~2.25 M frames at 100 fps | 6 styles (happy, sad, confident, scared, shy, ...) | release **not stated** | [arXiv HTML v2](https://arxiv.org/html/2609.14432v2) |

Consequences for every approach below:
- Only anger, fear, happiness, sadness (+ neutral) have gait mocap from tens of actors. Disgust
  and surprise have none in gait; the only six-emotion set (Zhang 2020) is not walking. They stay
  `UNSOURCED` whatever ML is used.
- Almost all sets record one (full) intensity per emotion. The intensity-scaling law (linear in
  intensity, as the current model assumes) is therefore **not identifiable** from these data.
  Destephe et al. had graded intensities (intermediate/high/exaggerated) but I found no data
  release ([ROBIO 2013 record](https://oa.mg/work/10.1109/robio.2013.6739640); release **UNVERIFIED**).
- None of the sets record blended emotions, so mixing behaviour stays a modelling assumption.

---

## 3. Normalising human gait parameters to a 1.32 m, ~33 kg humanoid

**Scaling rules.** The standard dimensionless gait normalisation divides lengths by leg length
`l` (hip joint to floor), cadence and angular velocity by `sqrt(g/l)`, and velocity by
`sqrt(g*l)`; Hof's 1996 Gait & Posture letter (4:222-223) gives dimensionless step length as
`s' = s/l` ([clinicalgaitanalysis.com normalisation page](https://clinicalgaitanalysis.com/faq/normalisation.html)).
The Froude number is `Fr = v^2/(g*l)` ([Donelan & Kram 1997, J. Exp. Biol. 200:3193-3201](https://www.sfu.ca/locomotionlab/assets/dynsimwalk1997.pdf)).
The dynamic similarity hypothesis says animals of different sizes at equal Froude number move in
dynamically similar ways (Alexander & Jayes 1983, J. Zool. 201:135-152, as summarised in
[Donelan & Kram 1997](https://www.sfu.ca/locomotionlab/assets/dynsimwalk1997.pdf)).

**Caveat with a primary source.** Donelan & Kram tested the hypothesis by simulating reduced
gravity. At equal Froude numbers, duty factor was nearly the same, but relative stride length
`Ls/Lleg` differed by as much as 67 %, and they rejected the hypothesis
([J. Exp. Biol. 1997](https://www.sfu.ca/locomotionlab/assets/dynsimwalk1997.pdf)). Froude
matching is therefore a first-order rule, not a law. That matters even more for a robot with
different mass distribution, a bent-knee walk and servo dynamics.

**G1 numbers.** These are *(computed)* from the Menagerie `unitree_g1/scene.xml` at qpos0, with
all joints at zero, pelvis at 0.79 m and total mass 33.34 kg, matching CLAUDE.md:
- hip-pitch joint height 0.690 m, so take `l_G1 ≈ 0.69 m` (straight leg);
- thigh about 0.337 m and shank about 0.318 m, both vertical joint-to-joint distances; ankle-roll
  joint at 0.033 m.

For comparison, Unitree lists a standing height of 1320 mm and about 35 kg with battery
([unitree.com/g1](https://www.unitree.com/en/mobile/g1)). A distributor quotes "calf + thigh
0.6 m" (secondary, **UNVERIFIED**). The repository's neutral gait (0.35 m steps at 1.8 steps/s)
works out to `s' ≈ 0.51`, normalised cadence `≈ 0.48`, `v' ≈ 0.24` and `Fr ≈ 0.06`
*(computed)*. Note that the README's walking pelvis height is 0.68 m (bent knee) against 0.79 m
standing, so the G1 walks crouched. It is not geometrically similar to a straight-legged human
walker.

**Recommended procedure (for approaches 1, 2 and 4):**
1. Measure `l` per actor from the mocap (hip joint centre to floor in a standing trial). Do not
   use height-fraction tables; I could not verify the commonly quoted ~0.53 x height figure.
2. Express every length parameter (step length, step height, step width, CoM bounce, lateral
   sway, CoM height) as a ratio to `l`. Express cadence as `cadence / sqrt(g/l)` and speed as
   `v / sqrt(g*l)` (Hof scaling, link above).
3. Model the emotion effect as a **within-actor delta from that actor's own neutral**, in
   dimensionless units. Then re-dimension with `l_G1` and add it to the G1's neutral. This
   transfers the *change*, not the human's absolute gait, and absorbs between-actor size
   differences.
4. Transfer angles (trunk pitch, head inclination, mean elbow flexion, arm-swing amplitude, arm
   abduction) as angles. Geometric scaling preserves angles. Then clip to G1 joint limits and
   report the clipped amount. Fear's crouch is already bound by the G1 ankle range (README).
5. Map head inclination onto `neck_pitch` with the repository's sign convention (positive = gaze
   down, CLAUDE.md). The G1 neck is injected, so human head-on-trunk angle is the right source
   quantity. Whether to measure it relative to the trunk or to the vertical is a definitional
   choice that must be fixed before fitting.
6. Because of the Donelan & Kram result, report the transferred deltas in *two* forms, (a)
   Froude-scaled and (b) raw ratio-to-neutral, and show that the cited-direction tests pass under
   both. That keeps the normalisation choice from silently driving a claim.

---

## 4. The approaches

### Approach 1: Data-derived parameter statistics (hierarchical linear model)

**What it learns, from what.** For each parameter `p`, fit the emotion effects `β[e,p]` (one per
sourced emotion, relative to neutral) from per-trial parameter values extracted from human
emotional-walk mocap. Use a random intercept per actor, and a random slope per actor x emotion
where the data support it. If several datasets are pooled, add a dataset term. The output is a
4 x 15 matrix of effects in normalised units, with 95 % CIs, plus the between-actor SD (how much
people differ). The low-rank variant constrains `B = W G` (emotion -> 3 axes -> parameters),
which *is* the current architecture with fitted numbers. That variant belongs to Approach 4.

**Methodological precedent.** This is how the source literature itself works at this sample
size:
- Roether: 25 actors, 3 repetitions per emotion. Mean joint flexion and gait velocity were
  extracted per trial, sparse regression selected emotion-specific features, and emotional trials
  were matched trial by trial to speed-matched neutral trials (anger 24, fear 18, happiness 30,
  sadness 20 matches) to isolate effects beyond speed
  ([thesis, sections 2.1 and 3.2](https://www.compsens.uni-tuebingen.de/compsens//files/publications/thesisroether2011.pdf)).
- Barliya: 13 naive subjects and 8 actors. Subjects also walked at slow and fast speeds so that
  emotion effects could be separated from speed effects by regression; the effects "could not be
  explained solely by a change in speed"
  ([preprint](https://www.compsens.uni-tuebingen.de/compsens//files/publications/Barliya2012.pdf)).
- Destephe et al. drove a WABIAN-2R robot gait from two actors' mocap, scaling "the actors'
  values to respect the hardware limits of the robot". They report 51.1 % (French) and 63.3 %
  (Japanese) recognition against 20 % chance
  ([Frontiers in Psychology 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4340147)). This is
  the closest robot precedent. It used hand-made patterns and gave no CI-bearing parameter model.
- Mixed models: Barr et al. 2013 argue for the maximal random-effects structure the design
  supports in confirmatory tests, and report that random-intercept-only models performed worse in
  their simulations ([PMC3881361](https://pmc.ncbi.nlm.nih.gov/articles/PMC3881361)). Tooling:
  [statsmodels MixedLM](https://www.statsmodels.org/stable/mixed_linear.html) (pure Python, which
  fits the package).

**What it needs.**
- Mocap (marker or BVH) of the four sourced emotions plus neutral. Ma/Paterson/Pollick (30
  actors, 3 emotions) and Halovic & Kroos (CC BY 4.0) look most accessible. Roether and Barliya
  data are the best designed (speed-matched neutral), but their availability is **UNVERIFIED**.
  Ask the authors.
- A deterministic extractor from mocap to the 14 measurable parameters: heel strikes from foot
  marker height or velocity, giving cadence, step length, step width, double-support fraction and
  step height; pelvis or CoM marker for height, bounce and sway; segment angles for trunk, head,
  shoulder and elbow. **No retargeting to G1 joints is needed**: the model lives in parameter
  space, which is the main advantage of this architecture.
- No pose estimation if mocap is used. Video-only sets (E-Gait) would need 3D pose estimation;
  avoid them for magnitudes because depth error contaminates step length and trunk pitch.
- Labels: the acted emotion category, plus intensity where it exists. Compute: trivial (CPU,
  seconds).

**Measurable claims it supports.**
1. "For emotion e and parameter p, the data give an effect of β ± CI (normalised units, N
   actors, dataset D)." Each evidence claim can be upgraded from a signed direction to a measured
   effect.
   - *Note:* CLAUDE.md forbids claims that *state a magnitude in degrees*. A data-derived effect
     with a CI is a different kind of object, so this needs an explicit design decision. A clean
     option: keep `evidence.py` directional, and put fitted effects in a separate generated data
     file with provenance.
2. "On held-out actors (leave-one-actor-out), the fitted map predicts each actor's emotional
   parameter deltas with error E_fit, against E_hand for the current hand-tuned gains." This is
   the single most useful number: it says whether the hand tuning was wrong, and by how much.
3. "The fitted effects agree in sign with every cited direction" (or they don't). This is a
   *data* check on the literature claims, a stronger version of the existing direction test.
4. Between-actor variability: if CIs are wide relative to effects, that is itself a reportable
   result. It matches Venture et al.'s finding that inter-subject recognition "is more prone to
   error, suggesting strong inter-personal differences"
   ([Int. J. Soc. Robotics 2014, Tsukuba release](https://www.tsukuba.ac.jp/en/research-news/20140714162012.html)).

**How it respects the constraints.** No RL and no control. The emotion is still applied to
parameters, and the generator still clamps. The evidence module and direction tests run
unchanged on the fitted map. Magnitudes are data-derived by construction, so long as the clip to
G1 limits is reported separately as "played vs data".

**Failure modes at small data.**
- Wide CIs on posture parameters with small effects (arm abduction, step width). Report them;
  do not shrink silently. If a prior is used, label it.
- Speed confound: most emotion effects co-vary with speed (Barliya; Roether). Without
  speed-matched neutral trials the model cannot say whether, for example, sad arm swing is "sad"
  or "slow". Fit a speed covariate where the data allow it, and state when they don't.
- Acted vs felt emotion: Roether and Barliya used mood induction, other sets are acted. Pooled
  effects mix protocols, so give each dataset its own term.
- Intensity: only the endpoint is identified. Linear interpolation from 0 to 1 remains an
  assumption, and should be labelled as such in the model.
- Disgust and surprise: nothing to fit. They stay hand-placed and flagged.
- Operational definitions: `elbow_swing_ratio` and `arm_swing_amp` need human-measurable
  definitions that match the generator's semantics (generator line 114), or the fitted numbers
  will not mean the same thing on the robot.
- Transfer: the G1's bent-knee walk and joint limits truncate some effects (fear crouch). The
  measured claim is about *human* effects, and the robot shows a clipped version. Say so.

**Effort (estimate).** About 1-2 weeks: extractor and validation against reported values (for
example Halovic & Kroos's stride and cadence pattern, [WSU abstract](https://researchers.westernsydney.edu.au/en/publications/not-all-is-noticed-kinematic-cues-of-emotion-specific-gait/)),
then about 2-3 days to fit, cross-validate and generate the data file. Most of the risk is in
data access and extraction, not modelling.

---

### Approach 2: A learned emotion classifier used as an automatic rater

**What it learns, from what.** A classifier from gait features to emotion category, trained on
human data. It is applied to features extracted the same way from the *simulated* G1 gait, and
its accuracy or confusion matrix is reported as a recognition-rate proxy.

**Prior work.**
- Karg, Kühnlenz & Buss 2010 (IEEE TSMC-B 40(4):1050-1061) compared PCA, KPCA and LDA features.
  Secondary summaries report much higher person-dependent than inter-individual accuracy
  ([PeerJ 2016 review](https://peerj.com/articles/2364)); I could not read the primary figures,
  so they are **UNVERIFIED** here.
- Venture et al. 2014: 4 actors, good intra-subject accuracy, poor inter-subject accuracy
  ([Tsukuba](https://www.tsukuba.ac.jp/en/research-news/20140714162012.html)).
- Randhavane et al. 2019 (EWalk): 80.07 % on 4 classes from video-derived 3D poses
  ([arXiv 1906.11884](https://arxiv.org/pdf/1906.11884)).
- STEP (AAAI 2020): ST-GCN with CVAE augmentation, reported 88 % on E-Gait with *perceived*
  labels ([AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/5490)).
- EMoG uses a Transformer classifier only to produce continuous style-intensity training labels,
  and measured recognition with humans, not with a classifier
  ([arXiv HTML](https://arxiv.org/html/2609.14432v2)).

**What it needs.** The same extractor as Approach 1 (for a feature-level classifier), or skeleton
retargeting between human and G1 (for a skeleton-level ST-GCN). Skeleton-level models face a
domain gap: different proportions, an injected neck, and a bent-knee walk. GMR is an MIT-licensed
human-to-G1 retargeter ([arXiv 2510.02252](https://arxiv.org/pdf/2510.02252)). Leave-one-actor-out
evaluation is mandatory.

**What it proves, and why that is weak.**
- If the classifier reads the 15-parameter vector, it is **circular**: the robot's parameters
  *are* the model output, so it only re-checks that the parameter vector sits in the right human
  region. Approach 1's held-out error already measures that more directly.
- If it reads features re-extracted from the *rendered* G1 joint trajectories, it measures
  something real but narrow: "the rendering preserves the parameter-level discriminability of
  human gait". That is a useful generator-fidelity regression test.
- It does not measure human perception. Its validity as a proxy has to be established against
  Phase 5 (for example, correlation between classifier and human confusion matrices). Until
  then it should not be reported as a recognition rate.
- Goodhart risk: tuning gains to maximise classifier accuracy turns the proxy into the target.
  Keep it out of the tuning loop.

**Constraints.** Compatible: no RL, and it touches nothing at runtime.

**Failure modes at small data.** Inter-subject collapse (Venture); over-fitting to actor identity;
4-class chance at 25 % with about 25-30 actors gives wide accuracy CIs; the human-to-robot domain
shift. Labels: acted versus perceived (E-Gait labels are perceived, Roether-type labels are
acted) measure different things.

**Effort (estimate).** About 3-5 days for a feature-level classifier on top of Approach 1's
extractor, and about 2+ weeks for a skeleton-level ST-GCN with retargeting.

---

### Approach 3: Style-transfer / motion-generation models retargeted to the G1

**Representative work.**
- **EMoG** (arXiv 2609.14432, Sep 2026). An MLP maps an emotion-style code with continuous
  intensity, plus commands, to periodic gait trajectories, and a **unified RL policy** tracks them
  on a **Unitree G1**. It uses 8 professional performers and 12,243 clips. In human five-way
  recognition it reaches 49.1 % (81/165), 54.5 % and 73.3 % (121/165) at intensities 0.3, 0.6 and
  0.9. In the blinded six-way survey, *direct retargeting of the human clips* scored 77.28 %
  against 64.50 % for EMoG + S on the reference rendering
  ([arXiv HTML v2](https://arxiv.org/html/2609.14432v2)).
- **PAMoR** (arXiv 2608.28213). A diffusion model with action and affect priors in a shared
  latent space, on a 29-DoF G1. Its valence-arousal coordinate is computed from robot kinematics
  (postural expansion, movement energy) with no human annotation. In a perceptual study, raters
  identified the commanded emotion on 0.38 of trials, against 0.44 reported for acted human
  bodies ([arXiv](https://arxiv.org/abs/2608.28213)).
- **SMooDi** (ECCV 2024): a style adaptor on a pretrained text-to-motion latent diffusion model,
  using 100STYLE ([arXiv 2407.12783](https://arxiv.org/pdf/2407.12783)).
- **Aberman et al.** (SIGGRAPH 2020): unpaired style transfer with AdaIN style codes, including
  from video ([arXiv 2005.05751](https://arxiv.org/abs/2005.05751v1)).
- **FEEL / EmotionT2M** (CVPR 2026 workshop): emotion text-to-motion, where emotion descriptions
  are partly LLM-generated over HumanML3D and IDEA400
  ([supplement](https://openaccess.thecvf.com/content/CVPR2026W/HuMoGen/supplemental/Choi_FEEL_Emotion-Enriched_Text-to-Motion_CVPRW_2026_supplemental.pdf)).
  LLM-written emotion labels are not data-derived emotion, so they fail the user's constraint.
- "Walk Like This" (named in the README): **not found** under that title. The closest match is a
  2026 physics-aware latent-diffusion style-transfer work on the G1
  ([preprint listing](https://preprints.kavlimeetings.org/2026/06/02/all/nano/caltech/261873/),
  **UNVERIFIED**). The README's citation should be checked.

**What it would need.** Thousands of clips (EMoG used 12k from 8 performers), retargeting (GMR),
and GPU training. For physical execution these systems use RL tracking (EMoG), which the user
excludes. The kinematic tier could play the generated trajectories without RL.

**Fit with the architecture: poor.**
- The output is joint trajectories, not `GaitParameters`, which violates "emotion is applied to
  parameters, never to joint angles" (CLAUDE.md).
- Magnitudes are network outputs, not citable statistics. The cited-direction tests could only
  be applied after the fact, by re-extracting parameters.
- Small data: mode collapse or memorisation of individual performers. There is no CI on any
  effect.

**What it can prove.** Only a capability ("it generates recognisable emotional gaits"), measured
by human ratings. That is not a claim about which kinematics carry emotion.

**The useful take-away is the EMoG comparison:** direct retargeting of the human clips was
recognised better than the learned generator. For this project that argues for a
**direct-retarget baseline condition in Phase 5** (human emotional mocap -> G1 via GMR, played
kinematically). It gives a ceiling against which the parametric gait's recognition can be
reported, with no ML in the generator.

**Effort (estimate).** Many weeks plus data that do not exist at the needed scale. Not
recommended.

---

### Approach 4: Learning, or testing, the three-axis structure

**What it would test.** Does the activation / valence / tension structure, with each parameter
driven by one axis (`AXIS_GAINS`), describe human emotional-gait parameters better than the
alternatives?

**Prior evidence for the structure.**
- Roether's discriminant analysis of posture features found a strong first function loading on
  head inclination, which separated sadness (the valence-like cue). A second function loaded on
  limb flexion and separated anger/fear from happiness/sadness (the tension-like cue). Movement
  size and speed carried activation, and average flexion was the cue that discriminated affects
  at similar activation
  ([thesis, abstract and section 3.3.2.3](https://www.compsens.uni-tuebingen.de/compsens//files/publications/thesisroether2011.pdf)).
  This is effectively the provenance of the project's three axes.
- Pollick et al. 2001 (arm movements, not gait): multidimensional scaling of perceived affect
  gave a circumplex-like space. Dimension 1 was activation, highly correlated with kinematics;
  dimension 2 was pleasantness, attributed to phase relations between segments
  ([Glasgow eprint](https://eprints.gla.ac.uk/25001)).
- Troje's walker framework linearises gait and computes linear discriminant axes from attributes
  or observer ratings, including sad-vs-happy and nervous-vs-relaxed axes
  ([biomotionlab.ca](https://www.biomotionlab.ca/walking/); Troje 2002, J. Vision 2:371-387).

**The honest statistical problem.** With 4 sourced emotions plus neutral, there are 5 group
means. Any 5 points in parameter space lie in a 4-dimensional affine subspace, so "a 3-axis model
fits" removes only one dimension and is weakly falsifiable. The testable parts are:
1. **Sparsity of loadings**: is each parameter driven by a single axis (current model), or does
   it need two? Compare held-out-actor likelihood of (a) the constrained one-axis-per-parameter
   model, (b) unconstrained rank-3 reduced-rank regression, (c) full rank-4 per-emotion effects.
2. **Axis identity**: does the leading component of the emotion effects align with
   speed/size (activation), as Roether and Pollick predict? Is tension (mean flexion) separable
   from valence (head and trunk)?
3. **Category offsets**: are fear's override offsets needed, that is, does fear's held-out error
   drop when they are included? This directly tests a CLAUDE.md design choice.

Tools: CCA and PLS ([scikit-learn cross_decomposition](https://scikit-learn.org/stable/modules/cross_decomposition.html)),
factor analysis ([scikit-learn decomposition](https://scikit-learn.org/stable/modules/decomposition.html)).
CCA between emotion indicators and parameters gives at most K-1 = 4 canonical dimensions, which
is the same identifiability limit.

**Measurable claim.** "The one-axis-per-parameter three-axis model loses (or does not lose) X in
held-out-actor log-likelihood against the per-emotion model." It is a model-comparison number,
falsifiable, with the low-power caveat stated. A dataset with more categories (Emilya's 8, with
11 actors, [ACL](https://aclanthology.org/L14-1295/)) or graded intensities would strengthen it.

**Constraints, failure modes, effort.** Fully compatible. It reuses Approach 1's data and model,
and adds about 2-3 days (estimate). Failure mode: over-reading a component analysis on 5 group
means. Report the comparison, not "the data discovered three axes".

---

### Approach 5: Better-supported alternatives

**5a. Perception model fitted to Phase 5 ratings (recommended as the follow-on).** Phase 5 renders
G1 stimuli whose parameters are known exactly. Fit a multinomial (or per-emotion logistic) mixed
model of the forced-choice response on the parameter vector, with random effects for rater and
stimulus. The result gives the **robot's own** cue weights: which parameters actually move
perceived emotion on this body. That is Brunswik-lens-style cue utilisation, and the parametric
analogue of Troje's rating-derived discriminant axes ([biomotionlab.ca](https://www.biomotionlab.ca/walking/)).
- Sripathy et al. 2022 learned a single map from robot trajectories to valence-arousal-dominance
  from user labels and used it as a cost for any emotion, on a vacuum robot and Cassie
  ([arXiv 2203.02091](https://arxiv.org/abs/2203.02091)). That is the same idea in learned form.
- Measurable claim: "parameter p shifts the probability of response e by β ± CI on the G1",
  which directly tests whether a human-derived cue survives on the robot.
- Requirement: Phase 5 must sample parameters *off* the six presets (for example, a design that
  varies parameters independently), or cue weights are not identifiable.
- Planning: with stimuli as a random factor, power plateaus as raters are added and is capped by
  the number of stimuli (Westfall, Kenny & Judd 2014, [PubMed 25111580](https://pubmed.ncbi.nlm.nih.gov/25111580)).
  Plan the stimulus count, not just the rater count.
- No RL, parametric, citable. Rank 3 only because it depends on Phase 5 data.

**5b. Direct-retarget ceiling condition (not ML, but decision-relevant).** See the end of
Approach 3. Adding human mocap -> G1 (GMR, [arXiv 2510.02252](https://arxiv.org/pdf/2510.02252))
as a Phase 5 condition turns "people recognise our gait at R %" into "at R % against R_human-on-G1 %".
EMoG's 77.28 % (direct retarget) vs 64.50 % (learned) shows that the gap can be large
([arXiv HTML](https://arxiv.org/html/2609.14432v2)).

---

## 5. Ranking rationale (claim strength x small-data feasibility x architectural fit)

1. **Approach 1, hierarchical linear parameter model.** It produces per-parameter effects with
   CIs and a held-out comparison against the hand-tuned gains. 13-30 actors is the regime the
   source studies (Roether, Barliya) analysed successfully. It plugs into the seam without
   touching joint angles.
2. **Approach 4, structure test.** A falsifiable model comparison of the project's core
   assumption on the same data, at low marginal cost. It is limited by having 4 sourced emotions.
3. **Approach 5a, Phase 5 perception regression.** The strongest claim *about the robot*. It is
   parametric and citable, but needs a Phase 5 design that varies parameters.
4. **Approach 2, classifier as rater.** Circular on parameters, and only a fidelity test on
   rendered kinematics. Inter-subject accuracy is poor at this data size. Keep it as a CI check.
5. **Approach 3, generative style models.** No citable magnitudes, data-hungry, outputs joint
   angles, and physical execution in prior work uses RL. Use only as a comparison stimulus.

## 6. Honest limitations of the top choice (Approach 1)

- **It measures human effects, not robot perception.** A data-derived magnitude says how much
  humans change step length when acting sad. It does not say a G1 doing so is read as sad. Only
  Phase 5 (or 5a) can support the perceptual claim.
- **Intensity is not identified.** Single-intensity data fix the endpoint only. The current
  linear-in-intensity law, and all mixtures, remain assumptions.
- **Disgust and surprise stay unsourced.** No gait data exist for them among the sets found.
- **Data access is the critical path.** The best-designed sets (Roether 25 actors with
  speed-matched neutral; Barliya 21 subjects) have no confirmed public release.
  Ma/Paterson/Pollick lacks fear; Halovic & Kroos's actor count is unverified.
- **Speed confound.** Without speed-matched neutral trials, emotion and speed effects cannot be
  separated, and the fitted "sad arm swing" may be "slow arm swing".
- **Acted vs induced vs perceived labels** differ across datasets. Pooling needs a dataset term
  and may still blur effects.
- **Transfer is approximate.** Froude scaling is not exact even in humans (Donelan & Kram 1997:
  up to 67 % relative stride length differences at equal Fr). The G1 walks crouched, and its
  joint limits clip some effects (fear's crouch). The robot shows a clipped, rescaled version of
  a human effect. Report data value and played value separately.
- **Small effects get wide CIs.** Several posture parameters may come out "direction
  consistent, magnitude uncertain". That is a legitimate result and should be published as such,
  not tuned away.
- **`expansiveness` is not measurable** independently of arm swing, abduction and step height;
  it should be fixed rather than fitted.
- **CLAUDE.md conflict:** "No claim may state a magnitude in degrees". Data-derived effects need a
  separate, generated, provenance-carrying data file, or a deliberate change to that rule.

## 7. Sources (primary or nearest available)

- Roether PhD thesis (Roether et al. 2009 J. Vision 9(6)): https://www.compsens.uni-tuebingen.de/compsens//files/publications/thesisroether2011.pdf
- Barliya et al. 2013 preprint: https://www.compsens.uni-tuebingen.de/compsens//files/publications/Barliya2012.pdf
- Halovic & Kroos data (CC BY 4.0): https://data.mendeley.com/datasets/x8zr2b2gmt/1 ; abstract: https://researchers.westernsydney.edu.au/en/publications/not-all-is-noticed-kinematic-cues-of-emotion-specific-gait/
- Ma, Paterson & Pollick 2006: https://ueaeprints.uea.ac.uk/id/eprint/90416
- Montepare et al. 1987 (DOI): https://dx.doi.org/10.1007/BF00999605
- Michalak et al. 2009 (secondary description): https://link.springer.com/article/10.1007/s10608-021-10215-7
- Emilya: https://aclanthology.org/L14-1295/
- Zhang et al. 2020: https://pmc.ncbi.nlm.nih.gov/articles/PMC7478954/
- Bandai Namco dataset: https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset/blob/master/README.md
- Venture et al. 2014: https://www.tsukuba.ac.jp/en/research-news/20140714162012.html ; DOI https://unpaywall.org/10.1007%2FS12369-014-0243-1
- Karg et al. (secondary): https://peerj.com/articles/2364
- Randhavane et al. 2019: https://arxiv.org/pdf/1906.11884
- STEP (Bhattacharya et al. 2020): https://ojs.aaai.org/index.php/AAAI/article/view/5490
- Destephe et al. 2015: https://pmc.ncbi.nlm.nih.gov/articles/PMC4340147 ; ROBIO 2013: https://oa.mg/work/10.1109/robio.2013.6739640
- EMoG: https://arxiv.org/html/2609.14432v2
- PAMoR: https://arxiv.org/abs/2608.28213
- SMooDi: https://arxiv.org/pdf/2407.12783
- Aberman et al. 2020: https://arxiv.org/abs/2005.05751v1
- FEEL / EmotionT2M: https://openaccess.thecvf.com/content/CVPR2026W/HuMoGen/supplemental/Choi_FEEL_Emotion-Enriched_Text-to-Motion_CVPRW_2026_supplemental.pdf
- GMR: https://arxiv.org/pdf/2510.02252
- Pollick et al. 2001: https://eprints.gla.ac.uk/25001
- Troje walker framework: https://www.biomotionlab.ca/walking/
- Sripathy et al. 2022: https://arxiv.org/abs/2203.02091
- Barr et al. 2013: https://pmc.ncbi.nlm.nih.gov/articles/PMC3881361
- Westfall, Kenny & Judd 2014: https://pubmed.ncbi.nlm.nih.gov/25111580
- Hof normalisation: https://clinicalgaitanalysis.com/faq/normalisation.html
- Donelan & Kram 1997: https://www.sfu.ca/locomotionlab/assets/dynsimwalk1997.pdf
- Unitree G1 spec: https://www.unitree.com/en/mobile/g1
- statsmodels MixedLM: https://www.statsmodels.org/stable/mixed_linear.html ; scikit-learn: https://scikit-learn.org/stable/modules/cross_decomposition.html , https://scikit-learn.org/stable/modules/decomposition.html
