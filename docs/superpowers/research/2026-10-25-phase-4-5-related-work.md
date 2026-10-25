# Related work: emotional / expressive gait for humanoids and characters (survey as of 2026-10-25)

Method: web search + arXiv abstract/HTML fetches. Everything below carries a URL. "UNVERIFIED" = not confirmed from a primary page. Search coverage of HRI venues (RO-MAN/HRI/IROS 2024-2026) was thin; absence of a hit is NOT evidence of absence.

## 0. What PAMoR is (README names it as "closest in approach")
**PAMoR: Parameterized Affective Motion Generation in Real Time for Humanoid Robots** - Yan Pan, Lingfan Bao, Tianhu Peng, Chengxu Zhou (institution not stated on arXiv page). arXiv 2608.28213, v1 28 Aug 2026, v2 21 Sep 2026, cs.RO, "under review at IEEE RA-L". https://arxiv.org/abs/2608.28213 (HTML: https://arxiv.org/html/2608.28213v2)
- Does: generates whole-body G1 motion where affect is a continuous valence-arousal (V-A) control parameter. An action prior and two affect priors share a latent space and are combined at each denoising step (diffusion, autoregressive, real time); action and affect are separately editable.
- ML: yes. Latent diffusion priors trained on BABEL-annotated AMASS + video-reconstructed motions (GVHMR) + teleoperated G1 recordings; 15 action classes, 10,095 train / 2,406 held-out chains at 20 Hz. Learned = motion priors. V-A "labels" are NOT human-annotated: closed-form from 14 FK keypoints (valence = z-scored arm expansiveness + root height + trunk tilt; arousal = z-scored keypoint speed + acceleration). Paper itself says equal weights are "a convention", not empirically calibrated.
- Robot: Unitree G1 29-DoF, physical deployment. Balance/locomotion: NOT learned by the paper; execution by the existing SONIC whole-body tracker (RL, NVIDIA); sim replay with SONIC used to filter untrackable data. Open-loop, no affect feedback.
- Measures: perceptual study, 12 raters, 6 per set, 70 trials each = 840 trials. Commanded emotion named first on 0.384 of trials (chance 0.125), top-3 0.845 (chance 0.375); per-emotion range 0.21 (relaxed) to 0.49 (excited); compared indirectly with 0.44 for acted humans (Emilya-based benchmark). Baselines: TextOp (emotive prompt), SMooDi (style clip; covers 4 of 8 emotions), ECHO (text-to-motion). Tracking of commanded V-A across range also reported.
- Magnitudes: learned end-to-end (priors) with a hand-chosen V-A definition. Code/data: no release statement found.
- Caveat for positioning: it is "parameterized" in the sense of a continuous V-A knob, but it is a learned diffusion generator, NOT an analytic parameter model. The README's "closest in approach" holds for the V-A-knob idea only; mechanistically it is the opposite of emotive-gait. It is also generic action+affect, not walking-specific (gait is one of 15 action classes; punches/kicks read as aggressive regardless of affect, per authors).

## 1. Directly competing: emotional humanoid locomotion with learning (2026)

**EMoG: Emotion-Modulated Gait Generation for Expressive Humanoid Locomotion** - Yi Lu, Tianhao Jiang, Honglong Tian, Yumeng Zhang, Qingrui Zhao, Zhengtao Wang, Xiao-Xiao Long, Qiu Shen, Xun Cao. arXiv 2609.14432, v1 13 Sep 2026, v2 20 Sep 2026, cs.RO. https://arxiv.org/abs/2609.14432 ; HTML https://arxiv.org/html/2609.14432v2 ; project https://jtthhhh.github.io/Emog-web
- Does: emotion-style code with continuous intensity conditions a lightweight MLP generating periodic gait trajectories; an RL tracker executes them while following velocity commands; an LLM parser maps free text to style/params (mean latency 2.18 s).
- ML: MLP trained on a new dataset: ~2.25M frames, 12,243 clips, 100 fps, 8 professional performers, six categories (happy, confident, sad, scared, shy, neutral). A Transformer classifier's softmax gives intensity labels.
- Robot: Unitree G1 per full-text HTML, physical (README says "unverified"; the abstract page does not name it). Balance: learned, but not by EMoG - frozen SONIC tracker and a BeyondMimic-based tracker (training details not given).
- Measures: human recognition. Five-way n=99 at intensities 0.3/0.6/0.9 -> 49.1/54.5/73.3% overall; six-way n=73 at 0.9; blinded six-way reference/execution surveys n=141/106. Table II six-way reference/executed: direct retargeting 77.28/65.25%; MotionGPT3 23.06/21.23%; DART 25.71/24.73%; EMoG+SONIC 64.50/61.32%; EMoG+BeyondMimic 64.50/53.93%. Classifier accuracy not reported; authors state intensity is "not perceptually calibrated".
- Magnitudes: learned from performer data. Code/data: none stated; project site only; dataset appears proprietary/unreleased.
- This is the single most relevant competitor to a learned-from-data emotion model.

**PAMoR** - see section 0.

**Bionic Human-Motion Style Transfer for Physically Executable Whole-Body Control of Humanoid Robots** - Tianchen Huang, Mingkuan Zhao, Yang Gao, Feiyang Yuan, Junchi Gu, Xiaohu Zhang, Dongdong Zhao, Shi Yan, Yu Wang, Wei Gao, Shiwu Zhang. arXiv 2606.03536, 2 Jun 2026. https://arxiv.org/abs/2606.03536 ; project https://huangtc233.github.io/bionic-style-transfer/
- Style (not emotion) transfer: physics-aware multi-condition latent diffusion (style exemplar + content + trajectory), classifier-free guidance as a style-intensity knob, contact/smoothness regularisation, plus a preview-based whole-body tracking RL policy (cluster-and-distill). G1, sim + real; 96.0% success over 125 real trials (execution success, not human recognition). Datasets not named on abstract (UNVERIFIED). Magnitude: guidance scale, learned. Code: none linked on arXiv page.

**EmoPose** - Daojie Peng, Bingtao Wang, Fulong Ma, Wenjun Yue, Liang Zhang, Jun Ma. arXiv 2609.23414, 20 Sep 2026. https://arxiv.org/abs/2609.23414
- Gesture, not gait. A VLM (GPT-5.5 in tests) turns language/dialogue into a gesture plan (class, library variant, intensity, speech anchor); deterministic robot-side modules build trajectories from a hand-authored library (29 variants). G1 physical + MuJoCo preview. Measures: planning accuracy on EmoPose-Bench 98.25% Easy / 76.50% overall; no human recognition study seen. Code: none stated. The only G1 + MuJoCo + emotion item found, but upper-body gestures.

**EMOTION: Expressive Motion Sequence Generation for Humanoid Robots with In-Context Learning** - Peide Huang, Yuhan Hu, Nataliya Nechyporenko, Daehwa Kim, Walter Talbott, Jian Zhang. arXiv 2410.23234, 30 Oct 2024 (no venue listed). https://arxiv.org/abs/2410.23234
- LLM in-context learning composes gesture sequences (10 gestures); online user studies (n not on abstract page); robot unspecified; gestures, not gait. Code none linked.

## 2. Whole-body tracking / teleop stacks (G1/H1) - tracking, NOT emotion
All are RL (sim-to-real) trackers of human mocap. None models emotion; "expressive"/"style" means "can reproduce whatever the reference does". They matter because EMoG and PAMoR delegate balance to them (SONIC), and a learned emotion model would likely do the same.
- **ExBody** (Expressive Whole-Body Control for Humanoid Robots) - Cheng, Ji, Chen, Yang, Yang, Wang. arXiv 2402.16796, Feb 2024. https://arxiv.org/abs/2402.16796. Upper body imitates mocap, legs follow velocity command; RL in sim; "walk in different styles" comes from reference data; emotion not addressed; robot not named on abstract.
- **ExBody2** - Ji et al. arXiv 2412.13196, Dec 2024. https://arxiv.org/abs/2412.13196. Teacher-student RL tracker; no explicit emotion/style mechanism on abstract.
- **H2O** - He, Luo, Xiao, Zhang, Kitani, Liu, Shi. arXiv 2403.04436, Mar 2024. https://arxiv.org/abs/2403.04436. RGB teleop tracker.
- **OmniH2O** - CoRL 2024 (PMLR v270), arXiv 2406.08858. https://proceedings.mlr.press/v270/he25b.html ; https://arxiv.org/html/2406.08858v1. Teleop + autonomy; releases OmniH2O-6 dataset.
- **HumanPlus** - Fu, Zhao, Wu, Wetzstein, Finn. arXiv 2406.10454, Jun 2024; custom 33-DoF humanoid. https://arxiv.org/abs/2406.10454
- **HOVER** - He et al., ICRA 2025, multi-mode distillation. https://rpl.cs.utexas.edu/publications/2025/05/19/he-icra25-hover/ (arXiv id not verified).
- **ASAP** - He et al., RSS 2025, arXiv 2502.01143; sim-to-real residual dynamics; real G1. https://arxiv.org/abs/2502.01143
- **AMO** - Li, Cheng, Huang, Yang, Qiu, Wang. arXiv 2505.03738, May 2025; G1; RL + trajectory optimisation, dexterous reaching. https://arxiv.org/abs/2505.03738
- **TWIST** - Ze, Chen, Araujo, Cao, Peng, Wu, Liu. arXiv 2505.02833, May 2025; RL+BC tracker with mocap teleop; lists "expressive movement" as a capability. https://arxiv.org/abs/2505.02833
- **GMT** - Chen, Ji, Cheng, Peng, Peng, Wang. arXiv 2506.14770, Jun 2025; adaptive sampling + motion MoE. https://arxiv.org/abs/2506.14770
- **SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control** - Luo, Yuan, Zhu et al. (NVIDIA). arXiv 2511.07820 (v1 Nov 2025, v2 Dec 2025). https://arxiv.org/abs/2511.07820. 1.2M-42M param trackers, >100M frames / 700 h mocap, real-time kinematic planner; used as executor by both EMoG and PAMoR. Whether SONIC itself has any style/emotion conditioning: UNVERIFIED (not found).
- **Puppeteer** (Hierarchical World Models as Visual Whole-Body Humanoid Controllers) - Hansen, Jyothir S V, Sobal, LeCun, Wang, Su. arXiv 2405.18418. https://arxiv.org/abs/2405.18418. Simulated 56-DoF humanoid, RL; motions "broadly preferred by humans" - naturalness, not emotion. Code: https://nicklashansen.com/rlpuppeteer
- Other items surfaced, not read: UniTracker https://arxiv.org/html/2507.07356v2 ; "Robust and Generalized Humanoid Motion Tracking" https://arxiv.org/pdf/2601.23080 ; PCHC preference-conditioned humanoid control https://arxiv.org/abs/2603.24047 (style-like knob via multi-objective RL, not emotion; beyond-abstract content UNVERIFIED). A title "One Policy, Many Styles: Where to Aggregate Preferences for Steerable Humanoid Control?" appeared in https://github.com/YanjieZe/awesome-humanoid-robot-learning but I could not locate the paper: UNVERIFIED.
Verdict: no tracker above handles emotion; style enters only through the reference. Code/data release for each: project sites exist; per-paper status not checked (UNVERIFIED).

## 3. Style / emotion for characters (kinematic, no balance)
- **SMooDi: Stylized Motion Diffusion Model** - Lei Zhong, Yiming Xie, Varun Jampani, Deqing Sun, Huaizu Jiang. ECCV 2024. https://arxiv.org/abs/2407.12783 ; https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/147_ECCV_2024_paper.php. Content text + style-motion sequence on MLD with style guidance and adaptor; trained on retargeted 100STYLE. Kinematic, simulated character only; automatic metrics (user-study details UNVERIFIED). Used as a baseline by PAMoR. Code stated as "to be released upon acceptance"; current state UNVERIFIED.
- **100STYLE** (Mason, Starke, Komura 2022 - attribution from memory, UNVERIFIED this session); described as locomotion-only content in https://eccv2024.ecva.net/media/eccv-2024/Slides/1010.pdf.
- Findlay et al., **Denoising Diffusion Probabilistic Models for Styled Walking Synthesis**, arXiv 2209.14828 (2022; nearest walking-specific diffusion style work). https://arxiv.org/abs/2209.14828
- Hu et al., **On-the-fly Learning to Transfer Motion Style with Diffusion Models: A Semantic Guidance Approach**, arXiv 2405.06646 (2024). https://arxiv.org/html/2405.06646v1
- Alexanderson et al., **Listen, Denoise, Action!**, arXiv 2211.09707 (2022): style strength via classifier-free guidance, also locomotion. https://arxiv.org/abs/2211.09707
- "Walk Like This" (README: "stylized locomotion via example-driven diffusion"): I could NOT find a paper by this title. UNVERIFIED.
- Emotion-enriched text-to-motion via LLM limb-level manipulation, AMUSE, HIAER, FEEL, L3EM, ARDY, MotionGPT3: named only inside the PAMoR/EMoG reference lists (https://arxiv.org/html/2608.28213v2, https://arxiv.org/html/2609.14432v2); not individually verified - UNVERIFIED. Per EMoG's Table II, MotionGPT3 and DART with emotional prompts are recognised at only ~21-26% six-way (chance 16.7%), i.e. text-prompted emotion in generic generators is weak.
- Marmpena et al., **Data-driven emotional body language generation for social robotics**, arXiv 2205.00763 (2022; CVAE conditioned on V-A, small hand-designed set, Pepper inferred from repo name; most levels perceived as distinct). https://arxiv.org/abs/2205.00763 - prior art for the "learned V-A knob".

## 4. UMD / Manocha group (gait emotion PERCEPTION, not generation)
No 2023-2025 follow-ups were found (UNVERIFIED that none exist); I could not find a paper titled exactly "Learning Gait Emotions" other than the MIG 2022 one below. Verified items, all recognition:
- **Identifying Emotions from Walking using Affective and Deep Features** (Randhavane, Bhattacharya, Kapsaskis, Gray, Bera, Manocha) arXiv 1906.11884, 2019: autoencoder/LSTM + affective features; origin of E-Gait data. https://arxiv.org/abs/1906.11884
- **STEP: Spatial Temporal Graph Convolutional Networks for Emotion Perception from Gaits**, AAAI 2020. https://ojs.aaai.org/index.php/AAAI/article/view/5490
- **Take an Emotion Walk**, ECCV 2020, arXiv 1911.08708. https://arxiv.org/abs/1911.08708
- **ProxEmo** (Narayanan, Bhattacharya, Bera, Manocha), arXiv 2003.01062 (2020): gait emotion classifier for socially aware navigation; 82.47% mean AP on Emotion-Gait. https://arxiv.org/abs/2003.01062
- **Learning Gait Emotions Using Affective and Deep Features** (Randhavane et al.), ACM MIG 2022, best paper per Purdue. https://dl.acm.org/doi/pdf/10.1145/3561975.3562957 ; https://www.cs.purdue.edu/news/articles/2022/bera-best-paper-migs-2022.html. 80.07% on EWalk (per that news summary). EWalk = RGB-video-derived 3D poses with crowd-perceived emotion labels (not performer intent).
- Bhattacharya says current work is generating gestures/gaits per emotion (Text2Gestures, IEEE VR 2021): https://www.cs.umd.edu/~uttaranb/ . No gait-generation-for-robots outcome found.
Learned: classifiers on perceived labels. No robot gait generated; no emotion->parameter magnitudes published.

## 5. Datasets / benchmarks for emotional gait recognition (2024-2026)
No new 2025-2026 public emotional-gait dataset or benchmark paper was found. Found:
- Survey: **Emotion Recognition from Skeleton Data: A Comprehensive Survey**, arXiv 2507.18026, Jul 2025. https://arxiv.org/abs/2507.18026
- **CGTGait** - Zhou, Xiong, Lu, Lin, Feng; IJCB 2025; arXiv 2509.16623; Emotion-Gait + ELMD; code https://github.com/githubzjj1/CGTGait. https://arxiv.org/abs/2509.16623
- Self-supervised gait-based emotion representation learning, arXiv 2405.04900 (2024); uses E-Gait and Emilya walking subset (anger, neutral, joy, sadness). https://arxiv.org/pdf/2405.04900
- **Single-Stream Multi-Feature Fusion with Temporal Robustness for Gait Emotion Recognition (SV-GCN)** - Lyu, Quan, Ding, Wang; ICANN 2026; arXiv 2609.11680; E-Gait; states E-Gait is the only popular public set (2,177 real + 1,000 synthetic, 16 joints). https://arxiv.org/abs/2609.11680
- **Emilya** (Fourati & Pelachaud, LREC 2014; IEEE TAC 2018): 11 actors x 8 emotions x 7 actions, Xsens 28 joints at 120 Hz, with a perception study. https://aclanthology.org/L14-1295/ ; https://www.lisn.upsaclay.fr/publications/perception-of-emotions-and-body-movement-in-the-emilya-database/ . Licence/availability: UNVERIFIED.
- Understanding Emotional Body Expressions via LLMs, arXiv 2412.12581 (2024). https://arxiv.org/pdf/2412.12581 (not read).
Caution: E-Gait labels are perceived emotion on 16-joint skeletons with partly synthetic data; Emilya is acted with intent labels but small (11 actors).

## 6. Affective locomotion in HRI / other robots
- **Animated Cassie** - Zhongyu Li, Christine Cummings, Koushil Sreenath; IROS 2020; arXiv 2009.02846. Animation -> dynamics optimiser -> walking controller on Cassie hardware. https://arxiv.org/abs/2009.02846 (outside window).
- Sripathy et al. (V-A emotive motion on Cassie), Suguitan et al. (Blossom), Marmpena et al. (Pepper), HIAER: named only in PAMoR's related-work list - UNVERIFIED. https://arxiv.org/html/2608.28213v2
- **Enhancing the Mobile Humanoid Robot's Emotional Expression with Affective Vertical-Oscillations**, Int. J. Social Robotics 2024 (DOI 10.1007/s12369-024-01142-0). Springer page blocked; only a secondary news digest found; details UNVERIFIED. https://link.springer.com/article/10.1007/s12369-024-01142-0 . Predecessors: HRI 2020 and IEEE Access 2021 (android ibuki), hand-designed vertical oscillation + user study.
- **Emotional Motion Design and Evaluation of Humanoid Robot Based on PAD Model**, Int. J. Social Robotics, 2026 (DOI 10.1007/s12369-026-01428-5). Page blocked; contents UNVERIFIED. https://link.springer.com/article/10.1007/s12369-026-01428-5
- **Gait-to-Gait Emotional HRI ... Spatial-Temporal Transformer**, Sensors 2025, 25(3), 734: NAO walks with preset emotional gait in response to predicted human emotion (per search summary; MDPI page returned 403, so details UNVERIFIED). https://www.mdpi.com/1424-8220/25/3/734
- **Uni-Mo** (Liu, Gao, Qian, Liu, Cai, Tang), arXiv 2606.28237, Jun 2026: LLM + video diffusion -> quadruped (Go2) tracking policies; Quad-Imaginarium 7,488 motions / 18.5 h released; 96.7% real deployment success; "expressive/performative", no emotion measure. https://arxiv.org/abs/2606.28237
- Outside window: e-Inu (RL quadruped emotion in sim, 2023) https://arxiv.org/abs/2301.00964 ; Hainsworth 2017 thesis (hand-designed emotive quadruped gaits) https://collections.lib.utah.edu/details?id=1423247
- RO-MAN/HRI/IROS 2024-2026 coverage: no walking-emotion paper on NAO/Pepper/Go2 conclusively identified beyond the above; treat as a coverage gap (RO-MAN 2025 program, not read in full: https://ras.papercept.net/conferences/conferences/ROMAN25/program/ROMAN25_ContentListWeb_3.html).

## 7. MuJoCo + G1 + expressive motion
EmoPose (MuJoCo preview, gestures) is the only emotion + G1 + MuJoCo hit. MuJoCo G1 pipelines exist for tracking/sim2sim (https://arxiv.org/html/2605.12347v1 ; https://github.com/GaganBhusal/unitree-g1-motion-imitation) with no emotion. No analytic/parametric emotion-to-gait model on G1 found (search-based negative, not proof).

## 8. Positioning of emotive-gait
Today emotive-gait is analytic, literature-cited (Montepare, Michalak, Gross, Barliya, Halovic & Kroos, Randhavane), 3-axis (activation/valence/tension), rendered by task-space IK on simulated G1, no RL, no human study, signed directions with grounding levels, disgust/surprise flagged unsourced. The field splits into (a) learned generators with perceptual studies on real G1 (EMoG, PAMoR) that delegate balance to a learned tracker (SONIC/BeyondMimic), and (b) classifier work on E-Gait/EWalk/Emilya that never drives a robot. Nothing found publishes an interpretable parametric map from emotion to gait parameters with data-derived magnitudes.

**Genuinely new (if built and measured):**
1. Magnitudes fitted from PUBLIC mocap (Emilya; E-Gait/EWalk only with the label caveat) with confidence intervals, released openly. EMoG's dataset is proprietary/unreleased; PAMoR's V-A formula is uncalibrated by its own admission. An open, data-derived emotion -> gait-parameter table (cadence, trunk lean, arm swing, head pitch per emotion) was not found in anything reviewed.
2. Interpretability: effect sizes per parameter per emotion, versus latent codes (EMoG) or diffusion guidance (PAMoR); and the project's existing literature-cited signs become testable hypotheses.
3. A measurable recognition claim that needs no robot: a classifier on rendered skeletons with a cross-actor split, reported against chance and against the same classifier on source mocap. New for a parametric pipeline, but only a proxy for human perception.
4. A fully open MuJoCo+G1 stack (no proprietary data, no RL tracker) as a reproducible baseline.

**Re-implementation (do not oversell):**
- An emotion-conditioned gait generator trained on performer mocap and rated by humans is what EMoG already does (6 classes, 8 performers, n up to 141, six-way 61-65% executed). A black-box learned generator would be a re-implementation.
- A V-A knob on G1 is PAMoR's headline. A new gait-emotion classifier on E-Gait is a crowded field (CGTGait, SV-GCN, STEP, ProxEmo) and would be incremental.
- Style-from-reference (SMooDi, Bionic style transfer, 100STYLE) already covers stylised locomotion without emotion labels.

**Claims it must NOT make:**
- "First emotional humanoid gait" or "first emotion on G1": EMoG and PAMoR (Sep/Aug 2026) exist; EMoG used G1 per its full text.
- "Human-perceived emotion recognised" unless a human study is actually run. A classifier-on-skeleton score is machine recognition; E-Gait/EWalk labels are perceived-emotion labels on partly synthetic 16-joint data, not performer intent. Do not compare numbers with EMoG's 61-65% or PAMoR's 0.384 (different class sets, chance levels, raters, renderings); PAMoR itself calls its comparison to acted humans indirect.
- "Physically validated / balance-robust walking" while rendering in-place by IK on a simulated G1; the project docs say stride is not rendered and a free base falls in 0.74-2.25 s. EMoG/PAMoR run on hardware via learned trackers.
- "Data-derived" for any magnitude still hand-tuned; degree magnitudes from literature (project rule). Coverage of all six emotions: disgust and surprise lack sources, and Emilya (anger, anxiety, fear, joy, neutral, panic, sadness, shame) has no disgust/surprise, so the mapping from project categories needs explicit justification.
- Generalisation beyond the data (Emilya 11 actors; EMoG authors caution that 8 performers do not establish universal interpretations).
- "State of the art" against UMD classifiers: different task.
- README fixes suggested (no files changed): EMoG robot is G1 per full text; PAMoR is a learned latent-diffusion V-A generator, so "closest in approach" should be reworded; "Walk Like This" has no verifiable source; consider adding Bionic style transfer, SONIC, SMooDi, Emilya, Animated Cassie.

## Verification gaps
Springer/MDPI pages blocked; SONIC emotion conditioning, 100STYLE authorship, SMooDi user study, HOVER arXiv id, HRI venue coverage, per-paper code/data release of tracking papers, and PAMoR/EMoG supplementary materials were not verified. arXiv "Code" sections list only generic aggregators, so "no release stated" means not found, not confirmed absent.
