# Public datasets of emotional human gait and emotional body movement

Survey date: 2026-10-25. Purpose: training or validating a learned emotion-to-gait model for a simulated Unitree G1 (MuJoCo, 29 DoF).

Conventions:
- Every factual claim carries a URL. **UNVERIFIED** marks anything I could not confirm from a primary or near-primary source.
- "Works today" means I checked on 2026-10-25 with an HTTP HEAD request or a page fetch. I did not download or open any dataset archive.
- Nothing here is legal advice. The licence notes describe what the licence text says, not whether a given use is allowed.

---

## 1. Summary table

| # | Dataset (year, institution) | Modality / format | Walking? | Emotions / actors / size | Labels | Licence / access | Works 2026-10-25? |
|---|---|---|---|---|---|---|---|
| 1 | Roether, Omlor, Christensen & Giese 2009 (Tübingen) | Vicon 612, 41 PlugInGait markers, 120 Hz, 3-D marker positions plus computed joint angles | **Yes, walking only** (straight line, ~5 m, 3 reps) | anger, fear, happiness, sadness + neutral; 25 lay actors | perceptual classification + intensity ratings in the papers | **No public release found** (UNVERIFIED) | No download found |
| 2 | Body Movement Library, Ma, Paterson & Pollick 2006 (Glasgow) | Optical mocap, 3-D positions: CSM (Character Studio marker) files and PTD 15-joint point-light files, 60 Hz | **Yes** (walk is 1 of 4 actions, two directions) | angry, happy, sad, neutral; 30 non-professional actors; 4,080 movements | none in the library itself; perceived labels for BML walks exist in E-Gait/EWalk | open download, **no licence stated** | **Yes** (both zips HTTP 200, modified 2025-04-15) |
| 3 | Emilya, Fourati & Pelachaud 2014 (Telecom ParisTech) | Xsens MVN, 17 sensors / 23 segments, plus 4 HD videos | **Yes** (simple walk + walk carrying an object, back and forth) | joy, anger, panic fear, anxiety, sadness, shame, pride, neutral; 11 actors; ~7,084 single-trial sequences | Likert perception study (MTurk) | "will offer to the research community"; **no access route found** (UNVERIFIED) | No download found |
| 4 | KIT Whole-Body Human Motion Database (KIT) | Vicon C3D + MMM XML, 100 Hz | **Yes, but tiny**: 1 sad walk + 1 neutral walk entry (5 trials each), plus CMU imports | sad, neutral (native); 1 subject | none | free account required; no data licence stated | Site up |
| 5 | Emotion-Gait / E-Gait, Bhattacharya et al. 2020 (UMD GAMMA) | 3-D positions, 16 joints, sequences of 75 time steps, HDF5 | **Yes, gait only** | happy, sad, angry, neutral; 2,177 real gaits (342 compiled + 1,835 from ELMD) + synthetic | perceived labels, 10 annotators per gait (per survey) | Google Drive folder, **no dataset licence stated** (code is MIT) | **Yes** (folder lists files) |
| 6 | EWalk, Randhavane, Bhattacharya et al. 2019 (UMD) | RGB video → 3-D pose (16 joints) | **Yes** | happy, angry, sad, neutral; 24 subjects; 1,136 video-extracted gaits (1,384 in total with mocap sources) | MTurk perceived labels + valence/arousal | paper calls it "public domain"; **no download URL found** (UNVERIFIED) | No download found |
| 7 | CMU Graphics Lab MoCap (CMU) | Vicon, 41 markers, 120 Hz; ASF/AMC joint angles + C3D | **Yes, a few single-trial emotional walks** | e.g. sad, happy, depressed, elated, scared, shy, confident, stern (subjects 82, 91, 104, 105, 142) | none native; some CMU walks have perceived labels in E-Gait/EWalk | "free for all uses", no reselling the data | **Yes** (HTTP 200; TLS certificate chain incomplete) |
| 8 | GEMEP, Bänziger, Mortillaro & Scherer 2012 (Geneva CISA) | Audio-video | **No** | 18 affective states; 10 actors; Core Set of 145 files | validation ratings | senior academics only, no public display, IP stays with UNIGE | Page up |
| 9 | MPI Emotional Body Expressions DB, Volkova et al. 2014 (MPI Tübingen) | Xsens MVN, 120 fps, BVH (22 joints + root) / MVNX (23 segments) | **No** (actors seated) | 11 categories; 8 amateur actors; >1,400 sequences | intended + observer labels | BVH on figshare **CC BY 4.0**; interactive site closed | **Partly** (figshare BVH yes, site "no longer available") |
| 10 | BoLD, Luo et al. 2020 (Penn State) | In-the-wild movie video clips | **Not gait-specific** | 26 categories + valence/arousal/dominance; 9,876 clips, 13,239 instances | crowd-sourced | login + email approval by Prof. Wang; **no licence stated** | Page up; approval email flow reported broken |
| 11 | KDAE, Zhang et al. 2020 (Scientific Data) | Noitom Perception Neuron, 17 IMUs, 125 Hz, BVH (72 nodes) | **No** (1 m x 1 m stage; neutral includes "marking time") | 6 basic + neutral; 22 actors; 1,402 trials | 7-AFC validation by 70 students | PhysioNet **Restricted Health Data Licence 1.5.0**, signed DUA | Page up (restricted) |
| 12 | 100STYLE, Mason, Starke & Komura 2022 (Edinburgh) | Xsens, 60 fps, BVH, 28-bone skeleton | **Yes, locomotion only** (forward/back/sidestep walk + run, idle, transitions) | 100 styles incl. Angry, Depressed, Elated, Proud, Neutral; 1 actor; >4 M frames | none | **CC BY 4.0**, open | **Yes** (Zenodo 200, 1.47 GB BVH zip) |
| 13 | Bandai Namco Research Motiondataset-1/-2 (2022) | BVH, 30 fps | **Yes** (ds-1: walk, walk-back/left/right in all 15 styles) | ds-1 styles incl. happy, sad, angry, proud, not-confident, tired; 3 professional actors | none | **CC BY-NC 4.0** (repo LICENSE) | **Yes** (GitHub) |
| 14 | EMoG dataset, Lu et al. 2026 (Nanjing Univ.) | SMPL, 100 fps | **Yes, gait only** | happy, confident, sad, scared, shy, neutral; 8 professional performers; 12,243 clips | intensity-graded | **Not released** ("Code (Soon)") | No |
| 15 | BME, Hicheur et al. 2013 (Paris) | Vicon V8, 43 landmarks, 120 Hz | **Yes** (U-shaped path) | neutral, joy, anger, sadness, fear; 8 actors; ~200 trials | — | **no access route found** (UNVERIFIED) | No download found |
| 16 | Halovic & Kroos 2017 (Mendeley Data) | SPSS summary tables only | derived from walking | basic emotions (not listed on the page) | perception data | **CC BY 4.0** | Yes, but only ~16 KB `.sav` files, no trajectories |
| 17 | Motorica Dance | mocap, 120 fps | **No** | 8 dance styles, no emotion labels found | — | UNVERIFIED | not checked |

Walking content at a glance:
- **Contain walking:** Roether, BML, Emilya, KIT (tiny), E-Gait, EWalk, CMU (few), 100STYLE, Bandai Namco, EMoG (unreleased), BME.
- **Do not contain walking:** GEMEP, MPI EBMDB (seated), KDAE (stationary stage), Motorica Dance. BoLD has no gait-specific content.
- **Human3.6M** has walking but no emotion labels. Its walks appear only inside E-Gait/EWalk, relabelled by perception (14 walking videos per [EWalk paper](https://arxiv.org/abs/1906.11884)). I found no standalone "emotional Human3.6M" subset.

---

## 2. Dataset details

### 2.1 Roether / Giese emotional gait (Tübingen, 2009)
- **Paper:** Roether, Omlor, Christensen & Giese, "Critical features for the perception of emotion from gait", *Journal of Vision* 9(6), 2009, pp. 1–32 (citation as given in a later reference list: [Springer chapter refs](https://link.springer.com/10.1007/978-3-642-36368-9_6)). The data and methods are written up in Roether's dissertation: [thesisroether2011.pdf](https://www.compsens.uni-tuebingen.de/compsens//files/publications/thesisroether2011.pdf).
- **Recording** (dissertation §2.1, same URL):
  - 25 individuals: 13 with lay-theatre experience and 12 left-handed novices.
  - Emotions were induced by imagination plus expressive gesture and voice. For fear, only slower-than-normal fearful gait was kept.
  - Straight-line walking over about 5 m (about six step cycles), each condition repeated 3 times: neutral first, then anger, fear, happiness and sadness in counterbalanced order. A subset of actors also walked neutral at four speeds.
  - Eight-camera Vicon 612, 120 Hz, 41 PlugInGait markers. Joint centres were estimated and joint angles computed in MATLAB.
- **Labels:** perception experiments (classification and intensity) are in the dissertation and paper. The dissertation reports categorisation accuracies of 70% to >98% for its movement database (same URL).
- **Access:** I found no public download, licence or request form. The Tübingen lab pages describe the project ([compsens project 19](https://www.compsens.uni-tuebingen.de/compsens/index.php/people/project/19)) but link no data. **UNVERIFIED / likely by request only.**
- **Uses:** cited as a foundational emotional-gait feature study by EWalk ([arXiv 1906.11884](https://arxiv.org/abs/1906.11884)) and Emilya ([LREC 2014](https://aclanthology.org/L14-1295/)).

### 2.2 Body Movement Library (BML), Ma, Paterson & Pollick 2006 (Glasgow)
- **Paper:** "A motion capture library for the study of identity, gender, and emotion perception from biological motion", *Behavior Research Methods* 38(1):134–141, [doi:10.3758/BF03192758](https://doi.org/10.3758/BF03192758). There are 30 non-professional actors (15 female). They performed walking, knocking, lifting and throwing in angry, happy, neutral and sad styles, giving 4,080 movements ([UEA record](https://ueaeprints.uea.ac.uk/id/eprint/90416)).
- **Walking:** included. The data page names walk files "record 1 = anti-clockwise, record 2 = clockwise" ([PACO Data page](https://paco.psy.gla.ac.uk/?page_id=14973)). The EWalk paper describes the BML walks as walking in a triangle for 30 s in each direction, 4 styles x 2 directions = 240 gaits ([arXiv 1906.11884](https://arxiv.org/abs/1906.11884)).
- **Format** ([PACO Data page](https://paco.psy.gla.ac.uk/?page_id=14973)):
  - CSM files (Character Studio marker format) at 60 Hz, with per-actor `.cal`/`.fig` body-dimension files.
  - PTD point-light files at 60 Hz: frame number plus 45 columns, the x, y, z of 15 joints.
  - A 2025 survey names the capture system as Falcon Analog optical mocap ([arXiv 2507.18026](https://arxiv.org/abs/2507.18026)). That survey's table rows are misaligned, so its joint count and frame rate for BML are unreliable. The 60 Hz figure comes from the data page itself.
- **Labels:** the library contains acted (intended) labels only. Perceived labels for the BML walks were later collected by the UMD group in EWalk and E-Gait.
- **Access and licence:**
  - Open download links: `full_set_of_csm_data.zip` (1,015,636,919 bytes) and `full_set_of_ptd_data.zip` (263,914,453 bytes).
  - Both returned HTTP 200 with Last-Modified 2025-04-15 on 2026-10-25 ([csm](http://paco.psy.gla.ac.uk/wp-content/uploads/2018/05/full_set_of_csm_data.zip), [ptd](http://paco.psy.gla.ac.uk/wp-content/uploads/2018/05/full_set_of_ptd_data.zip)).
  - **The data page states no licence, terms or citation requirement** ([PACO Data page](https://paco.psy.gla.ac.uk/?page_id=14973)). Absent a licence, the default is "all rights reserved", so ask the PACO lab before committing derived parameters to a public repository.
- **Uses:** a source for E-Gait subset A ([arXiv 2507.18026](https://arxiv.org/abs/2507.18026)) and EWalk training ([arXiv 1906.11884](https://arxiv.org/abs/1906.11884)).

### 2.3 Emilya, Fourati & Pelachaud (LREC 2014; IEEE TAC 2018)
- **Papers:** [LREC 2014 L14-1295](https://aclanthology.org/L14-1295/) ([PDF](https://aclanthology.org/L14-1295.pdf), [HAL](https://hal.telecom-paris.fr/hal-02287074v1)) and the perception study "Perception of emotions and body movement in the Emilya database", IEEE TAC 2018 ([HAL](https://hal.archives-ouvertes.fr/hal-02287454)).
- **Content** (all from the [LREC PDF](https://aclanthology.org/L14-1295.pdf)):
  - 11 graduate-student actors (6 F, 5 M), trained by a professional acting director.
  - Emotions: Joy, Anger, Panic Fear, Anxiety, Sadness, Shame, Pride, Neutral, induced by written scenarios.
  - 7 tasks, including simple walking and walking with an object in hand, back and forth along the long side of the room. Each task was repeated 4 times.
  - Xsens MVN: 17 sensors, 23 segments, plus 4 synchronised HD cameras.
  - About 7,084 single-trial mocap sequences and 176 continuous sequences. Walk segmentation was automatic.
  - A 2025 survey counts 8,206 samples and gives 120 Hz in its text and 125 Hz in its table ([arXiv 2507.18026](https://arxiv.org/abs/2507.18026)), so the frame rate is **UNVERIFIED**.
- **Labels:** an MTurk Likert study on avatar renderings. Anger, Sadness, Neutral, Joy and Panic Fear were recognised significantly. Shame was confused with Sadness and Pride with Joy ([LREC PDF](https://aclanthology.org/L14-1295.pdf)).
- **Access:** the LREC paper says the authors "will offer this database to the research community" after further analyses ([LREC PDF](https://aclanthology.org/L14-1295.pdf)). I found no download page, licence or request form (searches on ACL Anthology, HAL and LISN). **UNVERIFIED; treat as contact-the-authors.**
- **Uses:** heavily used for emotion recognition (Fourati et al., Crenn et al.), summarised in [arXiv 2507.18026](https://arxiv.org/abs/2507.18026). Its walking condition with four emotions was used for gait emotion recognition ([arXiv 2405.04900](https://arxiv.org/pdf/2405.04900)).

### 2.4 KIT Whole-Body Human Motion Database (emotion subset)
- **Database:** Vicon C3D and MMM-model XML, both at 100 Hz. Downloads need a free account. The aim is free availability to the scientific community, and some files (non-anonymised video) are protected. Users are asked to cite Mandery et al. The FAQ gives no data licence ([KIT FAQ](https://motion-database.humanoids.kit.edu/faq)).
- **Emotion content (native KIT):** I scanned the database listing pages. The only entries are:
  - #961 "Emotional Motion: Neutral" (forward walk) and #962 "Emotional Motion: Sad" (forward walk), 5 C3D trials each ([list p.108](https://motion-database.humanoids.kit.edu/list/motions/?page=108)).
  - #963/#964, neutral and sad right-arm waves by subject #583 ([list p.109](https://motion-database.humanoids.kit.edu/list/motions/?page=109)).
  - The site's emotion filter offers only "neutral" and "sad".
- **CMU imports:** the database also mirrors CMU subjects:
  - #1089 "jumping; pushing; emotional walks" (CMU subject 82, [KIT #1089](https://motion-database.humanoids.kit.edu/details/motions/1089/?listpage=3)).
  - #1027 "Stylized Walks" (CMU subject 142) and #1007 "Stylized Motions" (CMU subject 137) ([list p.146](https://motion-database.humanoids.kit.edu/list/motions/?page=146)).
  - See §2.7.
- **Verdict:** walking yes, but one subject and one emotion. Useful only as a sad-vs-neutral sanity check.

### 2.5 Emotion-Gait (E-Gait), Bhattacharya et al., AAAI 2020 (UMD GAMMA)
- **Paper:** STEP ([AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/5490), [arXiv 1910.12906](https://arxiv.org/abs/1910.12906)).
- **Content** ([arXiv HTML](https://arxiv.org/html/1910.12906v3)):
  - 2,177 real gaits: 342 recorded or compiled by the authors plus 1,835 from the Edinburgh Locomotion MOCAP Database (ELMD). There are also synthetic STEP-Gen gaits; the count is reported inconsistently.
  - Emotions: angry, neutral, happy, sad.
  - Each gait is 16 joints x 3-D position x 75 time steps. The frame rate is not stated in the paper.
  - The AAAI version reports 4,227 human gaits (per [AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/5490) as summarised by search), which is inconsistent with the arXiv version.
- **Composition and labels:** a 2025 survey says subset A draws on BML, ICT, CMU-MoCap and Human3.6M. It says each gait was labelled by 10 annotators with majority vote, and gives 21 captured joints standardised to 16 and 60 Hz ([arXiv 2507.18026](https://arxiv.org/abs/2507.18026)). The original paper only says "domain experts" labelled them, so **the annotator count is from the survey, not the primary paper.**
- **ELMD origin:** cited as Habibie et al., BMVC 2017 ([survey refs](https://arxiv.org/abs/2507.18026)). The Edinburgh locomotion data is distributed inside Holden et al.'s `motionsynth_data.zip` (5.65 GB, HTTP 200). Its README says the data is "free for academic and non-commercial purposes" and that the constituent databases keep their own licences ([Holden page](https://theorangeduck.com/page/deep-learning-framework-character-motion-synthesis-and-editing)). That ELMD is exactly this Edinburgh subset is **UNVERIFIED**.
- **Access:**
  - The STEP repo README links the dataset at [go.umd.edu/emotion-gait](https://go.umd.edu/emotion-gait), which redirects to a public [Google Drive folder](https://drive.google.com/drive/folders/1g9fv2jwwBZrAl88qbzV8e_X6qKn4TVkh).
  - The folder lists `features.h5` (4.8 MB), `features_ELMD.h5` (162 MB), `labels.h5`, `labels_ELMD.h5`, `labels_edin_locomotion.zip`, the CVAE synthetic sets and `README.md`. I could not read the README without a Google sign-in.
  - The code repo is MIT ([STEP repo](https://github.com/UttaranB127/STEP)), last pushed 2022-12-08. **No dataset licence is visible.**
- **Uses:** the standard benchmark for gait emotion recognition (STEP, Take an Emotion Walk ECCV 2020 [arXiv 1911.08708](https://arxiv.org/abs/1911.08708), MSA-GCN [arXiv 2209.08988](https://arxiv.org/pdf/2209.08988), and others in [arXiv 2507.18026](https://arxiv.org/abs/2507.18026)).

### 2.6 EWalk, Randhavane, Bhattacharya et al. 2019 (UMD)
- **Content** ([arXiv 1906.11884](https://arxiv.org/abs/1906.11884)):
  - 24 campus subjects (16 M, 8 F), filmed indoors and outdoors, walking towards a camera from 7 m and asked to walk "as if" experiencing an emotion.
  - Videos were cropped to one walk cycle. 3-D poses (16 joints) were extracted with TimePoseNet.
  - 1,136 video gaits. Together with Human3.6M, CMU, ICT, BML and SIG gaits, the total is 1,384.
  - Labels: perceived happy, angry, sad or neutral from 688 MTurk participants (at least 10 responses per video), plus valence/arousal.
- **Access:** the paper calls it "a new public domain dataset" but gives no URL. I found no download location, so it is **UNVERIFIED and probably not separately downloadable**. It may overlap with E-Gait's 342-gait subset (**UNVERIFIED**).
- **Note:** it is video-derived 3-D pose from a single camera, so lower fidelity than mocap.

### 2.7 CMU Graphics Lab Motion Capture Database
- **System:** 12 Vicon MX-40 cameras at 120 Hz, 41 markers. Files are ASF/AMC (joint angles; lengths in inches, multiply by 2.54/100) and C3D ([CMU FAQ](http://mocap.cs.cmu.edu/faqs.php)).
- **Emotional walks** (one trial each, single actor per subject; [CMU search](http://mocap.cs.cmu.edu/search.php?subjectnumber=%25&motion=sad)):
  - Subject 82: confident, sad, "happy or fast" walk forward ([s82](http://mocap.cs.cmu.edu/search.php?subjectnumber=82)).
  - Subjects 91 and 105: SadWalk, DepressedWalk, ShyWalk, ScaredWalk, MachoWalk, TooCoolWalk and more ([s91](http://mocap.cs.cmu.edu/search.php?subjectnumber=91), [s105](http://mocap.cs.cmu.edu/search.php?subjectnumber=105)).
  - Subject 104: HappyGoLuckWalk, ExcitedWalk, SternWalk ([s104](http://mocap.cs.cmu.edu/search.php?subjectnumber=104)).
  - Subject 142 "Stylized Walks", 22 trials including Depressed, Elated, Happy, Joy, Sad, Scared x2, Shy, Cool, Rushed ([s142](http://mocap.cs.cmu.edu/search.php?subjectnumber=142)).
  - "Angry" appears only as hand gestures in a quarrel (subjects 18/19), not as a walk.
- **Licence:** the home page says the data is "free for all uses" and "free for use in research projects". It may be included in commercially sold products but "you may not resell this data directly, even in converted form", and a citation and acknowledgement are requested ([CMU home](http://mocap.cs.cmu.edu/)). This is the most permissive licence among the walking sets alongside 100STYLE.
- **Works today:** subject pages and `142_15.amc` returned HTTP 200, but the server's TLS chain fails strict verification (certificate error in a standard fetcher).
- **Uses:** 49 CMU stylised walks were used in EWalk/E-Gait with perceived labels ([arXiv 1906.11884](https://arxiv.org/abs/1906.11884)).

### 2.8 GEMEP (Geneva)
- **Content:** audio-video portrayals by 10 actors of 18 affective states. The distributed Core Set has 145 files ([GEMEP Core Set](https://www.unige.ch/cisa/gemep/coreset), [GEMEP](https://www.unige.ch/cisa/gemep)). Face and upper-body portrayals, **no walking**.
- **Access:** free for research, but only for researchers with a stable academic appointment (assistant professor or above). No public display in talks or articles; IP stays with UNIGE ([Core Set terms](https://www.unige.ch/cisa/gemep/coreset)).
- **Verdict:** unsuitable for gait and incompatible with a public repository.

### 2.9 MPI Emotional Body Expressions Database (Volkova et al., PLoS ONE 2014)
- **Content** ([PMC4252031](https://pmc.ncbi.nlm.nih.gov/articles/PMC4252031/)):
  - 8 amateur actors narrating stories, captured with an Xsens MVN suit (17 sensors) at 120 fps.
  - Files: BVH (22 joints + root) and MVNX (23 segments).
  - 11 categories: amusement, joy, pride, relief, surprise, anger, disgust, fear, sadness, shame, neutral.
  - Over 1,400 sequences, with intended plus observer labels.
  - **Actors were seated on a stool: no walking.**
- **Access:**
  - The BVH archive is on figshare under **CC BY 4.0**: `bvh.zip`, 50 MB, published 2014-10-28 ([figshare API record](https://api.figshare.com/v2/articles/1220428), [figshare page](https://figshare.com/articles/MPI_EMBM_Database_Mocap_Files/1220428)).
  - The interactive site [ebmdb.tuebingen.mpg.de](http://ebmdb.tuebingen.mpg.de/) now says "Website closed. This database is no longer available."
- **Verdict:** permissive and downloadable, but upper-body only. Possibly useful for neck pitch and arm posture cues, not gait.

### 2.10 BoLD, Luo et al., IJCV 2020 (Penn State)
- **Content:** movie clips; 9,876 clips and 13,239 human instances (the page also says 9,827). Labels are 26 categorical emotions plus valence/arousal/dominance ([BoLD dataset page](https://cydar.ist.psu.edu/emotionchallenge/dataset.php)).
- **Access:** log in, then email Prof. James Wang for manual approval. The site notes that a server update blocked automatic confirmation emails. Users must cite ARBEE (IJCV 128(1)) ([BoLD index](https://cydar.ist.psu.edu/emotionchallenge/index.php)). **No licence text is visible.**
- **Walking:** not gait-specific; any walking would be incidental in-the-wild footage (**UNVERIFIED**). It is 2-D video and would need 3-D pose estimation.

### 2.11 KDAE, Zhang et al., Scientific Data 2020
- **Content** ([PMC7478954](https://pmc.ncbi.nlm.nih.gov/articles/PMC7478954/)):
  - Noitom Perception Neuron, 17 IMUs, 125 Hz, BVH with 72 nodes.
  - 22 semi-professional actors, 1,402 trials (mean 7.22 s).
  - Emotions: happiness, sadness, anger, fear, disgust, surprise, neutral.
  - 7-AFC validation with 70 students.
- **Walking:** **no.** Performances were on a 1 m x 1 m stage about 0.5 m from a wall. The authors criticise earlier datasets for using walking for all emotions. One neutral scenario is "marking time", i.e. stepping in place ([PMC7478954](https://pmc.ncbi.nlm.nih.gov/articles/PMC7478954/)).
- **Access:** PhysioNet, **Restricted Health Data License 1.5.0**, with a registered account and a signed DUA ([PhysioNet v2.1.0](https://physionet.org/content/kinematic-actors-emotions/2.1.0/)). Incompatible with publishing derived data without checking the DUA.

### 2.12 100STYLE, Mason, Starke & Komura 2022
- **Content** ([100STYLE page](https://www.ianxmason.com/100style/)):
  - A single actor for all 100 styles.
  - For each style: forward, backward and sidestep walking and running, idling, and transitions (files `<Style>_FW.bvh`, `_BW`, `_SW`, `_FR`, `_BR`, `_SR`, `_ID`, `_TR*`).
  - Xsens at 60 fps, BVH, 28-bone skeleton (Hips, Chest..Chest4, Neck, Head, collar/shoulder/elbow/wrist, hip/knee/ankle/toe).
  - Over 4 M frames in total ([arXiv 2201.04439](https://arxiv.org/abs/2201.04439)).
- **Emotion-like styles** in the style table ([100STYLE page](https://www.ianxmason.com/100style/)):
  - Emotion styles: **Angry, Depressed, Elated, Proud**, plus **Neutral** as a baseline.
  - Affect-adjacent styles: Old, Rushed, Stiff, Strutting, Heavyset, Crouched, LookUp, BentForward, LeanBack.
  - There is **no** Sad, Happy or Fear style by those names.
- **Labels:** style names only, no perceptual ratings.
- **Licence:** **CC BY 4.0**. The page asks for citation in academic work and a credit such as "The 100STYLE Dataset - Ian Mason" for creative or commercial work ([100STYLE page](https://www.ianxmason.com/100style/), [Zenodo 8127870](https://zenodo.org/record/8127870)).
- **Works today:** `100STYLE.zip` returned HTTP 200 with content-length 1,468,515,354. The labelled data zip is 14.8 GB ([Zenodo](https://zenodo.org/record/8127870)).

### 2.13 Bandai Namco Research Motiondataset (2022)
- **Content** ([repo README](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset)):
  - BVH plus JSON at 30 fps, from 3 professional actors.
  - Dataset-1: 175 entries, 36,673 frames, 15 styles: normal, happy, sad, angry, proud, not-confident, masculinity, feminine, childish, old, tired, active, musical, giant, chimpira ([ds-1 README](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset/blob/master/dataset/Bandai-Namco-Research-Motiondataset-1/README.md)).
  - Dataset-2: 2,902 entries, 384,931 frames, 7 styles (active, elderly, exhausted, feminine, masculine, normal, youthful) with walk and walk-turn contents but no emotion styles.
- **Walking:** the dataset-1 repository tree contains `walk_<style>_001/002.bvh` for all 15 styles, plus single walk-back, walk-left and walk-right files per style. That gives 2 forward-walk clips each for happy, sad, angry, proud, not-confident and tired ([GitHub tree API](https://api.github.com/repos/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset/git/trees/master?recursive=1)).
- **Licence:** the README and the per-dataset LICENSE file say **CC BY-NC 4.0** ("Attribution-NonCommercial 4.0 International", [LICENSE](https://raw.githubusercontent.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset/master/dataset/Bandai-Namco-Research-Motiondataset-1/LICENSE)). A 2022 press article said CC BY-NC-ND ([CG Channel](https://www.cgchannel.com/?p=136588)), so the licence appears to have been relaxed; the repository text governs. The ds-1 README separately shows "© 2022 Bandai Namco Research Inc. All Rights Reserved."
- **Works today:** the GitHub repo is live (last push 2023-07-04).

### 2.14 2023–2026 additions
- **EMoG (Lu et al., arXiv 2609.14432, Sep 2026, Nanjing University)** ([arXiv](https://arxiv.org/abs/2609.14432), [HTML](https://arxiv.org/html/2609.14432)):
  - An emotion-annotated gait dataset from 8 professional performers, covering six styles (happy, confident, sad, scared, shy, neutral) with a continuous intensity code.
  - About 2.25 M frames and 12,243 clips at 100 fps, in SMPL.
  - Retargeted with GMR to a **Unitree G1** and tracked by an RL policy; MuJoCo is used for a baseline.
  - It is the most directly relevant work to this project, but **the dataset is not released**. The project page says "Code (Soon)" and gives no data link ([project page](https://jtthhhh.github.io/Emog-web/)).
- **BME (Hicheur et al. 2013):** 8 professional actors walking a U-shaped path with neutral, joy, anger, sadness and fear (≥5 trials each, ~200 samples). Vicon V8, 120 Hz ([chapter](https://doi.org/10.1007/978-3-642-36368-9_6); details per [arXiv 2507.18026](https://arxiv.org/abs/2507.18026)). **No public access route found.**
- **EMOGAIT (Sheng & Li, Pattern Recognition 2021):** 60 subjects, 1,440 RGB-video gait sequences (2-D skeletons, 16 joints), induced by film clips; happy, sad, neutral, angry (per [arXiv 2507.18026](https://arxiv.org/abs/2507.18026)). **Access UNVERIFIED.**
- **Halovic & Kroos (2017):** Vicon-recorded emotional walking with perception data, but the public deposit is two ~16 KB SPSS `.sav` files of analysed kinematics, not trajectories. **CC BY 4.0** ([Mendeley x8zr2b2gmt](https://data.mendeley.com/datasets/x8zr2b2gmt/1), [Mendeley 76bfnphwsm](https://data.mendeley.com/datasets/76bfnphwsm)). It could serve as a citable effect-size check if the variables are documented in the linked Human Movement Science papers.
- **Motorica Dance:** 8 dance styles, 120 fps; no emotion labels found and no walking ([arXiv 2511.20469](https://arxiv.org/pdf/2511.20469)). Licence **UNVERIFIED**.
- **AMASS:** re-hosts CMU and KIT (and with them the emotional walks above) in SMPL under a "non-commercial scientific research" licence ([AMASS licence](https://amass.is.tue.mpg.de/license.html)). The CMU originals are more permissive.
- **PHS Motion Library (Filmakademie):** CC BY-NC-SA 4.0, no redistribution ([terms](https://animationsinstitut.de/en/research/terms-of-use/terms-of-use-phs-motion-library)). Emotional or walking content **UNVERIFIED**.
- **Not found:** I found no 2023–2026 *released* emotional-walking mocap dataset beyond those above. Searches covered EMoG, the 2025 skeleton-emotion survey ([arXiv 2507.18026](https://arxiv.org/abs/2507.18026)) and generic searches.

---

## 3. What a builder can actually get

Scoring is (a) walking content, (b) 3-D skeleton suitability for retargeting to the 29-DoF G1, (c) licence fit for a public MIT repo that commits **derived parameters only**, and (d) whether the download works today.

| Rank | Dataset | (a) Walking | (b) 3-D skeleton | (c) Licence for public derived params | (d) Works today |
|---|---|---|---|---|---|
| 1 | **100STYLE** | Good: FW/BW/SW walks per style, long clips; but 1 actor, 4 emotion styles (Angry, Depressed, Elated, Proud) + Neutral | Excellent: BVH 60 fps, 28 bones, rotations | **Best: CC BY 4.0**, attribution only | **Yes** |
| 2 | **CMU MoCap emotional walks** | Thin: ~1 trial per style; subjects 82, 91, 104, 105, 142 | Excellent: ASF/AMC joint angles, 120 Hz | **Very good**: free for all uses; no reselling the raw data | **Yes** (TLS quirk) |
| 3 | **Body Movement Library (BML)** | Strong: 30 actors x 4 emotions x 2 directions, ~30 s each | Good: 3-D positions (15-joint PTD / CSM), 60 Hz; no rotations in PTD | **Unclear: no licence stated**, so ask PACO lab; facts-level aggregates are lower risk (not legal advice) | **Yes** |
| 4 | **E-Gait** | Strong: 2,177 gaits, perceived labels | Fair: 16-joint positions, 75 steps, frame rate unstated | **Unclear**: no dataset licence; ELMD part derived from academic/non-commercial data | **Yes** (Drive) |
| 5 | **Bandai Namco ds-1** | Moderate: 2 forward walks x happy/sad/angry/proud/not-confident/tired + side/back walks | Excellent: BVH 30 fps | **CC BY-NC 4.0**: non-commercial; an MIT repo allows commercial reuse, so publishing derived parameters is a licence-compatibility question | **Yes** |
| 6 | Emilya | Strong (8 emotions, 11 actors, walking 2 ways) | Excellent (Xsens) | Unknown | **No route found** |
| 7 | Roether/Giese | Strong (25 actors, gait-only, 41 markers) | Excellent | Unknown | **No route found** |
| — | EMoG | Ideal (G1, intensities) | SMPL | Unknown | **Not released** |
| — | MPI EBMDB, KDAE, GEMEP, BoLD, Motorica | **No walking** | — | — | — |

### Top three in practice

**1. 100STYLE (CC BY 4.0, works).** Use it as the main source for the arousal/valence directions it covers: Angry, Elated (high activation), Depressed (low activation, negative valence), Proud (posture) and Neutral, from one actor. The limitation is a single performer and no fear or sadness by name.

Pipeline:
1. Parse the BVH and run forward kinematics on the 28-bone skeleton to get world joint positions at 60 fps.
2. Trim with `Frame_Cuts.csv`.
3. Use only `_FW` files.
4. Detect heel strike and toe off from foot (Ankle/Toe) height and velocity thresholds.
5. Per gait cycle, extract:
   - **cadence**: steps/s between contralateral strikes
   - **step length**: horizontal ankle distance at strike
   - **step height**: peak swing-toe clearance
   - **trunk pitch**: sagittal angle of the Hips→Chest4/Neck vector from vertical
   - **neck pitch**: Head relative to Chest4, sagittal
   - **arm swing amplitude**: shoulder sagittal flexion range, or wrist AP excursion
   - **elbow flexion**: mean and range of the elbow angle
   - **lateral sway**: Hips ML peak-to-peak
   - **bounce**: Hips vertical peak-to-peak
6. Express each feature as **emotion minus Neutral** for the same actor. Scale lengths by leg length (hip-to-ankle) before mapping to the G1. Relative deltas sidestep most skeleton-mismatch problems.

No IK retargeting to the G1 is needed for parameter extraction. Full motion retargeting (e.g. GMR as in EMoG) is only needed for imitation-style training.

**2. CMU MoCap emotional walks (free for all uses, works).** About 15–20 usable emotional walk trials across 5 subjects, including Sad, Depressed, Scared, Shy, Happy, Elated, Joy and Stern, with matching Normal or Neutral walks per subject.

Pipeline:
1. ASF/AMC already gives joint angles. Run FK with the subject's ASF and convert inches to metres (×2.54/100).
2. Apply the same gait-event detection and feature list as above.
3. Compute deltas against the same subject's normal walk.

The main value is filling the fear and sadness gaps in 100STYLE. With one trial per style it is validation-grade, not training-grade.

**3. Body Movement Library (works; licence unstated).** This is the only openly downloadable multi-actor (30) emotional walking set, with angry, happy, sad and neutral, which makes it the best basis for population means and variance.

Pipeline:
1. Use the PTD files: 15-joint 3-D positions at 60 Hz, no rotations.
2. Compute angles geometrically from positions:
   - trunk pitch from the pelvis-to-neck vector
   - neck pitch from the neck-to-head vector relative to the trunk
   - elbow flexion from the shoulder-elbow-wrist angle
   - arm swing from the wrist's AP excursion relative to the pelvis
3. Walks are around a closed path, so segment out the straight portions or rotate into a path-tangent frame per frame before computing AP/ML quantities. Use the start/end frame files.
4. Cadence and step parameters come from ankle-joint events.
5. Compute per-actor emotion minus neutral deltas.

Before committing derived parameters to the MIT repo, email the PACO lab (Pollick) for written permission or a licence statement, because the data page states none ([PACO Data page](https://paco.psy.gla.ac.uk/?page_id=14973)).

### Also worth noting
- **E-Gait** is the only large set with *perceived* (observer) labels on gaits, which suits validating that the robot's output reads as the intended emotion. Its 16-joint, 75-step clips with an unstated frame rate make cadence unrecoverable without assumptions. Use it for posture and relative arm-swing features, not timing. The licence is unclear.
- **Bandai Namco ds-1** has clean BVH and covers happy, sad, angry, proud, not-confident and tired walks. CC BY-NC 4.0 makes publishing derived parameters in an MIT (commercial-permitting) repo a licence question to resolve first.
- **Emilya, Roether and EMoG** would be the best gait sources (multi-actor, many emotions, gait-specific; EMoG is even G1-retargeted), but none has a working public download as of 2026-10-25. Emailing the authors is the only route.
- **Coverage gap:** none of the downloadable, permissively licensed sets covers **disgust or surprise during walking**, and fear is covered only by CMU single trials (Scared, ×2 in subject 142, plus subjects 91/105). This matches the project's existing "unsourced" flag for disgust and surprise.
