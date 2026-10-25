# Phase 4.5 Learned Emotion Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A data-derived emotion model beside the analytic one: gait parameters extracted from public emotional-walking mocap on any skeleton, a mixed-effects fit with held-out-actor validation against the hand-tuned gains, a three-axis structure test, and a selectable `LearnedEmotionModel` under the existing cited-direction tests, with every number committed as a provenance-carrying aggregate.

**Architecture:** `mocap/` reads BVH, ASF/AMC and PTD into a `SkeletonClip`, detects gait events in a path frame, and measures the 14 measurable `GaitParameters` fields per cycle through a `LandmarkMap` (no robot anywhere). `learn/` builds a per-cycle table from a hashed manifest, fits one `statsmodels` `MixedLM` per parameter, runs leave-one-actor-out against the hand-tuned gains and a zero predictor, runs the structure test, and writes the artefact and report. `emotion/learned.py` loads the artefact and re-dimensions it through `RobotProfile.leg_length`; `emotion/model.py` gains an `EmotionModel` protocol and a `model=` argument on `modulate`. `cli.py` gains `fit`, `fetch`, `--model` and `evidence --model`.

**Tech Stack:** Python >= 3.11, NumPy 2, SciPy, `pyyaml` (runtime); `statsmodels>=0.14` and `pandas>=2` in a `fit` extra (included in `dev`); `mujoco>=3.17,<3.18` untouched; pytest, ruff 0.17.2.

**Spec:** `docs/superpowers/specs/2026-10-25-phase-4-5-learned-emotion-model-design.md` (approved 2026-10-25). The parent design `2026-10-25-emotive-gait-professionalization-design.md` stays binding where the spec is silent.

## Global Constraints

- **Runtime package gains no dependency.** `mocap/` imports NumPy, SciPy and `gait.params` only; `emotion/learned.py` uses `pyyaml`. `statsmodels` and `pandas` are reached only through `learn/`, behind the `fit` extra; nothing imports `learn/` except `cli.py` (spec 5).
- **Layering (spec 5):** `mocap/` imports nothing from `robot/`, `emotion/`, `sim/`, `learn/`. `emotion/` imports nothing from `mocap/`, `learn/`, `sim/`. `tests/mocap/` may import `sim/`; the package may not. `learn/` may import `mocap/`, `gait.params`, `emotion.evidence`, `emotion.model`, `emotion.learned`.
- **Only `robot/` change:** `RobotProfile.leg_length` (G1 0.69 m), spec 4.4. The G1's geometric elbow flexion and arm abduction differ from the joint angles by rig constants (the stand pose reads 1.28 rad on the elbow joint with the arm hanging nearly straight), so the round trip compares posture *deltas against neutral*, which need no constant; the learned model only ever adds deltas.
- **Re-dimensioning lives in `gait/scaling.py`** (`Scaling`, `G`, `to_gait_parameters`), which both `mocap/` and `emotion/` may import; `emotion/` never imports `mocap/`.
- **The artefact is generated, never hand-edited** (spec anti-goals). Only `emotive-gait fit` writes `src/emotive_gait/emotion/data/learned_coefficients.yaml`.
- **Aggregates only enter git.** `data/raw/` and `data/derived/` are git-ignored. Body Movement Library numbers enter the committed artefact and report only when its manifest entry says `committable: true` with a `permission` field (spec 4.1); a local fit may include it, and its output stays local.
- **`expansiveness` is fixed at 1.0** in the learned model and folded into arm swing, abduction and step height wherever analytic parameters are compared with measurements (spec 4.2).
- **Grounding per cell:** `fitted` (>= 3 actors), `thin` (1-2 actors: no interval, actor count printed, excluded from headline numbers), `missing` (delegated to the analytic model) (spec 4.3, 4.5).
- **Intervals** are 95 % Wald from `MixedLM`, present only on `fitted` cells. Standardised errors divide by the between-actor SD of observed per-actor deltas for that parameter, pooled over emotions (spec 4.3).
- **Normalisation (spec 4.2):** Hof scaling, `g = 9.81`: lengths / `l`; cadence x `sqrt(l / g)`; speed / `sqrt(g * l)`; angles, `double_support_frac`, `elbow_swing_ratio` unchanged. Pelvis-height delta to `com_height_scale`: x `leg_length` / `nominal_pelvis_height` (G1 0.69 / 0.68). `ratio` scaling alongside; `froude` default.
- **Sign conventions:** positive `trunk_pitch` is forward inclination; positive `neck_pitch` is gaze down (head on trunk); `elbow_swing_ratio` is the signed slope of elbow flexion on upper-arm sagittal deviation.
- **Analytic model is bit-identical** through the refactor: `AXIS_GAINS`, `AXIS_WEIGHTS`, `CATEGORY_OFFSETS`, `PRESET_PARAMS` and every existing test keep their values.
- **Repository conventions:** no magnitude in degrees in `evidence.py`; disgust and surprise stay `UNSOURCED` with the CLI warning; CLI exit codes 2 (user) and 1 (environment), no traceback escapes `main`; the Z-rail survival gate runs on the analytic presets only.
- **Commit messages carry no Claude attribution trailer** (user convention). Merging and pushing are the user's decisions; master already contains Phase 4 (PR #2 merged), and merging it into this branch is the user's call.

## Review Focus

Input classes the spec implies but no happy-path test exercises. Each has its test pinned to the owning task.

1. **A clip whose landmark map lacks a point** (BML without a head) must yield `NaN` for the dependent fields and real values for the rest, never zeros and never an exception. Owned by Task 3.
2. **A clip with fewer than two complete cycles, or turning throughout,** must summarise to zero cycles with a reason count, and the table builder must skip it with a logged line rather than raise. Owned by Tasks 3 and 4.
3. **A manifest file that is missing or whose hash differs** must stop `fit` and `fetch` with exit 2, naming the file, the expected and the actual hash; no traceback. Owned by Tasks 4 and 8.
4. **`demo --model learned` with `statsmodels` not installed** must work (runtime needs no fit dependency), and with thin or missing cells must print one warning naming them. Owned by Tasks 7 and 8.
5. **A fitted delta that drives an amplitude negative** (sad arm swing past zero) must floor at 0.0 exactly as the analytic model does; a fitted preset the IK cannot reach must fail the feasibility test naming the preset, with the artefact untouched. Owned by Tasks 7 and 9.

## Execution notes

Suggested model per task, implementer / reviewer (user preference: cheap for transcription, standard for integration, most capable for load-bearing review and the final whole-branch review): T1 sonnet / opus; T2 sonnet / opus; T3 opus / fable (the operational definitions are load-bearing); T4 sonnet / opus; T5 opus / fable; T6 sonnet / opus; T7 opus / fable; T8 sonnet / opus; T9 opus / fable (every data decision is recorded here); T10 haiku / sonnet; whole-branch review fable.

Branch: `phase-4-5-learned-emotion-model` (exists; base 7ddd27f, the merged Phase 4 head). Work in place: the `.venv` editable install resolves to this checkout. Every task ends with `ruff check . && ruff format --check . && pytest -q` green; the steps name only the task's own tests and that gate is implied. The venv has NumPy 2.4.6, SciPy 1.17.1, MuJoCo 3.17.0 on Python 3.11.3; `statsmodels` 0.15.0 has a cp311 Windows wheel.

Task 9 is the only task that touches real data and the only one that can block on the outside world (downloads, the PACO reply). Tasks 1-8 are complete with synthetic fixtures alone. Drop order if the schedule slips (spec 10): `fetch` (Task 4's download half); the CSM reader (not planned; add only if Task 9 finds no head point in PTD); the speed-adjusted fit (Task 5 step 9); the `S1 + fear offsets` row (Task 6); the `ratio` scaling; the structure test (Task 6). Never dropped: Tasks 1-3, 5 (headline), 7, the manifest and provenance, the README corrections.

---

### Task 1: `mocap/skeleton.py` and the three readers

**Files:**
- Create: `src/emotive_gait/mocap/__init__.py`, `src/emotive_gait/mocap/skeleton.py`, `src/emotive_gait/mocap/readers/__init__.py`, `src/emotive_gait/mocap/readers/bvh.py`, `src/emotive_gait/mocap/readers/asf_amc.py`, `src/emotive_gait/mocap/readers/ptd.py`
- Test: `tests/mocap/__init__.py`, `tests/mocap/test_readers.py`, `tests/mocap/fixtures/` (three tiny hand-written files)

**Interfaces:**
- Consumes: nothing from the project.
- Produces:
  - `class Landmark(str, Enum)` with members `PELVIS, HIP_L, HIP_R, KNEE_L, KNEE_R, ANKLE_L, ANKLE_R, TOE_L, TOE_R, SHOULDER_L, SHOULDER_R, ELBOW_L, ELBOW_R, WRIST_L, WRIST_R, NECK, HEAD`.
  - `LandmarkMap = Mapping[Landmark, str | tuple[str, ...]]` (a tuple means: average those joints).
  - `@dataclass(frozen=True) class SkeletonClip: joint_names: tuple[str, ...]; parents: tuple[int, ...] | None; positions: np.ndarray  # (T, J, 3) metres, +z up; fps: float; source: str` with `def landmark_positions(self, landmarks: LandmarkMap) -> dict[Landmark, np.ndarray]` returning `(T, 3)` arrays and *omitting* landmarks whose joints are absent (never raising), and `def frames(self) -> int`.
  - `read_bvh(path: Path, *, scale: float, up_axis: Literal["y", "z"]) -> SkeletonClip` (forward kinematics over the BVH hierarchy; `scale` converts the file's unit to metres; a y-up file is rotated so +z is up and +x is preserved).
  - `read_asf_amc(asf: Path, amc: Path) -> SkeletonClip` (CMU: file lengths x `(1 / 0.45) * 0.0254` to metres; the ASF `axis` blocks and AMC channel orders honoured; +y-up in the file becomes +z up).
  - `read_ptd(path: Path, *, joint_names: tuple[str, ...], scale: float, up_axis: Literal["y", "z"], fps: float = 60.0) -> SkeletonClip` (frame number plus 3 x 15 columns; `joint_names` gives the column order).
  - Reader errors raise `MocapFormatError(ValueError)` with the path and line number.

- [ ] **Step 1: Write the fixtures**

`tests/mocap/fixtures/two_bone.bvh`: ROOT `Hips` with OFFSET 0 100 0 and CHANNELS 6, one JOINT `Knee` with OFFSET 0 -40 0 and CHANNELS 3 Zrotation Xrotation Yrotation, End Site OFFSET 0 -40 0; 3 frames at 0.0333333 s: frame 1 all zeros, frame 2 Knee Xrotation 90, frame 3 root Ypos 110. `tests/mocap/fixtures/two_bone.asf` and `two_bone.amc`: a root plus one bone `lfemur` of length 20 (file units) along -y with `axis 0 0 0 XYZ` and dof `rx`, two AMC frames (0 and 90 degrees). `tests/mocap/fixtures/three_point.ptd`: two frames, three joints, integer coordinates.

- [ ] **Step 2: Write the failing reader tests**

```python
# tests/mocap/test_readers.py
def test_bvh_forward_kinematics_rotates_the_child_about_the_parent():
    clip = read_bvh(FIX / "two_bone.bvh", scale=0.01, up_axis="y")
    assert clip.fps == pytest.approx(30.0, abs=0.01) and clip.frames() == 3
    knee0 = clip.positions[0, 1]            # straight down: (0, 0, 0.6) after y-up -> z-up
    assert knee0 == pytest.approx([0.0, 0.0, 0.6], abs=1e-6)
    knee1 = clip.positions[1, 1]            # 90 deg about x: the bone now points along -y or +y
    assert abs(knee1[2] - 1.0) < 1e-6 and abs(abs(knee1[1]) - 0.4) < 1e-6
    assert clip.positions[2, 0, 2] == pytest.approx(1.1, abs=1e-6)   # root translated

def test_bvh_bone_length_is_preserved_in_every_frame():
    clip = read_bvh(FIX / "two_bone.bvh", scale=0.01, up_axis="y")
    lengths = np.linalg.norm(clip.positions[:, 1] - clip.positions[:, 0], axis=1)
    assert lengths == pytest.approx(0.4, abs=1e-9)

def test_asf_amc_scales_cmu_units_to_metres_and_rotates():
    clip = read_asf_amc(FIX / "two_bone.asf", FIX / "two_bone.amc")
    length = np.linalg.norm(clip.positions[0, 1] - clip.positions[0, 0])
    assert length == pytest.approx(20 * (1 / 0.45) * 0.0254, rel=1e-6)
    assert clip.positions[1, 1, 2] != pytest.approx(clip.positions[0, 1, 2])  # rotated frame

def test_ptd_columns_follow_the_given_joint_order():
    clip = read_ptd(FIX / "three_point.ptd", joint_names=("a", "b", "c"), scale=0.001, up_axis="z")
    assert clip.joint_names == ("a", "b", "c") and clip.parents is None and clip.frames() == 2

def test_landmark_positions_omit_missing_joints_and_average_tuples():
    clip = read_ptd(FIX / "three_point.ptd", joint_names=("a", "b", "c"), scale=0.001, up_axis="z")
    got = clip.landmark_positions({Landmark.PELVIS: ("a", "b"), Landmark.HEAD: "zzz"})
    assert Landmark.HEAD not in got
    assert got[Landmark.PELVIS][0] == pytest.approx((clip.positions[0, 0] + clip.positions[0, 1]) / 2)

def test_malformed_bvh_names_the_line(tmp_path):
    bad = tmp_path / "bad.bvh"; bad.write_text("HIERARCHY\nROOT Hips\n{\nOFFSET 0 0\n")
    with pytest.raises(MocapFormatError, match="line 4"):
        read_bvh(bad, scale=1.0, up_axis="y")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/mocap/test_readers.py -q`
Expected: FAIL with `ModuleNotFoundError: emotive_gait.mocap`

- [ ] **Step 4: Implement `skeleton.py`, then the three readers**

BVH: Euler rotations composed in the file's channel order, each channel a rotation about that axis in the parent's frame (the standard BVH convention: `R = R_c1 @ R_c2 @ R_c3` in channel order). ASF/AMC: per-bone `axis` rotation `C`, bone-local motion `M` from the AMC in the bone's `dof` order, world transform `parent @ C @ M @ C^-1`, translation by `length * direction`. Both readers keep the hierarchy in `parents`. Up-axis change: `(x, y, z) -> (x, -z, y)` for a y-up file.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/mocap/test_readers.py -q`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add src/emotive_gait/mocap tests/mocap
git commit -m "feat(mocap): SkeletonClip, Landmark map and BVH / ASF-AMC / PTD readers"
```

---

### Task 2: `mocap/events.py`: path frame and gait events

**Files:**
- Create: `src/emotive_gait/mocap/events.py`
- Test: `tests/mocap/test_events.py`

**Interfaces:**
- Consumes: nothing beyond NumPy and SciPy.
- Produces:
  - `@dataclass(frozen=True) class PathFrame: origin: np.ndarray  # (T, 3) pelvis; forward: np.ndarray  # (T, 3) unit, horizontal; left: np.ndarray  # (T, 3) unit; turning: np.ndarray  # (T,) bool` with `def to_path(self, world: np.ndarray) -> np.ndarray` giving `(T, 3)` coordinates `(forward, left, up)` relative to `origin`.
  - `path_frame(pelvis: np.ndarray, fps: float, *, smooth_s: float = 0.25, yaw_rate_max: float = 0.5) -> PathFrame` (forward = direction of the smoothed horizontal pelvis velocity; `turning` where the yaw rate of `forward` exceeds `yaw_rate_max` rad/s). Raises `ValueError("path frame needs forward travel")` when mean horizontal speed is below 0.05 m/s (spec 4.2: in-place is out of scope).
  - `@dataclass(frozen=True) class FootEvents: strikes: np.ndarray  # frame indices, int; toe_offs: np.ndarray  # int; stance: np.ndarray  # (T,) bool`.
  - `foot_events(foot: np.ndarray, frame: PathFrame, fps: float, *, speed_frac: float = 0.15, height_band_m: float = 0.03) -> FootEvents`: stance where the foot's path-frame horizontal speed (relative to the world, not the pelvis) is below `speed_frac` x mean pelvis speed and its height is within `height_band_m` of its own 5th-percentile height; strike = stance onset, toe-off = stance end; stance runs shorter than 0.1 s are dropped.
  - `cycles(left: FootEvents, right: FootEvents) -> list[tuple[int, int]]`: `(start, end)` frame pairs for consecutive left strikes that contain exactly one right strike.

- [ ] **Step 1: Write the failing tests on a synthetic walker**

Build the fixture in the test module: a pelvis moving at 0.6 m/s along +x for 6 s at 120 Hz with 0.015 m sway; a left foot that alternates 0.5 s stance (fixed world x) and 0.5 s swing (smoothstep advance of 0.6 m with a 0.05 m sine clearance); the right foot half a cycle later. Then a second fixture that walks a 90-degree arc at the same speed.

```python
def test_path_frame_forward_follows_travel_and_marks_turns():
    pf = path_frame(straight.pelvis, 120.0)
    assert np.allclose(pf.forward[60:-60], [1.0, 0.0, 0.0], atol=0.02)
    assert not pf.turning.any()
    pf2 = path_frame(arc.pelvis, 120.0)
    assert pf2.turning[len(arc.pelvis) // 2]

def test_strikes_and_toe_offs_match_the_synthetic_schedule():
    pf = path_frame(straight.pelvis, 120.0)
    ev = foot_events(straight.left_foot, pf, 120.0)
    assert np.allclose(ev.strikes[1:] - ev.strikes[:-1], 120, atol=6)      # one stride per second
    assert np.allclose(ev.toe_offs - ev.strikes[: len(ev.toe_offs)], 60, atol=6)   # smoothstep ends are slow

def test_cycles_pair_left_strikes_that_bracket_one_right_strike():
    pf = path_frame(straight.pelvis, 120.0)
    cyc = cycles(foot_events(straight.left_foot, pf, 120.0), foot_events(straight.right_foot, pf, 120.0))
    assert len(cyc) == 5 and all(e - s == pytest.approx(120, abs=6) for s, e in cyc)

def test_in_place_motion_is_refused():
    with pytest.raises(ValueError, match="forward travel"):
        path_frame(np.tile([0.0, 0.0, 0.9], (600, 1)), 120.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/mocap/test_events.py -q`
Expected: FAIL with `ImportError` on `emotive_gait.mocap.events`

- [ ] **Step 3: Implement `events.py`**

Smoothing: `scipy.ndimage.uniform_filter1d` over `round(smooth_s * fps)` frames on the pelvis velocity. Yaw rate from the finite difference of `atan2(forward_y, forward_x)`, unwrapped.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/mocap/test_events.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/mocap/events.py tests/mocap/test_events.py
git commit -m "feat(mocap): path frame, foot events and cycle pairing"
```

---

### Task 3: `gait/scaling.py`, `mocap/extract.py`, `mocap/normalise.py`; the round trip through the kinematic tier

**Files:**
- Create: `src/emotive_gait/gait/scaling.py`, `src/emotive_gait/mocap/extract.py`, `src/emotive_gait/mocap/normalise.py`
- Modify: `src/emotive_gait/robot/roles.py` (`RobotProfile`: add `leg_length: float = 0.69`, documented as hip-pitch joint height at `qpos0`), `src/emotive_gait/robot/profiles/g1.py` (comment: `hip_pitch_link` to `knee_link` 0.340 m plus `knee_link` to `ankle_pitch_link` 0.300 m plus ankle-pitch stance height 0.051 m = 0.691 m)
- Test: `tests/mocap/test_extract.py`, `tests/mocap/test_roundtrip.py`, `tests/mocap/g1_landmarks.py` (helper: G1 `LandmarkMap` and a function that records a `SkeletonClip` from the kinematic tier), `tests/gait/test_scaling.py`

**Interfaces:**
- Consumes: Task 1 `SkeletonClip`, `Landmark`, `LandmarkMap`; Task 2 `path_frame`, `foot_events`, `cycles`; `gait.params.GaitParameters` field names; for tests only: `sim.kinematic.play_kinematic`, `gait.generator.generate`, `robot.loader.load_robot`, `emotion.model.PRESET_PARAMS`.
- Produces:
  - `gait/scaling.py`: `G = 9.81`; `Scaling = Literal["froude", "ratio"]`; `MEASURED_FIELDS: tuple[str, ...]` = the 14 `GaitParameters` names except `expansiveness`; `LENGTH_FIELDS = ("step_length", "step_height", "feet_spacing", "com_bounce_amp", "lateral_sway_amp")`; `to_gait_parameters(delta: Mapping[str, float], base: GaitParameters, *, scaling: Scaling, leg_length: float, nominal_pelvis_height: float) -> GaitParameters` (`froude`: lengths x `l`, `cadence` x `sqrt(G / l)`, `com_height_scale` x `l / nominal_pelvis_height`, the rest added as-is; `ratio`: lengths, `cadence` and `com_height_scale` multiplied by the base value, the rest added as-is). This module imports only `gait.params`.
  - `mocap/extract.py` measurement names use `pelvis_height`; `mocap/normalise.py` stores its normalised value under `com_height_scale` so every table column is a `MEASURED_FIELDS` name.
  - `@dataclass(frozen=True) class GaitMeasurement` with float fields `cadence, step_length, step_height, double_support_frac, feet_spacing, pelvis_height, com_bounce_amp, lateral_sway_amp, trunk_pitch, neck_pitch, arm_swing_amp, arm_abduction, elbow_flexion, elbow_swing_ratio, speed, swing_fraction` and `start_frame: int, end_frame: int`; `NaN` for anything a missing landmark prevents.
  - `extract_cycles(clip: SkeletonClip, landmarks: LandmarkMap) -> list[GaitMeasurement]` (one per cycle from Task 2; cycles overlapping a `turning` frame are skipped).
  - `leg_length(clip: SkeletonClip, landmarks: LandmarkMap) -> float` (median hip-knee plus median knee-ankle distance plus the ankle's 5th-percentile height; left and right averaged).
  - `@dataclass(frozen=True) class ClipSummary: mean: GaitMeasurement; n_cycles: int; n_discarded: int; discard_reasons: Mapping[str, int]; leg_length: float` and `summarise(clip, landmarks) -> ClipSummary` (`n_cycles == 0` is a valid summary with `NaN` means and a non-empty `discard_reasons`, never an exception).
  - `normalise.py`: `normalise(m: GaitMeasurement, *, leg_length: float) -> dict[str, float]` keyed by `MEASURED_FIELDS` names (Hof scaling; `pelvis_height / l` stored under `com_height_scale`); `ratio_delta(m: GaitMeasurement, neutral: GaitMeasurement) -> dict[str, float]` (lengths, cadence and pelvis height as fractions of the neutral value, angles and fractions as differences). Imports `gait.scaling` for the constants and field lists.

- [ ] **Step 1: Write the failing unit tests for the geometric definitions**

Use the Task 2 synthetic walker, extended with a trunk (pelvis to neck 0.5 m, pitched 0.1 rad forward), a head (neck to head 0.2 m, pitched a further 0.3 rad), and arms whose upper-arm sagittal angle is `0.3 * sin(2 pi phi)` with elbow flexion `0.8 + 0.4 * (0.3 sin)`.

```python
def test_cadence_step_length_and_double_support_from_the_schedule():
    m = summarise(walker.clip, walker.landmarks).mean
    assert m.cadence == pytest.approx(2.0, rel=0.02)          # two steps per 1 s cycle
    assert m.step_length == pytest.approx(0.3, rel=0.02)      # 0.6 m per cycle
    assert m.speed == pytest.approx(0.6, rel=0.02)
    assert m.double_support_frac == pytest.approx(walker.double_support, abs=0.05)

def test_posture_angles_follow_the_construction():
    m = summarise(walker.clip, walker.landmarks).mean
    assert m.trunk_pitch == pytest.approx(0.1, abs=0.01)
    assert m.neck_pitch == pytest.approx(0.3, abs=0.01)
    assert m.arm_swing_amp == pytest.approx(0.3, abs=0.01)
    assert m.elbow_flexion == pytest.approx(0.8, abs=0.01)
    assert m.elbow_swing_ratio == pytest.approx(0.4, abs=0.03)

def test_missing_head_gives_nan_neck_pitch_and_nothing_else():   # Review Focus 1
    lm = {k: v for k, v in walker.landmarks.items() if k is not Landmark.HEAD}
    m = summarise(walker.clip, lm).mean
    assert math.isnan(m.neck_pitch) and not math.isnan(m.trunk_pitch) and not math.isnan(m.cadence)

def test_too_short_clip_summarises_to_zero_cycles_with_a_reason():   # Review Focus 2
    short = replace(walker.clip, positions=walker.clip.positions[:100])
    s = summarise(short, walker.landmarks)
    assert s.n_cycles == 0 and math.isnan(s.mean.cadence) and sum(s.discard_reasons.values()) >= 1

def test_normalise_then_to_gait_parameters_returns_the_measured_value():
    m = summarise(walker.clip, walker.landmarks).mean
    n = normalise(m, leg_length=0.9)
    p = to_gait_parameters(n, GaitParameters(**dict.fromkeys(MEASURED_FIELDS, 0.0)) if False else ZERO_BASE,
                           scaling="froude", leg_length=0.9, nominal_pelvis_height=m.pelvis_height)
    assert p.step_length == pytest.approx(m.step_length) and p.cadence == pytest.approx(m.cadence)

# tests/gait/test_scaling.py
def test_froude_re_dimensioning_uses_leg_length_and_nominal_height():
    base = GaitParameters()
    d = {"step_length": 0.1, "cadence": 0.05, "trunk_pitch": 0.2, "com_height_scale": -0.03}
    p = to_gait_parameters(d, base, scaling="froude", leg_length=0.69, nominal_pelvis_height=0.68)
    assert p.step_length == pytest.approx(0.35 + 0.1 * 0.69)
    assert p.cadence == pytest.approx(1.8 + 0.05 * math.sqrt(G / 0.69))
    assert p.trunk_pitch == pytest.approx(0.2)
    assert p.com_height_scale == pytest.approx(1.0 - 0.03 * 0.69 / 0.68)

def test_ratio_re_dimensioning_multiplies_the_base():
    p = to_gait_parameters({"step_length": -0.2, "cadence": 0.1, "neck_pitch": 0.3}, GaitParameters(), scaling="ratio", leg_length=0.69, nominal_pelvis_height=0.68)
    assert p.step_length == pytest.approx(0.35 * 0.8) and p.cadence == pytest.approx(1.8 * 1.1) and p.neck_pitch == pytest.approx(0.3)
```

`ZERO_BASE` is `GaitParameters` with every `MEASURED_FIELDS` value 0.0 except `com_height_scale` 1.0 (the `pelvis_height` normalisation needs a positive nominal height, hence the `nominal_pelvis_height=m.pelvis_height` argument).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/mocap/test_extract.py tests/gait/test_scaling.py -q`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `gait/scaling.py`, `extract.py` and `normalise.py`**

Angles: trunk vector `NECK - PELVIS`; sagittal angle = `atan2(forward component, up component)` in the path frame; head on trunk = sagittal angle of `HEAD - NECK` minus the trunk's; upper arm `ELBOW - SHOULDER` relative to the trunk vector; abduction = `asin` of the arm's lateral component (sign so that away from the body is positive); elbow flexion = `pi - angle(SHOULDER - ELBOW, WRIST - ELBOW)`. Bounce and sway: detrend the pelvis path-frame up and left coordinates over the cycle (linear), then half peak-to-peak. `feet_spacing`: mean |left ankle lateral - right ankle lateral|. `elbow_swing_ratio`: `np.polyfit(arm_dev, elbow_dev, 1)[0]` over the cycle, averaged over arms.

- [ ] **Step 4: Add `leg_length` to `RobotProfile` and set it for the G1**

`leg_length: float = 0.69` on `RobotProfile`, with the G1 comment from the Files list. Run `pytest tests/robot -q` to confirm nothing else moves.

- [ ] **Step 5: Write the G1 landmark helper and the failing round-trip test**

`tests/mocap/g1_landmarks.py`: `G1_LANDMARKS` mapping `PELVIS -> "pelvis"`, `HIP_L -> "left_hip_pitch_link"`, `KNEE_L -> "left_knee_link"`, `ANKLE_L -> "left_ankle_pitch_link"`, `SHOULDER_L -> "left_shoulder_roll_link"`, `ELBOW_L -> "left_elbow_link"`, `WRIST_L -> "left_wrist_roll_link"`, mirrored for the right, `NECK -> "head_link"` (the neck joint sits at the head body origin), and `HEAD -> "head_top"`, a synthetic point 0.10 m along the head body's local +z computed from `data.xmat`. `record_clip(robot, traj, *, duration, fps) -> SkeletonClip` iterates `play_kinematic` and stacks body positions (`data.xpos`) with the extra `head_top` joint.

```python
# tests/mocap/test_roundtrip.py
# Module-scoped: one robot, one trajectory and one recorded clip per preset (neutral, sad, angry, happy).
POSTURE = ("trunk_pitch", "neck_pitch", "elbow_flexion", "arm_abduction")   # absolute on the robot, so compared as deltas vs neutral

def half_ptp_shoulder_pitch(traj) -> float:
    angles = [traj.angles_at(k / 240)[JointRole.SHOULDER_PITCH_L] for k in range(240)]
    return (max(angles) - min(angles)) / 2

@pytest.mark.parametrize("name", ["neutral", "sad", "angry", "happy"])
def test_extractor_recovers_the_parameters_that_generated_the_motion(recorded, name):
    p, traj, s = recorded[name].params, recorded[name].traj, recorded[name].summary
    m = s.mean
    assert s.n_cycles >= 4
    assert m.cadence == pytest.approx(p.cadence, rel=0.02)
    assert m.step_length == pytest.approx(p.step_length, rel=0.02)
    assert m.step_height == pytest.approx(p.expansiveness * p.step_height, abs=0.003)
    assert m.com_bounce_amp == pytest.approx(p.com_bounce_amp, abs=0.003)
    assert m.lateral_sway_amp == pytest.approx(p.lateral_sway_amp, abs=0.003)
    assert m.double_support_frac == pytest.approx(p.double_support_frac, abs=0.05)
    assert m.feet_spacing == pytest.approx(p.feet_spacing, abs=0.02)
    assert m.arm_swing_amp == pytest.approx(half_ptp_shoulder_pitch(traj), abs=0.02)   # what was rendered, after the hip-normalised drive
    assert m.elbow_swing_ratio == pytest.approx(p.elbow_swing_ratio, abs=0.05)
    assert s.leg_length == pytest.approx(0.69, abs=0.01)
    assert m.pelvis_height == pytest.approx(0.68 * p.com_height_scale, abs=0.01)

@pytest.mark.parametrize("name", ["sad", "angry", "happy"])
@pytest.mark.parametrize("field", POSTURE)
def test_posture_deltas_against_neutral_match_the_parameter_deltas(recorded, name, field):
    # Geometric angles differ from joint angles by rig constants; deltas do not. Also pins the sign conventions.
    measured = getattr(recorded[name].summary.mean, field) - getattr(recorded["neutral"].summary.mean, field)
    expected = getattr(recorded[name].params, field) - getattr(recorded["neutral"].params, field)
    if field == "arm_abduction":
        expected = recorded[name].params.expansiveness * recorded[name].params.arm_abduction - recorded["neutral"].params.arm_abduction
    assert measured == pytest.approx(expected, abs=0.03), field

def test_neutral_trunk_and_head_read_level():
    m = recorded["neutral"].summary.mean
    assert m.trunk_pitch == pytest.approx(0.0, abs=0.03) and m.neck_pitch == pytest.approx(0.0, abs=0.03)

def test_noisy_sixty_hertz_clip_stays_within_margin(recorded):
    p, traj = recorded["neutral"].params, recorded["neutral"].traj
    clip = record_clip(recorded.robot, traj, duration=8.0, fps=60.0)
    rng = np.random.default_rng(0)
    noisy = replace(clip, positions=clip.positions + rng.normal(0, 0.003, clip.positions.shape))
    m = summarise(noisy, G1_LANDMARKS).mean
    assert m.cadence == pytest.approx(p.cadence, rel=0.03) and m.step_length == pytest.approx(p.step_length, rel=0.03)
    assert m.step_height == pytest.approx(p.step_height, abs=0.006)
```

The `recorded` fixture is module-scoped: `load_robot()` once, then for each preset `generate`, `record_clip(..., duration=8.0, fps=120.0)` and `summarise(clip, G1_LANDMARKS)`, stored as `(params, traj, summary)` (spec: tests share trajectories). The noisy test below reuses the neutral trajectory at 60 fps.

- [ ] **Step 6: Run the round trip and freeze the thresholds**

Run: `pytest tests/mocap/test_roundtrip.py -q`
Expected: PASS. If a preset fails, adjust only Task 2's defaults (`speed_frac`, `height_band_m`, `smooth_s`) and the detrending, re-run Task 2's tests, and record the final values in a comment at the top of `events.py`. Do not loosen the tolerances above.

- [ ] **Step 7: Run the whole suite**

Run: `pytest -q`
Expected: 259 previously passing tests still pass plus the new ones; the robot and emotion suites are unaffected by the new profile field.

- [ ] **Step 8: Commit**

```bash
git add src/emotive_gait/gait/scaling.py src/emotive_gait/mocap src/emotive_gait/robot tests/mocap tests/gait/test_scaling.py
git commit -m "feat(mocap): per-cycle gait measurements, Hof normalisation and re-dimensioning, G1 round trip; profile leg_length"
```

---

### Task 4: `learn/manifest.py`, `learn/fetch.py`, `learn/table.py`; landmark maps for the three datasets

**Files:**
- Create: `src/emotive_gait/learn/__init__.py`, `src/emotive_gait/learn/manifest.py`, `src/emotive_gait/learn/fetch.py`, `src/emotive_gait/learn/table.py`, `src/emotive_gait/mocap/landmarks.py`, `data/manifest.yaml` (skeleton: the three dataset entries with URLs, licences, `committable`, empty `files:` lists to be frozen in Task 9)
- Modify: `pyproject.toml` (`[project.optional-dependencies] fit = ["statsmodels>=0.14", "pandas>=2"]`; `dev` gains the same two), `.gitignore` (`data/raw/`, `data/derived/`)
- Test: `tests/learn/__init__.py`, `tests/learn/test_manifest.py`, `tests/learn/test_fetch.py`, `tests/learn/test_table.py`

**Interfaces:**
- Consumes: Task 1 readers and `LandmarkMap`; Task 3 `summarise`, `extract_cycles`, `leg_length`, `normalise`, `MEASURED_FIELDS`.
- Produces:
  - `mocap/landmarks.py`: `STYLE100: LandmarkMap`, `CMU: LandmarkMap`, `BML_PTD: LandmarkMap` and `BML_PTD_JOINTS: tuple[str, ...]` (the 15 PTD column names; Task 9 verifies them against the data page), plus `LANDMARKS: Mapping[str, LandmarkMap]` keyed by dataset id `"100style"`, `"cmu"`, `"bml"`.
  - `manifest.py`: `@dataclass(frozen=True) class ClipEntry: path: str; sha256: str; subject: str; actor: str; emotion: str  # project emotion or "neutral"; label: str  # dataset's own label; label_kind: Literal["exact", "mapped"]; start_frame: int | None; end_frame: int | None; extra: Mapping[str, str]` (e.g. the ASF path for CMU); `@dataclass(frozen=True) class DatasetEntry: id: str; name: str; url: str; archive: str | None; archive_sha256: str | None; archive_bytes: int | None; licence: str; committable: bool; permission: str; reader: Literal["bvh", "asf_amc", "ptd"]; reader_args: Mapping[str, object]; clips: tuple[ClipEntry, ...]; excluded: tuple[Mapping[str, str], ...]`; `@dataclass(frozen=True) class Manifest: datasets: tuple[DatasetEntry, ...]; sha256: str  # of the manifest file itself`; `load_manifest(path: Path) -> Manifest` (raises `ManifestError(ValueError)` naming the key on a schema problem); `sha256_of(path: Path) -> str`; `@dataclass(frozen=True) class HashProblem: path: str; expected: str; actual: str | None  # None = missing`; `verify(manifest: Manifest, root: Path, *, datasets: Iterable[str] | None = None) -> list[HashProblem]`.
  - `fetch.py`: `Downloader = Callable[[str, Path], None]`; `fetch(manifest: Manifest, root: Path, *, from_paths: Mapping[str, Path] = {}, downloader: Downloader = urlretrieve_to, datasets: Iterable[str] | None = None) -> list[HashProblem]` (copies `from_paths[dataset_id]` into `root/<id>/` when given, otherwise downloads `url`; extracts archives; verifies; returns the problems and never raises on a hash mismatch).
  - `table.py`: `build_table(manifest: Manifest, root: Path, *, datasets: Iterable[str] | None = None, log: Callable[[str], None] = print) -> pandas.DataFrame` with columns `dataset, subject, actor, emotion, label_kind, clip, cycle, leg_length` plus the 14 `MEASURED_FIELDS` raw values; clips with zero cycles are logged and skipped (Review Focus 2); `normalise_table(df: DataFrame, scaling: Scaling) -> DataFrame` (same columns, values normalised per row by `leg_length`; `ratio` divides by the actor's neutral mean per parameter).

- [ ] **Step 1: Write the failing manifest and fetch tests**

```python
# tests/learn/test_manifest.py
def test_manifest_round_trips_and_hashes_itself(tmp_path):
    m = load_manifest(write_minimal_manifest(tmp_path))   # helper writes one dataset, one clip
    assert m.datasets[0].id == "toy" and m.datasets[0].clips[0].label_kind == "exact"
    assert m.sha256 == sha256_of(tmp_path / "manifest.yaml")

def test_verify_reports_missing_and_mismatched_files(tmp_path):     # Review Focus 3
    m = load_manifest(write_minimal_manifest(tmp_path))
    problems = verify(m, tmp_path / "raw")
    assert problems[0].actual is None                                  # missing
    (tmp_path / "raw" / "toy").mkdir(parents=True); (tmp_path / "raw" / "toy" / "walk.bvh").write_text("x")
    problems = verify(m, tmp_path / "raw")
    assert problems[0].expected != problems[0].actual and problems[0].path.endswith("walk.bvh")

def test_unknown_label_kind_names_the_key(tmp_path):
    with pytest.raises(ManifestError, match="label_kind"):
        load_manifest(write_minimal_manifest(tmp_path, label_kind="guessed"))

# tests/learn/test_fetch.py
def test_fetch_prefers_a_local_archive_and_verifies_it(tmp_path):
    m = load_manifest(write_archive_manifest(tmp_path))   # one dataset whose archive is a zip of walk.bvh
    problems = fetch(m, tmp_path / "raw", from_paths={"toy": tmp_path / "toy.zip"}, downloader=fail_if_called)
    assert problems == [] and (tmp_path / "raw" / "toy" / "walk.bvh").exists()

def test_fetch_downloads_when_no_local_copy_and_reports_a_bad_hash(tmp_path):
    calls = []
    def dl(url, dest): calls.append(url); dest.write_bytes(b"wrong")
    problems = fetch(load_manifest(write_minimal_manifest(tmp_path)), tmp_path / "raw", downloader=dl)
    assert calls == ["http://example.test/walk.bvh"] and problems and problems[0].actual is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/learn -q`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `manifest.py` and `fetch.py`; add the `fit` extra and the `.gitignore` lines; install the extra**

Run after editing `pyproject.toml`: `.venv\Scripts\pip install -e ".[dev]"`. Archive extraction: `zipfile` for `.zip`; extract only the paths named in `clips` (never the whole 100STYLE archive). The default downloader uses `urllib.request.urlretrieve` and does not verify TLS differently from the standard library (CMU's chain is incomplete; its files are fetched over `http://`).

- [ ] **Step 4: Write the failing table test**

Uses the Task 2/3 synthetic walker written to a BVH through a tiny writer in the test (two actors, neutral plus "angry" with cadence 2.4), a manifest naming the four clips with reader `bvh`.

```python
def test_build_table_has_one_row_per_cycle_and_skips_empty_clips(tmp_path, capsys):
    df = build_table(manifest, tmp_path / "raw", log=print)
    assert set(df.columns) >= {"dataset", "actor", "emotion", "cycle", "leg_length", *MEASURED_FIELDS}
    assert set(df.emotion) == {"neutral", "anger"} and (df.groupby(["actor", "emotion"]).size() >= 4).all()
    assert "0 cycles" in capsys.readouterr().out          # the deliberately short fifth clip

def test_normalise_table_divides_lengths_by_leg_length():
    n = normalise_table(df, "froude")
    row = df.iloc[0]; nrow = n.iloc[0]
    assert nrow.step_length == pytest.approx(row.step_length / row.leg_length)
    assert nrow.cadence == pytest.approx(row.cadence * math.sqrt(row.leg_length / G))
```

- [ ] **Step 5: Implement `table.py` and `landmarks.py`; run the tests**

`landmarks.py` names: 100STYLE bones `Hips, LeftUpLeg/LeftLeg/LeftFoot/LeftToe, Neck, Head, LeftShoulder/LeftArm/LeftForeArm/LeftHand` (the builder confirms against one 100STYLE BVH header in Task 9 and edits here); CMU `root, lhipjoint/lfemur/ltibia/lfoot, thorax, head, lclavicle/lhumerus/lradius/lwrist`; BML PTD column names per the data page (Task 9 verifies).

Run: `pytest tests/learn tests/mocap -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/emotive_gait/learn src/emotive_gait/mocap/landmarks.py data/manifest.yaml pyproject.toml .gitignore tests/learn
git commit -m "feat(learn): hashed data manifest, fetch, per-cycle table; dataset landmark maps; fit extra"
```

---

### Task 5: `learn/fit.py`: mixed model per parameter, held-out actors, hand-tuned and zero baselines

**Files:**
- Create: `src/emotive_gait/learn/fit.py`, `src/emotive_gait/learn/synthetic.py` (synthetic table generator shared by tests and the report's self-check)
- Test: `tests/learn/test_fit.py`

**Interfaces:**
- Consumes: Task 4 `normalise_table` output; `emotion.model.modulate`, `emotion.evidence.Emotion`, `GaitParameters`; Task 3 `gait.scaling` (`G`, `MEASURED_FIELDS`, `LENGTH_FIELDS`).
- Produces:
  - `FITTED_EMOTIONS = ("anger", "fear", "happiness", "sadness")`; `MIN_ACTORS_FITTED = 3`.
  - `@dataclass(frozen=True) class Cell: emotion: str; parameter: str; estimate: float; ci_low: float | None; ci_high: float | None; between_actor_sd: float; residual_sd: float; n_actors: int; n_cycles: int; n_datasets: int; n_mapped_actors: int; grounding: Literal["fitted", "thin", "missing"]; converged_maximal: bool; per_dataset: Mapping[str, float]`.
  - `fit_parameter(df: DataFrame, parameter: str) -> list[Cell]` (one `MixedLM` with formula `parameter ~ C(emotion, Treatment("neutral"))`, `groups="actor"`, `re_formula="~ C(emotion, Treatment('neutral'))"`; REML; on non-convergence or a singular random-effects covariance, refit with `re_formula="1"` and set `converged_maximal=False`; rows with `NaN` in `parameter` are dropped first; a cell with `n_actors == 0` is `missing`).
  - `fit_all(df: DataFrame) -> dict[str, list[Cell]]` keyed by scaling-independent parameter name (called once per scaling by Task 8).
  - `actor_deltas(df: DataFrame) -> DataFrame` with columns `actor, dataset, emotion, parameter, delta` (emotion-clip mean minus neutral-clip mean per actor, one row per actor-emotion-parameter).
  - `between_actor_sd(deltas: DataFrame) -> dict[str, float]` (SD of `delta` around its emotion mean, pooled over emotions, per parameter).
  - `hand_tuned_deltas(scaling: Scaling, *, leg_length: float = 0.69, nominal_pelvis_height: float = 0.68) -> dict[tuple[str, str], float]` keyed `(emotion, parameter)`: `modulate(GaitParameters(), {e: 1.0}, G1_PROFILE_WITH_NECK)` minus `GaitParameters()`, `expansiveness` folded (`arm_swing_amp`, `arm_abduction`, `step_height` multiplied by the modulated `expansiveness` before differencing), then forward-normalised.
  - `@dataclass(frozen=True) class HeldOut: parameter: str; e_fit: float; e_hand: float; e_zero: float; e_fit_natural: float; e_hand_natural: float; n_pairs: int` and `leave_one_actor_out(df: DataFrame, cells: Mapping[str, list[Cell]], hand: Mapping[tuple[str, str], float]) -> list[HeldOut]` (fitted cells only; targets are `actor_deltas`; standardised by `between_actor_sd`; plus an `overall` entry whose errors are the unweighted means).
  - `speed_adjusted(df: DataFrame) -> dict[tuple[str, str], float]`: the same fixed effects with the actor's normalised speed delta added as a covariate (report only).

- [ ] **Step 1: Write the synthetic generator and the failing recovery test**

`synthetic.py`: `synthetic_table(*, n_actors: int, emotions: Sequence[str], effects: Mapping[tuple[str, str], float], actor_sd: float, cycle_sd: float, cycles_per_clip: int, seed: int) -> DataFrame` in the normalised-table layout, with a per-actor random intercept (SD 0.05) and random slopes (`actor_sd`), neutral included.

```python
# tests/learn/test_fit.py
EFFECTS = {("anger", "cadence"): 0.08, ("sadness", "cadence"): -0.10, ("anger", "trunk_pitch"): 0.12}

def test_fixed_effects_are_recovered_within_their_intervals():
    df = synthetic_table(n_actors=20, emotions=("anger", "sadness"), effects=EFFECTS, actor_sd=0.03, cycle_sd=0.02, cycles_per_clip=10, seed=1)
    cells = {c.emotion: c for c in fit_parameter(df, "cadence")}
    assert cells["anger"].ci_low < 0.08 < cells["anger"].ci_high
    assert cells["sadness"].estimate == pytest.approx(-0.10, abs=0.02)
    assert cells["anger"].grounding == "fitted" and cells["anger"].n_actors == 20
    assert cells["anger"].between_actor_sd == pytest.approx(0.03, abs=0.015)

def test_two_actor_emotion_is_thin_without_an_interval():
    df = synthetic_table(n_actors=2, emotions=("fear",), effects={("fear", "cadence"): 0.05}, actor_sd=0.02, cycle_sd=0.02, cycles_per_clip=3, seed=2)
    cell = fit_parameter(df, "cadence")[0]
    assert cell.grounding == "thin" and cell.ci_low is None and cell.n_actors == 2

def test_missing_parameter_rows_make_a_missing_cell():
    df = synthetic_table(n_actors=5, emotions=("anger",), effects={}, actor_sd=0.01, cycle_sd=0.01, cycles_per_clip=4, seed=3)
    df.loc[:, "neck_pitch"] = float("nan")
    assert fit_parameter(df, "neck_pitch")[0].grounding == "missing"

def test_hand_tuned_deltas_fold_expansiveness_and_normalise():
    h = hand_tuned_deltas("froude")
    assert h[("anger", "cadence")] == pytest.approx(0.60 * math.sqrt(0.69 / G))
    angry = modulate(GaitParameters(), {Emotion.ANGER: 1.0}, G1_PROFILE_WITH_NECK)
    assert h[("anger", "arm_swing_amp")] == pytest.approx(angry.expansiveness * angry.arm_swing_amp - 0.35)

def test_held_out_fit_beats_zero_and_a_wrong_hand_tuning_on_synthetic_data():
    df = synthetic_table(n_actors=12, emotions=("anger", "sadness"), effects=EFFECTS, actor_sd=0.03, cycle_sd=0.02, cycles_per_clip=8, seed=4)
    wrong = {k: -v for k, v in EFFECTS.items()}
    out = {h.parameter: h for h in leave_one_actor_out(df, fit_all(df), wrong)}
    assert out["cadence"].e_fit < out["cadence"].e_zero < out["cadence"].e_hand
    assert 0.8 < out["cadence"].e_fit < 1.5          # near the between-actor floor
    assert out["overall"].n_pairs == 24
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/learn/test_fit.py -q`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `fit.py`**

Wald intervals from `MixedLMResults.conf_int()`; `between_actor_sd` for a cell from the random-slope variance of that emotion (`cov_re`), or from `actor_deltas` when the maximal structure fell back. Non-convergence is detected by `ConvergenceWarning` (caught with `warnings.catch_warnings(record=True)`) or `result.converged is False`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/learn/test_fit.py -q`
Expected: PASS (5 tests, under 60 s)

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/learn/fit.py src/emotive_gait/learn/synthetic.py tests/learn/test_fit.py
git commit -m "feat(learn): mixed-effects fit per parameter, held-out-actor validation, hand-tuned and zero baselines"
```

---

### Task 6: `learn/structure.py`: the three-axis structure test

**Files:**
- Create: `src/emotive_gait/learn/structure.py`
- Test: `tests/learn/test_structure.py`

**Interfaces:**
- Consumes: Task 5 `actor_deltas`, `between_actor_sd`; `emotion.axes.AXIS_WEIGHTS`, `emotion.model.AXIS_GAINS`, `emotion.axes.CATEGORY_OFFSETS`.
- Produces: `STRUCTURES = ("S1_analytic", "S1_fear_offsets", "S2_fixed_axes_free_loadings", "S3_rank3", "S4_full")`; `@dataclass(frozen=True) class StructureResult: name: str; n_free: int; r2_in_sample: float; e_heldout: float` and `structure_test(deltas: DataFrame, sd: Mapping[str, float], *, parameters: Sequence[str]) -> list[StructureResult]` (leave-one-actor-out on actor-emotion means, each structure refit on the training actors by least squares; `S3` by truncated SVD of the training `B`; `S4` is the per-emotion mean; errors standardised by `sd` and averaged over `parameters`).

- [ ] **Step 1: Write the failing tests**

```python
def test_full_model_fits_in_sample_at_least_as_well_as_every_constraint():
    res = {r.name: r for r in structure_test(deltas_rank3, SD, parameters=PARAMS)}
    assert res["S4_full"].r2_in_sample >= res["S3_rank3"].r2_in_sample >= res["S2_fixed_axes_free_loadings"].r2_in_sample
    assert res["S4_full"].n_free == 4 * len(PARAMS) and res["S1_analytic"].n_free == len(AXIS_GAINS)

def test_rank3_truth_is_recovered_by_rank3_but_rank4_overfits_held_out():
    # deltas_rank3: 30 synthetic actors whose emotion means are exactly W_true @ G_true plus actor noise
    res = {r.name: r for r in structure_test(deltas_rank3, SD, parameters=PARAMS)}
    assert res["S3_rank3"].e_heldout <= res["S4_full"].e_heldout + 0.05

def test_fear_offsets_row_frees_exactly_four_more_parameters():
    res = {r.name: r for r in structure_test(deltas_rank3, SD, parameters=PARAMS)}
    assert res["S1_fear_offsets"].n_free == res["S1_analytic"].n_free + 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/learn/test_structure.py -q`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `structure.py`; run the tests**

`W` for S1/S2 is `AXIS_WEIGHTS` for the four fitted emotions as a 4 x 3 matrix; S1's `G` has one non-zero per parameter at the axis `AXIS_GAINS` names (parameters absent from `AXIS_GAINS` are zero rows); `S1_fear_offsets` adds a free fear-row entry for `com_height_scale`, `elbow_swing_ratio`, `cadence`, `neck_pitch`.

Run: `pytest tests/learn/test_structure.py -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/emotive_gait/learn/structure.py tests/learn/test_structure.py
git commit -m "feat(learn): three-axis structure test (constrained, reduced-rank, full) on held-out actors"
```

---

### Task 7: The seam: `EmotionModel` protocol, `AnalyticEmotionModel`, `LearnedEmotionModel`, artefact schema

**Files:**
- Create: `src/emotive_gait/emotion/learned.py`, `src/emotive_gait/emotion/data/__init__.py` (empty, so the data directory is a package for `importlib.resources`)
- Modify: `src/emotive_gait/emotion/model.py` (protocol, `AnalyticEmotionModel`, `model=` on `modulate` and `unexpressible`, `learned_presets`), `pyproject.toml` (`[tool.setuptools.package-data] "emotive_gait.emotion" = ["data/*.yaml"]`), `src/emotive_gait/emotion/evidence.py` (docstring sentence: fitted magnitudes live in the artefact)
- Test: `tests/emotion/test_model.py` (parametrise the direction test), `tests/emotion/test_learned.py`, `tests/emotion/artefacts.py` (helper building in-memory artefacts)

**Interfaces:**
- Consumes: Task 3 `gait.scaling.to_gait_parameters`, `Scaling` (never anything from `mocap/`); `RobotProfile.leg_length`, `nominal_pelvis_height`; `evidence.claims_for`, `GROUNDED_EMOTIONS`.
- Produces:
  - `model.py`: `class EmotionModel(Protocol): def deltas(self, intensities: Mapping[Emotion, float], profile: RobotProfile) -> Mapping[str, float]: ...` (robot units, validated intensities); `class AnalyticEmotionModel` (wraps today's `_deltas`); `ANALYTIC: AnalyticEmotionModel`; `modulate(params, intensities, profile, *, model: EmotionModel | None = None, hold_speed=None, speed_mode=...)`; `unexpressible(intensities, profile, model: EmotionModel | None = None)`; `learned_presets(model: EmotionModel) -> Mapping[str, GaitParameters]` (same four keys as `PRESET_PARAMS`).
  - `learned.py`: `@dataclass(frozen=True) class ArtefactCell` (the Task 5 `Cell` fields plus `direction_check: str`); `@dataclass(frozen=True) class Artefact: cells: Mapping[tuple[str, str, str], ArtefactCell]  # (scaling, emotion, parameter); contradictions: tuple[Mapping[str, object], ...]; provenance: Mapping[str, object]`; `load_artefact(path: Path) -> Artefact`; `default_artefact_path() -> Path` (package data); `class ArtefactMissingError(FileNotFoundError)`; `class LearnedEmotionModel(EmotionModel)` with `__init__(self, artefact: Artefact, *, scaling: Scaling = "froude", fallback: EmotionModel = ANALYTIC)`, `@classmethod default(cls, *, scaling="froude") -> LearnedEmotionModel`, properties `thin_cells -> tuple[tuple[str, str], ...]`, `analytic_cells -> tuple[tuple[str, str], ...]` (missing cells plus every cell of disgust and surprise), and `deltas(...)` = sum over emotions of `intensity * re-dimensioned effect` for fitted and thin cells, plus the fallback model's deltas restricted to the analytic cells.
  - `DIRECTION_CHECKS = ("agrees", "agrees_ci_spans_zero", "contradicts", "contradicts_ci_spans_zero", "thin_unchecked", "no_estimate", "no_claim")` and `direction_check(estimate: float, ci: tuple[float, float] | None, grounding: str, claim_direction: int | None) -> str` (in `learned.py`, plain arguments so `learn/` can call it without `emotion/` knowing `learn/`'s `Cell`; used by Task 8's artefact writer).
  - `ArtefactCell` duplicates the `Cell` field list by hand (the two layers may not share a class); `tests/emotion/test_learned.py` asserts `set(fields(ArtefactCell)) == set(fields(Cell)) | {"direction_check"}` so they cannot drift.

- [ ] **Step 1: Write the failing refactor test (bit-identical analytic model)**

```python
def test_analytic_model_object_matches_modulate_defaults():
    for e in Emotion:
        a = modulate(GaitParameters(), {e: 0.7}, G1_PROFILE_WITH_NECK)
        b = modulate(GaitParameters(), {e: 0.7}, G1_PROFILE_WITH_NECK, model=ANALYTIC)
        assert a == b
    assert PRESET_PARAMS["angry"] == modulate(GaitParameters(), {Emotion.ANGER: 1.0}, G1_PROFILE_WITH_NECK)
```

- [ ] **Step 2: Refactor `model.py`; run the whole emotion suite unchanged**

Run: `pytest tests/emotion -q`
Expected: every existing test passes with no edit to its values.

- [ ] **Step 3: Write the artefact helper and the failing learned-model tests**

`tests/emotion/artefacts.py`: `make_artefact(effects: Mapping[tuple[str, str], float], *, thin: Iterable[tuple[str, str]] = (), contradictions=()) -> Artefact` filling both scalings with the same numbers; every (fitted emotion, parameter) cell absent from `effects` is `missing`, cells named in `thin` get `grounding="thin"` and no interval, the others get `grounding="fitted"` with an interval of +-0.01 around the estimate.

```python
# tests/emotion/test_learned.py
AGREEING = {("sadness", "cadence"): -0.05, ("sadness", "neck_pitch"): 0.3, ("anger", "cadence"): 0.08, ("anger", "elbow_flexion"): 0.4}

def test_learned_deltas_re_dimension_through_the_profile():
    m = LearnedEmotionModel(make_artefact(AGREEING))
    d = m.deltas({Emotion.SADNESS: 1.0}, G1_PROFILE_WITH_NECK)
    assert d["cadence"] == pytest.approx(-0.05 * math.sqrt(G / 0.69)) and d["neck_pitch"] == pytest.approx(0.3)
    half = m.deltas({Emotion.SADNESS: 0.5}, G1_PROFILE_WITH_NECK)
    assert half["neck_pitch"] == pytest.approx(0.15)                       # linear in intensity

def test_missing_cells_and_unsourced_emotions_delegate_to_the_analytic_model():
    m = LearnedEmotionModel(make_artefact(AGREEING))                          # (sadness, trunk_pitch) is absent, so missing
    d = m.deltas({Emotion.SADNESS: 1.0, Emotion.DISGUST: 1.0}, G1_PROFILE_WITH_NECK)
    a = ANALYTIC.deltas({Emotion.SADNESS: 1.0, Emotion.DISGUST: 1.0}, G1_PROFILE_WITH_NECK)
    assert ("sadness", "trunk_pitch") in m.analytic_cells and ("disgust", "trunk_pitch") in m.analytic_cells
    assert d["trunk_pitch"] == pytest.approx(a["trunk_pitch"])             # both shares analytic, so the sum equals the analytic sum
    assert d["cadence"] == pytest.approx(-0.05 * math.sqrt(G / 0.69) + ANALYTIC.deltas({Emotion.DISGUST: 1.0}, G1_PROFILE_WITH_NECK)["cadence"])   # fitted share plus analytic share

def test_thin_cells_are_used_and_listed():
    m = LearnedEmotionModel(make_artefact(AGREEING, thin=[("anger", "cadence")]))
    assert ("anger", "cadence") in m.thin_cells
    assert m.deltas({Emotion.ANGER: 1.0}, G1_PROFILE_WITH_NECK)["cadence"] > 0

def test_negative_amplitude_from_a_fitted_delta_floors_at_zero():       # Review Focus 5
    m = LearnedEmotionModel(make_artefact({("sadness", "arm_swing_amp"): -2.0}))
    assert modulate(GaitParameters(), {Emotion.SADNESS: 1.0}, G1_PROFILE_WITH_NECK, model=m).arm_swing_amp == 0.0

def test_default_without_an_artefact_raises_a_named_error(monkeypatch, tmp_path):
    monkeypatch.setattr("emotive_gait.emotion.learned.default_artefact_path", lambda: tmp_path / "none.yaml")
    with pytest.raises(ArtefactMissingError, match="emotive-gait fit"):
        LearnedEmotionModel.default()

def test_direction_check_vocabulary():
    assert direction_check(0.2, (0.1, 0.3), "fitted", +1) == "agrees"
    assert direction_check(0.2, (-0.1, 0.3), "fitted", -1) == "contradicts_ci_spans_zero"
    assert direction_check(0.2, (-0.1, 0.3), "fitted", 0) == "agrees_ci_spans_zero"
    assert direction_check(0.2, (0.1, 0.3), "fitted", 0) == "contradicts"          # a no-change claim with an interval off zero
    assert direction_check(0.2, None, "thin", 0) == "thin_unchecked"
    assert direction_check(0.2, None, "thin", +1) == "agrees"                       # sign only
    assert direction_check(0.2, (0.1, 0.3), "fitted", None) == "no_claim"
```

`deltas` sums per-emotion shares: for each emotion with non-zero intensity and each parameter, the share is `intensity x re-dimensioned effect` when that cell is fitted or thin, and that emotion's analytic delta for that parameter when the cell is missing or the emotion is unsourced. `analytic_cells` lists the cells that take the second route.

- [ ] **Step 4: Parametrise the direction test over both models**

In `tests/emotion/test_model.py`, `test_modulation_direction_agrees_with_every_grounded_claim` gains `@pytest.mark.parametrize("model", [ANALYTIC, learned_fixture_model])` where the fixture model is `LearnedEmotionModel(make_artefact(FULL_AGREEING))`, a table that agrees with every grounded claim. For the learned model the body implements spec 4.5: non-zero claims on fitted or thin cells must match in sign *or* appear in `artefact.contradictions`; `expansiveness` claims are skipped; claims on analytic cells are checked as for the analytic model. Add:

```python
def test_an_unreported_contradiction_fails_the_direction_test():
    m = LearnedEmotionModel(make_artefact({**FULL_AGREEING, ("sadness", "arm_swing_amp"): +0.3}))
    with pytest.raises(AssertionError, match="sadness.arm_swing_amp"):
        check_directions(m)                       # the body of the parametrised test, factored out

def test_a_reported_contradiction_passes():
    m = LearnedEmotionModel(make_artefact({**FULL_AGREEING, ("sadness", "arm_swing_amp"): +0.3}, contradictions=[{"emotion": "sadness", "parameter": "arm_swing_amp"}]))
    check_directions(m)
```

- [ ] **Step 5: Add the learned presets to the feasibility test**

In `tests/gait/test_generator.py::test_every_preset_is_feasible`, extend `cases` with `{f"learned_{k}": v for k, v in learned_presets(LearnedEmotionModel.default()).items()}` *only if* `default_artefact_path().exists()` (so Tasks 7 and 8 stay green before Task 9 commits the artefact), with an `assert` message naming the preset and the report (Review Focus 5).

- [ ] **Step 6: Run the emotion, gait and CLI suites**

Run: `pytest tests/emotion tests/gait tests/test_cli.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/emotive_gait/emotion pyproject.toml tests/emotion tests/gait/test_generator.py
git commit -m "feat(emotion): EmotionModel protocol, AnalyticEmotionModel, LearnedEmotionModel with per-cell delegation; direction tests on both"
```

---

### Task 8: `learn/artefact.py`, `learn/report.py`, and the CLI (`fit`, `fetch`, `--model`, `evidence --model`)

**Files:**
- Create: `src/emotive_gait/learn/artefact.py`, `src/emotive_gait/learn/report.py`, `src/emotive_gait/learn/pipeline.py`
- Modify: `src/emotive_gait/cli.py` (new subcommands and flags; `_demo` passes `model=`; `_evidence` takes `model`), `tests/test_cli.py`
- Test: `tests/learn/test_pipeline.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: Tasks 4-7.
- Produces:
  - `artefact.py`: `write_artefact(path: Path, cells: Mapping[Scaling, Mapping[str, list[Cell]]], *, provenance: Mapping[str, object]) -> Artefact` (computes `direction_check` per cell against `evidence.claims_for`, builds `contradictions` from every `contradicts*` cell, writes YAML with sorted keys, returns what `load_artefact` would read).
  - `report.py`: `write_report(path: Path, *, held_out: list[HeldOut], structure: list[StructureResult], artefact: Artefact, speed_adjusted: Mapping, provenance: Mapping) -> None` (Markdown: headline table E_fit / E_hand / E_zero per parameter and overall; per-cell table with interval or actor count; per-dataset diagnostics; structure table with the spec's caveat sentence verbatim; contradictions with citations; the nine limitations of spec 6).
  - `pipeline.py`: `run_fit(manifest_path: Path, root: Path, *, out_artefact: Path, out_report: Path, derived: Path, datasets: Iterable[str] | None, include_uncommittable: bool, seed: int = 0) -> FitSummary` (verify -> table -> both scalings -> fit -> LOAO -> structure -> speed-adjusted -> artefact -> report; refuses to write to the package artefact path when any included dataset is `committable: false` unless `include_uncommittable=True` *and* `out_artefact` is outside `src/`), raising `FitInputError(ValueError)` for manifest or hash problems (message lists each `HashProblem`) and `FitEnvironmentError(RuntimeError)` when `statsmodels` is missing.
  - CLI: `emotive-gait fetch [--root data/raw] [--from ID=PATH ...] [--datasets ID ...]`; `emotive-gait fit [--root data/raw] [--manifest data/manifest.yaml] [--datasets ID ...] [--include-uncommittable --out PATH] [--check]` (`--check` refits to a temporary file and compares every `estimate` with the artefact at `--out` or the package path within 1e-6; a difference is exit 2 naming the first differing cell); `demo --model {analytic,learned} --scaling {froude,ratio}`; `evidence --model {analytic,learned}`. Exit 2 for `FitInputError`, 1 for `FitEnvironmentError`, `ArtefactMissingError` and download failures.

- [ ] **Step 1: Write the failing pipeline test on synthetic data**

```python
def test_run_fit_writes_a_loadable_artefact_and_a_report(tmp_path, synthetic_dataset_on_disk):
    summary = run_fit(synthetic_dataset_on_disk.manifest, synthetic_dataset_on_disk.root,
                      out_artefact=tmp_path / "coef.yaml", out_report=tmp_path / "report.md",
                      derived=tmp_path / "derived", datasets=None, include_uncommittable=False)
    art = load_artefact(tmp_path / "coef.yaml")
    assert ("froude", "anger", "cadence") in art.cells and ("ratio", "anger", "cadence") in art.cells
    assert art.provenance["manifest_sha256"] == summary.manifest_sha256 and art.provenance["seed"] == 0
    text = (tmp_path / "report.md").read_text()
    assert "E_fit" in text and "E_zero" in text and "five group means" in text

def test_run_fit_refuses_uncommittable_data_into_the_package_path(tmp_path, synthetic_dataset_on_disk):
    with pytest.raises(FitInputError, match="committable"):
        run_fit(..., datasets=["toy_uncommittable"], out_artefact=default_artefact_path(), include_uncommittable=False)

def test_hash_mismatch_is_a_fit_input_error_naming_the_file(tmp_path, synthetic_dataset_on_disk):   # Review Focus 3
    (synthetic_dataset_on_disk.root / "toy" / "walk_a.bvh").write_text("tampered")
    with pytest.raises(FitInputError, match="walk_a.bvh"):
        run_fit(...)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/learn/test_pipeline.py -q`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `artefact.py`, `report.py`, `pipeline.py`; run the test**

Run: `pytest tests/learn -q`
Expected: PASS

- [ ] **Step 4: Write the failing CLI tests**

```python
def test_demo_learned_without_artefact_exits_one_with_the_fit_hint(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr("emotive_gait.emotion.learned.default_artefact_path", lambda: tmp_path / "none.yaml")
    assert main(["demo", "--emotion", "sadness=1.0", "--model", "learned", "--dry-run"]) == 1
    assert "emotive-gait fit" in capsys.readouterr().err

def test_demo_learned_prints_the_model_and_warns_on_thin_cells(capsys, fixture_artefact_path, monkeypatch):   # Review Focus 4
    monkeypatch.setattr("emotive_gait.emotion.learned.default_artefact_path", lambda: fixture_artefact_path)
    assert main(["demo", "--emotion", "anger=1.0", "--model", "learned", "--dry-run"]) == 0
    out, err = capsys.readouterr()
    assert "model: learned (froude)" in out and "thin" in err and "anger.cadence" in err

def test_evidence_learned_adds_estimate_columns(capsys, fixture_artefact_path, monkeypatch):
    monkeypatch.setattr("emotive_gait.emotion.learned.default_artefact_path", lambda: fixture_artefact_path)
    assert main(["evidence", "--model", "learned"]) == 0
    assert "ESTIMATE" in capsys.readouterr().out

def test_fit_with_a_missing_manifest_exits_two(capsys, tmp_path):
    assert main(["fit", "--manifest", str(tmp_path / "nope.yaml")]) == 2
    assert "nope.yaml" in capsys.readouterr().err

def test_fit_without_statsmodels_exits_one(capsys, monkeypatch, tmp_path, synthetic_dataset_on_disk):
    monkeypatch.setitem(sys.modules, "statsmodels", None)
    assert main(["fit", "--manifest", str(synthetic_dataset_on_disk.manifest), "--root", str(synthetic_dataset_on_disk.root), "--include-uncommittable", "--out", str(tmp_path / "c.yaml")]) == 1
    assert "pip install" in capsys.readouterr().err

def test_fit_check_passes_on_a_matching_artefact_and_fails_on_a_changed_one(tmp_path, synthetic_dataset_on_disk):
    out = tmp_path / "c.yaml"
    args = ["fit", "--manifest", str(synthetic_dataset_on_disk.manifest), "--root", str(synthetic_dataset_on_disk.root), "--include-uncommittable", "--out", str(out)]
    assert main(args) == 0 and main([*args, "--check"]) == 0
    out.write_text(out.read_text().replace("estimate: 0.0", "estimate: 9.9", 1))
    assert main([*args, "--check"]) == 2
```

`fixture_artefact_path` writes `make_artefact(FULL_AGREEING, thin=[("anger", "cadence")])` through `write_artefact` to a temp file. `demo --model learned` must not import `statsmodels` (assert with `monkeypatch.setitem(sys.modules, "statsmodels", None)` in the second test too).

- [ ] **Step 5: Implement the CLI changes; run the CLI tests and the whole suite**

Run: `pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/emotive_gait/learn src/emotive_gait/cli.py tests/learn tests/test_cli.py
git commit -m "feat(cli): fit, fetch, --model and evidence --model; artefact writer and fit report"
```

---

### Task 9: Real data: fetch, freeze the manifest, fit, commit the artefact and report

**Files:**
- Modify: `data/manifest.yaml` (frozen file lists, hashes, label map, actor identities, exclusions), `src/emotive_gait/mocap/landmarks.py` (verified joint names), `tests/gait/test_generator.py` (the learned presets now run, from Task 7 step 5)
- Create: `src/emotive_gait/emotion/data/learned_coefficients.yaml`, `docs/reports/phase-4-5-fit-report.md`
- Test: `tests/emotion/test_artefact_committed.py`

**Interfaces:**
- Consumes: everything above.
- Produces: the committed artefact and report; `tests/emotion/test_artefact_committed.py`.

- [ ] **Step 1: Fetch**

Run: `emotive-gait fetch --from 100style=C:\Users\attan\Downloads\100STYLE.zip`
Expected: the CMU ASF/AMC files (subjects 82, 91, 104, 105, 142, every walk trial named in the manifest plus each subject's normal walk) and BML's `full_set_of_ptd_data.zip` download; 100STYLE is copied, not downloaded. The user's copy is 1,468,168,639 bytes against the 1,468,515,354 the datasets report saw: fetch the Zenodo record's checksum (`https://zenodo.org/api/records/8127870`), put it in the manifest, and if the local file does not match, download afresh and tell the user. Any URL that fails: print the URL and expected hash, continue, and report at the end; the user fetches that file by hand.

- [ ] **Step 2: Verify the dataset facts the spec left to the builder, and freeze the manifest**

Record each in the manifest's `notes:` and in the commit message:
- 100STYLE: bone names from one BVH header; the unit (`scale`) from the hip height of the Neutral idle (a 1.7 m actor's hips sit near 0.9-1.0 m); `Frame_Cuts.csv` start and end for `Angry_FW`, `Depressed_FW`, `Elated_FW`, `Neutral_FW`; `Proud` excluded with reason.
- CMU: for each subject the trial numbers of the normal walk and of Sad/Depressed, Happy/Elated/Joy/HappyGoLucky, Scared; whether 91 and 105 are one performer (spec 4.1: merged unless the pages say otherwise); subjects without a usable normal walk excluded with reason; styles that are not basic emotions excluded.
- BML PTD: the 15 column names from the data page; whether a head and a neck point exist (if not, `neck_pitch` is `missing` for BML and the CSM fallback is *not* built in this phase: record it as a follow-up); the start/end-frame files; `record 1` and `record 2` directions.
- Label kinds per spec 4.1: `exact` for angry, happy, sad, scared, neutral/normal; `mapped` for Depressed, Elated, Joy, HappyGoLucky.
- BML entry stays `committable: false` unless the user has forwarded the PACO lab's written permission; then set `committable: true` and paste the reference into `permission:`.

- [ ] **Step 3: Run the fit on the committable tier and read the report**

Run: `emotive-gait fit`
Expected: the artefact at the package path and the report at `docs/reports/phase-4-5-fit-report.md`; the console prints the headline row, the thin cells, the contradictions and any `converged_maximal: false` cells. If `fit` reports zero cycles for a clip, inspect `data/derived/cycles.csv` and the discard reasons before touching any threshold; thresholds were frozen in Task 3 and change only with a new round-trip run.

- [ ] **Step 4: Run the fit including BML locally (not committed unless permitted)**

Run: `emotive-gait fit --include-uncommittable --out data/derived/with_bml/coef.yaml`
Expected: a local artefact and report under `data/derived/`, used to check the pipeline on 30 actors and to have the numbers ready for the day permission arrives. Nothing from this run is pasted anywhere committed.

- [ ] **Step 5: Write the committed-artefact tests**

```python
# tests/emotion/test_artefact_committed.py
def test_committed_artefact_loads_and_matches_the_manifest():
    art = load_artefact(default_artefact_path())
    assert art.provenance["manifest_sha256"] == sha256_of(Path("data/manifest.yaml"))
    assert all(ds["committable"] for ds in art.provenance["datasets_included"])

def test_every_cell_has_a_grounding_and_fitted_cells_have_intervals():
    for cell in load_artefact(default_artefact_path()).cells.values():
        assert cell.grounding in ("fitted", "thin", "missing")
        assert (cell.ci_low is not None) == (cell.grounding == "fitted")

def test_committed_direction_checks_are_consistent_with_evidence():
    # the parametrised direction test of Task 7 now also runs on LearnedEmotionModel.default()
    check_directions(LearnedEmotionModel.default())
    check_directions(LearnedEmotionModel.default(scaling="ratio"))
```

Run: `pytest tests/emotion/test_artefact_committed.py tests/gait/test_generator.py::test_every_preset_is_feasible -q`
Expected: PASS. If a learned preset is infeasible, STOP: do not edit the artefact or the thresholds; report the preset, the parameter and the IK residual to the user (spec 4.5) and continue with the remaining tasks only after their decision.

- [ ] **Step 6: Run the whole suite and commit**

Run: `pytest -q`

```bash
git add data/manifest.yaml src/emotive_gait/mocap/landmarks.py src/emotive_gait/emotion/data/learned_coefficients.yaml docs/reports/phase-4-5-fit-report.md tests/emotion/test_artefact_committed.py
git commit -m "data: frozen manifest (100STYLE, CMU; BML pending permission); fitted coefficients and fit report"
```

Controller: after this commit, tell the user the headline numbers, the thin cells, the contradictions, and whether BML is in or out.

---

### Task 10: README, `CLAUDE.md`, docs, version

**Files:**
- Modify: `README.md` (Architecture `emotion/` paragraph; new "Two emotion models" section with the results table from the report; Related work bullets per spec 8; attributions; the Quickstart gains `emotive-gait demo --emotion sadness=0.8 --model learned` and `emotive-gait fit`), `CLAUDE.md` (Commands; Architecture: `mocap/` and `learn/` with the spec 5 dependency rules; Facts that bite: artefact is generated, grounding per cell, thin and missing, `leg_length` and the posture-delta round trip, aggregates-only and the BML switch, the no-change claim rule; Conventions: `fit` exit codes), `docs/future-learned-model.md` (replace the body with a pointer to the spec and the report), `pyproject.toml` and `src/emotive_gait/__init__.py` (both to `0.3.0`; the package currently says `0.1.0` against `0.2.0` in `pyproject.toml`), `tests/test_packaging.py` (assert the two versions agree)
- Test: `tests/test_packaging.py`

**Interfaces:** none new.

- [ ] **Step 1: Write the failing version test**

```python
def test_package_version_matches_pyproject():
    import tomllib
    data = tomllib.loads(Path("pyproject.toml").read_text())
    assert emotive_gait.__version__ == data["project"]["version"] == "0.3.0"
```

- [ ] **Step 2: Set both versions; run the test**

Run: `pytest tests/test_packaging.py -q`
Expected: PASS

- [ ] **Step 3: README edits**

Related work, exactly per spec 8: PAMoR's bullet becomes "a learned latent-diffusion generator with a continuous valence-arousal control: close in the knob, opposite in mechanism"; the "Walk Like This" bullet is removed; EMoG's "which robot it used is unverified" changes to name the Unitree G1 *only if* the builder opens `https://arxiv.org/html/2609.14432v2` and quotes the sentence in the commit message, otherwise it stays. The results section copies the headline table and the contradictions from `docs/reports/phase-4-5-fit-report.md`, states N actors and D datasets, prints the spec 6 limitations 1, 2, 5 and 7 in one short paragraph, and keeps "No recognition rate has been measured". Attribution lines: "The 100STYLE Dataset - Ian Mason" (CC BY 4.0); CMU Graphics Lab Motion Capture Database acknowledgement; BML citation only if included.

- [ ] **Step 4: `CLAUDE.md` and `docs/future-learned-model.md`**

- [ ] **Step 5: Final gate and commit**

Run: `ruff check . && ruff format --check . && pytest -q`
Expected: all green.

```bash
git add README.md CLAUDE.md docs/future-learned-model.md pyproject.toml src/emotive_gait/__init__.py tests/test_packaging.py
git commit -m "docs: Phase 4.5 learned emotion model; README positioning and corrections; version 0.3.0"
```

Controller: hand the branch to a whole-branch review (most capable model), then present the user with the merge decision and the Phase 4.6 hand-off. Follow-ups to list, not do: the CSM reader if BML lacks a head point; BML inclusion on permission (`committable: true`, re-run `fit`, commit); the `0.1.0` / `0.2.0` version drift that this plan fixed in passing.
