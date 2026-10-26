# Phase 4.6 Emotion for Any Rig Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read any BVH walk, measure its gait parameters with the existing extractor, move them by an emotion through either emotion model re-dimensioned to that rig, re-synthesise the walk on a MuJoCo chain compiled from the BVH hierarchy with the existing leg IK, and write a BVH back with the same hierarchy; prove round-trip fidelity and zero foot slip in CI; expose it as a CLI and an MCP server.

**Architecture:** A new top package `src/emotive_gait/rig/` sits above the six existing ones. `mocap/readers/bvh.py` keeps the whole file (`BvhFile`) and `mocap/writers/bvh.py` writes one back. `rig/compile.py` turns a `BvhFile` into a MuJoCo chain (`full`: every channel a hinge, for the FK gate and probes; `driven`: only role-bound hinges, with the actor's mean walking pose baked in). `rig/mapping.py` binds bone names to `Landmark`s and `JointRole`s by a name lexicon, rest-pose geometry and probes, and `rig/calibration.py` measures the reference pose and the affine posture map that makes the generator's absolute postures mean on a rig what they mean on the G1. `rig/synthesis.py` runs the pipeline end to end and measures the round trip and the slip; `rig/cli.py` and `rig/server.py` expose it. The G1 path is bit-identical.

**Tech Stack:** Python >= 3.11, NumPy 2.4, SciPy 1.17, `mujoco>=3.17,<3.18`, `mink 3.17`, `pyyaml`; `mcp` (official Python SDK, 2.x) in a new `mcp` extra included in `dev`; pytest, ruff 0.17.2.

**Spec:** `docs/superpowers/specs/2026-10-26-phase-4-6-emotion-for-any-rig-design.md` (approved 2026-10-26, the "modified gait from parameters" reading). The Phase 4.5 design stays binding for the extractor.

## Global Constraints

- **Layering (spec 5):** `rig/` may import everything except `learn/` and `sim/`; only `cli.py` imports `rig/`. `rig/server.py` is the only module that imports `mcp`, inside the function that starts the server. `mocap/` still imports only NumPy, SciPy and `gait.params`. `emotion/prompt.py` imports only `emotion.evidence` and `pyyaml`. `robot/` imports nothing from the other packages.
- **G1 bit-identical:** every existing test passes unchanged; `generate` gains `welded: RobotModel | None = None` and reads `profile.posture_map` (identity when empty); `solve_leg_cycle` is unchanged; the G1's two round-trip xfails stay.
- **No LLM inside the package**, no `anthropic` dependency, no API key (spec 4.4, 4.8).
- **Extractor thresholds and definitions are frozen** (`mocap/events.py`, `mocap/extract.py` constants): never retuned for a rig (spec 3).
- **One foot point everywhere (spec 4.6):** the toe landmark when the rig has one (a toe bone, else the foot's End Site), the ankle otherwise; the IK site, the slip measurement and the extractor all use it. The floor is the median height of that point over the source's stance frames, both feet pooled.
- **Role sign convention (spec 4.5):** `ROLE_CONVENTIONS` in `robot/roles.py`; on a rig every sign is baked into the hinge axis so `sign_for` is all `+1.0`; the G1 elbow row is a strict xfail.
- **Usable rig minimum (spec 4.4):** `PELVIS`, both `HIP`, `KNEE`, `ANKLE` landmarks and all twelve `gait.ik.LEG_ROLES`; otherwise exit 2 naming what is missing.
- **Ranges (spec 4.3), radians about the calibrated zero, in the convention's positive direction:** knee `[0, 2.6]`; hip pitch `[-1.2, 2.0]`; hip roll and yaw `[-0.8, 0.8]`; ankle pitch `[-1.0, 1.0]`; ankle roll `[-0.6, 0.6]`; waist pitch `[-0.8, 0.8]`; neck pitch `[-0.8, 0.8]`; shoulder pitch `[-3.0, 3.0]`; shoulder roll `[-0.3, 2.5]`; elbow `[0, 2.6]`.
- **Inference (spec 4.4):** up-axis = the rest-hierarchy axis of largest pelvis-to-head extent; scale = the one of `(1, 0.01, 0.0254, 0.1, 0.001)` that puts the hip-to-ankle rest distance nearest 0.8 m, accepted only inside `[0.3, 1.3]` m, else exit 2 asking for `--scale`.
- **Claims (spec 4.6, 4.7):** round trip within cadence and step length 2 %, step height, bounce and sway 3 mm, angles 0.02 rad, `elbow_swing_ratio` 0.05, double support against `rule_double_support`; slip: horizontal travel of the foot point over each commanded stance <= 2 mm and height within 2 mm of the floor, on feasible parameters; infeasible is flagged and still written.
- **Naming fixed here:** hinge joints and actuators are named `f"{joint}/{channel}"` (`LeftUpLeg/Xrotation`); End Sites appear in a clip as `f"{joint}_End"`; foot sites are `"foot_l"` and `"foot_r"`; the compiled root body is `"pelvis"`; the probe angle is 0.05 rad; the `axis_purity` threshold is 0.8.
- **Data (spec 2, 4.7):** only the 100STYLE excerpt (CC BY 4.0, attribution sidecar) is committed as a fixture; Bandai Namco (CC BY-NC 4.0) and the cgspeed CMU conversion are fetched through the manifest and never committed; Mixamo files are never committed in any form; the validation report carries aggregates only.
- **Repository conventions:** exit codes 2 (user input) and 1 (environment), no traceback escapes `main`; no magnitude in degrees from a citation; disgust and surprise stay unsourced with the warning; `docs/reports/*.md` are generated, LF-pinned; commit messages carry no Claude attribution trailer; merging and pushing are the user's.

## Review Focus

Input classes the spec implies but no happy-path test exercises. Each has its test pinned to the owning task.

1. **A BVH whose root is a floor-level reference bone with the pelvis as a child** must map, compile rooted at the pelvis, and write back with the ancestors held and the root translation corrected; the output must re-read with the pelvis where the pipeline put it. Owned by Tasks 3, 6 and 8.
2. **A Blender export (z-up, metres) and a file whose units cannot be inferred** must respectively infer correctly and stop with exit 2 asking for `--scale`, never produce a 100 m character. Owned by Tasks 5 and 10.
3. **A clip with no usable cycle (standing, in place, turning throughout)** must stop with exit 2 and the discard reasons in the CLI, and return a tool error, not a crash, over MCP. Owned by Tasks 8, 10 and 11.
4. **A mapping override naming a bone or channel the file does not have** must stop with exit 2 naming it. Owned by Tasks 5 and 10.
5. **An emotion that puts the rig beyond its reach** must print the existing one-line infeasibility warning, still write the file, and report the slip numbers rather than hide them. Owned by Tasks 8 and 10.

## Execution notes

Suggested model per task, implementer / reviewer (user preference: cheap for transcription, standard for integration, most capable for load-bearing geometry and the final review): T1 opus / fable; T2 sonnet / opus; T3 sonnet / opus; T4 opus / fable; T5 opus / fable; T6 opus / fable; T7 opus / fable; T8 opus / fable; T9 sonnet / opus; T10 sonnet / opus; T11 sonnet / opus; T12 opus / fable (real data and the committed report); T13 haiku / sonnet; whole-branch review fable.

Branch: `phase-4-6-emotion-for-any-rig` (exists; base 149776e, the Phase 4.5 head; the design is committed as 7cee34a). Work in place: the `.venv` editable install resolves to this checkout. Every task ends with `ruff check . && ruff format --check . && pytest -q` green; the steps name only the task's own tests and that gate is implied. The venv has NumPy 2.4.6, SciPy 1.17.1, MuJoCo 3.17.0, mink 3.17.0 on Python 3.11.3; `mcp` is not installed until Task 11. `data/raw/100style/Neutral_FW.bvh` is on disk (fetched in Phase 4.5; 60 fps, centimetres, y-up, FW cut 327-7445).

Drop order if the schedule slips (spec 9): the Bandai and CMU rows of Task 12; the `rig-mapping-schema` resource (Task 11); `--json` (Task 10); negation in the affect lexicon (Task 9); `rig validate --report` (Task 12 then fills the report by hand from the test output and says so). Never dropped: Tasks 1-8, Task 11's five tools and smoke test, Task 10's `inspect` output.

---

### Task 1: Role conventions, the posture map, `generate(welded=)`, `rule_double_support`

**Files:**
- Create: `src/emotive_gait/robot/conventions.py`
- Modify: `src/emotive_gait/robot/roles.py` (`RobotProfile`), `src/emotive_gait/gait/generator.py` (`generate`, `GaitTrajectory._procedural`), `src/emotive_gait/gait/footsteps.py`, `tests/mocap/test_roundtrip.py` (import `rule_double_support` from `footsteps`, delete the local copy)
- Test: `tests/robot/test_conventions.py`, `tests/gait/test_generator.py` (one new test), `tests/gait/test_footsteps.py` (one new test)

**Interfaces:**
- Consumes: `RobotModel`, `weld_robot_for`, `JointRole`, `GaitTrajectory`.
- Produces:
  - `robot/roles.py`: `POSTURE_PARAMETERS: tuple[str, ...] = ("trunk_pitch", "neck_pitch", "elbow_flexion", "arm_abduction")`; `POSTURE_ROLES: tuple[JointRole, ...] = (WAIST_PITCH, NECK_PITCH, ELBOW_L, ELBOW_R, SHOULDER_ROLL_L, SHOULDER_ROLL_R)`; `RobotProfile.posture_map: Mapping[JointRole, Mapping[str, float]] = field(default_factory=dict)`, one row per posture role with keys from `POSTURE_PARAMETERS` plus `"offset"`; a role without a row copies its own parameter as today (rolls times `sign`).
  - `robot/conventions.py`: `ROLE_CONVENTIONS: Mapping[JointRole, tuple[str, str]]`, role -> (landmark value moved, direction), with directions `"forward" | "backward" | "outward" | "down" | "toward_shoulder"`: `HIP_PITCH_L: ("knee_l", "backward")`, `KNEE_L: ("ankle_l", "backward")`, `ANKLE_PITCH_L: ("toe_l", "down")`, `HIP_ROLL_L: ("knee_l", "outward")`, `SHOULDER_PITCH_L: ("elbow_l", "backward")`, `SHOULDER_ROLL_L: ("elbow_l", "outward")`, `ELBOW_L: ("wrist_l", "toward_shoulder")`, `WAIST_PITCH: ("neck", "forward")`, `NECK_PITCH: ("head", "forward")`, right side mirrored; `HIP_YAW_*`, `ANKLE_ROLL_*`, `WAIST_YAW`, `WAIST_ROLL` are absent (IK-solved or unused, symmetric ranges, no probe). `PROBE_RAD = 0.05`. `def convention_holds(role: JointRole, before: Mapping[str, np.ndarray], after: Mapping[str, np.ndarray], *, forward: np.ndarray, left: np.ndarray, up: np.ndarray) -> bool` (`before`/`after` are landmark value -> world point; "outward" is `+left` for `_l` roles and `-left` for `_r`; "toward_shoulder" is a shorter distance to `shoulder_<side>`).
  - `gait/footsteps.py`: `def rule_double_support(params: GaitParameters, samples: int = 4000) -> float` (the body of the test helper, unchanged).
  - `gait/generator.py`: `def generate(params, robot, *, welded: RobotModel | None = None, offsets=None, overrides=None) -> GaitTrajectory`; with `welded` the leg solve runs on it, else on `weld_robot_for(robot.profile)`. `GaitTrajectory` gains `_posture_map: tuple[tuple[JointRole, tuple[tuple[str, float], ...]], ...]`.

- [ ] **Step 1: Write the failing convention test on the G1**

```python
# tests/robot/test_conventions.py
G1_POINTS = {**{lm.value: body for lm, body in G1_LANDMARKS.items() if body != HEAD_TOP}}  # from mocap.g1_landmarks
# toe_l / toe_r: the foot site position plus 0.10 m along the site's own x axis (G1 has no toe body)
# head: head_link position plus HEAD_TOP_OFFSET

@pytest.mark.parametrize("role", [r for r in ROLE_CONVENTIONS if r is not JointRole.ELBOW_L and r is not JointRole.ELBOW_R])
def test_g1_positive_joint_motion_follows_the_convention(weld, role):
    before, after = probe_g1(weld, role)   # qpos = stand; after adds PROBE_RAD * profile.sign(role) at the role's joint
    assert convention_holds(role, before, after, forward=X, left=Y, up=Z)

@pytest.mark.xfail(strict=True, reason="G1 elbow joint straightens when positive (Phase 4 rendering finding)")
@pytest.mark.parametrize("role", [JointRole.ELBOW_L, JointRole.ELBOW_R])
def test_g1_elbow_follows_the_convention(weld, role): ...

def test_convention_holds_is_pure_geometry():
    before = {"knee_l": np.array([0.0, 0.1, 0.4])}; after = {"knee_l": np.array([-0.01, 0.1, 0.4])}
    assert convention_holds(JointRole.HIP_PITCH_L, before, after, forward=X, left=Y, up=Z)
    assert not convention_holds(JointRole.HIP_PITCH_L, after, before, forward=X, left=Y, up=Z)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/robot/test_conventions.py -v`
Expected: FAIL with `ModuleNotFoundError: emotive_gait.robot.conventions`

- [ ] **Step 3: Implement `robot/conventions.py` and the `RobotProfile` fields**

`convention_holds` projects `after - before` of the named landmark on the direction's unit vector and requires a positive component (`toward_shoulder`: distance decreases). Add `POSTURE_PARAMETERS`, `POSTURE_ROLES`, `posture_map` to `roles.py`. If a G1 row other than the elbow fails, correct that row of `ROLE_CONVENTIONS` (the G1 is the reference, spec 4.5) and say so in the commit message.

- [ ] **Step 4: Run the convention tests**

Run: `pytest tests/robot/test_conventions.py -v`
Expected: PASS, with the two elbow cases XFAIL

- [ ] **Step 5: Write the failing generator and footsteps tests**

```python
# tests/gait/test_generator.py
def test_posture_map_writes_an_affine_combination_of_the_posture_parameters(robot):
    profile = replace(robot.profile, posture_map={JointRole.WAIST_PITCH: {"trunk_pitch": 0.5, "neck_pitch": 0.0, "elbow_flexion": 0.0, "arm_abduction": 0.0, "offset": 0.1}})
    traj = generate(GaitParameters(trunk_pitch=0.2), replace(robot, profile=profile), welded=weld_robot_for(robot.profile))
    assert traj.angles_at(0.3)[JointRole.WAIST_PITCH] == pytest.approx(0.2)   # 0.5 * 0.2 + 0.1
    plain = generate(GaitParameters(trunk_pitch=0.2), robot)
    assert plain.angles_at(0.3)[JointRole.WAIST_PITCH] == pytest.approx(0.2)  # identity path unchanged

# tests/gait/test_footsteps.py
def test_rule_double_support_reads_above_the_parameter_on_the_generator_shape():
    assert rule_double_support(GaitParameters()) == pytest.approx(0.41, abs=0.02)   # 0.20 + the ~0.21 G1 offset
```

- [ ] **Step 6: Run them to verify they fail**

Run: `pytest tests/gait/test_generator.py::test_posture_map_writes_an_affine_combination_of_the_posture_parameters tests/gait/test_footsteps.py::test_rule_double_support_reads_above_the_parameter_on_the_generator_shape -v`
Expected: FAIL (`TypeError: unexpected keyword 'welded'`; `ImportError: rule_double_support`)

- [ ] **Step 7: Implement `generate(welded=)`, the posture map in `_procedural`, and move `rule_double_support`**

In `_procedural`, build `posture = {"trunk_pitch": p.trunk_pitch, "neck_pitch": p.neck_pitch, "elbow_flexion": p.elbow_flexion, "arm_abduction": abduction}` and a local `mapped(role, default)` returning `row["offset"] + sum(row[k] * posture[k])` when the role has a row, else `default`; use it for the rolls (default `signs[roll] * abduction`), the elbows (default `p.elbow_flexion`, the swing term still added after), `WAIST_PITCH` and `NECK_PITCH`. `tests/mocap/test_roundtrip.py` imports `rule_double_support` from `emotive_gait.gait.footsteps`.

- [ ] **Step 8: Run the whole suite to prove the G1 is bit-identical**

Run: `pytest -q`
Expected: PASS, same xfails as before (the round trip's elbow and trunk rows)

- [ ] **Step 9: Commit**

```bash
git add src/emotive_gait/robot src/emotive_gait/gait tests/robot tests/gait tests/mocap/test_roundtrip.py
git commit -m "feat(robot,gait): role sign conventions with a G1 probe test, RobotProfile.posture_map, generate(welded=), rule_double_support in footsteps"
```

---

### Task 2: `BvhFile`: read everything, write it back, Euler decomposition; the 100STYLE fixture

**Files:**
- Modify: `src/emotive_gait/mocap/readers/bvh.py`
- Create: `src/emotive_gait/mocap/writers/__init__.py`, `src/emotive_gait/mocap/writers/bvh.py`, `tests/mocap/fixtures/style100_neutral_4s.bvh`, `tests/mocap/fixtures/style100_neutral_4s.ATTRIBUTION.txt`
- Test: `tests/mocap/test_bvh_file.py`

**Interfaces:**
- Consumes: the existing parser (`_parse_hierarchy`, `rotation_about`), `SkeletonClip`, `to_z_up`.
- Produces:
  - `@dataclass(frozen=True) class BvhJoint: name: str; parent: int; offset: np.ndarray  # (3,) file units; channels: tuple[str, ...]; end_site: np.ndarray | None`
  - `@dataclass(frozen=True) class BvhFile: joints: tuple[BvhJoint, ...]; motion: np.ndarray  # (T, C) file units and degrees; frame_time: float; scale: float; up_axis: Literal["y", "z"]; source: str` with `fps` (property), `channel_index(joint: str, channel: str) -> int`, `channel_slice(joint: str) -> slice`, `to_clip(*, end_sites: bool = False) -> SkeletonClip` (End Sites appended as joints named `f"{joint}_End"`, parent = that joint), `with_motion(motion: np.ndarray) -> BvhFile`, `world_rotations(frame: int) -> np.ndarray  # (J, 3, 3) in the file's own axes`.
  - `read_bvh_file(path: Path, *, scale: float, up_axis: Literal["y", "z"]) -> BvhFile`; `read_bvh` keeps its signature and returns `read_bvh_file(...).to_clip()`.
  - `euler_from_matrix(r: np.ndarray, channels: tuple[str, ...]) -> np.ndarray  # degrees, one per rotation channel, such that composing them in channel order with rotation_about gives r` (all six orders; gimbal lock resolved by zeroing the last channel).
  - `mocap/writers/bvh.py`: `write_bvh(file: BvhFile, path: Path) -> None` (hierarchy in file order, `Frames:` and `Frame Time:` from the object, motion with `%.6f`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/mocap/test_bvh_file.py
def test_read_bvh_file_keeps_hierarchy_channels_and_motion():
    f = read_bvh_file(FIX / "two_bone.bvh", scale=0.01, up_axis="y")
    assert [j.name for j in f.joints] == ["Hips", "Knee"] and f.joints[1].channels == ("Zrotation", "Xrotation", "Yrotation")
    assert f.motion.shape == (3, 9) and f.channel_index("Knee", "Xrotation") == 7
    assert f.joints[1].end_site == pytest.approx([0.0, -40.0, 0.0])

def test_to_clip_matches_read_bvh_bit_for_bit():
    f = read_bvh_file(FIX / "two_bone.bvh", scale=0.01, up_axis="y")
    assert np.array_equal(f.to_clip().positions, read_bvh(FIX / "two_bone.bvh", scale=0.01, up_axis="y").positions)

def test_to_clip_can_append_end_sites_as_joints():
    clip = read_bvh_file(FIX / "two_bone.bvh", scale=0.01, up_axis="y").to_clip(end_sites=True)
    assert clip.joint_names[-1] == "Knee_End" and clip.parents[-1] == 1
    assert clip.positions[0, -1] == pytest.approx([0.0, 0.0, 0.2], abs=1e-6)

@pytest.mark.parametrize("order", ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"])
def test_euler_from_matrix_inverts_the_reader_composition(order):
    channels = tuple(f"{a}rotation" for a in order)
    rng = np.random.default_rng(0)
    for deg in rng.uniform(-170, 170, size=(50, 3)):
        r = compose(channels, deg)           # test helper: rotation_about in channel order
        assert np.allclose(compose(channels, euler_from_matrix(r, channels)), r, atol=1e-9)

def test_write_then_read_reproduces_the_file(tmp_path):
    f = read_bvh_file(FIX / "two_bone.bvh", scale=0.01, up_axis="y")
    write_bvh(f, tmp_path / "copy.bvh")
    g = read_bvh_file(tmp_path / "copy.bvh", scale=0.01, up_axis="y")
    assert [j.name for j in g.joints] == [j.name for j in f.joints] and g.joints[1].channels == f.joints[1].channels
    assert np.allclose(g.motion, f.motion, atol=1e-6) and g.frame_time == pytest.approx(f.frame_time)
    assert np.allclose(g.to_clip().positions, f.to_clip().positions, atol=1e-9)

def test_style100_excerpt_reads_and_has_240_frames():
    f = read_bvh_file(FIX / "style100_neutral_4s.bvh", scale=0.01, up_axis="y")
    assert f.motion.shape[0] == 240 and f.fps == pytest.approx(60.0, abs=0.01) and "LeftToe" in [j.name for j in f.joints]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/mocap/test_bvh_file.py -v`
Expected: FAIL with `ImportError: read_bvh_file`

- [ ] **Step 3: Implement `BvhFile`, `read_bvh_file`, `euler_from_matrix`, `write_bvh`**

`read_bvh_file` reuses `_parse_hierarchy` (extend `_Joint` with the End Site offset) and the motion parse; `to_clip` runs today's FK and converts with `to_z_up`. `euler_from_matrix` solves the three angles for the given order analytically (asin on the middle angle; when `|sin| > 1 - 1e-9` set the last channel to 0 and solve the first from the remaining matrix). The writer never reorders channels.

- [ ] **Step 4: Make the fixture excerpt**

Run (a one-off in the venv, not committed as code): read `data/raw/100style/Neutral_FW.bvh` with `read_bvh_file(scale=0.01, up_axis="y")`, keep motion rows 600 to 839, `write_bvh` to `tests/mocap/fixtures/style100_neutral_4s.bvh`. Write `style100_neutral_4s.ATTRIBUTION.txt`: "Excerpt (frames 600-839 of Neutral_FW.bvh) from 100STYLE, Mason, Starke & Komura 2022, https://doi.org/10.5281/zenodo.8127870, CC BY 4.0; unmodified except trimming." Add both paths to `.gitattributes` as `text eol=lf`.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/mocap/test_bvh_file.py tests/mocap/test_readers.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/emotive_gait/mocap tests/mocap .gitattributes
git commit -m "feat(mocap): BvhFile keeps hierarchy, channels and motion; BVH writer; Euler decomposition; 100STYLE neutral excerpt fixture (CC BY 4.0)"
```

---

### Task 3: The synthetic rig family and its test-side walker

**Files:**
- Create: `tests/rig/__init__.py`, `tests/rig/rigs.py`
- Test: `tests/rig/test_rigs.py`

**Interfaces:**
- Consumes: `BvhFile`, `BvhJoint`, `write_bvh`, `read_bvh_file`, `euler_from_matrix`.
- Produces (test-side only):
  - `@dataclass(frozen=True) class RigSpec: family: str = "mixamo"; order: str = "ZXY"; up_axis: str = "y"; units_per_m: float = 100.0; toes: bool = True; spine_bones: int = 3; fingers: bool = True; clavicles: bool = True; root_above_pelvis: bool = False; swap_rest_sides: bool = False; leg_length_m: float = 0.9`
  - `FAMILIES: Mapping[str, Mapping[str, str]]`, family -> canonical part -> bone name, for `mixamo` (`mixamorig:Hips`, `mixamorig:LeftUpLeg`, `LeftLeg`, `LeftFoot`, `LeftToeBase`, `Spine`, `Spine1`, `Spine2`, `Neck`, `Head`, `LeftShoulder`, `LeftArm`, `LeftForeArm`, `LeftHand`, `LeftHandIndex1`), `motionbuilder` (cgspeed CMU: `Hips`, `LHipJoint`, `LeftUpLeg`, `LeftLeg`, `LeftFoot`, `LeftToeBase`, `LowerBack`, `Spine`, `Spine1`, `Neck`, `Neck1`, `Head`, `LeftShoulder`, `LeftArm`, `LeftForeArm`, `LeftHand`), `style100` (`Hips`, `Chest`..`Chest4`, `Neck`, `Head`, `LeftCollar`, `LeftShoulder`, `LeftElbow`, `LeftWrist`, `LeftHip`, `LeftKnee`, `LeftAnkle`, `LeftToe`), `rokoko` (`Hips`, `LeftThigh`, `LeftShin`, `LeftFoot`, `LeftToe`, `Spine`, `Spine1`, `Spine2`, `Neck`, `Head`, `LeftShoulder`, `LeftArm`, `LeftForeArm`, `LeftHand`), `xsens` (`Hips`, `LeftUpperLeg`, `LeftLowerLeg`, `LeftFoot`, `LeftToe`, `Spine`, `Spine1`, `Spine2`, `Spine3`, `Neck`, `Head`, `LeftShoulder`, `LeftArm`, `LeftForeArm`, `LeftHand`), `smpl` (`Pelvis`, `L_Hip`, `L_Knee`, `L_Ankle`, `L_Foot` (the toe), `Spine1`..`Spine3`, `Neck`, `Head`, `L_Collar`, `L_Shoulder`, `L_Elbow`, `L_Wrist`, `L_Hand`), `rigify` (`DEF-spine`, `DEF-thigh.L`, `DEF-shin.L`, `DEF-foot.L`, `DEF-toe.L`, `DEF-spine.001`.., `DEF-shoulder.L`, `DEF-upper_arm.L`, `DEF-forearm.L`, `DEF-hand.L`), `unity` (`Hips`, `LeftUpperLeg`, `LeftLowerLeg`, `LeftFoot`, `LeftToes`, `Spine`, `Chest`, `UpperChest`, `Neck`, `Head`, `LeftShoulder`, `LeftUpperArm`, `LeftLowerArm`, `LeftHand`). The builder verifies spellings against the files in hand and documented references; a spelling it cannot verify is marked in a comment.
  - `build_rig(spec: RigSpec) -> BvhFile` (rest pose: A-pose arms 20 degrees from the trunk, legs straight, one zero frame; proportions scaled from `leg_length_m`; `root_above_pelvis` adds a `reference` root bone at the floor with the pelvis as its child; `swap_rest_sides` puts the bone named Left on the right of the pelvis).
  - `write_walk(spec: RigSpec, path: Path, *, seconds: float = 6.0, cadence: float = 1.8, step_length: float = 0.5, fps: float = 60.0) -> BvhFile` (joint-space walk: hip pitch sinusoid, half-wave knee flexion in swing, counter-phase arm swing, root translating at `step_length * cadence` along the family's forward, written in the spec's units, order and up-axis).
  - `FAMILY_VARIANTS: tuple[RigSpec, ...]`: the base spec plus one variant per factor (each other family; `XYZ`, `ZYX`; z-up; metres, inches, decimetres; no toes; one spine bone; no fingers; no clavicles; root above pelvis; swapped sides; 0.6 m and 1.1 m legs).
  - `walk_file(spec: RigSpec, directory: Path, **walk_kwargs) -> Path` (writes `write_walk` to `directory / f"{spec.family}.bvh"`), `standing_file(spec: RigSpec, directory: Path) -> Path` (120 frames of the rest pose), `walk_and_clip(spec, directory) -> tuple[BvhFile, SkeletonClip]` (read back at the spec's scale and up-axis, End Sites included), `pelvis_name(spec) -> str`, `toe_rest_height_m(spec) -> float`, `hip_to_ankle_rest_m(file) -> float`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/rig/test_rigs.py
@pytest.mark.parametrize("spec", FAMILY_VARIANTS, ids=lambda s: s.family + ("-" + s.order) + ("-z" if s.up_axis == "z" else ""))
def test_every_variant_reads_back_with_a_human_leg_and_cycles(tmp_path, spec):
    f = write_walk(spec, tmp_path / "walk.bvh")
    g = read_bvh_file(tmp_path / "walk.bvh", scale=1.0 / spec.units_per_m, up_axis=spec.up_axis)
    clip = g.to_clip(end_sites=True)
    assert abs(hip_to_ankle_rest_m(g) - spec.leg_length_m * 0.9) < 0.05     # thigh + shin are 90 % of leg_length_m
    assert clip.positions[-1, 0, 0] - clip.positions[0, 0, 0] > 3.0          # walked forward along +x

def test_root_above_pelvis_variant_has_a_reference_root():
    f = build_rig(replace(RigSpec(), root_above_pelvis=True))
    assert f.joints[0].name == "reference" and f.joints[1].parent == 0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/rig/test_rigs.py -v`
Expected: FAIL with `ModuleNotFoundError: rig.rigs` (add `tests/rig/__init__.py` and the `sys.path` line in `tests/conftest.py` as `tests/emotion` has)

- [ ] **Step 3: Implement `tests/rig/rigs.py`**

Joint angles to channels through `euler_from_matrix` for the spec's order; the walk's forward is +x in z-up terms, converted by the inverse of `to_z_up` when `up_axis == "y"`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/rig/test_rigs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/rig tests/conftest.py
git commit -m "test(rig): synthetic rig family (eight naming families, orders, units, up-axes, toes, spines, root above pelvis) with a joint-space walker"
```

---

### Task 4: `rig/compile.py` (`full` mode) and the FK-equivalence gate

**Files:**
- Create: `src/emotive_gait/rig/__init__.py`, `src/emotive_gait/rig/compile.py`
- Test: `tests/rig/test_compile.py`

**Interfaces:**
- Consumes: `BvhFile`, `RobotModel`, `to_z_up`.
- Produces:
  - `@dataclass(frozen=True) class CompiledRig: model: mujoco.MjModel; file: BvhFile; pelvis: str  # bone name; hinges: tuple[str, ...]  # hinge names in qpos order; body_of: Mapping[str, str]  # bone name -> body name ("pelvis" for the pelvis bone, the bone name otherwise)` with `qpos_from_motion(self, frame: int) -> np.ndarray` (every rotation channel in radians; only meaningful in `full` mode) and `landmark_sites: Mapping[str, str]  # clip joint name (incl. _End) -> site name`.
  - `compile_full(file: BvhFile, *, pelvis: str, pelvis_height: float = 1.0) -> CompiledRig`: one body per bone from `pelvis` down (ancestors of the pelvis are not compiled), `pos` = offset in metres and +z up, one hinge per rotation channel in channel order about that channel's axis (also mapped to +z up), range unlimited, one position actuator per hinge, a site at every bone origin and every End Site, root body `"pelvis"` at `(0, 0, pelvis_height)` with no joint.
  - `heading_matrix(forward: np.ndarray) -> np.ndarray  # 3x3 rotation taking +x to the horizontal unit forward, +z fixed`.

- [ ] **Step 1: Write the failing gate test**

```python
# tests/rig/test_compile.py
@pytest.mark.parametrize("spec", [RigSpec(), replace(RigSpec(), order="XYZ"), replace(RigSpec(), order="ZYX"), replace(RigSpec(), up_axis="z"), replace(RigSpec(), root_above_pelvis=True)])
def test_full_chain_reproduces_the_reader_positions_relative_to_the_pelvis(tmp_path, spec):
    f = write_walk(spec, tmp_path / "w.bvh"); f = read_bvh_file(tmp_path / "w.bvh", scale=1.0 / spec.units_per_m, up_axis=spec.up_axis)
    rig = compile_full(f, pelvis=pelvis_name(spec))
    clip = f.to_clip(end_sites=True); data = mujoco.MjData(rig.model)
    for frame in (0, 17, 150):
        data.qpos[:] = rig.qpos_from_motion(frame); mujoco.mj_forward(rig.model, data)
        for name, site in rig.landmark_sites.items():
            expected = clip.positions[frame, clip.joint_names.index(name)] - clip.positions[frame, clip.joint_names.index(pelvis_name(spec))]
            got = data.site_xpos[rig.model.site(site).id] - data.xpos[rig.model.body("pelvis").id]
            assert np.allclose(got, expected, atol=1e-6)

def test_full_chain_reproduces_style100():
    f = read_bvh_file(FIX / "style100_neutral_4s.bvh", scale=0.01, up_axis="y")
    ...same check on frames (0, 100, 239) with pelvis="Hips"...
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/rig/test_compile.py -v`
Expected: FAIL with `ModuleNotFoundError: emotive_gait.rig`

- [ ] **Step 3: Implement `compile_full` with `MjSpec`**

Hinges on one body compose intrinsically in definition order (verified on 3.17 during the design's builder check). The channel axes are the file's X/Y/Z mapped through the same permutation `to_z_up` applies to positions. A relative tolerance failure here is a convention bug in the compile, never in the reader.

- [ ] **Step 4: Run the gate**

Run: `pytest tests/rig/test_compile.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/rig tests/rig/test_compile.py
git commit -m "feat(rig): compile a BVH hierarchy into a MuJoCo chain (full mode) under an FK-equivalence gate"
```

---

### Task 5: `rig/mapping.py`: landmarks by name and geometry, units and up-axis inference, overrides

**Files:**
- Create: `src/emotive_gait/rig/mapping.py`, `src/emotive_gait/rig/data/bone_names.yaml`
- Modify: `pyproject.toml` (`[tool.setuptools.package-data] "emotive_gait.rig" = ["data/*.yaml"]`)
- Test: `tests/rig/test_mapping.py`

**Interfaces:**
- Consumes: `BvhFile`, `SkeletonClip`, `Landmark`, `LandmarkMap`, `mocap.events.path_frame`, `mocap.landmarks.STYLE100` (test only).
- Produces:
  - `@dataclass(frozen=True) class Decision: what: str; chosen: str; source: Literal["name", "geometry", "user"]; note: str = ""`
  - `class RigError(ValueError)` (user-input problems: unmapped leg, unknown override bone, uninferable units; the message names the item).
  - `@dataclass(frozen=True) class LandmarkMapping: landmarks: LandmarkMap; family: str | None; decisions: tuple[Decision, ...]; warnings: tuple[str, ...]`
  - `def infer_scale_and_up(file_at_unit_scale: BvhFile) -> tuple[float, Literal["y", "z"], tuple[Decision, ...]]` (constants from Global Constraints; raises `RigError` outside the window).
  - `def map_landmarks(file: BvhFile, clip: SkeletonClip, *, overrides: Mapping[str, str] | None = None) -> LandmarkMapping` (`overrides`: landmark value -> bone name; a name not in the file raises `RigError`). Names first: pick the family with the largest overlap of its bone names with the file (at least 6), exact names win; otherwise tokens from the lexicon. Geometry referees: side from the sign of the candidate's lateral coordinate relative to the pelvis in `path_frame(pelvis, fps).left` averaged over the clip (a disagreeing name is corrected with a warning); hip, knee, ankle must be an ancestor chain; the lowest spine bone is the trunk root and `NECK` is the neck bone else the highest spine bone; `HEAD` is the head bone else its End Site (`_End`); `TOE` is the toe bone else the foot's End Site. The minimum set missing raises `RigError`.
  - `def required_landmarks_present(landmarks: LandmarkMap) -> tuple[Landmark, ...]  # the missing ones from PELVIS, HIP, KNEE, ANKLE both sides`.
  - `bone_names.yaml`: `families: {mixamo: {pelvis: "mixamorig:Hips", hip_l: ..., ...}, ...}` for the eight families of Task 3 plus `tokens: {side_left: [left, l_, "_l", ".l", "lft"], side_right: [...], hip: [upleg, thigh, hip, femur, upperleg], knee: [leg, shin, knee, tibia, lowerleg, calf], ankle: [foot, ankle], toe: [toebase, toe, toes], spine: [spine, chest, lowerback], neck: [neck], head: [head], shoulder: [shoulder, clavicle, collar], upper_arm: [arm, upperarm, humerus], forearm: [forearm, elbow, radius, lowerarm], hand: [hand, wrist]}` with a header stating it is data, not evidence.

- [ ] **Step 1: Write the failing tests**

```python
# tests/rig/test_mapping.py
@pytest.mark.parametrize("spec", FAMILY_VARIANTS, ids=...)
def test_every_family_maps_the_seventeen_landmarks(tmp_path, spec):
    f, clip = walk_and_clip(spec, tmp_path)
    m = map_landmarks(f, clip)
    assert set(m.landmarks) >= {Landmark.PELVIS, Landmark.HIP_L, Landmark.KNEE_R, Landmark.ANKLE_L, Landmark.NECK, Landmark.HEAD, Landmark.WRIST_R}
    assert (Landmark.TOE_L in m.landmarks) and m.landmarks[Landmark.TOE_L].endswith("_End") == (not spec.toes)

def test_style100_fixture_maps_to_the_phase_4_5_landmark_map():
    f = read_bvh_file(FIX / "style100_neutral_4s.bvh", scale=0.01, up_axis="y")
    assert dict(map_landmarks(f, f.to_clip(end_sites=True)).landmarks) == dict(STYLE100)

def test_swapped_rest_sides_are_corrected_by_geometry_with_a_warning(tmp_path):
    f, clip = walk_and_clip(replace(RigSpec(), swap_rest_sides=True), tmp_path)
    m = map_landmarks(f, clip)
    assert m.landmarks[Landmark.HIP_L] == FAMILIES["mixamo"]["hip_r"] and any("side" in w for w in m.warnings)

def test_override_with_an_unknown_bone_is_a_rig_error(tmp_path):
    f, clip = walk_and_clip(RigSpec(), tmp_path)
    with pytest.raises(RigError, match="NoSuchBone"):
        map_landmarks(f, clip, overrides={"head": "NoSuchBone"})

def test_a_rig_without_knees_is_refused_naming_them(tmp_path):
    f, clip = walk_and_clip(RigSpec(), tmp_path)
    f = drop_bones(f, ("mixamorig:LeftLeg", "mixamorig:RightLeg"))       # test helper: splice the bones out, keep offsets summed
    with pytest.raises(RigError, match="knee_l"):
        map_landmarks(f, f.to_clip(end_sites=True))

@pytest.mark.parametrize("spec", [RigSpec(), replace(RigSpec(), up_axis="z", units_per_m=1.0), replace(RigSpec(), units_per_m=39.37), replace(RigSpec(), units_per_m=10.0)])
def test_scale_and_up_axis_are_inferred(tmp_path, spec):
    f = write_walk(spec, tmp_path / "w.bvh"); raw = read_bvh_file(tmp_path / "w.bvh", scale=1.0, up_axis="z")
    scale, up, _ = infer_scale_and_up(raw)
    assert up == spec.up_axis and scale == pytest.approx(1.0 / spec.units_per_m, rel=0.02)

def test_uninferable_units_are_a_rig_error(tmp_path):
    f = write_walk(replace(RigSpec(), units_per_m=3.0), tmp_path / "w.bvh")       # 0.3 m leg at scale 1, 3 m at 10
    with pytest.raises(RigError, match="--scale"):
        infer_scale_and_up(read_bvh_file(tmp_path / "w.bvh", scale=1.0, up_axis="z"))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/rig/test_mapping.py -v`
Expected: FAIL with `ModuleNotFoundError: emotive_gait.rig.mapping`

- [ ] **Step 3: Implement `mapping.py` and the lexicon**

Token matching strips a namespace prefix up to `:` and the `DEF-` prefix, lower-cases, and tests substrings in the order given (`toebase` before `toe`, `forearm` before `arm`); the family match is decided before any token is tried.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/rig/test_mapping.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/rig pyproject.toml tests/rig/test_mapping.py
git commit -m "feat(rig): landmark mapping by name lexicon and rest-pose geometry, units and up-axis inference, overrides"
```

---

### Task 6: Reference pose, role bindings by probe, the `driven` chain, `RigProfile`

**Files:**
- Create: `src/emotive_gait/rig/calibration.py`, `src/emotive_gait/rig/ranges.py`, `src/emotive_gait/rig/profile.py`
- Modify: `src/emotive_gait/rig/compile.py` (`compile_driven`), `src/emotive_gait/rig/mapping.py` (`RigMapping`, `bind_roles`)
- Test: `tests/rig/test_calibration.py`, `tests/rig/test_compile.py` (driven tests)

**Interfaces:**
- Consumes: `CompiledRig`, `compile_full`, `LandmarkMapping`, `ROLE_CONVENTIONS`, `convention_holds`, `PROBE_RAD`, `extract_cycles` (usable cycle frames), `path_frame`, `RobotModel`, `RobotProfile`, `POSTURE_ROLES`.
- Produces:
  - `rig/calibration.py`: `@dataclass(frozen=True) class ReferencePose: local_rotations: np.ndarray  # (J, 3, 3) per bone, the chordal mean of the local rotation over the usable cycle frames, the pelvis's after removing each frame's heading; heading: np.ndarray  # unit horizontal forward, the mean heading; floor_z: float; pelvis_height: float  # mean pelvis height above floor_z; foot_stance_rotations: Mapping[str, np.ndarray]  # bone name -> stance-only chordal mean for the two ankle bones; stance_frames: tuple[np.ndarray, np.ndarray]` and `def reference_pose(file: BvhFile, clip: SkeletonClip, landmarks: LandmarkMap) -> ReferencePose` (usable frames = union of `extract_cycles` cycles; stance masks from `foot_events` on the foot point; raises `RigError("no usable cycle: ...")` with the `ClipSummary.discard_reasons` text when there is none).
  - `rig/ranges.py`: `RANGES: Mapping[JointRole, tuple[float, float]]` (Global Constraints).
  - `rig/mapping.py`: `@dataclass(frozen=True) class RoleBinding: hinge: str; sign: float; axis_purity: float`; `@dataclass(frozen=True) class RigMapping: landmarks: LandmarkMap; roles: Mapping[JointRole, RoleBinding]; leg_length: float; scale: float; up_axis: str; family: str | None; decisions: tuple[Decision, ...]; warnings: tuple[str, ...]; skewed: tuple[JointRole, ...]`; `def bind_roles(full: CompiledRig, reference: ReferencePose, landmarks: LandmarkMapping, *, overrides: Mapping[str, str] | None = None) -> RigMapping` (`overrides`: role value -> `"Bone/Xrotation"`). For each bone that owns a landmark (hip, knee, ankle bones; the lowest spine bone; the neck bone; the upper-arm bones; the forearm bones), pose the full chain at the reference, probe each of its hinges by `+PROBE_RAD`, project the child landmark's displacement on `(forward, left, up)` of the reference heading: largest sagittal share is pitch, largest lateral share is roll, the remaining hinge is yaw; `axis_purity` = winning share; `sign` = `+1` if `convention_holds` else `-1`; yaw and ankle-roll signs are `+1`. Roles whose purity is below 0.8 go in `skewed`; a skewed role in `POSTURE_ROLES` or `SHOULDER_PITCH_*` is left unbound with a warning. Missing `LEG_ROLES` raise `RigError` naming them. `leg_length` is `mocap.extract.leg_length(full.file.to_clip(end_sites=True), landmarks.landmarks)`; `scale` and `up_axis` come from `full.file`.
  - `rig/compile.py`: `def compile_driven(file: BvhFile, mapping: RigMapping, reference: ReferencePose) -> CompiledRig`: bodies from the pelvis down with `quat` = the bone's reference local rotation (pelvis: heading-normalised, then the chain is turned so the reference heading is +x); a hinge only for bound roles, named as the bound channel, axis = `R_post.T @ e_k` where `R_post` is the product of the reference rotations of that bone's channels *after* k in channel order (identity for the last channel), multiplied by the binding's sign, range from `RANGES`; the two ankle bones use `foot_stance_rotations` as their reference; a site `foot_l` / `foot_r` on each ankle body at the toe bone's offset (or the End Site offset, or the origin without a toe); landmark sites as in `full`; root at `(0, 0, reference.pelvis_height)`.
  - `rig/profile.py`: `def rig_profile(name: str, mapping: RigMapping, reference: ReferencePose, *, posture_map: Mapping[JointRole, Mapping[str, float]]) -> RobotProfile` (`actuator_for` = role -> hinge name, `sign_for` all `1.0`, `nominal_pelvis_height = reference.pelvis_height`, `foot_site_floor_height` = the foot site's height above `floor_z` at the reference pose, `foot_sites = ("foot_l", "foot_r")`, `leg_length = mapping.leg_length`, `scene_entry = ""`), and `def rig_robot(driven: CompiledRig, profile: RobotProfile) -> RobotModel` (`stand_qpos` all zeros).

- [ ] **Step 1: Write the failing tests**

```python
# tests/rig/test_calibration.py
def test_reference_pose_heading_is_the_walk_direction_and_floor_is_the_toe_stance_height(tmp_path):
    f, clip = walk_and_clip(RigSpec(), tmp_path); lm = map_landmarks(f, clip).landmarks
    ref = reference_pose(f, clip, lm)
    assert np.allclose(ref.heading, [1.0, 0.0, 0.0], atol=0.05) and abs(ref.floor_z - toe_rest_height_m(RigSpec())) < 0.01
    assert 0.7 < ref.pelvis_height < 1.0

def test_reference_pose_refuses_a_standing_clip(tmp_path):
    f = build_rig(RigSpec()); f = f.with_motion(np.repeat(f.motion, 120, axis=0))
    with pytest.raises(RigError, match="no usable cycle"):
        reference_pose(f, f.to_clip(end_sites=True), map_landmarks(f, f.to_clip(end_sites=True)).landmarks)

@pytest.mark.parametrize("spec", FAMILY_VARIANTS, ids=...)
def test_every_bound_role_satisfies_its_convention_on_the_driven_chain(tmp_path, spec):
    robot, mapping, ref = driven_rig(spec, tmp_path)   # test helper: map, reference, bind, compile_driven, rig_profile, rig_robot
    assert set(mapping.roles) >= set(LEG_ROLES)
    for role in ROLE_CONVENTIONS:
        if role in mapping.roles:                           # every bound row, the elbow included (its sign was chosen)
            before, after = probe_driven(robot, role)      # qpos zero, then +PROBE_RAD at the role's hinge; landmark sites
            assert convention_holds(role, before, after, forward=X, left=Y, up=Z), role

def test_driven_chain_at_zero_equals_full_chain_at_the_reference(tmp_path):
    ... both chains posed (full: reference channel values via euler_from_matrix; driven: zeros), landmark sites agree to 1e-9 after removing the heading ...

def test_xyz_and_zyx_orders_bind_the_same_anatomical_hinges(tmp_path):
    for order in ("XYZ", "ZYX"):
        _, mapping, _ = driven_rig(replace(RigSpec(), order=order), tmp_path / order)
        assert mapping.roles[JointRole.KNEE_L].axis_purity > 0.95 and not mapping.skewed
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/rig/test_calibration.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `calibration.py` (reference pose), `ranges.py`, `bind_roles`, `compile_driven`, `profile.py`**

Chordal mean: average the rotation matrices and project back with SVD (`U @ diag(1, 1, det) @ Vt`). The conventions' `toe_l` for a rig without toes is the ankle's End Site. `compile_driven` keeps the axis formula above exactly; a single bound channel changed by `delta` then reproduces the BVH's Euler semantics, and several bound channels on one bone compose self-consistently (the writer in Task 8 decomposes whatever they compose).

- [ ] **Step 4: Run the tests**

Run: `pytest tests/rig/test_calibration.py tests/rig/test_compile.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/rig tests/rig
git commit -m "feat(rig): reference pose, role bindings by probe with sign and axis purity, driven chain, rig profile"
```

---

### Task 7: Posture calibration: static posture angles, the Jacobian, the affine `posture_map`

**Files:**
- Modify: `src/emotive_gait/mocap/extract.py` (`posture_angles`), `src/emotive_gait/rig/calibration.py` (`posture_map`)
- Test: `tests/rig/test_calibration.py` (posture tests), `tests/mocap/test_extract.py` (one test)

**Interfaces:**
- Consumes: `_angles` (internal), `PathFrame`, `CompiledRig`, `RobotModel`, `POSTURE_PARAMETERS`, `POSTURE_ROLES`.
- Produces:
  - `mocap/extract.py`: `def posture_angles(tracks: Mapping[Landmark, np.ndarray], frame: PathFrame) -> dict[str, float]` keyed by `POSTURE_PARAMETERS` names (`trunk_pitch`, `neck_pitch`, `elbow_flexion` mean of both arms, `arm_abduction` mean of both arms), the frame-mean of `_angles` over the tracks given (any length; NaN where a landmark is absent). `mocap/` may not import `robot/`, so the key names are literal strings here and a test keeps them equal to `POSTURE_PARAMETERS`.
  - `rig/calibration.py`: `def posture_map(robot: RobotModel, mapping: RigMapping, *, landmark_sites: Mapping[Landmark, str]) -> tuple[Mapping[JointRole, Mapping[str, float]], dict[str, float]]`: with the driven chain at zero, read `p_ref = posture_angles(...)` from the landmark sites in the reference frame (`forward = +x`); probe each bound posture role by `+PROBE_RAD` (both elbows together, both shoulder rolls together) and build the Jacobian `J[param, role-group]`; `K = inv(J)` (least squares when a role group is unbound, with that parameter's row and column dropped); return rows `{param: K[role, param] ..., "offset": -(K @ p_ref)[role]}` for each bound posture role (left and right rows equal), and `p_ref`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/mocap/test_extract.py
def test_posture_angles_keys_match_the_posture_parameters():
    from emotive_gait.robot.roles import POSTURE_PARAMETERS   # tests may cross layers
    tracks = full_walker().tracks   # existing fixture helper
    assert tuple(posture_angles(tracks, path_frame(tracks[Landmark.PELVIS], FPS))) == POSTURE_PARAMETERS

# tests/rig/test_calibration.py
@pytest.mark.parametrize("spec", [RigSpec(), replace(RigSpec(), spine_bones=1), replace(RigSpec(), family="style100")], ids=...)
def test_posture_map_makes_commanded_posture_read_back(tmp_path, spec):
    robot, mapping, ref, pm, p_ref = calibrated_rig(spec, tmp_path)
    for command in ({}, {"trunk_pitch": 0.3}, {"neck_pitch": 0.3}, {"elbow_flexion": 0.5}, {"arm_abduction": 0.3}):
        target = {k: p_ref[k] + command.get(k, 0.0) for k in POSTURE_PARAMETERS}
        qpos = hinges_from_posture(robot, pm, target)      # test helper: offset + sum(coef * target) per posture role
        read = static_posture(robot, qpos, mapping)        # test helper: mj_forward, landmark sites, posture_angles with forward=+x
        for k in POSTURE_PARAMETERS:
            assert abs(read[k] - target[k]) < 0.02, (k, command)

def test_identity_posture_map_on_the_g1_is_the_default():
    assert G1_PROFILE_WITH_NECK.posture_map == {}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/rig/test_calibration.py -k posture tests/mocap/test_extract.py::test_posture_angles_keys_match_the_posture_parameters -v`
Expected: FAIL with `ImportError: posture_angles`

- [ ] **Step 3: Implement `posture_angles` and `posture_map`**

The trunk gain is about 0.77 on 100STYLE-like proportions (spec 4.3), so `K` is far from identity on the waist and neck rows; `J` is 4 x 4 at most and well conditioned because each role moves mainly one measure.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/rig/test_calibration.py tests/mocap/test_extract.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/mocap/extract.py src/emotive_gait/rig/calibration.py tests
git commit -m "feat(rig,mocap): static posture angles and the measured affine posture map (trunk gain, neck leak, elbow, abduction)"
```

---

### Task 8: `rig/synthesis.py`: the pipeline, the writer round trip, slip, the claims

**Files:**
- Create: `src/emotive_gait/rig/synthesis.py`, `src/emotive_gait/rig/scaled.py`
- Modify: `src/emotive_gait/emotion/model.py` (`profile_aware`), `src/emotive_gait/emotion/learned.py` (`profile_aware = True`)
- Test: `tests/rig/test_synthesis.py`, `tests/emotion/test_model.py` (one test)

**Interfaces:**
- Consumes: everything from Tasks 1-7; `modulate`, `unexpressible`, `EmotionModel`, `ANALYTIC`, `LearnedEmotionModel`, `to_gait_parameters`, `generate`, `FeasibilityReport`, `footsteps.foot_point`, `phase_at`, `pelvis_x`, `pelvis_offset`, `rule_double_support`, `summarise`, `foot_events`, `write_bvh`, `euler_from_matrix`.
- Produces:
  - `emotion/model.py`: `EmotionModel.profile_aware: bool` documented on the protocol (`False` when absent); `AnalyticEmotionModel.profile_aware = False`; `LearnedEmotionModel.profile_aware = True`.
  - `rig/scaled.py`: `class RigScaledModel(EmotionModel)` with `__init__(self, inner: EmotionModel, *, scaling: Scaling, p0: GaitParameters)` and `deltas(intensities, profile)`: `froude` + `profile_aware` inner: pass through; `froude` + not aware: make the inner deltas dimensionless as if on the G1 (`/ 0.69` for `LENGTH_FIELDS`, `* sqrt(0.69 / G)` for `cadence`, `* 0.68 / 0.69` for `com_height_scale`) then `to_gait_parameters(delta, GaitParameters(), scaling="froude", leg_length=profile.leg_length, nominal_pelvis_height=profile.nominal_pelvis_height)` minus the defaults; `ratio`: `delta / default_g1 * p0_value` for `LENGTH_FIELDS`, `cadence` and `com_height_scale`, pass-through otherwise. `profile_aware = True`.
  - `rig/synthesis.py` (imports `math`):
    - `def load_rig(source: Path, *, scale: float | None, up_axis: str | None, mapping_overrides: Mapping[str, str] | None) -> LoadedRig` where `@dataclass(frozen=True) class LoadedRig: file: BvhFile; clip: SkeletonClip; mapping: RigMapping; reference: ReferencePose; summary: ClipSummary; p0: GaitParameters; raw_double_support: float; robot: RobotModel; profile: RobotProfile; p_ref: dict[str, float]` (`mapping_overrides` keys are landmark values or role values; `p0` is `GaitParameters` from `summary.mean` with `com_height_scale = 1.0`, `expansiveness = 1.0`, `double_support_frac = invert_double_support(summary.mean.double_support_frac, p0)`).
    - `def invert_double_support(raw: float, params: GaitParameters) -> float` (bisection on `d` in `[0, 0.5]` of `rule_double_support(replace(params, double_support_frac=d)) - raw`, 1e-4).
    - `def resynthesise(rig: LoadedRig, p1: GaitParameters, *, duration_s: float) -> tuple[BvhFile, GaitTrajectory]`: `generate(p1, rig.robot, welded=rig.robot)`; per output frame at the source fps, `qpos = traj.qpos_at(robot, phase_at(p1, t))`, `mj_forward`, each compiled bone's local rotation `R_parent.T @ R_body` turned back by the heading and into the file's axes, `euler_from_matrix` in the bone's channel order; pelvis ancestors and uncompiled bones take their reference channel values; the root's position channels put the pelvis at `heading_matrix(source first-frame heading) @ (pelvis_x(p1, t), sway, nominal * com_height_scale + bounce) + source first-frame pelvis horizontal position + floor_z` (minus the held root-to-pelvis offset when the pelvis is not the root); `BvhFile.with_motion`.
    - `@dataclass(frozen=True) class SlipReport: max_travel_m: float; max_height_error_m: float; stances: int` and `def measure_slip(output: BvhFile, landmarks: LandmarkMap, p1: GaitParameters, floor_z: float) -> SlipReport` (foot point tracks from `output.to_clip(end_sites=True)`; for every commanded stance interval of each foot, from `foot_point(p1, phase, side).in_stance` sampled at the output frames, the horizontal travel = max distance from the stance's first-frame position, the height error = max `|z - floor_z|`).
    - `@dataclass(frozen=True) class RoundTrip: commanded: GaitParameters; measured: ClipSummary; residuals: Mapping[str, float]  # measured minus commanded per MEASURED_FIELDS, double support against rule_double_support(commanded)` and `def round_trip(output: BvhFile, landmarks: LandmarkMap, p1: GaitParameters) -> RoundTrip`.
    - `@dataclass(frozen=True) class WalkResult: rig: LoadedRig; intensities: Mapping[Emotion, float]; intensities_source: Literal["flags", "lexicon", "caller"]; model_name: str; scaling: Scaling; p1: GaitParameters; unexpressible: tuple[tuple[Emotion, str], ...]; feasibility: FeasibilityReport; slip: SlipReport; round_trip: RoundTrip; output: BvhFile`.
    - `def emotional_walk(source: Path, *, intensities: Mapping[Emotion, float], intensities_source: str, model: EmotionModel, model_name: str, scaling: Scaling = "froude", duration_s: float | None = None, mapping_overrides=None, scale=None, up_axis=None) -> WalkResult` (`duration_s` default = the source's usable length; `p1 = modulate(p0, intensities, profile, model=RigScaledModel(model, scaling=scaling, p0=p0))`).
    - `TOLERANCES: Mapping[str, float]` (Global Constraints) and `def within_tolerance(rt: RoundTrip) -> dict[str, bool]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/emotion/test_model.py
def test_models_declare_profile_awareness():
    assert ANALYTIC.profile_aware is False and LearnedEmotionModel.profile_aware is True

# tests/rig/test_synthesis.py   (module-scoped fixtures: one LoadedRig per spec in a small set)
ZERO = {}
PRESETS = {"sad": {Emotion.SADNESS: 1.0}, "angry": {Emotion.ANGER: 1.0}, "happy": {Emotion.HAPPINESS: 1.0}, "fear": {Emotion.FEAR: 1.0}}

@pytest.mark.parametrize("spec", FAMILY_VARIANTS, ids=...)
def test_zero_emotion_round_trip_recovers_p0_on_every_rig(loaded, spec):
    r = emotional_walk(loaded[spec].path, intensities=ZERO, intensities_source="flags", model=ANALYTIC, model_name="analytic")
    assert all(within_tolerance(r.round_trip).values()), {k: v for k, v in r.round_trip.residuals.items() if not within_tolerance(r.round_trip)[k]}
    assert r.feasibility.feasible and r.slip.max_travel_m <= 0.002 and r.slip.max_height_error_m <= 0.002

@pytest.mark.parametrize("spec", [RigSpec(), replace(RigSpec(), up_axis="z"), replace(RigSpec(), toes=False)], ids=...)
@pytest.mark.parametrize("preset", list(PRESETS))
def test_emotion_round_trip_recovers_p1(loaded, spec, preset):
    r = emotional_walk(loaded[spec].path, intensities=PRESETS[preset], intensities_source="flags", model=ANALYTIC, model_name="analytic")
    ok = within_tolerance(r.round_trip)
    assert all(ok.values()), [k for k, v in ok.items() if not v]
    if r.feasibility.feasible:
        assert r.slip.max_travel_m <= 0.002

def test_style100_fixture_round_trips_at_zero_and_sad():
    for intensities in (ZERO, PRESETS["sad"]):
        r = emotional_walk(FIX / "style100_neutral_4s.bvh", intensities=intensities, intensities_source="flags", model=ANALYTIC, model_name="analytic", scale=0.01, up_axis="y")
        assert all(within_tolerance(r.round_trip).values()) and r.slip.max_travel_m <= 0.002

def test_output_keeps_the_source_hierarchy_channels_and_fps(loaded):
    r = emotional_walk(loaded[RigSpec()].path, intensities=ZERO, ...)
    assert [j.name for j in r.output.joints] == [j.name for j in r.rig.file.joints] and r.output.frame_time == r.rig.file.frame_time
    assert all(a.channels == b.channels for a, b in zip(r.output.joints, r.rig.file.joints))

def test_root_above_pelvis_output_puts_the_pelvis_where_commanded(loaded):
    spec = replace(RigSpec(), root_above_pelvis=True)
    r = emotional_walk(loaded[spec].path, intensities=ZERO, ...)
    clip = r.output.to_clip(); pelvis = clip.joint_names.index(FAMILIES["mixamo"]["pelvis"])
    assert abs(clip.positions[0, pelvis, 2] - (r.rig.reference.floor_z + r.rig.reference.pelvis_height)) < 0.01

def test_learned_model_goes_through_the_rig_profile(loaded, fixture_artefact_path):
    m = LearnedEmotionModel(fixture_artefact_path, scaling="froude")
    r = emotional_walk(loaded[RigSpec()].path, intensities=PRESETS["sad"], intensities_source="flags", model=m, model_name="learned")
    assert r.p1.cadence < r.rig.p0.cadence

def test_ratio_scaling_is_relative_to_the_rigs_own_neutral(loaded, fixture_artefact_path):
    ... RigScaledModel(LearnedEmotionModel(fixture_artefact_path, scaling="ratio"), scaling="ratio", p0=p0).deltas({Emotion.SADNESS: 1.0}, profile)["step_length"] == pytest.approx(cell_estimate * p0.step_length, rel=1e-6) ...

def test_unreachable_stride_is_flagged_and_still_written(loaded):
    rig = load_rig(loaded[RigSpec()].path, scale=None, up_axis=None, mapping_overrides=None)
    out, traj = resynthesise(rig, replace(rig.p0, step_length=1.2), duration_s=2.0)   # 1.2 m steps on 0.9 m legs
    assert not traj.feasibility.feasible and out.motion.shape[0] == 120
    slip = measure_slip(out, rig.mapping.landmarks, replace(rig.p0, step_length=1.2), rig.reference.floor_z)
    assert slip.stances > 0 and math.isfinite(slip.max_travel_m)                    # reported, never raised

def test_invert_double_support_is_the_inverse_of_the_rule():
    p = GaitParameters()
    assert invert_double_support(rule_double_support(p), p) == pytest.approx(0.20, abs=1e-3)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/rig/test_synthesis.py tests/emotion/test_model.py::test_models_declare_profile_awareness -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `scaled.py`, `synthesis.py`, `profile_aware`**

If a residual exceeds its tolerance on a rig, the first suspects in order are: the writer's Euler decomposition for that rig's order, the `R_post` axis formula in `compile_driven`, the stance-mean foot reference, and only then the extractor (whose thresholds are frozen; a genuine extractor limit is a finding for the report, and the test is then narrowed to the rigs that pass with the exclusion named).

- [ ] **Step 4: Run the tests**

Run: `pytest tests/rig/test_synthesis.py -v`
Expected: PASS within about two minutes

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/rig src/emotive_gait/emotion tests
git commit -m "feat(rig): end-to-end synthesis with the writer round trip, slip and round-trip claims on the synthetic family and 100STYLE; RigScaledModel"
```

---

### Task 9: `emotion/prompt.py`: free text to six intensities

**Files:**
- Create: `src/emotive_gait/emotion/prompt.py`, `src/emotive_gait/emotion/data/affect_lexicon.yaml`
- Test: `tests/emotion/test_prompt.py`

**Interfaces:**
- Consumes: `Emotion`, `UNSOURCED_EMOTIONS`, `pyyaml`.
- Produces: `@dataclass(frozen=True) class PromptResult: intensities: Mapping[Emotion, float]; matched: tuple[tuple[str, Emotion, float], ...]; unmatched: tuple[str, ...]; grounded: bool` and `def parse_emotion_prompt(text: str) -> PromptResult` (lower-case, tokens on non-letters, bigram terms first (`grossed out`), intensifiers `very`/`extremely`/`really` x1.0 (clamp), `slightly`/`a bit`/`somewhat` x0.5, `not`/`no longer`/`never` zero the next term; several terms for one emotion take the max; `unmatched` = content words not in the lexicon and not stop words; `grounded` false when any matched emotion is in `UNSOURCED_EMOTIONS`; an empty `matched` is a valid result with all-zero intensities). `affect_lexicon.yaml`: `terms: {furious: [anger, 1.0], annoyed: [anger, 0.4], ..., gloomy: [sadness, 0.6], elated: [happiness, 1.0], nervous: [fear, 0.5], "grossed out": [disgust, 0.7], startled: [surprise, 0.8], ...}` about 150 terms, header: tuned, not cited.

- [ ] **Step 1: Write the failing tests**

```python
def test_plain_words_map_to_their_emotion():
    r = parse_emotion_prompt("she walks in, furious")
    assert r.intensities[Emotion.ANGER] == 1.0 and r.grounded and "furious" in [m[0] for m in r.matched]

def test_intensifiers_and_negation():
    assert parse_emotion_prompt("slightly gloomy").intensities[Emotion.SADNESS] == pytest.approx(0.3)
    assert parse_emotion_prompt("not angry at all, just tired").intensities.get(Emotion.ANGER, 0.0) == 0.0

def test_several_terms_take_the_maximum_not_the_sum():
    assert parse_emotion_prompt("annoyed and furious").intensities[Emotion.ANGER] == 1.0

def test_unsourced_emotions_are_flagged():
    r = parse_emotion_prompt("grossed out by the smell")
    assert r.intensities[Emotion.DISGUST] == pytest.approx(0.7) and not r.grounded

def test_unknown_words_are_listed_and_nothing_is_invented():
    r = parse_emotion_prompt("like a lighthouse keeper")
    assert not r.matched and "lighthouse" in r.unmatched and all(v == 0.0 for v in r.intensities.values())

def test_every_lexicon_term_names_a_known_emotion_and_a_weight_in_range():
    for term, (emotion, w) in LEXICON.items():
        Emotion(emotion); assert 0.0 < w <= 1.0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/emotion/test_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `prompt.py` and write the lexicon**

- [ ] **Step 4: Run the tests**

Run: `pytest tests/emotion/test_prompt.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/emotion tests/emotion/test_prompt.py
git commit -m "feat(emotion): free-text emotion prompt to six intensities through a shipped, tuned lexicon"
```

---

### Task 10: `rig/cli.py`: `rig inspect`, `rig walk`, `rig validate`; `--json`

**Files:**
- Create: `src/emotive_gait/rig/cli.py`
- Modify: `src/emotive_gait/cli.py` (register the `rig` group and `serve`; `main` dispatch)
- Test: `tests/test_cli_rig.py`

**Interfaces:**
- Consumes: `load_rig`, `emotional_walk`, `parse_emotion_prompt`, `LearnedEmotionModel`, `_parse_emotions`, `_print_params`, `_err` (import from `cli.py` is circular: move `_err`, `_parse_emotions`, `_print_params` and `_add_model_arguments` into a new `src/emotive_gait/cli_common.py` that both import).
- Produces: `def add_rig_parser(sub) -> None` (subcommands `inspect`, `walk`, `validate` with the options of spec 4.10; every subcommand takes `--json`), `def run_rig(args) -> int`, `def add_serve_parser(sub) -> None`, `def run_serve(args) -> int` (Task 11 fills it; here it returns 1 with "install the mcp extra" when `import mcp` fails). `cli.main` dispatches `rig` and `serve`. `inspect` prints the decisions (`what  chosen  source  note` columns), warnings, the family, scale and up-axis, leg length, `P0` with the raw double support beside it, and the cycle counts; `walk` prints the same plus `unexpressible`, `P0` and `P1` side by side, the feasibility warning line (`warning: parameters exceed the robot's reach` as `demo` does), the slip line (`slip: max travel 0.4 mm, max height error 0.3 mm over 11 stances`), the round-trip residual table with an `ok` column, and writes `--out`; `validate` runs zero plus the four presets and prints one row per run (`--report` is Task 12). `--json` prints one object with the same content. Exit codes: 2 for `RigError`, `MocapFormatError`, `IntensityError`, a `--prompt` with nothing matched (message lists `unmatched`), both or neither of `--emotion`/`--prompt`; 1 for `OSError` on write and `ArtefactMissingError`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli_rig.py
def test_rig_inspect_prints_decisions_and_p0(tmp_path, capsys):
    path = walk_file(RigSpec(), tmp_path)
    assert main(["rig", "inspect", str(path)]) == 0
    out = capsys.readouterr().out
    assert "mixamo" in out and "cadence" in out and "source" in out

def test_rig_walk_writes_a_bvh_and_prints_slip(tmp_path, capsys):
    path = walk_file(RigSpec(), tmp_path); out = tmp_path / "sad.bvh"
    assert main(["rig", "walk", str(path), "--emotion", "sadness=0.8", "--out", str(out)]) == 0
    assert out.is_file() and "slip:" in capsys.readouterr().out

def test_rig_walk_with_a_prompt_uses_the_lexicon(tmp_path, capsys):
    assert main(["rig", "walk", str(walk_file(RigSpec(), tmp_path)), "--prompt", "very gloomy", "--out", str(tmp_path / "o.bvh")]) == 0
    assert "lexicon" in capsys.readouterr().out

def test_rig_walk_with_an_unmatched_prompt_exits_two_listing_the_words(tmp_path, capsys):
    assert main(["rig", "walk", str(walk_file(RigSpec(), tmp_path)), "--prompt", "lighthouse keeper", "--out", str(tmp_path / "o.bvh")]) == 2
    assert "lighthouse" in capsys.readouterr().err

def test_rig_walk_needs_exactly_one_of_emotion_and_prompt(tmp_path, capsys):
    assert main(["rig", "walk", str(walk_file(RigSpec(), tmp_path)), "--out", str(tmp_path / "o.bvh")]) == 2

def test_standing_clip_exits_two_with_the_reasons(tmp_path, capsys):
    path = standing_file(RigSpec(), tmp_path)
    assert main(["rig", "inspect", str(path)]) == 2
    assert "no usable cycle" in capsys.readouterr().err

def test_uninferable_units_exit_two_asking_for_scale(tmp_path, capsys):
    assert main(["rig", "inspect", str(walk_file(replace(RigSpec(), units_per_m=3.0), tmp_path))]) == 2
    assert "--scale" in capsys.readouterr().err

def test_blender_style_z_up_metres_file_is_inferred(tmp_path, capsys):
    assert main(["rig", "inspect", str(walk_file(replace(RigSpec(), up_axis="z", units_per_m=1.0), tmp_path)), "--json"]) == 0
    mapping = json.loads(capsys.readouterr().out)["mapping"]
    assert mapping["up_axis"] == "z" and mapping["scale"] == pytest.approx(1.0)

def test_mapping_override_with_an_unknown_bone_exits_two(tmp_path, capsys):
    (tmp_path / "m.yaml").write_text("landmarks:\n  head: NoSuchBone\n")
    assert main(["rig", "inspect", str(walk_file(RigSpec(), tmp_path)), "--mapping", str(tmp_path / "m.yaml")]) == 2
    assert "NoSuchBone" in capsys.readouterr().err

def test_infeasible_emotion_warns_and_still_writes(tmp_path, capsys):
    out = tmp_path / "o.bvh"
    source = walk_file(replace(RigSpec(), leg_length_m=0.6), tmp_path, step_length=0.6)   # a 0.6 m stride on 0.6 m legs
    assert main(["rig", "walk", str(source), "--emotion", "anger=1.0", "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert out.is_file() and "exceed the robot's reach" in captured.err and "slip:" in captured.out

def test_rig_json_output_is_parseable(tmp_path, capsys):
    assert main(["rig", "inspect", str(walk_file(RigSpec(), tmp_path)), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["p0"]["cadence"] > 0

def test_serve_without_mcp_exits_one_with_the_install_hint(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "mcp", None)
    assert main(["serve"]) == 1 and "mcp" in capsys.readouterr().err

def test_rig_validate_prints_five_rows(tmp_path, capsys):
    assert main(["rig", "validate", str(walk_file(RigSpec(), tmp_path))]) == 0
    assert capsys.readouterr().out.count("\n") >= 5
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_cli_rig.py -v`
Expected: FAIL (`argparse` error: invalid choice `rig`)

- [ ] **Step 3: Implement `cli_common.py`, `rig/cli.py`, register in `cli.py`**

The mapping YAML has two top-level keys, `landmarks:` (landmark value -> bone) and `roles:` (role value -> `Bone/Channel`); `inspect --json` emits exactly that shape under `"mapping"` so its output can be edited and fed back. The existing `test_cli.py` tests must stay green after the move to `cli_common.py`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_cli_rig.py tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait tests
git commit -m "feat(cli): rig inspect/walk/validate with --json, serve stub, shared CLI helpers"
```

---

### Task 11: `rig/server.py`: the MCP server and its smoke test

**Files:**
- Create: `src/emotive_gait/rig/server.py`
- Modify: `pyproject.toml` (`mcp = ["mcp>=2,<3"]` after verifying the installed major; `dev` includes it), `src/emotive_gait/rig/cli.py` (`run_serve`)
- Test: `tests/rig/test_server.py`

**Interfaces:**
- Consumes: `load_rig`, `emotional_walk`, `parse_emotion_prompt`, `EVIDENCE`, `LearnedEmotionModel.default()` (for thin cells), `write_bvh`.
- Produces: `def build_server() -> "MCPServer"` (name `emotive-gait`), tools `inspect_rig(bvh_path: str, scale: float | None = None, up_axis: str | None = None, mapping: dict | None = None) -> dict`, `measure_gait(bvh_path, scale=None, up_axis=None, mapping=None) -> dict`, `emotional_walk(bvh_path: str, out_path: str, intensities: dict[str, float] | None = None, prompt: str | None = None, model: str = "analytic", scaling: str = "froude", duration_s: float | None = None, mapping: dict | None = None, scale: float | None = None, up_axis: str | None = None) -> dict` (returns the `WalkResult` as the CLI's `--json` object minus the motion; `intensities_source` is `caller` when `intensities` is given, `lexicon` when only `prompt`), `parse_emotion_prompt(text: str) -> dict`, `list_emotions() -> dict`; resources `emotive-gait://evidence` (the `evidence` table text) and `emotive-gait://rig-mapping-schema` (the YAML shape with an example); prompt `emotional_walk_request(bvh_path: str, description: str) -> str` with the spec 4.9 instructions. `RigError`, `MocapFormatError` and `IntensityError` become tool errors (the SDK's error result), never a crash. `def serve() -> None` runs stdio. `run_serve` in `rig/cli.py` imports `serve` inside the function and returns 1 with the install hint on `ImportError`.

- [ ] **Step 1: Install the extra and write the failing smoke test**

Run: `python -m pip install -e ".[dev]"` after adding the extra; record the installed `mcp` version in the commit message.

```python
# tests/rig/test_server.py
pytest.importorskip("mcp")

@pytest.fixture(scope="module")
def server():
    return build_server()

def test_server_lists_the_five_tools_two_resources_and_the_prompt(server):
    ... via the SDK's in-memory client (or a stdio subprocess running `python -m emotive_gait.rig.server`): tool names == {"inspect_rig", "measure_gait", "emotional_walk", "parse_emotion_prompt", "list_emotions"}; resource URIs; prompt name ...

def test_parse_emotion_prompt_tool_returns_intensities(server):
    result = call_tool(server, "parse_emotion_prompt", {"text": "very gloomy"})
    assert result["intensities"]["sadness"] == pytest.approx(0.6)

def test_emotional_walk_tool_writes_the_file_and_reports(server, tmp_path):
    path = walk_file(RigSpec(), tmp_path)
    result = call_tool(server, "emotional_walk", {"bvh_path": str(path), "out_path": str(tmp_path / "o.bvh"), "intensities": {"sadness": 0.8}})
    assert (tmp_path / "o.bvh").is_file() and result["intensities_source"] == "caller" and "slip" in result

def test_standing_clip_is_a_tool_error_not_a_crash(server, tmp_path):
    err = call_tool_expecting_error(server, "inspect_rig", {"bvh_path": str(standing_file(RigSpec(), tmp_path))})
    assert "no usable cycle" in err

def test_stdio_server_starts_and_answers_list_tools(tmp_path):
    ... subprocess `python -m emotive_gait.rig.server` with the SDK stdio client, list tools, terminate; 20 s timeout ...
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/rig/test_server.py -v`
Expected: FAIL with `ImportError: build_server`

- [ ] **Step 3: Implement `server.py` and `run_serve`**

Never print to stdout inside the server (stdio transport); log to stderr. Tool results are plain JSON-able dicts.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/rig/test_server.py tests/test_cli_rig.py::test_serve_without_mcp_exits_one_with_the_install_hint -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/rig pyproject.toml tests/rig/test_server.py
git commit -m "feat(rig): MCP server over stdio (five tools, two resources, one prompt) with a smoke test; mcp extra"
```

---

### Task 12: `rig validate --report`, manifest entries, real-rig validation report

**Files:**
- Create: `src/emotive_gait/rig/validate.py`, `docs/reports/phase-4-6-rig-validation.md`
- Modify: `src/emotive_gait/rig/cli.py` (`--report`), `data/manifest.yaml` (two new datasets), `.gitattributes` (the report is already covered by `docs/reports/*.md`)
- Test: `tests/rig/test_validate.py`

**Interfaces:**
- Consumes: `emotional_walk`, `within_tolerance`, `PRESET_PARAMS` intensities (`sad`, `angry`, `happy` at 1.0 plus `fear` 1.0), `load_manifest`, `fetch` (CLI only).
- Produces: `def validate_rig(source: Path, *, scale, up_axis, mapping_overrides) -> ValidationTable` with `@dataclass(frozen=True) class ValidationRow: run: str; feasible: bool; max_residual: Mapping[str, float]; within: bool; slip_m: float; height_m: float` and `def render_report(tables: Mapping[str, ValidationTable], *, not_run: Mapping[str, str]) -> str` (Markdown: a header naming the commit, the tolerances, one table per rig with the five runs, a "not run" list with reasons, a sentence on how many rigs hit `skewed`). `rig validate --report PATH` writes `render_report` for that one rig; the committed report is assembled by running it on each real rig and concatenating with the `not_run` list (a one-off, documented in the report's header as the command lines used).

- [ ] **Step 1: Write the failing tests**

```python
def test_validate_rig_runs_zero_and_four_presets(tmp_path):
    t = validate_rig(walk_file(RigSpec(), tmp_path), scale=None, up_axis=None, mapping_overrides=None)
    assert [r.run for r in t.rows] == ["neutral", "sad", "angry", "happy", "fear"] and t.rows[0].within

def test_render_report_lists_not_run_rigs():
    text = render_report({}, not_run={"bandai": "archive not fetched"})
    assert "bandai" in text and "archive not fetched" in text
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/rig/test_validate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `validate.py` and `--report`**

- [ ] **Step 4: Add the manifest entries and fetch**

Add to `data/manifest.yaml`: dataset `cmu_bvh` (the cgspeed MotionBuilder-named conversion; URL as found on its page, `archive` the zip it ships in, `committable: false`, `licence` quoting CMU's terms and the conversion README's sentence, reader `bvh`, `reader_args` with the verified scale and `up_axis: y`, one neutral walk file, e.g. subject 07 trial 01, hashed after download) and dataset `bandai2` (Bandai-Namco-Research-Motiondataset-2, GitHub release archive, `licence: CC BY-NC 4.0`, `committable: false`, one `walk` normal-style file, hashed). Run `emotive-gait fetch --datasets cmu_bvh bandai2`. A dataset that cannot be fetched (URL gone, hash unknown) is left in the manifest with its `excluded` reason and goes into the report's `not_run` list; nothing is guessed.

- [ ] **Step 5: Run the validation on the real rigs and commit the report**

Run `emotive-gait rig validate <path> --report <tmp>` for the 100STYLE excerpt, `data/raw/100style/Neutral_FW.bvh` (full clip, `--scale 0.01 --up y`), the CMU conversion file and the Bandai file; assemble `docs/reports/phase-4-6-rig-validation.md` with `render_report`. The report carries numbers only; no clip, no frame data.

- [ ] **Step 6: Run the tests and commit**

Run: `pytest tests/rig/test_validate.py -v`
Expected: PASS

```bash
git add src/emotive_gait/rig tests/rig/test_validate.py data/manifest.yaml docs/reports/phase-4-6-rig-validation.md
git commit -m "feat(rig): validate --report; manifest entries for the CMU BVH conversion and Bandai Namco (local only); rig validation report"
```

---

### Task 13: README, `CLAUDE.md`, version 0.4.0

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `pyproject.toml` (version), `src/emotive_gait/__init__.py` (`__version__`), `tests/test_packaging.py` (`"0.4.0"`)

- [ ] **Step 1: README section "Emotion for any rig"**

What goes in and what comes out (one paragraph, stating in its first sentence that the output is a parametric walk with the source's parameters, not the source's frames); the three commands with an example each; the Blender round trip (export BVH with Blender's exporter: the tool infers Z-up and metres, or pass `--up z --scale 1`; import the result with the same settings); the Mixamo path (download FBX, import to Blender, export BVH, never commit the files); the MCP registration (`claude mcp add emotive-gait -- emotive-gait serve`, and the `claude_desktop_config.json` block); the measured claim with the report's headline numbers (max residuals and max slip per rig) and a link to the report; the limitations list of spec 6, verbatim in sense.

- [ ] **Step 2: `CLAUDE.md`**

Commands (`rig inspect/walk/validate`, `serve`); the `rig/` package in the architecture list with its import rules; facts that bite: the calibrated zero is an affine `posture_map` measured by probing (trunk gain about 0.77 on human proportions); one foot point everywhere; the floor definition; `full` versus `driven` compile and the FK gate; `ROLE_CONVENTIONS` with the G1 elbow xfail; `generate(..., welded=)`; the twelve-leg-role minimum; the double-support inversion through `rule_double_support`; the hinge naming `Bone/Channel`; End Sites as `_End` joints; no LLM inside; which clips may never be committed; `mcp>=2,<3` and stdout silence in the server.

- [ ] **Step 3: Version 0.4.0 and the whole suite**

Run: `ruff check . && ruff format --check . && pytest -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md pyproject.toml src/emotive_gait/__init__.py tests/test_packaging.py
git commit -m "docs: Phase 4.6 emotion for any rig; README section, CLAUDE.md facts; version 0.4.0"
```
