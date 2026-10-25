# Emotive Gait Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `pip install -e .` package where six continuous emotion intensities drive a parameterized in-place walk cycle on a Unitree G1 with an injected neck joint, rendered kinematically and simulated under physics on a Z-rail root.

**Architecture:** Four layers, each depending only on the one before. `robot/` resolves semantic joint roles to a concrete model's actuator names and injects the neck. `gait/` turns a `GaitParameters` dataclass plus optional periodic spline offsets into per-role joint angles over a normalized phase. `emotion/` maps six intensities onto three affective axes which modulate those parameters — never the generator's output. `sim/` plays a trajectory either kinematically (`mj_forward`) or under physics (`mj_step`) and reports balance metrics.

**Tech Stack:** Python 3.12+, MuJoCo 3.2.5+, `mujoco-menagerie`, NumPy, SciPy (`CubicSpline`), PyYAML, pytest, ruff, ffmpeg (already on PATH).

**Spec:** `docs/superpowers/specs/2026-10-25-emotive-gait-professionalization-design.md`

**Scope:** This plan covers design Phases 1 to 3, which the design names a shippable milestone. Phases 4 to 7 (task-space IK and true stride, evaluation harness, GUI, presentation) get their own plans once this one lands and there is something to regress against.

## Global Constraints

- **Python >= 3.12.** `mujoco-menagerie` calls `tarfile.extractall(filter='data')` (PEP 706). The author's 3.11.3 lacks it. This is the project's only manual install.
- **MuJoCo floor `mujoco>=3.2.0`.** Do not raise it in this plan; `mink` and the 3.17 upgrade are Phase 4.
- **Load `scene.xml`, never `g1.xml`.** `g1.xml` has no ground plane and the robot free-falls through the void.
- **All angles in radians.** G1's MJCF is `compiler angle="radian"`.
- **Iterate `range(model.nu)` and resolve `model.actuator_trnid[i, 0]` -> `model.jnt_qposadr[j]`.** Never `range(model.nq)` with `model.joint(i)`; that yields the freejoint and is off-by-one.
- **Roll channels are mirrored.** `left_shoulder_roll_joint` is [-1.5882, 2.2515] and `right_shoulder_roll_joint` is [-2.2515, 1.5882]; hips likewise. Symmetric modulation negates roll, never copies it.
- **Neck mass split, exact:** head 1.8025 kg, diaginertia [0.008112, 0.007909, 0.002908], centroid [0.003687, 0.0000248, 0.429132]; torso drops 7.818 -> 6.0155. Total must stay 33.3411 kg.
- **The Z-rail is the physics tier.** A free root fell at 0.74-2.25 s across all presets, symmetric or antiphase. Never promise free-root stepping.
- **Disgust and surprise are unsourced** and must be flagged as such in code, docs and any UI.
- **No literature-derived magnitudes in degrees.** Only walking speeds in m/s and swing-phase percentages have published values. Everything else is directional.
- **License: MIT.**

## Review Focus

Input classes the spec implies but no task's happy-path tests exercise. Each has a test added to the task that owns the code.

1. **Emotion intensity outside [0, 1]** — negative, greater than one, or NaN. A slider or API caller will produce these; silently propagating NaN into `qpos` corrupts the whole trajectory. Owned by Task 14.
2. **Parameters driving a joint past its range** — the entire point is pushing extremes, and `waist_pitch` is only +/-0.52 rad while `ankle_pitch` is [-0.8727, 0.5236]. Must clamp to the model's own range, never emit invalid `qpos`. Owned by Task 8.
3. **`hold_speed_constant` when cadence or step_length is at or near zero** — sadness drives both down, and the proportional renormalization divides by their product and takes a square root. Owned by Task 6.
4. **A profile missing a role the emotion model wants** — the design promises explicit degradation, not a `KeyError` deep in a loop. Owned by Task 3.
5. **Model download unavailable (offline or first run)** — the design accepted a network dependency on first use; it must fail with an actionable message naming the prefetch command, not a bare traceback. Owned by Task 4.

---

## Task 1: Package scaffolding, tooling, license

**Files:**
- Create: `pyproject.toml`, `LICENSE`, `.gitignore`, `.github/workflows/ci.yml`, `src/emotive_gait/__init__.py`, `tests/test_packaging.py`

**Interfaces:**
- Consumes: nothing.
- Produces: importable `emotive_gait` package with `__version__: str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_packaging.py
def test_package_imports_and_has_version():
    import emotive_gait
    assert isinstance(emotive_gait.__version__, str)
    assert emotive_gait.__version__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_packaging.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'emotive_gait'`

- [ ] **Step 3: Create the package and its configuration**

`pyproject.toml`: setuptools backend, `src` layout, `requires-python = ">=3.12"`, dependencies `mujoco>=3.2.0`, `mujoco-menagerie`, `numpy`, `scipy`, `pyyaml`; optional `dev` extra with `pytest`, `ruff`. Configure `[tool.ruff]` with line-length 100 and `[tool.pytest.ini_options]` with `testpaths = ["tests"]`.
`src/emotive_gait/__init__.py` defines `__version__ = "0.1.0"`.
`LICENSE` is the MIT license, copyright 2026.
`.gitignore` covers `.venv/`, `__pycache__/`, `*.pyc`, `MUJOCO_LOG.TXT`, `*.npy`, `*.mp4`, `build/`, `dist/`, `*.egg-info/`.
`.github/workflows/ci.yml` runs on `ubuntu-latest` and `windows-latest` with Python 3.12: install `.[dev]`, run `ruff check`, `ruff format --check`, `pytest`.

- [ ] **Step 4: Install and run**

Run: `pip install -e ".[dev]" && pytest tests/test_packaging.py -v && ruff check .`
Expected: install succeeds, test PASSES, ruff reports no errors

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml LICENSE .gitignore .github src tests
git commit -m "feat: package scaffolding, MIT license, ruff and pytest config, CI"
```

---

## Task 2: Remove superseded files

**Files:**
- Delete: `nao.urdf`, `nao (1).urdf`, `NAOH25V60.urdf`, `robot.urdf`, `atlas_v4_with_multisense.urdf`, `atlas_minimal_contact.urdf`, `atlas_minimal_contact.xml`, `test.xml`, `scene.xml`, `naoV50_generated_urdf/`, `meshes/`, `MUJOCO_LOG.TXT`, `__pycache__/`, `clipped_spline_points.npy`, `TaiChi.motion`, `another.py`, `new.py`, `new2.py`, `emotive-gait-pybullet.py`, `emotive-gait-mujoco.py`
- Move: `emotive-gait-mujoco.ipynb` -> `docs/archive/emotive-gait-mujoco.ipynb`
- Create: `docs/archive/README.md`, `docs/future-learned-model.md`
- Keep for now: `emotive-gait.py` (Task 10 ports `compute_zmp` from it; deleted in Task 15)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing importable. This task clears ground.

- [ ] **Step 1: Delete the superseded files with `git rm -r`**

History preserves them. Do not delete `emotive-gait.py` yet.

- [ ] **Step 2: Write the archive and future-work notes**

`docs/archive/README.md`: one paragraph saying the notebook is a working transcript of the MuJoCo LQR tutorial adapted to the retired Atlas model, kept for provenance, not maintained, and not expected to run.
`docs/future-learned-model.md`: transcribe `emotionmodel.py`'s dataset shortlist (EWalk, Emotion-Gait, Human3.6M, CMU Mocap, ICT, BML, SIG) and its three-step plan, then state that the analytic emotion model in `emotion/` is the seam a learned model would replace, naming `emotion.model.modulate` as the function to swap.

- [ ] **Step 3: Verify the tree is clean**

Run: `pytest -q && ruff check .`
Expected: PASS, no missing-file errors

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove superseded Atlas/NAO assets and stranded Webots code"
```

---

## Task 3: Semantic joint roles

**Files:**
- Create: `src/emotive_gait/robot/__init__.py`, `src/emotive_gait/robot/roles.py`
- Test: `tests/robot/test_roles.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class JointRole(str, Enum)` with members `HIP_PITCH_L/R`, `HIP_ROLL_L/R`, `HIP_YAW_L/R`, `KNEE_L/R`, `ANKLE_PITCH_L/R`, `ANKLE_ROLL_L/R`, `WAIST_YAW`, `WAIST_ROLL`, `WAIST_PITCH`, `SHOULDER_PITCH_L/R`, `SHOULDER_ROLL_L/R`, `SHOULDER_YAW_L/R`, `ELBOW_L/R`, `NECK_PITCH`
  - `@dataclass(frozen=True) class RobotProfile` with fields `name: str`, `scene_entry: str`, `actuator_for: Mapping[JointRole, str]`, `sign_for: Mapping[JointRole, float]`
  - `RobotProfile.has(role) -> bool`
  - `RobotProfile.missing(roles: Iterable[JointRole]) -> tuple[JointRole, ...]`
  - `RobotProfile.actuator(role) -> str` raising `MissingRoleError(role, profile_name)` when absent
  - `class MissingRoleError(KeyError)` carrying `.role` and `.profile_name`
  - `JointRole.mirror() -> JointRole` returning the opposite-side role, or `self` for midline roles

- [ ] **Step 1: Write the failing tests**

```python
# tests/robot/test_roles.py
from emotive_gait.robot.roles import JointRole, RobotProfile, MissingRoleError

def _profile(**kw):
    return RobotProfile(name="t", scene_entry="scene",
                        actuator_for={JointRole.ELBOW_L: "left_elbow_joint"},
                        sign_for={JointRole.ELBOW_L: 1.0}, **kw)

def test_actuator_resolves_known_role():
    assert _profile().actuator(JointRole.ELBOW_L) == "left_elbow_joint"

def test_missing_role_raises_with_context():
    with pytest.raises(MissingRoleError) as e:
        _profile().actuator(JointRole.NECK_PITCH)
    assert e.value.role is JointRole.NECK_PITCH
    assert e.value.profile_name == "t"

# Review Focus 4: degradation is explicit and queryable, not an exception deep in a loop
def test_missing_reports_absent_roles_without_raising():
    p = _profile()
    assert p.has(JointRole.ELBOW_L) is True
    assert p.has(JointRole.NECK_PITCH) is False
    assert p.missing([JointRole.ELBOW_L, JointRole.NECK_PITCH]) == (JointRole.NECK_PITCH,)

def test_mirror_swaps_sides_and_fixes_midline():
    assert JointRole.ELBOW_L.mirror() is JointRole.ELBOW_R
    assert JointRole.ELBOW_R.mirror() is JointRole.ELBOW_L
    assert JointRole.WAIST_PITCH.mirror() is JointRole.WAIST_PITCH
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/robot/test_roles.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Implement `roles.py`**

`mirror()` derives the partner by swapping a trailing `_L`/`_R` in the member name and returning `self` when there is none.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/robot/test_roles.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/robot tests/robot
git commit -m "feat: semantic joint roles and robot profile with explicit degradation"
```

---

## Task 4: Model loader and G1 profile

**Files:**
- Create: `src/emotive_gait/robot/profiles/__init__.py`, `src/emotive_gait/robot/profiles/g1.py`, `src/emotive_gait/robot/loader.py`
- Test: `tests/robot/test_loader.py`

**Interfaces:**
- Consumes: Task 3's `JointRole`, `RobotProfile`.
- Produces:
  - `G1_PROFILE: RobotProfile` — `name="unitree_g1"`, `scene_entry="scene"`, mapping every role except `NECK_PITCH` to its G1 actuator name (which equals the joint name), `sign_for` = -1.0 for all `*_ROLL_R` roles and +1.0 otherwise
  - `@dataclass class RobotModel` with `model: mujoco.MjModel`, `profile: RobotProfile`, `stand_qpos: np.ndarray`
  - `RobotModel.qpos_adr(role) -> int`, `RobotModel.ctrl_index(role) -> int`, `RobotModel.joint_range(role) -> tuple[float, float]`
  - `load_robot(profile: RobotProfile = G1_PROFILE, *, with_neck: bool = True) -> RobotModel`
  - `class ModelUnavailableError(RuntimeError)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/robot/test_loader.py
NEEDED = [JointRole.HIP_PITCH_L, JointRole.KNEE_R, JointRole.ANKLE_PITCH_L,
          JointRole.WAIST_PITCH, JointRole.SHOULDER_PITCH_R, JointRole.ELBOW_L]

def test_profile_resolves_core_roles_to_real_actuators():
    r = load_robot(with_neck=False)
    for role in NEEDED:
        assert r.ctrl_index(role) >= 0
        assert mujoco.mj_id2name(r.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                 r.ctrl_index(role)) == G1_PROFILE.actuator(role)

def test_g1_has_no_neck_without_injection():
    assert G1_PROFILE.has(JointRole.NECK_PITCH) is False

def test_roll_signs_are_mirrored_not_copied():
    assert G1_PROFILE.sign_for[JointRole.SHOULDER_ROLL_L] == 1.0
    assert G1_PROFILE.sign_for[JointRole.SHOULDER_ROLL_R] == -1.0

def test_stand_qpos_matches_model_and_is_upright():
    r = load_robot(with_neck=False)
    assert r.stand_qpos.shape == (r.model.nq,)
    d = mujoco.MjData(r.model); d.qpos[:] = r.stand_qpos
    mujoco.mj_forward(r.model, d)
    tid = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    assert d.xmat[tid].reshape(3, 3)[2, 2] > 0.99

def test_known_ranges_match_the_spec():
    r = load_robot(with_neck=False)
    assert r.joint_range(JointRole.WAIST_PITCH) == pytest.approx((-0.52, 0.52))
    assert r.joint_range(JointRole.ANKLE_PITCH_L) == pytest.approx((-0.87267, 0.5236), abs=1e-4)

# Review Focus 5: an unavailable download names the fix
def test_unavailable_model_raises_actionable_error(monkeypatch):
    import mujoco_menagerie
    monkeypatch.setattr(mujoco_menagerie, "get",
                        lambda n: (_ for _ in ()).throw(OSError("no network")))
    with pytest.raises(ModelUnavailableError) as e:
        load_robot()
    assert "prefetch" in str(e.value)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/robot/test_loader.py -v`
Expected: FAIL, `ImportError`

- [ ] **Step 3: Implement `profiles/g1.py` and `loader.py`**

`load_robot` calls `mujoco_menagerie.get(profile.name).model(profile.scene_entry)`, wrapping any `OSError`/`mujoco_menagerie.MenagerieError` in `ModelUnavailableError` whose message names `python -m mujoco_menagerie prefetch unitree_g1`. With `with_neck=True` it routes through Task 5's `build_with_neck` instead. `stand_qpos` comes from the compiled model's `stand` keyframe when present, else from Task 5's name-based rebuild.

**Signature note for Task 5.** In this task `profile` defaults to `G1_PROFILE` and `with_neck` defaults to `False`, because the neck does not exist yet. Task 5 flips the default to `with_neck=True` and makes `profile=None` mean "pick `G1_PROFILE_WITH_NECK` when `with_neck` else `G1_PROFILE`", so the default profile and the default neck setting cannot disagree. Write the signature as `load_robot(profile: RobotProfile | None = None, *, with_neck: bool = False)` now so only the default changes later.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/robot/test_loader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/robot tests/robot
git commit -m "feat: G1 profile and model loader with actionable download failure"
```

---

## Task 5: Neck injection

**Files:**
- Create: `src/emotive_gait/robot/neck.py`
- Modify: `src/emotive_gait/robot/profiles/g1.py` (add `NECK_PITCH` to a `G1_PROFILE_WITH_NECK`)
- Test: `tests/robot/test_neck.py`

**Interfaces:**
- Consumes: Task 3's roles, Task 4's `RobotModel`.
- Produces:
  - `@dataclass(frozen=True) class NeckConfig` with defaults `parent_body="torso_link"`, `mesh_match="head"`, `pivot=(0.0, 0.0, 0.32)`, `mass=1.8025`, `diaginertia=(0.008112, 0.007909, 0.002908)`, `centroid=(0.003687, 0.0000248, 0.429132)`, `parent_mass_after=6.0155`, `joint_range=(-0.5, 0.8)`, `kp=300.0`, `kv=30.0`, `joint_name="neck_pitch_joint"`
  - `inject_neck(spec: mujoco.MjSpec, cfg: NeckConfig = NeckConfig()) -> mujoco.MjSpec`
  - `build_with_neck(entry_xml: str, cfg=NeckConfig(), *, offwidth=960, offheight=720) -> mujoco.MjModel`
  - `standing_qpos(src_model, new_model) -> np.ndarray` mapping the `stand` keyframe by joint name

- [ ] **Step 1: Write the failing tests**

```python
# tests/robot/test_neck.py
BASELINE_MASS = 33.3411

def test_neck_joint_and_actuator_exist_with_expected_range():
    r = load_robot(with_neck=True)
    assert r.profile.has(JointRole.NECK_PITCH)
    assert r.joint_range(JointRole.NECK_PITCH) == pytest.approx((-0.5, 0.8))
    assert r.ctrl_index(JointRole.NECK_PITCH) >= 0

def test_total_mass_is_conserved():
    r = load_robot(with_neck=True)
    assert r.model.body_subtreemass[0] == pytest.approx(BASELINE_MASS, abs=1e-3)

def test_injection_adds_exactly_one_dof_and_one_actuator():
    bare, necked = load_robot(with_neck=False), load_robot(with_neck=True)
    assert necked.model.nq == bare.model.nq + 1
    assert necked.model.nu == bare.model.nu + 1

def test_both_head_geoms_move_to_the_new_body():
    r = load_robot(with_neck=True)
    head = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_BODY, "head_link")
    on_head = [g for g in range(r.model.ngeom) if r.model.geom_bodyid[g] == head]
    assert len(on_head) == 2  # one visual (group 2), one collision (group 3)
    assert {int(r.model.geom_group[g]) for g in on_head} == {2, 3}

def test_commanding_neck_pitch_moves_the_head_down():
    r = load_robot(with_neck=True)
    d = mujoco.MjData(r.model); d.qpos[:] = r.stand_qpos
    mujoco.mj_forward(r.model, d)
    head = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_BODY, "head_link")
    z_up = d.xpos[head][2]
    d.qpos[r.qpos_adr(JointRole.NECK_PITCH)] = 0.8
    mujoco.mj_forward(r.model, d)
    assert d.xpos[head][2] < z_up

def test_standing_pose_survives_the_keyframe_rebuild():
    r = load_robot(with_neck=True)
    bare = load_robot(with_neck=False)
    for role in (JointRole.ELBOW_L, JointRole.SHOULDER_PITCH_R, JointRole.SHOULDER_ROLL_R):
        assert r.stand_qpos[r.qpos_adr(role)] == pytest.approx(bare.stand_qpos[bare.qpos_adr(role)])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/robot/test_neck.py -v`
Expected: FAIL, `ImportError`

- [ ] **Step 3: Implement `neck.py`**

Mirror the verified Phase 0 procedure. Set `getattr(spec.visual, "global").offwidth/offheight` before compiling. Find head geoms by iterating `body.first_geom()` / `body.next_geom(g)` and matching `g.meshname`. Create the child body at `cfg.pivot` with `explicitinertial=True`, copy each geom at `pos - pivot` with `density=0.0`, call `src.delete()`, set `torso.mass = cfg.parent_mass_after`. Actuator `gainprm` and `biasprm` must be length 10: `[kp] + [0]*9` and `[0, -kp, -kv] + [0]*7`. Delete all keyframes before compiling, since adding a joint invalidates them, then rebuild the pose with `standing_qpos`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/robot/test_neck.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/robot tests/robot
git commit -m "feat: inject neck pitch joint into G1 with mass-conserving inertia split"
```

---

## Task 6: Gait parameters

**Files:**
- Create: `src/emotive_gait/gait/__init__.py`, `src/emotive_gait/gait/params.py`
- Test: `tests/gait/test_params.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class SpeedMode(str, Enum)`: `PROPORTIONAL`, `HOLD_CADENCE`, `HOLD_STEP_LENGTH`
  - `@dataclass(frozen=True) class GaitParameters` with the 16 fields and defaults from design 4.3: `cadence=1.8`, `step_length=0.35`, `step_height=0.05`, `double_support_frac=0.20`, `feet_spacing=0.15`, `com_height_scale=1.0`, `com_bounce_amp=0.02`, `lateral_sway_amp=0.015`, `trunk_pitch=0.0`, `neck_pitch=0.0`, `knee_flexion_bias=0.0`, `arm_swing_amp=0.35`, `arm_abduction=0.05`, `elbow_flexion=0.25`, `elbow_swing_ratio=0.5`, `expansiveness=1.0`
  - `GaitParameters.speed -> float` (property, `step_length * cadence`, read-only)
  - `GaitParameters.with_speed(target: float, mode: SpeedMode = SpeedMode.PROPORTIONAL) -> GaitParameters`
  - `GaitParameters.to_yaml(path) -> None`, `GaitParameters.from_yaml(path) -> GaitParameters`
  - `class SpeedConstraintError(ValueError)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/gait/test_params.py
def test_speed_is_derived_and_read_only():
    p = GaitParameters(cadence=2.0, step_length=0.4)
    assert p.speed == pytest.approx(0.8)
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        p.speed = 1.0

@pytest.mark.parametrize("mode", list(SpeedMode))
def test_with_speed_hits_the_target_in_every_mode(mode):
    p = GaitParameters(cadence=2.4, step_length=0.5)
    assert p.with_speed(0.8, mode).speed == pytest.approx(0.8)

def test_proportional_mode_preserves_cadence_to_stride_ratio():
    p = GaitParameters(cadence=2.4, step_length=0.5)
    q = p.with_speed(0.8, SpeedMode.PROPORTIONAL)
    assert q.cadence / q.step_length == pytest.approx(p.cadence / p.step_length)

def test_hold_modes_pin_the_named_quantity():
    p = GaitParameters(cadence=2.4, step_length=0.5)
    assert p.with_speed(0.8, SpeedMode.HOLD_CADENCE).cadence == pytest.approx(2.4)
    assert p.with_speed(0.8, SpeedMode.HOLD_STEP_LENGTH).step_length == pytest.approx(0.5)

# Review Focus 3: degenerate inputs to the speed constraint
@pytest.mark.parametrize("cadence,step_length,target", [
    (0.0, 0.35, 0.8), (1.8, 0.0, 0.8), (0.0, 0.0, 0.8), (1.8, 0.35, 0.0), (1.8, 0.35, -1.0),
])
def test_degenerate_speed_constraints_raise_not_nan(cadence, step_length, target):
    p = GaitParameters(cadence=cadence, step_length=step_length)
    with pytest.raises(SpeedConstraintError):
        p.with_speed(target)

def test_yaml_round_trip_preserves_every_field(tmp_path):
    p = GaitParameters(cadence=2.1, trunk_pitch=0.3, expansiveness=1.2)
    f = tmp_path / "g.yaml"; p.to_yaml(f)
    assert GaitParameters.from_yaml(f) == p

def test_from_yaml_rejects_unknown_keys(tmp_path):
    f = tmp_path / "g.yaml"; f.write_text("cadence: 2.0\nnot_a_param: 1\n")
    with pytest.raises(ValueError, match="not_a_param"):
        GaitParameters.from_yaml(f)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/gait/test_params.py -v`
Expected: FAIL, `ImportError`

- [ ] **Step 3: Implement `params.py`**

`PROPORTIONAL` scales both by `k = sqrt(target / (cadence * step_length))`. Raise `SpeedConstraintError` when `cadence <= 0`, `step_length <= 0`, or `target <= 0`, before any division or square root, so no NaN can escape. `from_yaml` compares the loaded mapping's keys against the dataclass fields and raises naming the offenders.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/gait/test_params.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/gait tests/gait
git commit -m "feat: GaitParameters with derived speed, pinning modes, YAML round-trip"
```

---

## Task 7: Periodic curves

**Files:**
- Create: `src/emotive_gait/gait/spline.py`
- Test: `tests/gait/test_spline.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass(frozen=True) class PeriodicCurve` with `knots_phase: np.ndarray`, `knots_value: np.ndarray`
  - `PeriodicCurve.from_knots(phase, value) -> PeriodicCurve` — closes the loop itself if `value[0] != value[-1]` is not already satisfied by appending the first point at phase 1.0
  - `PeriodicCurve.__call__(phase) -> np.ndarray` — accepts scalar or array, wraps phase modulo 1.0
  - `PeriodicCurve.derivative(phase, order=1) -> np.ndarray`
  - `class NonPeriodicKnotsError(ValueError)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/gait/test_spline.py
def test_curve_interpolates_its_knots():
    c = PeriodicCurve.from_knots([0.0, 0.25, 0.5, 0.75], [0.0, 1.0, 0.0, -1.0])
    assert c(0.25) == pytest.approx(1.0)
    assert c(0.75) == pytest.approx(-1.0)

def test_phase_wraps_modulo_one():
    c = PeriodicCurve.from_knots([0.0, 0.25, 0.5, 0.75], [0.0, 1.0, 0.0, -1.0])
    assert c(1.25) == pytest.approx(c(0.25))
    assert c(-0.75) == pytest.approx(c(0.25))

def test_c2_continuous_across_the_wrap():
    """bc_type='periodic' matches first AND second derivatives, unlike the old
    GUI's hand-shared endpoint Y which only ever bought C0."""
    c = PeriodicCurve.from_knots([0.0, 0.25, 0.5, 0.75], [0.0, 1.0, 0.0, -1.0])
    eps = 1e-6
    for order in (1, 2):
        before = c.derivative(1.0 - eps, order)
        after = c.derivative(0.0 + eps, order)
        assert before == pytest.approx(after, abs=1e-3)

def test_vectorized_call_matches_scalar_calls():
    c = PeriodicCurve.from_knots([0.0, 0.5], [0.0, 1.0])
    xs = np.linspace(0, 1, 11)
    assert np.allclose(c(xs), [c(float(x)) for x in xs])

def test_explicitly_non_periodic_closing_knot_is_rejected():
    with pytest.raises(NonPeriodicKnotsError):
        PeriodicCurve.from_knots([0.0, 0.5, 1.0], [0.0, 1.0, 0.7])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/gait/test_spline.py -v`
Expected: FAIL, `ImportError`

- [ ] **Step 3: Implement `spline.py`**

Wrap `scipy.interpolate.CubicSpline(..., bc_type="periodic")`, which requires `value[0] == value[-1]`. When the caller supplies knots that do not include phase 1.0, append `(1.0, value[0])` to close the loop. When they do supply a closing knot whose value disagrees with the first, raise `NonPeriodicKnotsError` rather than silently correcting.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/gait/test_spline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/gait tests/gait
git commit -m "feat: C2-continuous periodic curve over normalized phase"
```

---

## Task 8: Gait generator

**Files:**
- Create: `src/emotive_gait/gait/generator.py`
- Test: `tests/gait/test_generator.py`

**Interfaces:**
- Consumes: Task 3 roles, Task 4 `RobotModel`, Task 6 `GaitParameters`, Task 7 `PeriodicCurve`.
- Produces:
  - `@dataclass(frozen=True) class GaitTrajectory` with `roles: tuple[JointRole, ...]`, `params: GaitParameters`
  - `GaitTrajectory.angles_at(phase: float) -> dict[JointRole, float]`
  - `GaitTrajectory.qpos_at(robot: RobotModel, phase: float) -> np.ndarray` — a full `qpos` starting from `robot.stand_qpos`
  - `GaitTrajectory.ctrl_at(robot: RobotModel, phase: float) -> np.ndarray`
  - `generate(params, robot, *, offsets=None, overrides=None) -> GaitTrajectory` where both are `Mapping[JointRole, PeriodicCurve] | None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/gait/test_generator.py
def test_legs_are_antiphase_and_arms_contralateral():
    t = generate(GaitParameters(), load_robot())
    a = t.angles_at(0.25)
    assert a[JointRole.HIP_PITCH_L] * a[JointRole.HIP_PITCH_R] < 0
    assert a[JointRole.HIP_PITCH_L] * a[JointRole.SHOULDER_PITCH_L] < 0

def test_cycle_is_continuous_across_the_phase_wrap():
    t = generate(GaitParameters(), load_robot())
    for role, v in t.angles_at(1.0 - 1e-6).items():
        assert v == pytest.approx(t.angles_at(0.0)[role], abs=1e-4)

def test_cycle_is_continuous_at_the_stance_to_swing_handoff():
    """The easier boundary to get wrong than the phase wrap."""
    t = generate(GaitParameters(), load_robot())
    for ph in (0.5, 0.0):
        for role, v in t.angles_at(ph - 1e-6).items():
            assert v == pytest.approx(t.angles_at(ph + 1e-6)[role], abs=1e-4)

def test_roll_channels_are_mirrored_between_sides():
    t = generate(GaitParameters(arm_abduction=0.3), load_robot())
    a = t.angles_at(0.0)
    assert a[JointRole.SHOULDER_ROLL_L] == pytest.approx(-a[JointRole.SHOULDER_ROLL_R])

def test_expansiveness_scales_procedural_amplitude_only():
    r = load_robot()
    base = generate(GaitParameters(expansiveness=1.0), r)
    big = generate(GaitParameters(expansiveness=1.4), r)
    amp = lambda t, role: max(abs(t.angles_at(p)[role]) for p in np.linspace(0, 1, 64))
    assert amp(big, JointRole.SHOULDER_PITCH_L) > amp(base, JointRole.SHOULDER_PITCH_L)

def test_offsets_are_additive_and_overrides_replace():
    r, p = load_robot(), GaitParameters()
    flat = PeriodicCurve.from_knots([0.0, 0.5], [0.2, 0.2])
    base = generate(p, r).angles_at(0.3)[JointRole.ELBOW_L]
    assert generate(p, r, offsets={JointRole.ELBOW_L: flat}).angles_at(0.3)[JointRole.ELBOW_L] \
        == pytest.approx(base + 0.2)
    assert generate(p, r, overrides={JointRole.ELBOW_L: flat}).angles_at(0.3)[JointRole.ELBOW_L] \
        == pytest.approx(0.2)

def test_expansiveness_does_not_rescale_hand_authored_offsets():
    r = load_robot()
    flat = PeriodicCurve.from_knots([0.0, 0.5], [0.2, 0.2])
    d = lambda e: (generate(GaitParameters(expansiveness=e), r, offsets={JointRole.ELBOW_L: flat})
                   .angles_at(0.3)[JointRole.ELBOW_L]
                   - generate(GaitParameters(expansiveness=e), r).angles_at(0.3)[JointRole.ELBOW_L])
    assert d(1.0) == pytest.approx(d(1.5))

def test_absent_role_is_skipped_not_raised():
    r = load_robot(with_neck=False)
    assert JointRole.NECK_PITCH not in generate(GaitParameters(neck_pitch=0.5), r).angles_at(0.0)

# Review Focus 2: extremes must clamp to the model's own ranges
@pytest.mark.parametrize("params", [
    GaitParameters(trunk_pitch=3.0), GaitParameters(trunk_pitch=-3.0),
    GaitParameters(arm_swing_amp=9.0), GaitParameters(knee_flexion_bias=5.0),
    GaitParameters(expansiveness=1.5, arm_swing_amp=3.0),
])
def test_extreme_parameters_stay_inside_joint_limits(params):
    r = load_robot()
    t = generate(params, r)
    for ph in np.linspace(0, 1, 64):
        for role, v in t.angles_at(float(ph)).items():
            lo, hi = r.joint_range(role)
            assert lo - 1e-9 <= v <= hi + 1e-9, f"{role}={v} outside [{lo},{hi}] at phase {ph}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/gait/test_generator.py -v`
Expected: FAIL, `ImportError`

- [ ] **Step 3: Implement `generator.py`**

Composition order is fixed by design 4.3.1: procedural base, then per-role override replaces or offset adds, then clamp to `robot.joint_range(role)` last. Apply `expansiveness` to the procedural amplitude group only (`step_length`, `step_height`, `arm_swing_amp`, `arm_abduction`, `com_bounce_amp`), as a post-multiplier computed in one place. Skip roles the profile lacks. Legs use `sin(2*pi*phase)` with the right leg at `phase + 0.5`; arms take the contralateral sign; roll channels multiply by `profile.sign_for[role]`.

**Parameters that are deliberately inert in this plan.** `step_length`, `step_height`, `feet_spacing`, `double_support_frac`, `com_height_scale`, `com_bounce_amp` and `lateral_sway_amp` describe foot placement and root motion, which an in-place gait does not produce. They are still real fields, because `speed` is derived from `step_length * cadence` and the emotion model and `hold_speed` both operate on them. Do **not** attempt foot placement here; Phase 4 consumes them via task-space IK. The generator reads only `cadence`, `trunk_pitch`, `neck_pitch`, `knee_flexion_bias`, `arm_swing_amp`, `arm_abduction`, `elbow_flexion`, `elbow_swing_ratio` and `expansiveness`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/gait/test_generator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/gait tests/gait
git commit -m "feat: parameterized in-place gait generator with clamping and offset composition"
```

---

## Task 9: Balance metrics

**Files:**
- Create: `src/emotive_gait/sim/__init__.py`, `src/emotive_gait/sim/metrics.py`
- Test: `tests/sim/test_metrics.py`

**Interfaces:**
- Consumes: Task 4 `RobotModel`.
- Produces:
  - `center_of_pressure(model, data) -> np.ndarray | None` — world xy, `None` when no foot contact
  - `zero_moment_point(model, data) -> np.ndarray | None`
  - `support_polygon(model, data) -> np.ndarray` — (N, 2) convex hull of foot contact points
  - `is_upright(model, data, *, threshold=0.6, body="torso_link") -> bool`
  - `uprightness(model, data, *, body="torso_link") -> float`

- [ ] **Step 1: Write the failing tests**

```python
# tests/sim/test_metrics.py
def test_uprightness_is_one_when_standing_and_drops_when_tipped():
    r = load_robot(); d = mujoco.MjData(r.model); d.qpos[:] = r.stand_qpos
    mujoco.mj_forward(r.model, d)
    assert uprightness(r.model, d) == pytest.approx(1.0, abs=1e-3)
    assert is_upright(r.model, d)
    d.qpos[3:7] = [0.707, 0.0, 0.707, 0.0]  # 90 deg pitch about y
    mujoco.mj_forward(r.model, d)
    assert not is_upright(r.model, d)

def test_cop_is_none_without_contact_and_inside_support_when_standing():
    r = load_robot(); d = mujoco.MjData(r.model)
    d.qpos[:] = r.stand_qpos; d.qpos[2] += 2.0  # lifted clear of the floor
    mujoco.mj_forward(r.model, d)
    assert center_of_pressure(r.model, d) is None

    d.qpos[:] = r.stand_qpos
    for _ in range(200):
        d.ctrl[:] = [r.stand_qpos[r.model.jnt_qposadr[r.model.actuator_trnid[i, 0]]]
                     for i in range(r.model.nu)]
        mujoco.mj_step(r.model, d)
    cop = center_of_pressure(r.model, d)
    assert cop is not None
    hull = support_polygon(r.model, d)
    assert hull[:, 0].min() <= cop[0] <= hull[:, 0].max()
    assert hull[:, 1].min() <= cop[1] <= hull[:, 1].max()

def test_cop_uses_real_contact_forces_not_a_scalar_vertical_estimate():
    """The old calculate_moments ignored horizontal reaction and per-contact
    normals; a one-footed stance is where that assumption visibly fails."""
    r = load_robot(); d = mujoco.MjData(r.model); d.qpos[:] = r.stand_qpos
    for _ in range(200):
        mujoco.mj_step(r.model, d)
    forces = np.zeros(6)
    total = 0.0
    for i in range(d.ncon):
        mujoco.mj_contactForce(r.model, d, i, forces)
        total += abs(forces[0])
    assert total > 0.5 * r.model.body_subtreemass[0] * 9.81
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/sim/test_metrics.py -v`
Expected: FAIL, `ImportError`

- [ ] **Step 3: Implement `metrics.py`**

Port `compute_zmp`'s formula from `emotive-gait.py:285-299` unchanged. **Replace** `calculate_moments`: iterate `data.contact`, keep contacts whose geoms belong to a foot body, call `mujoco.mj_contactForce` for each, rotate into world with `contact.frame`, and take the normal-force-weighted mean of contact positions for CoP. `uprightness` is `data.xmat[body].reshape(3,3)[2,2]`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/sim/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/sim tests/sim
git commit -m "feat: contact-force-based CoP, ZMP and uprightness metrics"
```

---

## Task 10: Kinematic and physical playback

**Files:**
- Create: `src/emotive_gait/sim/kinematic.py`, `src/emotive_gait/sim/physical.py`
- Test: `tests/sim/test_playback.py`

**Interfaces:**
- Consumes: Tasks 4, 8, 9.
- Produces:
  - `play_kinematic(robot, traj, *, duration, fps) -> Iterator[mujoco.MjData]` — sets `qpos`, calls `mj_forward`, never `mj_step`
  - `class RootMode(str, Enum)`: `FREE`, `Z_RAIL`, `WELD`
  - `to_root_mode(spec: mujoco.MjSpec, mode: RootMode) -> mujoco.MjSpec`
  - `@dataclass class SimResult`: `fell: bool`, `fell_at: float | None`, `pelvis_z: np.ndarray`, `uprightness: np.ndarray`, `cop: np.ndarray`
  - `simulate(robot, traj, *, duration, root=RootMode.Z_RAIL, ramp=0.5) -> SimResult`

- [ ] **Step 1: Write the failing tests**

```python
# tests/sim/test_playback.py
def test_kinematic_playback_never_integrates_physics():
    r = load_robot(); t = generate(GaitParameters(), r)
    frames = list(play_kinematic(r, t, duration=1.0, fps=30))
    assert len(frames) == 30
    assert all(f.time == 0.0 for f in frames)  # mj_forward does not advance time

def test_kinematic_playback_reaches_commanded_angles_exactly():
    r = load_robot(); t = generate(GaitParameters(trunk_pitch=0.4), r)
    f = next(iter(play_kinematic(r, t, duration=0.1, fps=10)))
    assert f.qpos[r.qpos_adr(JointRole.WAIST_PITCH)] == pytest.approx(0.4, abs=1e-9)

def test_z_rail_root_has_no_quaternion():
    r = load_robot(root=RootMode.Z_RAIL)
    assert r.model.nq == r.model.nv

# Local literals, not emotion.model.PRESET_PARAMS: that is Task 14 and this is Task 10.
# These mirror the Phase 0 spike's crude presets.
_PHASE0_PRESETS = {
    "neutral": GaitParameters(cadence=1.8, arm_swing_amp=0.30, elbow_flexion=1.28,
                              trunk_pitch=0.00, neck_pitch=0.00, knee_flexion_bias=0.25),
    "sad":     GaitParameters(cadence=1.0, arm_swing_amp=0.08, elbow_flexion=1.05,
                              trunk_pitch=0.32, neck_pitch=0.70, knee_flexion_bias=0.14),
    "angry":   GaitParameters(cadence=2.7, arm_swing_amp=0.60, elbow_flexion=1.85,
                              trunk_pitch=0.18, neck_pitch=-0.10, knee_flexion_bias=0.38),
    "happy":   GaitParameters(cadence=2.3, arm_swing_amp=0.62, elbow_flexion=0.95,
                              trunk_pitch=-0.12, neck_pitch=-0.35, knee_flexion_bias=0.36),
}

@pytest.mark.parametrize("preset", list(_PHASE0_PRESETS))
def test_z_rail_survives_six_seconds_on_every_preset(preset):
    """Measured in Phase 0. The free root fell at 0.74-2.25 s; the Z-rail did not."""
    r = load_robot(root=RootMode.Z_RAIL)
    res = simulate(r, generate(_PHASE0_PRESETS[preset], r), duration=6.0)
    assert not res.fell, f"{preset} fell at {res.fell_at}"

def test_ramp_avoids_the_startup_yank():
    r = load_robot(root=RootMode.Z_RAIL)
    t = generate(GaitParameters(knee_flexion_bias=0.4), r)
    assert not simulate(r, t, duration=1.0, ramp=0.5).fell

def test_fall_is_detected_by_uprightness_not_height():
    """A crouch lowers the pelvis without falling; only tipping is a fall."""
    r = load_robot(root=RootMode.Z_RAIL)
    res = simulate(r, generate(GaitParameters(knee_flexion_bias=0.5), r), duration=2.0)
    assert not res.fell
    assert res.pelvis_z.min() < res.pelvis_z[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/sim/test_playback.py -v`
Expected: FAIL, `ImportError`

- [ ] **Step 3: Implement both modules**

`to_root_mode` deletes the pelvis freejoint and, for `Z_RAIL`, adds a slide joint named `root_slide_z` on axis `[0,0,1]` with range `[-0.5, 0.5]`; for `WELD` it adds no joint. `load_robot` gains a `root: RootMode = RootMode.FREE` keyword. `simulate` blends `ctrl` from the standing pose to the trajectory target over `ramp` seconds, and flags a fall on `uprightness < 0.6`, never on height alone.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/sim/test_playback.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/sim tests/sim
git commit -m "feat: kinematic and Z-rail physical playback with uprightness-based fall detection"
```

---

## Task 11: Offscreen rendering

**Files:**
- Create: `src/emotive_gait/sim/render.py`
- Test: `tests/sim/test_render.py`

**Interfaces:**
- Consumes: Task 10.
- Produces:
  - `@dataclass(frozen=True) class CameraSpec`: `distance=3.2`, `azimuth=135.0`, `elevation=-8.0`, `lookat_z=0.9`
  - `render_frames(robot, frames: Iterable[mujoco.MjData], *, width=960, height=720, camera=CameraSpec()) -> Iterator[np.ndarray]`
  - `write_video(frames, path, *, fps=30, width=960, height=720) -> Path`
  - `class FfmpegUnavailableError(RuntimeError)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/sim/test_render.py
def test_frames_have_the_requested_shape():
    r = load_robot(); t = generate(GaitParameters(), r)
    f = next(iter(render_frames(r, play_kinematic(r, t, duration=0.1, fps=10),
                                width=320, height=240)))
    assert f.shape == (240, 320, 3) and f.dtype == np.uint8

def test_render_above_the_default_framebuffer_works():
    """Default offscreen buffer is 640x480; the model must raise offwidth/offheight."""
    r = load_robot(); t = generate(GaitParameters(), r)
    f = next(iter(render_frames(r, play_kinematic(r, t, duration=0.1, fps=10),
                                width=960, height=720)))
    assert f.shape == (720, 960, 3)

def test_write_video_produces_a_playable_file(tmp_path):
    r = load_robot(); t = generate(GaitParameters(), r)
    out = write_video(render_frames(r, play_kinematic(r, t, duration=0.5, fps=10),
                                    width=320, height=240),
                      tmp_path / "v.mp4", fps=10, width=320, height=240)
    assert out.is_file() and out.stat().st_size > 1000

def test_missing_ffmpeg_raises_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(FfmpegUnavailableError, match="ffmpeg"):
        write_video(iter([]), tmp_path / "v.mp4")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/sim/test_render.py -v`
Expected: FAIL, `ImportError`

- [ ] **Step 3: Implement `render.py`**

`write_video` pipes raw `rgb24` to `ffmpeg` on stdin with `-c:v libx264 -pix_fmt yuv420p`, checking `shutil.which("ffmpeg")` first. The offscreen framebuffer is raised at model build time in Task 5's `build_with_neck`, so `render_frames` only needs to construct `mujoco.Renderer` and close it in a `finally`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/sim/test_render.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/sim tests/sim
git commit -m "feat: offscreen rendering and ffmpeg video export"
```

---

## Task 12: Evidence module

**Files:**
- Create: `src/emotive_gait/emotion/__init__.py`, `src/emotive_gait/emotion/evidence.py`
- Test: `tests/emotion/test_evidence.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class Emotion(str, Enum)`: `ANGER`, `DISGUST`, `FEAR`, `HAPPINESS`, `SADNESS`, `SURPRISE`
  - `class Grounding(str, Enum)`: `MEASURED`, `DIRECTIONAL`, `UNSOURCED`
  - `@dataclass(frozen=True) class Claim`: `emotion: Emotion`, `parameter: str`, `direction: int` (-1, 0, +1), `grounding: Grounding`, `citation: str`, `note: str = ""`
  - `EVIDENCE: tuple[Claim, ...]` transcribed from design Appendix A.3
  - `claims_for(emotion) -> tuple[Claim, ...]`
  - `GROUNDED_EMOTIONS: frozenset[Emotion]`, `UNSOURCED_EMOTIONS: frozenset[Emotion]`
  - `MEASURED_SPEEDS: Mapping[Emotion, float]` — anger 1.82, happiness 1.31, fear 0.83, sadness 0.68

- [ ] **Step 1: Write the failing tests**

```python
# tests/emotion/test_evidence.py
def test_disgust_and_surprise_are_marked_unsourced():
    assert UNSOURCED_EMOTIONS == {Emotion.DISGUST, Emotion.SURPRISE}
    for e in UNSOURCED_EMOTIONS:
        assert all(c.grounding is Grounding.UNSOURCED for c in claims_for(e))

def test_the_four_grounded_emotions_have_citations():
    assert GROUNDED_EMOTIONS == {Emotion.ANGER, Emotion.FEAR, Emotion.HAPPINESS, Emotion.SADNESS}
    for e in GROUNDED_EMOTIONS:
        cs = claims_for(e)
        assert cs
        assert all(c.citation for c in cs)

def test_no_claim_asserts_a_magnitude_in_degrees():
    """Appendix A.2: only m/s speeds and swing-phase percentages have published values."""
    for c in EVIDENCE:
        assert "degree" not in c.note.lower() and "deg" not in c.note.lower().split()

def test_measured_speeds_order_matches_the_literature():
    s = MEASURED_SPEEDS
    assert s[Emotion.ANGER] > s[Emotion.HAPPINESS] > s[Emotion.FEAR] > s[Emotion.SADNESS]

def test_headline_cue_directions():
    def d(e, p): return next(c.direction for c in claims_for(e) if c.parameter == p)
    assert d(Emotion.SADNESS, "neck_pitch") > 0      # head down
    assert d(Emotion.SADNESS, "arm_swing_amp") < 0
    assert d(Emotion.ANGER, "elbow_flexion") > 0
    assert d(Emotion.FEAR, "elbow_flexion") > 0
    assert d(Emotion.HAPPINESS, "elbow_flexion") < 0
    assert d(Emotion.FEAR, "knee_flexion_bias") > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/emotion/test_evidence.py -v`
Expected: FAIL, `ImportError`

- [ ] **Step 3: Transcribe design Appendix A.3 into `EVIDENCE`**

One `Claim` per non-blank cell. `citation` is the short form used in the appendix ("Roether 2011 ch.3", "Barliya 2013", "Halovic & Kroos 2018"). Parameter names must exactly match `GaitParameters` field names so Task 14 can map them mechanically.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/emotion/test_evidence.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/emotion tests/emotion
git commit -m "feat: evidence module with per-claim grounding flags and citations"
```

---

## Task 13: Affective axes

**Files:**
- Create: `src/emotive_gait/emotion/axes.py`
- Test: `tests/emotion/test_axes.py`

**Interfaces:**
- Consumes: Task 12 `Emotion`.
- Produces:
  - `@dataclass(frozen=True) class AffectiveAxes`: `activation: float`, `valence: float`, `tension: float`
  - `AXIS_WEIGHTS: Mapping[Emotion, AffectiveAxes]` — per design 4.4 and Appendix A.6
  - `axes_from(intensities: Mapping[Emotion, float]) -> AffectiveAxes`
  - `CATEGORY_OFFSETS: Mapping[Emotion, Mapping[str, float]]` — fear's `knee_flexion_bias` and `elbow_swing_ratio`

- [ ] **Step 1: Write the failing tests**

```python
# tests/emotion/test_axes.py
def test_all_zero_intensities_give_zero_axes():
    a = axes_from({e: 0.0 for e in Emotion})
    assert (a.activation, a.valence, a.tension) == (0.0, 0.0, 0.0)

def test_activation_ordering_follows_measured_speeds():
    act = lambda e: axes_from({e: 1.0}).activation
    assert act(Emotion.ANGER) > act(Emotion.HAPPINESS) > 0 > act(Emotion.FEAR) > act(Emotion.SADNESS)

def test_valence_separates_sadness_from_happiness():
    assert axes_from({Emotion.SADNESS: 1.0}).valence < 0 < axes_from({Emotion.HAPPINESS: 1.0}).valence

def test_tension_groups_anger_and_fear_against_happiness_and_sadness():
    t = lambda e: axes_from({e: 1.0}).tension
    assert t(Emotion.ANGER) > 0 and t(Emotion.FEAR) > 0
    assert t(Emotion.HAPPINESS) < 0 and t(Emotion.SADNESS) < 0

def test_opposing_emotions_cancel_toward_neutral():
    """Documented consequence: averaging happy and sad yields neutral, not 'mixed'.
    Appendix A.6 found no evidence for blended-gait perception."""
    a = axes_from({Emotion.HAPPINESS: 1.0, Emotion.SADNESS: 1.0})
    assert a.valence == pytest.approx(0.0, abs=1e-9)

def test_intensity_scales_each_axis_monotonically():
    prev = -1e9
    for i in (0.0, 0.25, 0.5, 0.75, 1.0):
        cur = axes_from({Emotion.ANGER: i}).activation
        assert cur > prev
        prev = cur

def test_fear_carries_a_category_offset_for_knee_flexion():
    assert CATEGORY_OFFSETS[Emotion.FEAR]["knee_flexion_bias"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/emotion/test_axes.py -v`
Expected: FAIL, `ImportError`

- [ ] **Step 3: Implement `axes.py`**

`axes_from` is the intensity-weighted sum of `AXIS_WEIGHTS`. Weights are tuned, not cited; the docstring must say so and point at Appendix A.6 for the three-axis structure's provenance. Keep every weight in [-1, 1].

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/emotion/test_axes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/emotion tests/emotion
git commit -m "feat: activation/valence/tension affective axes from emotion intensities"
```

---

## Task 14: Emotion to gait modulation

**Files:**
- Create: `src/emotive_gait/emotion/model.py`
- Test: `tests/emotion/test_model.py`

**Interfaces:**
- Consumes: Tasks 3, 6, 12, 13.
- Produces:
  - `modulate(params: GaitParameters, intensities: Mapping[Emotion, float], profile: RobotProfile, *, hold_speed: float | None = None, speed_mode: SpeedMode = SpeedMode.PROPORTIONAL) -> GaitParameters`
  - `unexpressible(intensities, profile) -> tuple[tuple[Emotion, str], ...]` — (emotion, parameter) pairs the profile cannot show
  - `class IntensityError(ValueError)`
  - `PRESET_PARAMS: Mapping[str, GaitParameters]` for `neutral`, `sad`, `angry`, `happy`

- [ ] **Step 1: Write the failing tests**

```python
# tests/emotion/test_model.py
def test_neutral_is_the_identity():
    p = GaitParameters()
    assert modulate(p, {e: 0.0 for e in Emotion}, G1_PROFILE_WITH_NECK) == p

def test_sadness_slows_shortens_slumps_and_drops_the_head():
    p = GaitParameters()
    q = modulate(p, {Emotion.SADNESS: 1.0}, G1_PROFILE_WITH_NECK)
    assert q.cadence < p.cadence and q.step_length < p.step_length
    assert q.neck_pitch > p.neck_pitch and q.trunk_pitch > p.trunk_pitch
    assert q.arm_swing_amp < p.arm_swing_amp

def test_anger_speeds_up_and_flexes_the_elbows():
    p = GaitParameters()
    q = modulate(p, {Emotion.ANGER: 1.0}, G1_PROFILE_WITH_NECK)
    assert q.speed > p.speed and q.elbow_flexion > p.elbow_flexion

def test_modulation_direction_agrees_with_every_grounded_claim():
    p = GaitParameters()
    for emotion in GROUNDED_EMOTIONS:
        q = modulate(p, {emotion: 1.0}, G1_PROFILE_WITH_NECK)
        for c in claims_for(emotion):
            if c.direction == 0 or not hasattr(p, c.parameter):
                continue
            delta = getattr(q, c.parameter) - getattr(p, c.parameter)
            assert delta * c.direction >= 0, f"{emotion}.{c.parameter} contradicts {c.citation}"

def test_hold_speed_pins_speed_while_posture_still_varies():
    p = GaitParameters()
    q = modulate(p, {Emotion.SADNESS: 1.0}, G1_PROFILE_WITH_NECK, hold_speed=p.speed)
    assert q.speed == pytest.approx(p.speed)
    assert q.neck_pitch > p.neck_pitch and q.trunk_pitch > p.trunk_pitch

def test_profile_without_a_neck_reports_the_lost_cue():
    lost = unexpressible({Emotion.SADNESS: 1.0}, G1_PROFILE)  # no neck
    assert (Emotion.SADNESS, "neck_pitch") in lost

def test_modulation_degrades_rather_than_raising_without_a_neck():
    q = modulate(GaitParameters(), {Emotion.SADNESS: 1.0}, G1_PROFILE)
    assert q.cadence < GaitParameters().cadence  # other cues still applied

# Review Focus 1: out-of-range and non-finite intensities
@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_intensity_raises_before_any_nan_escapes(bad):
    with pytest.raises(IntensityError):
        modulate(GaitParameters(), {Emotion.ANGER: bad}, G1_PROFILE_WITH_NECK)

def test_every_preset_produces_finite_in_range_parameters():
    for name, p in PRESET_PARAMS.items():
        for f in dataclasses.fields(p):
            assert math.isfinite(getattr(p, f.name)), f"{name}.{f.name}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/emotion/test_model.py -v`
Expected: FAIL, `ImportError`

- [ ] **Step 3: Implement `model.py`**

Validate every intensity is finite and within [0, 1] first, raising `IntensityError` naming the emotion and value. Map `AffectiveAxes` onto parameter deltas, add `CATEGORY_OFFSETS`, drop deltas for parameters whose role the profile lacks (recording them for `unexpressible`), and apply `with_speed` last when `hold_speed` is given. `PRESET_PARAMS` is built by calling `modulate` at intensity 1.0, so the presets cannot drift from the model.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/emotion/test_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/emotion tests/emotion
git commit -m "feat: emotion-to-gait modulation over affective axes with evidence-checked directions"
```

---

## Task 15: CLI and end-to-end demo

**Files:**
- Create: `src/emotive_gait/cli.py`, `tests/test_cli.py`, `README.md`
- Modify: `pyproject.toml` (add `[project.scripts] emotive-gait = "emotive_gait.cli:main"`), `CLAUDE.md`
- Delete: `emotive-gait.py` (its `compute_zmp` now lives in `sim/metrics.py`)

**Interfaces:**
- Consumes: every prior task.
- Produces: `main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py
def test_demo_writes_a_video(tmp_path):
    out = tmp_path / "d.mp4"
    assert main(["demo", "--emotion", "sadness=1.0", "--duration", "0.5",
                 "--width", "320", "--height", "240", "--out", str(out)]) == 0
    assert out.is_file() and out.stat().st_size > 1000

def test_invalid_emotion_name_is_reported_not_traced(capsys):
    assert main(["demo", "--emotion", "ennui=1.0"]) == 2
    assert "ennui" in capsys.readouterr().err

def test_out_of_range_intensity_is_reported_not_traced(capsys):
    assert main(["demo", "--emotion", "anger=5.0"]) == 2
    assert "anger" in capsys.readouterr().err

def test_unsourced_emotions_warn_the_user(capsys):
    main(["demo", "--emotion", "disgust=1.0", "--duration", "0.2", "--dry-run"])
    assert "unsourced" in capsys.readouterr().err.lower()

def test_evidence_subcommand_lists_grounding(capsys):
    assert main(["evidence"]) == 0
    out = capsys.readouterr().out
    assert "Roether" in out and "UNSOURCED" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL, `ImportError`

- [ ] **Step 3: Implement `cli.py`, write the README, update CLAUDE.md**

Subcommands: `demo` (`--emotion NAME=VALUE` repeatable, `--duration`, `--fps`, `--width`, `--height`, `--root {free,z-rail,weld}` defaulting to `z-rail`, `--kinematic/--physics` defaulting to kinematic, `--hold-speed`, `--out`, `--dry-run`) and `evidence` (prints the claim table with grounding). Invalid input returns 2 with a message on stderr, never a traceback. Any unsourced emotion with non-zero intensity prints a warning naming it.

README covers: what the project is, the Python 3.12 requirement and why, quickstart (`pip install -e .`, `python -m mujoco_menagerie prefetch unitree_g1`, `emotive-gait demo`), the architecture in four sentences, an honest statement that the physics tier is in-place on a Z-rail and that forward stride is kinematic-only until Phase 4, the grounded-versus-unsourced emotion split, and the related-work positioning from design section 7.

Rewrite `CLAUDE.md` for the new architecture, deleting the Atlas, LQR and Webots sections entirely.

- [ ] **Step 4: Run the full suite**

Run: `pytest -v && ruff check . && ruff format --check .`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: CLI with demo and evidence subcommands, README, rewritten CLAUDE.md"
```
