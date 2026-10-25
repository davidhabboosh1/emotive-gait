# Phase 4 Forward Locomotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The G1 walks forward with emotion: legs solved by task-space IK against foot and pelvis paths so every `GaitParameters` field is rendered, on MuJoCo 3.17 with `mink`, with a per-trajectory feasibility verdict and a physics tier that reports foot slip and ZMP margin.

**Architecture:** `gait/footsteps.py` turns parameters and phase into pelvis-frame foot targets and a world pelvis path (pure functions). `gait/ik.py` solves those targets on a welded-root copy of the robot with `mink` (two warm-started laps, second kept) and measures feasibility with `mj_forward`. `gait/generator.py` fits the solved leg samples with `PeriodicCurve`, composes the procedural upper body on top, and exposes the root path; the old joint-space legs are deleted. `sim/kinematic.py` adds the world root pose; `sim/physical.py` plays a recorded projection of the parameters on constrained roots and gains the driven `XZ_RAIL`; `sim/metrics.py` gains the ZMP margin and foot slip that feed a `GaitReport`.

**Tech Stack:** Python >= 3.11, `mujoco>=3.17,<3.18`, `mink>=1.3,<2`, `qpsolvers[daqp]>=4,<5`, `mujoco-menagerie==2026.10.4`, NumPy 2, SciPy, pytest, ruff 0.17.2.

**Spec:** `docs/superpowers/specs/2026-10-25-phase-4-forward-locomotion-design.md` (the parent design `2026-10-25-emotive-gait-professionalization-design.md` stays binding where this one is silent).

## Global Constraints

- **Pin shape:** `mujoco>=3.17,<3.18`. Never an open upper bound (spec 4.1).
- **3.17 `MjSpec` substitutions, and only these:** `spec.find_body(n)` -> `spec.body(n)`; `obj.delete()` -> `spec.delete(obj)` for geoms, joints and keys. Everything else in `neck.py`/`root.py` is unchanged (spec 3).
- **Phase 0 numbers are the port's regression gate and are never re-baselined:** total mass 33.3411, torso remainder 6.0155, head 1.8025 (spec 4.1).
- **IK model is welded-root, pelvis frame, treadmill style.** Stance foot moves backward at `speed`; the world pelvis moves forward at `speed`; the stance foot is stationary in the world by construction (spec 4.2).
- **Seed the IK from a bent-knee pose, never the straight-leg `stand` keyframe** (spec 3): hip pitch -0.2, knee 0.4, ankle pitch -0.2, both sides.
- **Nominal pelvis height is a tuned profile constant, G1 = 0.70 m.** `com_height_scale = 1.0` means this height (spec 4.3).
- **Feasibility is a flag, never a clamp or a rescale.** `generate()` never raises on infeasibility and never edits parameters (spec 4.4). Thresholds: position <= 0.002 m, orientation <= 0.01 rad.
- **`expansiveness` scales `arm_swing_amp`, `arm_abduction`, `step_height`. Never `step_length`** (spec 4.3).
- **`knee_flexion_bias` is retired** (spec 4.5). Fear's crouch is `com_height_scale` -0.04 (tuned).
- **Amplitudes are relative, postures are absolute** (spec 4.5): shoulder-pitch swing is centred on the profile rest pose; `elbow_flexion`, `arm_abduction`, `trunk_pitch`, `neck_pitch`, `com_height_scale` are absolute targets.
- **Phase convention:** `phi = 0` is the start of the left swing; right swing starts at `phi = 0.5`. `phase = (t * cadence / 2) % 1`.
- **Sway goes to the stance side:** pelvis `y = -lateral_sway_amp * sin(2*pi*phi)` (G1's +y is the robot's left; the `left_foot` site is at y = +0.1185).
- **Constrained roots play a recorded projection** (spec 4.6 table): Z_RAIL zeroes `step_length` and `lateral_sway_amp`; XZ_RAIL zeroes `lateral_sway_amp`; WELD zeroes `step_length`, `lateral_sway_amp`, `com_bounce_amp` and forces `com_height_scale = 1.0`; FREE plays everything.
- **ZMP is instrumentation.** `zmp_margin` is reported and is never a fall criterion (spec 4.6).
- **No feedback control, no free-base walking, no heel-toe roll** (spec 2).
- **Repository conventions hold:** load `scene`, never `g1`; roll joints mirrored; positive `neck_pitch` is gaze down; no magnitudes in degrees; emotion on parameters only; CLI exit codes 2 (user) and 1 (environment), no traceback escapes `main`.
- **Commit messages carry no Claude attribution trailer** (user convention). Merging and pushing are the user's decisions.

## Review Focus

Input classes the spec implies but no happy-path test exercises. Each has its test pinned to the owning task.

1. **`step_length = 0` (in-place) must still lift the feet and swing the arms** — sadness pinned to zero speed, and every constrained-root projection, hits this path; a divide-by-travel in the arm oscillator or foot easing would produce NaN. Owned by Tasks 2 and 5.
2. **A parameter extreme that is infeasible must be flagged *and* still inside joint range** — the user will push sliders past the envelope; the clamp after offsets must hold even when the QP residual is large. Owned by Tasks 4 and 5.
3. **A YAML written before this phase carries `knee_flexion_bias`** — it must fail with the existing "unknown gait parameter" message, not a traceback. Owned by Task 7.
4. **The Z-rail starts at the straight-leg keyframe height (0.79) and the trajectory wants 0.70** — the ramp lowers the pelvis 9 cm; the bob bound must be measured after the ramp or every preset fails for the wrong reason. Owned by Task 5.
5. **Kinematic playback on a non-FREE root** — the root pose write would land on a slide joint; it must raise a clear `ValueError`, not corrupt `qpos`. Owned by Task 6.

## Execution notes

Suggested model per task, implementer / reviewer (user preference: cheap for transcription, standard for integration, most capable for load-bearing review and the final whole-branch review): T1 sonnet / fable; T2 sonnet / sonnet; T3 sonnet / opus; T4 opus / fable; T5 opus / fable; T6 sonnet / opus; T7 sonnet / fable (evidence claims are load-bearing); T8 sonnet / opus; T9 sonnet / opus; T10 haiku / sonnet; whole-branch review fable. Branch: `phase-4-forward-locomotion` from `master`. After Task 1 the user is asked whether the port merges on its own (spec 4.1).

Every task ends with `ruff check . && ruff format --check . && pytest -q` green; the step lists below name only the task's own tests, and that gate is implied.

Task order differs from the spec's section 5 in one place: the constrained-root projection (Task 3) lands before the IK generator (Task 5), because the Z-rail gate in Task 5 needs the in-place projection to exist or every stride would drag the stance foot on the rail. Drop order if the schedule slips (spec 5): Task 8 and Task 9 first, then the tracking camera half of Task 6; Tasks 1 to 5 and 7 are never dropped.

---

### Task 1: Port `robot/neck.py` and `robot/root.py` to MuJoCo 3.17 and raise the pin

**Files:**
- Modify: `src/emotive_gait/robot/neck.py` (`inject_neck`: `spec.find_body` at the top, `src.delete()` in the geom loop, `key.delete()` in the keyframe loop; `build_with_neck`: `key.delete()` loop)
- Modify: `src/emotive_gait/robot/root.py:36-39` (`spec.find_body`, `joint.delete()`)
- Modify: `pyproject.toml` (`mujoco>=3.17,<3.18`; rewrite the comment above it)
- Modify: `CLAUDE.md:50` (the pin paragraph)
- Test: `tests/robot/test_neck.py` (unchanged), `tests/robot/test_root_api.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: the same `inject_neck`, `build_with_neck`, `standing_qpos`, `to_root_mode` signatures, running on 3.17.

- [ ] **Step 1: Install 3.17 into the project venv and run the neck tests to see them fail on import**

Run: `.venv\Scripts\pip install "mujoco>=3.17,<3.18" && pytest tests/robot/test_neck.py -q`
Expected: FAIL with `AttributeError: 'MjSpec' object has no attribute 'find_body'`

- [ ] **Step 2: Write the failing API-level test**

```python
# tests/robot/test_root_api.py
import mujoco
from emotive_gait.robot.root import RootMode, to_root_mode

def test_root_modes_build_on_the_installed_mujoco():
    spec = mujoco.MjSpec()
    pelvis = spec.worldbody.add_body(name="pelvis")
    pelvis.add_freejoint()
    to_root_mode(spec, RootMode.Z_RAIL)
    model = spec.compile()
    assert model.njnt == 1 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_SLIDE

def test_weld_removes_every_root_joint():
    spec = mujoco.MjSpec()
    pelvis = spec.worldbody.add_body(name="pelvis")
    pelvis.add_freejoint()
    to_root_mode(spec, RootMode.WELD)
    assert spec.compile().njnt == 0
```

- [ ] **Step 3: Apply the three substitutions in `neck.py` and `root.py`**

`spec.body(name)` returns `None` for a missing name on 3.17 exactly as `find_body` did; keep the `KeyError` branches.

- [ ] **Step 4: Run the robot suite**

Run: `pytest tests/robot -q`
Expected: PASS, including `test_total_mass_is_conserved`, `test_whole_body_com_is_conserved`, `test_torso_remainder_inertia_is_positive` with no edits to their numbers.

- [ ] **Step 5: Run the whole suite and record any physics drift**

Run: `pytest -q`
Expected: 198 passed, 1 xfailed. If `test_z_rail_survives_six_seconds_on_every_preset` moves outside its bob or contact bounds, STOP and report the measured numbers; do not loosen the bounds (spec 4.1).

- [ ] **Step 6: Update `pyproject.toml` and the `CLAUDE.md` pin paragraph**

The `CLAUDE.md` paragraph now says: pinned to the 3.17 line, which `MjSpec` substitutions were made, and that raising the upper bound is a deliberate re-verify with `tests/robot/test_neck.py` as the mass/inertia gate.

- [ ] **Step 7: Commit, then stop for the user**

```bash
git add src/emotive_gait/robot/neck.py src/emotive_gait/robot/root.py pyproject.toml CLAUDE.md tests/robot/test_root_api.py
git commit -m "port: MjSpec body()/delete() for MuJoCo 3.17; pin mujoco>=3.17,<3.18"
```

Controller: push is the user's call; present "push for CI and merge the port alone" versus "continue on the branch" before Task 2 starts.

---

### Task 2: `gait/footsteps.py` — foot targets and pelvis path as pure functions

**Files:**
- Create: `src/emotive_gait/gait/footsteps.py`
- Test: `tests/gait/test_footsteps.py`

**Interfaces:**
- Consumes: `GaitParameters` (`gait/params.py`).
- Produces:
  - `class Side(str, Enum): LEFT = "left"; RIGHT = "right"`
  - `@dataclass(frozen=True) class FootPoint: x: float; z: float; in_stance: bool` — `x` in the pelvis frame (m, +forward), `z` clearance above the floor-contact height (m, 0 in stance).
  - `def swing_fraction(params: GaitParameters) -> float` = `(1 - params.double_support_frac) / 2`
  - `def foot_travel(params: GaitParameters) -> float` = `params.step_length * (1 + params.double_support_frac)`
  - `def foot_point(params: GaitParameters, phase: float, side: Side) -> FootPoint`
  - `def pelvis_offset(params: GaitParameters, phase: float) -> tuple[float, float]` -> `(dy, dz)` with `dy = -params.lateral_sway_amp * sin(2*pi*phase)`, `dz = params.com_bounce_amp * cos(4*pi*phase)`
  - `def pelvis_x(params: GaitParameters, t: float) -> float` = `params.speed * t`
  - `def phase_at(params: GaitParameters, t: float) -> float` = `(t * params.cadence / 2) % 1`
  - Constants `SWING_X_EASING` (quintic smoothstep `6u^5 - 15u^4 + 10u^3`) and clearance profile `sin(pi*u)**3` (both C2 at `u = 0` and `u = 1`). Imports nothing from `robot/`.

Timing per side: left swing is `phase in [0, sf)`, left stance `[sf, 1)`; the right foot uses `(phase + 0.5) % 1`. Stance local `u = (p - sf) / (1 - sf)`, `x = D/2 - D*u`. Swing local `u = p / sf`, `x = -D/2 + 2*step_length*S(u) - step_length*(1 - double_support_frac)*u` (world-frame smoothstep displacement of `2*step_length` minus the pelvis travel during the swing; the end slope then equals the stance slope `-2*step_length` per unit phase, so x is C2 at both handoffs — ruling R5, which corrected an earlier `x = -D/2 + D*S(u)` whose zero end slope left a velocity jump), `z = step_height * sin(pi*u)**3`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/gait/test_footsteps.py
import math
import numpy as np
import pytest
from emotive_gait.gait.footsteps import (Side, foot_point, foot_travel, pelvis_offset, pelvis_x, phase_at, swing_fraction)
from emotive_gait.gait.params import GaitParameters

P = GaitParameters()  # step_length 0.35, cadence 1.8, double_support_frac 0.20

def test_left_swings_first_and_right_half_a_cycle_later():
    sf = swing_fraction(P)
    assert not foot_point(P, 0.1 * sf, Side.LEFT).in_stance
    assert foot_point(P, 0.1 * sf, Side.RIGHT).in_stance
    assert not foot_point(P, 0.5 + 0.1 * sf, Side.RIGHT).in_stance

def test_both_feet_are_down_through_double_support():
    sf = swing_fraction(P)
    for ph in np.linspace(sf, 0.5, 20, endpoint=False):
        assert foot_point(P, ph, Side.LEFT).in_stance and foot_point(P, ph, Side.RIGHT).in_stance

def test_stance_foot_moves_back_at_walking_speed():
    sf = swing_fraction(P)
    dt = 1e-4
    t0 = 0.6 / (P.cadence / 2)  # a time inside left stance
    x0 = foot_point(P, phase_at(P, t0), Side.LEFT).x
    x1 = foot_point(P, phase_at(P, t0 + dt), Side.LEFT).x
    assert (x1 - x0) / dt == pytest.approx(-P.speed, rel=1e-3)

def test_foot_x_range_is_half_the_travel_each_way():
    xs = [foot_point(P, ph, Side.LEFT).x for ph in np.linspace(0, 1, 2001)]
    assert max(xs) == pytest.approx(foot_travel(P) / 2, abs=1e-6)
    assert min(xs) == pytest.approx(-foot_travel(P) / 2, abs=1e-6)

def test_swing_apex_is_step_height_and_stance_is_on_the_floor():
    sf = swing_fraction(P)
    assert foot_point(P, sf / 2, Side.LEFT).z == pytest.approx(P.step_height)
    assert all(foot_point(P, ph, Side.LEFT).z == 0.0 for ph in np.linspace(sf, 1, 50, endpoint=False))

@pytest.mark.parametrize("ph", [0.0, swing_fraction(P), 0.5, 0.5 + swing_fraction(P)])
def test_paths_are_c2_at_every_handoff(ph):
    h = 1e-3
    for side in Side:
        f = lambda p: foot_point(P, p % 1.0, side)
        for attr in ("x", "z"):
            g = lambda p: getattr(f(p), attr)
            left2 = (g(ph - h) - 2 * g(ph - 2 * h) + g(ph - 3 * h)) / h**2
            right2 = (g(ph + 3 * h) - 2 * g(ph + 2 * h) + g(ph + h)) / h**2
            assert g(ph - h) == pytest.approx(g(ph + h), abs=5e-3)
            assert left2 == pytest.approx(right2, abs=0.5)  # second derivative continuous

def test_in_place_when_step_length_is_zero():
    q = GaitParameters(step_length=0.0)
    pts = [foot_point(q, ph, Side.LEFT) for ph in np.linspace(0, 1, 100)]
    assert all(p.x == 0.0 and math.isfinite(p.z) for p in pts)
    assert max(p.z for p in pts) == pytest.approx(q.step_height)

def test_sway_goes_to_the_stance_side_and_bounce_has_two_periods():
    sf = swing_fraction(P)
    dy, _ = pelvis_offset(P, sf / 2)  # left swinging, right in stance: pelvis toward -y
    assert dy < 0
    for ph in (0.1, 0.3):
        assert pelvis_offset(P, ph)[1] == pytest.approx(pelvis_offset(P, ph + 0.5)[1])

def test_pelvis_x_is_speed_times_time():
    assert pelvis_x(P, 2.0) == pytest.approx(2.0 * P.speed)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/gait/test_footsteps.py -q`
Expected: FAIL with `ModuleNotFoundError: emotive_gait.gait.footsteps`

- [ ] **Step 3: Implement `footsteps.py` per the Interfaces block**

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/gait/test_footsteps.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/gait/footsteps.py tests/gait/test_footsteps.py
git commit -m "feat(gait): pelvis-frame foot targets and world pelvis path from GaitParameters"
```

---

### Task 3: Profile constants, root modes at the nominal height, projection in `simulate`

**Files:**
- Modify: `src/emotive_gait/robot/roles.py` (`RobotProfile` gains three fields)
- Modify: `src/emotive_gait/robot/profiles/g1.py`
- Modify: `src/emotive_gait/robot/root.py` (`to_root_mode` signature)
- Modify: `src/emotive_gait/robot/neck.py:` `build_with_neck` -> generalised builder; `src/emotive_gait/robot/loader.py` (`load_robot` lifts the "non-FREE root needs the neck" rule)
- Modify: `src/emotive_gait/sim/physical.py` (`SimResult.played_params`, projection, WELD initial pose)
- Test: `tests/robot/test_loader.py`, `tests/sim/test_playback.py`

**Interfaces:**
- Consumes: `RootMode`, `GaitParameters`.
- Produces:
  - `RobotProfile` fields: `nominal_pelvis_height: float = 0.70`, `foot_site_floor_height: float = 0.0331`, `foot_sites: tuple[str, str] = ("left_foot", "right_foot")` (left, right; a plain tuple because `robot/` imports nothing from `gait/`, so `Side` cannot appear here; `ik.py` indexes it with `0` for `Side.LEFT`, `1` for `Side.RIGHT`). `g1.py` sets all three on both profiles.
  - `def to_root_mode(spec, mode, *, pelvis_z: float | None = None) -> mujoco.MjSpec` — when `pelvis_z` is given and the mode is `WELD`, set `pelvis.pos[2] = pelvis_z` (Z_RAIL keeps the keyframe height: the rail is free and the ramp lowers it).
  - `def build_model(entry_xml: str, *, neck: NeckConfig | None, root: RootMode, pelvis_z: float | None, offwidth: int = 960, offheight: int = 720) -> mujoco.MjModel` in `neck.py`; `build_with_neck(...)` becomes a thin wrapper kept for callers.
  - `load_robot(profile=None, *, with_neck=True, root=RootMode.FREE)` works for every `(with_neck, root)` pair; on `WELD` it passes `pelvis_z=profile.nominal_pelvis_height`.
  - `def project_for_root(params: GaitParameters, mode: RootMode) -> GaitParameters` in `physical.py`, implementing the Global Constraints projection table (XZ_RAIL row added in Task 8).
  - `SimResult.played_params: GaitParameters` (the projected parameters actually simulated).
  - `simulate(robot, traj, *, duration, ramp=0.5)` now calls `generate(project_for_root(traj.params, mode), robot, offsets=..., overrides=...)` internally when the projection differs from `traj.params`, so callers keep passing the emotion's trajectory. On `WELD`, the initial `qpos` is the trajectory's phase-0 pose (feet on the floor by construction) and the ramp blends from that pose.

- [ ] **Step 1: Write the failing tests**

```python
# tests/robot/test_loader.py (append; add `import mujoco`, `import pytest`, `from emotive_gait.robot.root import RootMode`)
def test_profile_carries_the_nominal_height_and_foot_sites():
    r = load_robot()
    assert r.profile.nominal_pelvis_height == 0.70
    assert r.profile.foot_sites == ("left_foot", "right_foot")

def test_foot_site_floor_height_matches_the_stand_pose():
    r = load_robot()
    d = mujoco.MjData(r.model); d.qpos[:] = r.stand_qpos; mujoco.mj_forward(r.model, d)
    site = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    assert d.site_xpos[site][2] == pytest.approx(r.profile.foot_site_floor_height, abs=1e-4)

def test_weld_puts_the_pelvis_at_the_nominal_height():
    r = load_robot(root=RootMode.WELD)
    d = mujoco.MjData(r.model); mujoco.mj_forward(r.model, d)
    pelvis = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    assert d.xpos[pelvis][2] == pytest.approx(0.70)

def test_every_root_mode_builds_without_a_neck():
    for mode in RootMode:
        assert load_robot(with_neck=False, root=mode).model.nq > 0
```

```python
# tests/sim/test_playback.py (delete test_non_free_root_requires_the_neck_path; append;
# add `from emotive_gait.sim.physical import project_for_root`)
def test_z_rail_plays_an_in_place_projection():
    r = load_robot(root=RootMode.Z_RAIL)
    res = simulate(r, generate(GaitParameters(step_length=0.3, lateral_sway_amp=0.02), r), duration=0.2)
    assert res.played_params.step_length == 0.0 and res.played_params.lateral_sway_amp == 0.0
    assert res.played_params.com_bounce_amp == GaitParameters().com_bounce_amp

def test_weld_projection_freezes_everything_the_pelvis_would_move_for():
    p = project_for_root(GaitParameters(step_length=0.3, com_bounce_amp=0.03, com_height_scale=0.9, lateral_sway_amp=0.02), RootMode.WELD)
    assert (p.step_length, p.com_bounce_amp, p.lateral_sway_amp, p.com_height_scale) == (0.0, 0.0, 0.0, 1.0)

def test_free_root_is_not_projected():
    p = GaitParameters(step_length=0.3)
    assert project_for_root(p, RootMode.FREE) == p
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/robot/test_loader.py tests/sim/test_playback.py -q -k "nominal or floor_height or weld or without_a_neck or projection or not_projected"`
Expected: FAIL (`AttributeError` on the profile fields; `ValueError: non-FREE root modes require with_neck`)

- [ ] **Step 3: Implement the profile fields, `to_root_mode(pelvis_z=)`, `build_model`, the loader change, `project_for_root`, `SimResult.played_params`, and the WELD initial pose**

`load_robot` with `with_neck=False` and a non-FREE root builds from the spec via `build_model(entry_xml, neck=None, ...)` and uses `standing_qpos(model, new_model)` exactly like the neck path.

- [ ] **Step 4: Run the robot and sim suites**

Run: `pytest tests/robot tests/sim -q`
Expected: PASS (the old in-place generator ignores `step_length`, so the projection is a no-op for its legs; the survival tests are unchanged here).

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/robot tests/robot src/emotive_gait/sim/physical.py tests/sim/test_playback.py
git commit -m "feat(robot,sim): nominal pelvis height, root modes for every profile, projected parameters on constrained roots"
```

---

### Task 4: `gait/ik.py` — mink leg solver and feasibility report

**Files:**
- Modify: `pyproject.toml` (add `"mink>=1.3,<2"`, `"qpsolvers[daqp]>=4,<5"`)
- Create: `src/emotive_gait/gait/ik.py`
- Test: `tests/gait/test_ik.py`

**Interfaces:**
- Consumes: `footsteps.foot_point/pelvis_offset/Side`, `RobotModel` (welded), `RobotProfile.nominal_pelvis_height/foot_site_floor_height/foot_sites`, `JointRole`.
- Produces:
  - Constants: `QP_SOLVER = "daqp"`, `N_SAMPLES = 120`, `ITERATIONS = 8`, `IK_DT = 0.05`, `POSITION_TOL_M = 0.002`, `ORIENTATION_TOL_RAD = 0.01`, `LIMIT_MARGIN_RAD = 1e-3`, `SEED_LEG_POSE: Mapping[JointRole, float]` (hip pitch -0.2, knee 0.4, ankle pitch -0.2, both sides), `LEG_ROLES: tuple[JointRole, ...]` (the twelve hip/knee/ankle roles).
  - `@dataclass(frozen=True) class FeasibilityReport: max_foot_position_error_m: float; max_foot_orientation_error_rad: float; roles_at_limit: tuple[JointRole, ...]; worst_phase: float` with `@property feasible -> bool` (both tolerances met).
  - `@dataclass(frozen=True) class LegCycle: phases: np.ndarray` (`(n,)`, `k/n`), `angles: Mapping[JointRole, np.ndarray]` (`(n,)` per leg role, read-only arrays), `report: FeasibilityReport`.
  - `def weld_robot_for(profile: RobotProfile) -> RobotModel` — `functools.cache` over `profile.name`, returns `load_robot(profile, with_neck=profile.has(JointRole.NECK_PITCH), root=RootMode.WELD)`.
  - `def solve_leg_cycle(params: GaitParameters, robot: RobotModel, *, n_samples: int = N_SAMPLES, iterations: int = ITERATIONS) -> LegCycle` — raises `ValueError` unless `robot` is WELD-rooted.

Algorithm (what the signature does not determine): build `mink.Configuration(robot.model)` from `robot.stand_qpos` with `SEED_LEG_POSE` applied by role; one `mink.FrameTask(site, "site", position_cost=10.0, orientation_cost=5.0)` per foot with the target rotation taken from that site at the seed pose (level sole), one `mink.PostureTask(model, cost=1e-3)` toward the seed, `limits=[mink.ConfigurationLimit(model)]`. Per sample `k`, phase `k/n`: world target = pelvis world position + `(foot.x, ±feet_spacing/2 - dy, foot_site_floor_height - (nominal*com_height_scale + dz) + foot.z)`. Iterate `iterations` times `v = mink.solve_ik(cfg, tasks, IK_DT, QP_SOLVER, damping=1e-4, limits=limits); cfg.integrate_inplace(v, IK_DT)`. Run two laps over all samples with `cfg` carried across; keep the second lap's `q` per sample. Then, per kept sample, `mj_forward` on a scratch `MjData` and measure site position/orientation error against the target (the report is measured, not read from the solver), and record roles within `LIMIT_MARGIN_RAD` of a range limit.

- [ ] **Step 1: Add the dependencies and install**

Run: `.venv\Scripts\pip install -e ".[dev]" && .venv\Scripts\python -c "import mink, qpsolvers; assert 'daqp' in qpsolvers.available_solvers"`
Expected: no output, exit 0

- [ ] **Step 2: Write the failing tests**

```python
# tests/gait/test_ik.py
import numpy as np
import pytest
from emotive_gait.gait.footsteps import Side, foot_point, foot_travel
from emotive_gait.gait.ik import LEG_ROLES, solve_leg_cycle, weld_robot_for
from emotive_gait.gait.params import GaitParameters
from emotive_gait.robot.loader import load_robot
from emotive_gait.robot.profiles.g1 import G1_PROFILE_WITH_NECK
from emotive_gait.robot.roles import JointRole
from emotive_gait.robot.root import RootMode

@pytest.fixture(scope="module")
def weld():
    return weld_robot_for(G1_PROFILE_WITH_NECK)

def test_requires_a_welded_robot():
    with pytest.raises(ValueError, match="WELD"):
        solve_leg_cycle(GaitParameters(), load_robot())

def test_neutral_cycle_is_feasible_with_negligible_residual(weld):
    c = solve_leg_cycle(GaitParameters(), weld)
    assert c.report.feasible
    assert c.report.max_foot_position_error_m < 1e-4
    assert set(c.angles) == set(LEG_ROLES) and c.phases.shape == (120,)

def test_absurd_stride_is_flagged_at_the_ankle(weld):
    c = solve_leg_cycle(GaitParameters(step_length=0.75), weld)  # 0.9 m separation
    assert not c.report.feasible
    assert c.report.max_foot_position_error_m > 0.01
    assert {JointRole.ANKLE_PITCH_L, JointRole.ANKLE_PITCH_R} & set(c.report.roles_at_limit)

def test_solve_is_deterministic(weld):
    a, b = (solve_leg_cycle(GaitParameters(), weld) for _ in range(2))
    assert all(np.array_equal(a.angles[r], b.angles[r]) for r in LEG_ROLES)

def test_wrap_step_is_no_larger_than_the_largest_intra_cycle_step(weld):
    c = solve_leg_cycle(GaitParameters(), weld)
    for r in LEG_ROLES:
        q = c.angles[r]
        assert abs(q[0] - q[-1]) <= np.abs(np.diff(q)).max() + 1e-9

def test_left_foot_starts_its_swing_half_a_travel_behind_the_pelvis(weld):
    import mujoco
    c = solve_leg_cycle(GaitParameters(), weld)
    d = mujoco.MjData(weld.model); d.qpos[:] = weld.stand_qpos
    for r in LEG_ROLES:
        d.qpos[weld.qpos_adr(r)] = c.angles[r][0]
    mujoco.mj_forward(weld.model, d)
    site = mujoco.mj_name2id(weld.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    pelvis = mujoco.mj_name2id(weld.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    assert d.site_xpos[site][0] - d.xpos[pelvis][0] == pytest.approx(-foot_travel(GaitParameters()) / 2, abs=2e-3)

def test_legs_are_half_a_cycle_apart(weld):
    c = solve_leg_cycle(GaitParameters(), weld)
    n = len(c.phases)
    assert np.allclose(c.angles[JointRole.HIP_PITCH_L], np.roll(c.angles[JointRole.HIP_PITCH_R], n // 2), atol=2e-3)

def test_in_place_cycle_lifts_the_feet_without_travel(weld):
    c = solve_leg_cycle(GaitParameters(step_length=0.0), weld)
    assert c.report.feasible
    assert np.ptp(c.angles[JointRole.KNEE_L]) > 0.2
```

- [ ] **Step 3: Run to verify they fail**

Run: `pytest tests/gait/test_ik.py -q`
Expected: FAIL with `ModuleNotFoundError: emotive_gait.gait.ik`

- [ ] **Step 4: Implement `ik.py` per the Interfaces block and algorithm**

- [ ] **Step 5: Run to verify they pass**

Run: `pytest tests/gait/test_ik.py -q`
Expected: PASS. If `test_legs_are_half_a_cycle_apart` fails only by the `atol`, the hip-roll/yaw symmetry of the seed is the first thing to check, not the tolerance.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/emotive_gait/gait/ik.py tests/gait/test_ik.py
git commit -m "feat(gait): mink leg solver on a welded root with a measured feasibility report"
```

---

### Task 5: `generate()` composes IK legs with the procedural upper body; old legs deleted; Z-rail gate

**Files:**
- Modify: `src/emotive_gait/gait/generator.py` (rewrite `_procedural`, `generate`, `GaitTrajectory`; delete `_BASE_KNEE_AMPLITUDE`, `_HIP_PITCH_OFFSET`, `_KNEE_OFFSET`, `_HIP_SWING_GAIN`, `_ANKLE_PITCH_OFFSET`, `_ANKLE_SWING_GAIN`, `_LEG_SIDES`)
- Modify: `tests/gait/test_generator.py`, `tests/sim/test_playback.py`
- Test: `tests/gait/test_generator.py`

**Interfaces:**
- Consumes: `ik.solve_leg_cycle`, `ik.weld_robot_for`, `ik.FeasibilityReport`, `footsteps.pelvis_offset/pelvis_x/phase_at`, `PeriodicCurve.from_knots`.
- Produces:
  - `GaitTrajectory` gains `feasibility: FeasibilityReport` (public, part of equality) and `_legs: tuple[tuple[JointRole, PeriodicCurve], ...]`; keeps `roles`, `params`, `_ranges`, `_stand`, `_signs`, `_offsets`, `_overrides`, `angles_at`, `qpos_at`, `ctrl_at`.
  - `GaitTrajectory.root_offset_at(phase: float) -> np.ndarray` — `(0.0, dy, nominal*com_height_scale + dz)` from `pelvis_offset`; the nominal height is captured at `generate()` time as `_nominal_height: float`.
  - `GaitTrajectory.root_x_at(t: float) -> float` = `pelvis_x(params, t)`.
  - `def generate(params, robot, *, offsets=None, overrides=None) -> GaitTrajectory` — solves `solve_leg_cycle(replace(params, step_height=params.expansiveness * params.step_height), weld_robot_for(robot.profile))`, fits each leg role with `PeriodicCurve.from_knots(cycle.phases, cycle.angles[role])`, and composes the upper body.
  - Upper body: `arm = stand[arm] + arm_swing_amp * expansiveness * s_opp(phase)` where `s_opp` is the opposite leg's hip-pitch curve, mean-removed and normalised by its peak deviation (zero if the peak is below 1e-9); elbow, rolls, waist and neck as today (roll amplitude scaled by expansiveness as today).

- [ ] **Step 1: Update the generator tests**

Replace in `tests/gait/test_generator.py`:
- `test_cycle_is_continuous_at_the_stance_to_swing_handoff`: phases `(sf, 0.5 + sf, 0.0, 0.5)` with `sf = swing_fraction(GaitParameters())`.
- `test_expansiveness_scales_procedural_amplitude_only` and `test_expansiveness_scales_elbow_swing_too`: unchanged assertions; add `assert generate(GaitParameters(expansiveness=1.5), r).params.step_length == GaitParameters().step_length`.
- In `test_extreme_parameters_stay_inside_joint_limits`: replace `GaitParameters(knee_flexion_bias=5.0)` with `GaitParameters(step_length=0.75)` (infeasible, still clamped).

Add:

```python
def test_every_preset_is_feasible():
    from emotive_gait.emotion.evidence import Emotion
    from emotive_gait.emotion.model import PRESET_PARAMS, modulate
    from emotive_gait.robot.profiles.g1 import G1_PROFILE_WITH_NECK
    r = load_robot()
    cases = dict(PRESET_PARAMS)
    cases |= {e.value: modulate(GaitParameters(), {e: 1.0}, G1_PROFILE_WITH_NECK) for e in Emotion}
    for name, p in cases.items():
        rep = generate(p, r).feasibility
        assert rep.feasible, f"{name}: {rep}"

def test_infeasible_stride_is_flagged_not_rescaled():
    r = load_robot()
    t = generate(GaitParameters(step_length=0.75), r)
    assert not t.feasibility.feasible and t.params.step_length == 0.75

def test_arms_follow_the_opposite_hip():
    t = generate(GaitParameters(), load_robot())
    ph = np.linspace(0, 1, 64, endpoint=False)
    hip_r = np.array([t.angles_at(p)[JointRole.HIP_PITCH_R] for p in ph]); hip_r -= hip_r.mean()
    arm_l = np.array([t.angles_at(p)[JointRole.SHOULDER_PITCH_L] for p in ph]); arm_l -= arm_l.mean()
    assert np.corrcoef(hip_r, arm_l)[0, 1] > 0.99
    assert np.abs(arm_l).max() == pytest.approx(GaitParameters().arm_swing_amp, rel=1e-3)

def test_in_place_gait_still_swings_the_arms():
    t = generate(GaitParameters(step_length=0.0), load_robot())
    v = [t.angles_at(float(p))[JointRole.SHOULDER_PITCH_L] for p in np.linspace(0, 1, 32, endpoint=False)]
    assert max(v) - min(v) > 0.1 and all(math.isfinite(x) for x in v)

def test_root_path_carries_crouch_bounce_and_forward_travel():
    r = load_robot()
    t = generate(GaitParameters(com_height_scale=0.9, com_bounce_amp=0.02), r)
    assert t.root_offset_at(0.0)[2] == pytest.approx(0.70 * 0.9 + 0.02)
    assert t.root_x_at(2.0) == pytest.approx(2.0 * t.params.speed)
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `pytest tests/gait/test_generator.py -q`
Expected: FAIL (`feasibility`, `root_offset_at` missing; handoff continuity fails at the new phases)

- [ ] **Step 3: Rewrite `generator.py` per the Interfaces block**

- [ ] **Step 4: Run the generator suite**

Run: `pytest tests/gait -q`
Expected: PASS

- [ ] **Step 5: The Z-rail gate — recalibrate the survival test after the ramp, then run it**

In `tests/sim/test_playback.py`: `test_z_rail_survives_six_seconds_on_every_preset` measures `bob` and `no_contact` on `res.pelvis_z[post:]` / `res.cop[post:]` with `post = math.ceil(0.5 / r.model.opt.timestep)` (Review Focus 4); bounds stay `bob < 0.06`, `no_contact < 0.25`. `test_z_rail_detects_collapse` uses `GaitParameters(cadence=6, step_height=0.5, trunk_pitch=3, expansiveness=2)`; keep its assertions. The `_PHASE0_PRESETS` literals keep their `knee_flexion_bias` keys for now (inert; removed in Task 7).

Run: `pytest tests/sim/test_playback.py -q`
Expected: PASS. If a preset fails the bob or contact bound, tune in this order and record each value tried in the commit message: `GaitParameters.step_height` default, `double_support_frac` default, `nominal_pelvis_height`. If none of the three passes it, STOP and report (spec 4.7); do not resurrect the old legs.

- [ ] **Step 6: Run the whole suite**

Run: `pytest -q`
Expected: all pass, 1 xfailed (the crouch xfail stays strict until Task 7).

- [ ] **Step 7: Commit**

```bash
git add src/emotive_gait/gait/generator.py tests/gait/test_generator.py tests/sim/test_playback.py
git commit -m "feat(gait): legs from task-space IK, upper body procedural; old joint-space legs removed"
```

---

### Task 6: Kinematic tier walks forward; tracking camera; CLI feasibility warning

**Files:**
- Modify: `src/emotive_gait/sim/kinematic.py`
- Modify: `src/emotive_gait/sim/render.py` (`CameraSpec.tracking`, `make_camera`)
- Modify: `src/emotive_gait/cli.py` (`--camera`, feasibility warning)
- Test: `tests/sim/test_playback.py`, `tests/sim/test_render.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `GaitTrajectory.root_offset_at/root_x_at/qpos_at/feasibility`, `footsteps.phase_at`.
- Produces:
  - `play_kinematic(robot, traj, *, duration, fps)` — requires a FREE root (else `ValueError` mentioning "FREE"); per frame `t = k / fps`, `phase = phase_at(traj.params, t)`, `qpos = traj.qpos_at(robot, phase)`, then `qpos[0:3] = (traj.root_x_at(t), *traj.root_offset_at(phase)[1:])`, `qpos[3:7] = (1, 0, 0, 0)`.
  - `CameraSpec(distance=3.2, azimuth=135.0, elevation=-8.0, lookat_z=0.9, tracking: bool = True)`.
  - `def make_camera(model: mujoco.MjModel, spec: CameraSpec) -> mujoco.MjvCamera` — `mjCAMERA_TRACKING` with `trackbodyid` = pelvis when `tracking`, else the free camera as today; `render_frames` uses it.
  - CLI: `--camera {tracking,fixed}` (default `tracking`); after the first `generate` call in `_demo` (physics or video, whichever runs first), if `not traj.feasibility.feasible`, one stderr line `warning: parameters exceed the robot's reach: max foot error X mm; roles at limit: ...` and the run continues.

- [ ] **Step 1: Write the failing tests**

```python
# tests/sim/test_playback.py (append; add `import mujoco`,
# `from emotive_gait.gait.footsteps import Side, foot_point, phase_at`)
def test_kinematic_stance_foot_is_stationary_in_the_world():
    r = load_robot(); p = GaitParameters(); t = generate(p, r)
    site = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    fps = 200; xs = []
    for k, f in enumerate(play_kinematic(r, t, duration=2 / p.cadence, fps=fps)):
        if foot_point(p, phase_at(p, k / fps), Side.LEFT).in_stance:
            xs.append(f.site_xpos[site][:2].copy())
    xs = np.array(xs)
    assert len(xs) > 50 and np.ptp(xs, axis=0).max() < 1e-3

def test_kinematic_pelvis_advances_two_step_lengths_per_cycle():
    r = load_robot(); p = GaitParameters(cadence=1.5); t = generate(p, r)
    # 2 / cadence s at 30 fps is exactly 40 frames; the yielded MjData is reused, so read inside the loop.
    xs = [float(f.qpos[0]) for f in play_kinematic(r, t, duration=2 / p.cadence, fps=30)]
    assert len(xs) == 40
    assert xs[-1] - xs[0] == pytest.approx(2 * p.step_length * (39 / 40), abs=1e-6)

def test_kinematic_crouch_lowers_the_world_pelvis():
    r = load_robot()
    z = lambda s: next(iter(play_kinematic(r, generate(GaitParameters(com_height_scale=s), r), duration=0.1, fps=10))).qpos[2]
    assert z(1.0) - z(0.9) == pytest.approx(0.07, abs=1e-6)

def test_kinematic_playback_rejects_a_constrained_root():
    r = load_robot(root=RootMode.Z_RAIL)
    with pytest.raises(ValueError, match="FREE"):
        next(iter(play_kinematic(r, generate(GaitParameters(), r), duration=0.1, fps=10)))
```

```python
# tests/sim/test_render.py (append, no GL needed; add `import mujoco`,
# `from emotive_gait.sim.render import CameraSpec, make_camera`)
def test_tracking_camera_follows_the_pelvis():
    r = load_robot()
    cam = make_camera(r.model, CameraSpec())
    assert cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING
    assert cam.trackbodyid == mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

def test_fixed_camera_is_free():
    cam = make_camera(load_robot().model, CameraSpec(tracking=False))
    assert cam.type == mujoco.mjtCamera.mjCAMERA_FREE
```

```python
# tests/test_cli.py (append)
def test_infeasible_parameters_warn_but_still_render(tmp_path, capsys, monkeypatch):
    # --dry-run loads no model, so drive the warning through the physics path with no video
    code = main(["demo", "--physics", "--no-video", "--root", "free", "--duration", "0.2", "--hold-speed", "3.0"])
    err = capsys.readouterr().err
    assert code == 0 and "exceed the robot's reach" in err
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/sim/test_playback.py tests/sim/test_render.py tests/test_cli.py -q -k "kinematic or camera or infeasible"`
Expected: FAIL

- [ ] **Step 3: Implement `play_kinematic`, `make_camera`, `CameraSpec.tracking`, the CLI flag and warning**

- [ ] **Step 4: Run the sim and CLI suites**

Run: `pytest tests/sim tests/test_cli.py -q`
Expected: PASS (GL tests skip where no context)

- [ ] **Step 5: Render one clip and look at it**

Run: `emotive-gait demo --emotion anger=1.0 --out demo-angry.mp4 --gif`
Expected: `wrote demo-angry.mp4`; the robot crosses the checker floor with the camera following; feet do not slide. Delete the files after viewing (media is regenerated in Task 10).

- [ ] **Step 6: Commit**

```bash
git add src/emotive_gait/sim/kinematic.py src/emotive_gait/sim/render.py src/emotive_gait/cli.py tests/sim tests/test_cli.py
git commit -m "feat(sim): kinematic tier walks forward in the world with a tracking camera; CLI warns on infeasible parameters"
```

---

### Task 7: Retire `knee_flexion_bias`; re-home fear's crouch on `com_height_scale`

**Files:**
- Modify: `src/emotive_gait/gait/params.py` (remove the field)
- Modify: `src/emotive_gait/emotion/evidence.py:69,103-108` (two claims re-homed)
- Modify: `src/emotive_gait/emotion/axes.py` (`CATEGORY_OFFSETS[FEAR]`), `src/emotive_gait/emotion/model.py` (`_PARAMETER_ROLES`, `_NONNEGATIVE_AMPLITUDES`)
- Modify: `tests/emotion/test_axes.py:54`, `tests/emotion/test_evidence.py:51`, `tests/emotion/test_model.py:40`, `tests/sim/test_playback.py` (presets, ramp, crouch tests), `tests/gait/test_params.py`
- Modify: `CLAUDE.md` (facts list)
- Test: as listed

**Interfaces:**
- Consumes: Task 5's generator.
- Produces:
  - `GaitParameters` without `knee_flexion_bias` (15 fields).
  - Evidence: `(Emotion.FEAR, "com_height_scale", -1, _D, _ROETHER, "increased hip and knee flexion, widespread postural tension")`; `(Emotion.ANGER, "com_height_scale", 0, _D, _ROETHER, "hip/knee flexion not significant")`.
  - `CATEGORY_OFFSETS[FEAR] = {"com_height_scale": -0.04, "elbow_swing_ratio": 0.30, "cadence": 0.45, "neck_pitch": -0.25}`.
  - `_PARAMETER_ROLES` adds `com_height_scale`, `step_length`, `step_height`, `feet_spacing`, `double_support_frac`, `com_bounce_amp`, `lateral_sway_amp` -> `(HIP_PITCH_L, HIP_PITCH_R, KNEE_L, KNEE_R, ANKLE_PITCH_L, ANKLE_PITCH_R)`; `_NONNEGATIVE_AMPLITUDES` adds `step_length`.
  - `GaitParameters.feet_spacing` default moves from 0.15 to 0.20 (spec 4.5; tuned). The YAML round-trip and `from_yaml` default tests pass unchanged because they compare against the dataclass defaults.

- [ ] **Step 1: Update the tests**

- `tests/emotion/test_axes.py`: `test_fear_carries_a_category_offset_for_crouch`: `assert CATEGORY_OFFSETS[Emotion.FEAR]["com_height_scale"] < 0`.
- `tests/emotion/test_evidence.py:51`: `assert d(Emotion.FEAR, "com_height_scale") < 0`.
- `tests/emotion/test_model.py:40`: `NO_CHANGE = {(Emotion.ANGER, "neck_pitch"), (Emotion.ANGER, "com_height_scale")}`.
- `tests/gait/test_params.py` (append):

```python
def test_yaml_with_the_retired_knee_bias_is_rejected_by_name(tmp_path):
    f = tmp_path / "old.yaml"; f.write_text("knee_flexion_bias: 0.1\n")
    with pytest.raises(ValueError, match="knee_flexion_bias"):
        GaitParameters.from_yaml(f)
```

- `tests/sim/test_playback.py`: drop the `knee_flexion_bias` keys from `_PHASE0_PRESETS`; `test_ramp_avoids_the_startup_yank` and `test_crouch_and_neutral_do_not_fall_on_the_z_rail` use `com_height_scale` in `(0.9, 1.0)`; delete the `xfail` decorator and rewrite:

```python
def test_crouch_lowers_the_pelvis_at_least_3cm_below_neutral():
    r = load_robot(root=RootMode.Z_RAIL)
    crouch, neutral = (simulate(r, generate(GaitParameters(com_height_scale=s), r), duration=2.0) for s in (0.9, 1.0))
    assert _post_ramp_mean_pelvis_z(r, crouch) <= _post_ramp_mean_pelvis_z(r, neutral) - 0.03
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/emotion tests/gait/test_params.py tests/sim/test_playback.py -q`
Expected: FAIL (`KeyError: 'com_height_scale'` in offsets; the crouch test fails until the field goes)

- [ ] **Step 3: Remove the field, re-home the claims, set the offset, extend `_PARAMETER_ROLES` and the floor list**

`grep -rn knee_flexion_bias src tests` must return nothing afterwards.

- [ ] **Step 4: Run the whole suite, including the preset feasibility test from Task 5 with fear's new crouch**

Run: `pytest -q`
Expected: all pass, 0 xfailed. If `test_every_preset_is_feasible` fails on `fear`, reduce the crouch offset magnitude (spec 4.4 lever order) and record the value in the commit message.

- [ ] **Step 5: Update `CLAUDE.md` facts**

Delete the inert-fields fact and the leg-constants fact. Replace the knee-bias sentence in the fear bullet with the four current offsets. Add the posture rule verbatim: "amplitudes are relative, postures are absolute" (spec 4.5). Add: `com_height_scale = 1.0` is the profile's nominal pelvis height (G1 0.70 m), below the straight-leg 0.79 m stand; the IK is seeded bent-kneed because the keyframe knee is singular; the ankle pitch range binds at deep crouch and leg reach binds at long stride, and the preset feasibility test is the arbiter.

- [ ] **Step 6: Commit**

```bash
git add src tests CLAUDE.md
git commit -m "feat(emotion): retire knee_flexion_bias; fear's crouch is com_height_scale, rendered by the IK legs"
```

---

### Task 8: `XZ_RAIL` — driven forward rail for the physics tier

**Files:**
- Modify: `src/emotive_gait/robot/root.py` (`RootMode.XZ_RAIL`, `ROOT_SLIDE_X_JOINT = "root_slide_x"`, `ROOT_SLIDE_X_ACTUATOR = "root_slide_x"`, `_X_RANGE = (-1.0, 20.0)`, `RAIL_KP = 5000.0`, `RAIL_KV = 200.0` tuned)
- Modify: `src/emotive_gait/sim/physical.py` (`_root_mode`, projection row, rail command, root-lag criterion, `_MAX_ROOT_LAG = 0.10`)
- Modify: `src/emotive_gait/cli.py` (`_ROOTS["xz-rail"] = "XZ_RAIL"`)
- Test: `tests/sim/test_playback.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `GaitTrajectory.root_x_at`, `project_for_root`.
- Produces: `to_root_mode(spec, RootMode.XZ_RAIL)` adds slide X (range `_X_RANGE`) then slide Z (range as Z_RAIL) on the pelvis and a position actuator on slide X (`gainprm[0]=RAIL_KP`, `biasprm=[0, -RAIL_KP, -RAIL_KV]`, `ctrlrange=_X_RANGE`); `simulate` on XZ_RAIL writes `data.ctrl[rail] = w * traj.root_x_at(t)` with the same ramp weight `w` as the joints, applies every Z_RAIL criterion plus `"root lag: ..."` when `|cmd - qpos[slide_x]| > _MAX_ROOT_LAG` after the ramp; `played_params` zeroes only `lateral_sway_amp`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/sim/test_playback.py (append)
def test_xz_rail_has_two_slide_dofs_and_a_rail_actuator():
    r = load_robot(root=RootMode.XZ_RAIL)
    assert r.model.nq == r.model.nv == load_robot(root=RootMode.Z_RAIL).model.nq + 1
    assert mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "root_slide_x") >= 0

def test_xz_rail_walks_forward_for_four_seconds_on_neutral():
    r = load_robot(root=RootMode.XZ_RAIL); p = GaitParameters()
    res = simulate(r, generate(p, r), duration=4.0)
    assert not res.fell, res.fail_reason
    assert res.played_params.step_length == p.step_length and res.played_params.lateral_sway_amp == 0.0
    # The ramped command w * speed * t equals the unramped one at t = ramp, so no travel is lost.
    assert res.root_x[-1] == pytest.approx(p.speed * 4.0, abs=0.10)

def test_xz_rail_reports_root_lag_when_the_rail_cannot_keep_up():
    r = load_robot(root=RootMode.XZ_RAIL)
    res = simulate(r, generate(GaitParameters(cadence=8.0, step_length=1.0), r), duration=3.0)
    assert res.fell and res.fail_reason.startswith(("root lag", "foot contact lost", "rail stop"))
```

`SimResult` gains `root_x: np.ndarray` (`(N,)`, world pelvis x per step, on every root).

```python
# tests/test_cli.py (append)
def test_xz_rail_is_a_root_choice(capsys):
    assert main(["demo", "--physics", "--no-video", "--root", "xz-rail", "--duration", "0.5"]) == 0
    assert "root: xz-rail" in capsys.readouterr().out
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/sim/test_playback.py tests/test_cli.py -q -k xz`
Expected: FAIL (`AttributeError: XZ_RAIL`)

- [ ] **Step 3: Implement the root mode, detection, rail command, lag criterion, projection row and CLI choice**

`_root_mode` checks for `root_slide_x` before `root_slide_z`.

- [ ] **Step 4: Run the sim and CLI suites**

Run: `pytest tests/sim tests/test_cli.py -q`
Expected: PASS. If the neutral walk fails on root lag, raise `RAIL_KP` (tuned) before anything else and record the value.

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/robot/root.py src/emotive_gait/sim/physical.py src/emotive_gait/cli.py tests/sim/test_playback.py tests/test_cli.py
git commit -m "feat(sim): XZ_RAIL driven forward rail with a root-lag fall criterion"
```

---

### Task 9: ZMP margin, foot slip and `GaitReport`

**Files:**
- Modify: `src/emotive_gait/sim/metrics.py` (`zmp_margin`, `foot_contact_flags`)
- Modify: `src/emotive_gait/sim/physical.py` (`SimResult.zmp_margin`, `SimResult.foot_slip`, `GaitReport`, `summarize`)
- Modify: `src/emotive_gait/cli.py` (print the report after `--physics`)
- Test: `tests/sim/test_metrics.py`, `tests/sim/test_playback.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `zero_moment_point`, `support_polygon`, `_foot_contacts`, `RobotProfile.foot_sites`.
- Produces:
  - `def zmp_margin(model, data) -> float | None` — signed distance (m) from the ZMP to the nearest edge of `support_polygon`, positive inside, negative outside; `None` when the ZMP is `None` or the polygon has fewer than three vertices.
  - `def foot_contact_flags(model, data) -> tuple[bool, bool]` — (left, right) foot has a floor contact.
  - `SimResult.zmp_margin: np.ndarray` (`(N,)`, NaN where `None`), `SimResult.foot_slip: np.ndarray` (`(N, 2)` horizontal site speed in m/s of (left, right) `foot_sites`, finite-differenced between steps, NaN when that foot has no contact).
  - `@dataclass(frozen=True) class GaitReport: zmp_outside_frac: float; zmp_mean_margin_m: float; foot_slip_rms_m_s: tuple[float, float]; pelvis_bob_m: float; pelvis_held: bool; played_params: GaitParameters`, all over post-ramp steps.
  - `def summarize(res: SimResult, *, ramp: float = 0.5, timestep: float) -> GaitReport`; CLI prints one `key: value` line per field after the existing `pelvis bob` line, with `(pelvis held)` appended to the ZMP lines on WELD.

- [ ] **Step 1: Write the failing tests**

```python
# tests/sim/test_metrics.py (append; the module has no fixtures, only `_settle(r, steps)`)
def test_zmp_margin_is_positive_when_standing_still_on_both_feet():
    r = load_robot(root=RootMode.Z_RAIL)
    d = _settle(r)  # settled standing pose with both feet on the floor
    m = zmp_margin(r.model, d)
    assert m is not None and m > 0.02

def test_zmp_margin_is_none_without_three_contact_points():
    r = load_robot(root=RootMode.Z_RAIL)
    d = mujoco.MjData(r.model); d.qpos[:] = r.stand_qpos
    slide = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_JOINT, "root_slide_z")
    d.qpos[r.model.jnt_qposadr[slide]] = 0.3  # feet 30 cm in the air
    mujoco.mj_forward(r.model, d)
    assert zmp_margin(r.model, d) is None
```

(`_settle` returns the settled `MjData`; add `from emotive_gait.robot.root import RootMode` and `zmp_margin` to the module's imports.)

```python
# tests/sim/test_playback.py (append)
def test_report_on_the_z_rail_has_finite_numbers_and_no_slip_claim():
    r = load_robot(root=RootMode.Z_RAIL)
    res = simulate(r, generate(GaitParameters(), r), duration=2.0)
    rep = summarize(res, timestep=r.model.opt.timestep)
    assert 0.0 <= rep.zmp_outside_frac <= 1.0 and math.isfinite(rep.zmp_mean_margin_m)
    assert all(s < 0.05 for s in rep.foot_slip_rms_m_s)  # in place: feet do not slide
    assert not rep.pelvis_held

def test_report_on_xz_rail_measures_slip_below_a_tenth_of_walking_speed():
    r = load_robot(root=RootMode.XZ_RAIL); p = GaitParameters()
    rep = summarize(simulate(r, generate(p, r), duration=3.0), timestep=r.model.opt.timestep)
    assert max(rep.foot_slip_rms_m_s) < 0.1 * p.speed

def test_weld_report_is_labelled_pelvis_held():
    r = load_robot(root=RootMode.WELD)
    assert summarize(simulate(r, generate(GaitParameters(), r), duration=1.0), timestep=r.model.opt.timestep).pelvis_held
```

```python
# tests/test_cli.py (append)
def test_physics_prints_the_gait_report(capsys):
    assert main(["demo", "--physics", "--no-video", "--duration", "1"]) == 0
    out = capsys.readouterr().out
    assert "zmp_outside_frac:" in out and "foot_slip_rms_m_s:" in out
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/sim/test_metrics.py tests/sim/test_playback.py tests/test_cli.py -q -k "margin or report"`
Expected: FAIL (`ImportError: zmp_margin`, `summarize`)

- [ ] **Step 3: Implement `zmp_margin`, `foot_contact_flags`, the two `SimResult` arrays, `GaitReport`, `summarize`, and the CLI lines**

Signed distance to a convex polygon: minimum over edges of the signed distance to the edge line, with the polygon's vertices ordered counter-clockwise (reverse them if `ConvexHull` gives clockwise order).

- [ ] **Step 4: Run the sim and CLI suites**

Run: `pytest tests/sim tests/test_cli.py -q`
Expected: PASS. If the XZ_RAIL slip test fails, the rail gain or the position-servo tracking is the cause; report the RMS value and tune `RAIL_KP` (tuned) before touching the threshold.

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/sim tests/sim src/emotive_gait/cli.py tests/test_cli.py
git commit -m "feat(sim): ZMP margin, foot slip and a per-run GaitReport printed by the CLI"
```

---

### Task 10: Docs, media, version

**Files:**
- Modify: `README.md` (Limitations; `com_height_scale` meaning; the tiers paragraph), `CLAUDE.md` (architecture list adds `footsteps.py`, `ik.py`, `XZ_RAIL`; dependency rules add `gait/ik.py` imports `mink` and `robot/`, `gait/footsteps.py` imports nothing from `robot/`; the root-mode fall-criteria bullet gains XZ_RAIL's root lag; `Commands` adds `--root xz-rail` and `--camera fixed`), `pyproject.toml` (`version = "0.2.0"`; `tests/test_packaging.py` checks only that a version string exists, so it needs no edit)
- Regenerate: `docs/media/sad.gif`, `docs/media/angry.gif` (`*.mp4` stay ignored)

**Interfaces:** none.

- [ ] **Step 1: Rewrite the README Limitations**

Replace the "in-place in both tiers" bullet with: the kinematic tier walks forward with exact placement; the physics tier walks forward on a driven rail (`--root xz-rail`) and reports foot slip and ZMP margin, or steps in place on the Z rail; sway is lost on every rail; feet are flat (no heel-toe); no balance control. State that `com_height_scale = 1.0` is a tuned 0.70 m walking height.

- [ ] **Step 2: Update `CLAUDE.md` and bump the version**

- [ ] **Step 3: Regenerate the media**

Run: `emotive-gait demo --emotion sadness=1.0 --out docs/media/sad.mp4 --gif && emotive-gait demo --emotion anger=1.0 --out docs/media/angry.mp4 --gif`
Expected: both `wrote` lines; view both GIFs; the sad walker is slow and short-stepped, the angry one long and fast, both with stationary stance feet.

- [ ] **Step 4: Full gate**

Run: `ruff check . && ruff format --check . && pytest -q`
Expected: clean, all pass, 0 xfailed.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md pyproject.toml docs/media/sad.gif docs/media/angry.gif tests/test_packaging.py
git commit -m "docs: Phase 4 forward locomotion; regenerate demo media; version 0.2.0"
```

Then the whole-branch review (fable), fixes, and `superpowers:finishing-a-development-branch`; merge and push are the user's decisions.
