# Phase 4.7 Rig Reader Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read every BVH rig at its true size (position channels replace the OFFSET; the CMU unit is inferred), give two-channel bones a reference their channels can write back, and make the validation report's commit line true, then regenerate the report.

**Architecture:** One convention change in `BvhFile._kinematics` with its mirror in `rig/compile.py::bone_offset` and a one-line consequence in `rig/writer.py`; one new scale in `rig/mapping.py::SCALES` with a runner-up note; one new fitting step in `rig/calibration.py::reference_pose` whose decisions and warnings flow through `ReferencePose` into `load_rig`'s `RigMapping`; a rewritten `_commit` in `rig/validate.py`. No new module, no new dependency.

**Tech Stack:** Python 3.11, NumPy, SciPy (`scipy.optimize.least_squares`), MuJoCo 3.17 (`MjSpec`), pytest, ruff. Run tests with the worktree venv: `.venv/Scripts/python.exe -m pytest ...` (Windows); `ruff` likewise.

**Spec:** `docs/superpowers/specs/2026-10-26-phase-4-7-rig-reader-fidelity-design.md`

## Global Constraints

- `mocap/` imports NumPy, SciPy and `gait.params` only; `rig/` may import everything except `learn/` and `sim/`; only `cli.py` imports `rig/`.
- `data/manifest.yaml` is frozen; `data/rig-manifest.yaml`'s dataset entries do not change (its comments may). Bandai, the cgspeed CMU conversion and Mixamo files never enter git in any form. Aggregates only.
- The committed fit reads bit-identically: `emotive-gait fit --check` passes; `tests/mocap/test_bvh_file.py::test_to_clip_matches_read_bvh_bit_for_bit` passes; the 100STYLE fixture's FK is unchanged.
- The G1 path and its two round-trip xfails do not change; the extractor's thresholds and definitions do not change.
- `docs/reports/phase-4-6-rig-validation.md` is regenerated with `python -m emotive_gait.rig.validate docs/reports/phase-4-6-rig-validation.yaml --out docs/reports/phase-4-6-rig-validation.md`, never hand-edited; `docs/reports/*.md`, the manifests and the artefact are LF (`.gitattributes`).
- Zero-column and constant-column tolerance: `1e-9` file units (`CONSTANT_TRANSLATION_TOL`).
- Two-channel warning threshold `TWO_CHANNEL_WARN_RAD = 0.01`; fit exactness on a representable mean below `1e-9` rad.
- `SCALES = (1.0, 0.01, 0.0254, 0.0564444, 0.1, 0.001)`; `LEG_TARGET_M` and `LEG_WINDOW_M` unchanged.
- Commit messages carry no Claude attribution trailer. `ruff check . && ruff format --check . && pytest -q` green at every commit that completes a task.
- The validation data: `data/raw` in this worktree is a junction to `C:\emotive-gait\data\raw`; the report regeneration runs from the worktree root.

## Review Focus

1. A root with a non-zero OFFSET and absolute position channels (a Blender export): the output BVH must put the pelvis where the trajectory says, not a root OFFSET low. Test: Task 3's `root_height_in_channels` variant in the zero-emotion round trip.
2. A joint with position channels on only some axes (a root with only `Yposition`): the reader keeps the OFFSET on the other axes. Test: Task 1 `test_an_axis_without_a_position_channel_keeps_its_offset`.
3. A child whose position column varies: the reader follows it frame by frame and the compiler still refuses it naming the bone. Tests: Task 1 `test_a_varying_child_position_column_moves_the_bone`, existing `test_a_varying_translation_channel_below_the_pelvis_is_refused`.
4. A leg length that two known scales land inside the window: the chosen scale is the nearest and the decision names the runner-up. Test: Task 4 `test_an_ambiguous_leg_names_the_runner_up_scale`.
5. `git` missing or the report generated outside a checkout: the header still renders with `unknown`. Test: Task 6 `test_commit_line_outside_a_checkout_says_unknown`.

---

### Task 1: Position channels replace the OFFSET (reader, fixture, writer root rule)

**Files:**
- Modify: `src/emotive_gait/mocap/readers/bvh.py` (module docstring lines 1-8; `_kinematics` lines 180-206)
- Modify: `src/emotive_gait/rig/writer.py` (docstring bullet on the root, lines 15-18; `_held_row` docstring; root translation lines 126-130)
- Modify: `tests/mocap/fixtures/two_bone.bvh`
- Test: `tests/mocap/test_bvh_file.py`

**Interfaces:**
- Produces: the reader convention every later task relies on. `BvhFile._kinematics` unchanged in signature. Helper `_translation(joint: _Joint | BvhJoint, columns: np.ndarray) -> np.ndarray` is NOT required; the rule may live inline.
- Rule (per joint, per axis): an axis with no position channel keeps `offset[axis]`; an axis whose column is identically zero (`np.all(np.abs(column) <= 1e-9)`) keeps `offset[axis]`; otherwise `local_t[:, axis] = column` (no addition).

- [ ] **Step 1: Rewrite the fixture under the new convention**

`tests/mocap/fixtures/two_bone.bvh`: ROOT `OFFSET 0 0 0`; motion rows become

```
0 100 0 0 0 0 0 0 0
0 100 0 0 90 0 0 0 0
0 110 0 0 0 0 0 0 0
```

(Yposition carries the absolute height; the third frame's +10 is kept.) Every existing assertion on the fixture (`test_readers.py:19-26` knee at z 0.6, root z 1.1 in frame 3; `test_bvh_file.py` Knee_End at 0.2) keeps its numbers.

- [ ] **Step 2: Write the failing tests** in `tests/mocap/test_bvh_file.py`

Use a small builder in the test module: `_file(root_offset, root_channels, root_motion, child_offset, child_channels, child_motion) -> BvhFile` constructed from `BvhJoint` tuples and a `(T, n)` motion array, `scale=1.0`, `up_axis="z"`.

```python
def test_a_duplicate_constant_child_position_column_reads_the_bone_at_offset_length():
    # child OFFSET (0, 0, -0.4), child channels Xposition Yposition Zposition + ZXY, column == OFFSET
    f = _child_with_positions(offset=[0, 0, -0.4], column=[0, 0, -0.4])
    assert np.allclose(f.to_clip().positions[:, 1], [0, 0, 0.6])  # root at (0,0,1.0)

def test_a_constant_child_position_column_different_from_the_offset_positions_the_bone_at_the_column():
    f = _child_with_positions(offset=[0, 0, -0.4], column=[0, 0, -0.25])
    assert np.allclose(f.to_clip().positions[:, 1], [0, 0, 0.75])

def test_a_varying_child_position_column_moves_the_bone():
    f = _child_with_positions(offset=[0, 0, -0.4], column=[[0, 0, -0.4], [0, 0, -0.3], [0.1, 0, -0.4]])
    assert np.allclose(f.to_clip().positions[:, 1], [[0, 0, 0.6], [0, 0, 0.7], [0.1, 0, 0.6]])

def test_a_zero_child_position_column_under_a_non_zero_offset_keeps_the_offset():
    f = _child_with_positions(offset=[0, 0, -0.4], column=[0, 0, 0])
    assert np.allclose(f.to_clip().positions[:, 1], [0, 0, 0.6])

def test_root_position_channels_replace_a_non_zero_root_offset():
    f = _root(offset=[0, 0, 0.9], motion_xyz=[[0, 0, 0.95], [0.1, 0, 0.95]])
    assert np.allclose(f.to_clip().positions[:, 0], [[0, 0, 0.95], [0.1, 0, 0.95]])

def test_an_axis_without_a_position_channel_keeps_its_offset():
    # root channels ("Yposition", "Zrotation", "Xrotation", "Yrotation"), OFFSET (0.2, 0, 0.9), Y column 0.5
    assert np.allclose(f.to_clip().positions[0, 0], [0.2, 0.5, 0.9])
```

Keep `test_to_clip_matches_read_bvh_bit_for_bit` and the writer inverse test as they are.

- [ ] **Step 3: Run the new tests and the fixture tests; expect the new ones to FAIL**

Run: `.venv/Scripts/python.exe -m pytest tests/mocap/test_bvh_file.py tests/mocap/test_readers.py -q`
Expected: the five new tests fail on positions off by the OFFSET; the fixture tests fail until Step 4 (the rewritten fixture now reads 1.0 m high under the old reader).

- [ ] **Step 4: Implement the rule in `_kinematics`** and rewrite the module docstring's second sentence to state the convention: position channels replace the OFFSET axis by axis, a zero column keeps the OFFSET (the one concession to additive files), the reading of Blender's importer and MotionBuilder's exporter.

- [ ] **Step 5: Fix the writer's root rule**

In `write_trajectory`, `root_file = np.einsum("ij,tj->ti", p.T, targets - pelvis_held) / file.scale` becomes `targets - pelvis_held` plus the root OFFSET in file axes and units: `root_file = np.einsum(...) / file.scale + np.asarray(root.offset, dtype=float)`. Rewrite docstring bullet three and `_held_row`'s docstring: the held row's zero root columns leave the root OFFSET standing, so the written channels carry it.

- [ ] **Step 6: Run the reader, writer and synthesis tests**

Run: `.venv/Scripts/python.exe -m pytest tests/mocap tests/rig/test_synthesis.py -q`
Expected: all pass (the synthetic root OFFSET is at leg height; without Step 5 every zero-emotion round trip reads the pelvis 0.9 m low).

- [ ] **Step 7: Prove the committed fit is untouched**

Run: `.venv/Scripts/python.exe -m emotive_gait.cli fit --check` (or `.venv/Scripts/emotive-gait fit --check`), from the worktree root.
Expected: exit 0, every estimate equal to the artefact. Also `.venv/Scripts/python.exe -m pytest tests/rig/test_compile.py::test_full_chain_reproduces_style100 -q` passes.

- [ ] **Step 8: Commit**

```bash
git add src/emotive_gait/mocap/readers/bvh.py src/emotive_gait/rig/writer.py tests/mocap/fixtures/two_bone.bvh tests/mocap/test_bvh_file.py
git commit -m "feat(bvh): position channels replace the OFFSET axis by axis; writer carries the root OFFSET; two_bone fixture under the new convention"
```

---

### Task 2: The compile step takes the column; test helpers under the new convention

**Files:**
- Modify: `src/emotive_gait/rig/compile.py` (module header bullet lines 23-26; `bone_offset` lines 107-126)
- Modify: `tests/rig/rigs.py` (replace `fold_into_channels`, lines 434-456)
- Modify: `tests/rig/test_compile.py` (import line 17; `test_a_constant_translation_channel_is_folded_into_the_offset` lines 128-137)
- Modify: `tests/rig/test_synthesis.py` (import lines 45-53; `test_a_rig_with_constant_bone_translations_round_trips_and_keeps_them` lines 423-438)

**Interfaces:**
- Produces: `bone_offset(file: BvhFile, bone: int) -> np.ndarray` (name kept; semantics: the bone VECTOR under Task 1's rule: OFFSET where the column is absent or zero, the constant column otherwise; `RigError` naming the bone and channel on a varying column, as today).
- Produces (tests): `duplicate_offsets(f: BvhFile, bone: str) -> BvhFile` (position channels prepended, equal to the OFFSET) and `move_offset_into_channels(f: BvhFile, bone: str, vector) -> BvhFile` (position channels prepended, set to `vector` in file units; OFFSET left as is). Both keep `f.motion`'s other columns.

- [ ] **Step 1: Replace the helper and write the failing tests**

In `tests/rig/rigs.py` delete `fold_into_channels`; add the two helpers above. In `tests/rig/test_compile.py` replace the folded test with:

```python
@pytest.mark.parametrize("spec", [RigSpec(), replace(RigSpec(), up_axis="z")], ids=["y", "z"])
def test_duplicated_offsets_leave_the_reader_and_the_chain_unchanged(tmp_path, spec):
    ...  # for knee, spine2, head: f = duplicate_offsets(f, bone)
    assert np.allclose(f.to_clip().positions, before)
    _assert_chain_matches_reader(f, pelvis_name(spec), (0, 17, 150))

def test_a_constant_position_column_replaces_the_offset_in_the_chain(tmp_path):
    # knee moved to 0.8 of its OFFSET vector: reader shortens the thigh, chain agrees
    f = move_offset_into_channels(f, knee, np.asarray(offset) * 0.8)
    assert np.isclose(np.linalg.norm(knee_pos - hip_pos), 0.8 * thigh_m, atol=1e-9)
    _assert_chain_matches_reader(f, pelvis_name(RigSpec()), (0, 17, 150))
```

In `tests/rig/test_synthesis.py` the folded round-trip test uses `duplicate_offsets` for the knee and spine2 and keeps its assertions (constants kept in the output).

- [ ] **Step 2: Run; expect the second compile test to FAIL** (the chain adds OFFSET and column)

Run: `.venv/Scripts/python.exe -m pytest tests/rig/test_compile.py -q`

- [ ] **Step 3: Implement `bone_offset` under the rule** (per axis: skip an absent channel; a zero column keeps the OFFSET; a constant column replaces it; a varying column raises). Rewrite its docstring and the header bullet ("a translation channel below the pelvis that is constant over the clip is the bone vector on that axis; a varying one is refused").

- [ ] **Step 4: Run compile, calibration and synthesis tests**

Run: `.venv/Scripts/python.exe -m pytest tests/rig/test_compile.py tests/rig/test_calibration.py tests/rig/test_synthesis.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/rig/compile.py tests/rig/rigs.py tests/rig/test_compile.py tests/rig/test_synthesis.py
git commit -m "feat(rig): compiled bone vectors take a constant position column instead of adding it; duplicate_offsets and move_offset_into_channels test helpers"
```

---

### Task 3: The Blender-shaped root in the synthetic family

**Files:**
- Modify: `tests/rig/rigs.py` (`RigSpec` lines 40-52; `build_rig` lines 314-333; `write_walk` line 386; `_variants` lines 393-410)
- Modify: `tests/rig/test_compile.py` (parametrize list lines 42-52)
- Test: `tests/rig/test_synthesis.py` (existing `test_zero_emotion_round_trip_recovers_p0_on_every_rig` over `FAMILY_VARIANTS`)

**Interfaces:**
- Produces: `RigSpec.root_height_in_channels: bool = False`. When true, `build_rig` leaves the root OFFSET as it is (height at `leg_length_m` in the up axis) and `write_walk` writes the root's up-axis position column at that same height every frame (file units), the other two columns as today (X travel, the third zero).

- [ ] **Step 1: Add the field and the variant** to `_variants` (`replace(base, root_height_in_channels=True)`) and to `test_compile.py`'s explicit list with id `"root-height-in-channels"`.

- [ ] **Step 2: Run the FK gate and the round trip on the new variant; expect PASS only with Task 1's writer fix in place**

Run: `.venv/Scripts/python.exe -m pytest tests/rig/test_compile.py -k root-height tests/rig/test_synthesis.py -k root_height -q`
Expected: PASS. (Sanity: temporarily reverting the writer's `+ root.offset` must fail the round trip with a `com_height_scale` residual near -1; do not commit that state.)

- [ ] **Step 3: Run the rig suite**

Run: `.venv/Scripts/python.exe -m pytest tests/rig -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/rig/rigs.py tests/rig/test_compile.py
git commit -m "test(rig): synthetic variant with the root's height in its position channels (Blender export shape)"
```

---

### Task 4: The CMU ASF unit as an inference candidate, with a runner-up note

**Files:**
- Modify: `src/emotive_gait/rig/mapping.py` (`SCALES` line 72 and its comment; `infer_scale` lines 446-466)
- Test: `tests/rig/test_mapping.py`

**Interfaces:**
- Produces: `SCALES = (1.0, 0.01, 0.0254, 0.0564444, 0.1, 0.001)` with the comment "0.0564444: the CMU ASF length unit, 1/0.45 inch (the cgspeed BVH conversion keeps it)". `infer_scale` unchanged in signature; its `Decision.note` gains `"; runner-up {s:g} ({m:.3f} m); pass --scale to override"` when the second-nearest scale also lands the leg inside `LEG_WINDOW_M`.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_cmu_unit_is_inferred_from_a_fourteen_unit_leg(tmp_path):
    # rest hip-to-ankle is 0.9 * leg_length_m (thigh and shin OFFSETs 0.45 each); 0.809 m is
    # 14.33 units at the CMU unit
    write_walk(replace(RigSpec(), units_per_m=1 / 0.0564444, leg_length_m=0.809 / 0.9), tmp_path / "w.bvh")
    scale, _, decisions = infer_scale_and_up(read_bvh_file(tmp_path / "w.bvh", scale=1.0, up_axis="z"))
    assert scale == pytest.approx(0.0564444)
    note = next(d for d in decisions if d.what == "scale").note
    assert "runner-up 0.0254" in note and "0.364 m" in note and "--scale" in note

def test_an_unambiguous_leg_has_no_runner_up(tmp_path):
    write_walk(RigSpec(), tmp_path / "w.bvh")  # 90 units at 0.01: runner-up 0.0254 gives 2.29 m, outside
    ...
    assert "runner-up" not in note

def test_an_ambiguous_leg_names_the_runner_up_scale(tmp_path):
    # 19.6 units: 0.498 m at inches, 1.106 m at the CMU unit; both inside the window
    write_walk(replace(RigSpec(), units_per_m=39.3701, leg_length_m=0.498 / 0.9), tmp_path / "w.bvh")
    ...
    assert scale == pytest.approx(0.0564444) and "runner-up 0.0254" in note
```

Add `replace(RigSpec(), units_per_m=1 / 0.0564444)` to `test_scale_and_up_axis_are_inferred`'s parametrize list (its default 0.9 m leg gives a 0.81 m rest distance, 14.35 units).

- [ ] **Step 2: Run; expect FAIL** (`0.0254` chosen, no runner-up text)

- [ ] **Step 3: Implement** the new tuple and the runner-up note in `infer_scale` (sort the candidates by `abs(s * leg - LEG_TARGET_M)`; the second one is the runner-up when inside the window).

- [ ] **Step 4: Run mapping and synthesis tests**

Run: `.venv/Scripts/python.exe -m pytest tests/rig/test_mapping.py tests/rig/test_synthesis.py -q`
Expected: PASS (`test_uninferable_units_are_a_rig_error` still raises: 2.7 units fit nothing).

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/rig/mapping.py tests/rig/test_mapping.py
git commit -m "feat(rig): the CMU ASF length unit (1/0.45 inch) is an inference candidate; the scale decision names a runner-up inside the window"
```

---

### Task 5: Two-channel bones get a representable reference

**Files:**
- Create: `src/emotive_gait/rig/decision.py` (`Source`, `Decision`, moved from `rig/mapping.py:90-100`)
- Modify: `src/emotive_gait/rig/mapping.py` (import `Source`, `Decision` from `rig/decision.py`; keep both names exported)
- Modify: `src/emotive_gait/rig/calibration.py` (`ReferencePose` lines 92-108; `reference_pose` lines 186-250; new `fit_two_channel`; constant `TWO_CHANNEL_WARN_RAD`)
- Modify: `src/emotive_gait/rig/synthesis.py` (`load_rig` decisions merge, lines 355-385)
- Modify: `tests/rig/rigs.py` (`RigSpec.two_channel_limbs: bool = False`; `build_rig` channel construction line 320; `write_walk` lines 377-385; `_variants`)
- Test: `tests/rig/test_calibration.py`, `tests/rig/test_synthesis.py` (existing suites over `FAMILY_VARIANTS`)

**Interfaces:**
- Produces: `fit_two_channel(mean: np.ndarray, channels: Sequence[str]) -> tuple[np.ndarray, float]` in `rig/calibration.py`: `channels` are the bone's two rotation channel names in order; returns the nearest (Frobenius) `R_c1(a) @ R_c2(b)` to `mean` and the residual angle in radians (`arccos((trace(mean.T @ fitted) - 1) / 2)`, clipped). Start from `channel_angles(mean, channels)`; `scipy.optimize.least_squares` on the nine entries of `R(a, b) - mean`, `xtol=ftol=gtol=1e-14`.
- Produces: `TWO_CHANNEL_WARN_RAD = 0.01`.
- Produces: `ReferencePose.decisions: tuple[Decision, ...] = ()` and `ReferencePose.warnings: tuple[str, ...] = ()`. `Decision(what=f"reference:{bone}", chosen="fitted", source="geometry", note=f"two rotation channels ({' '.join(channels)}): reference fitted, residual {residual:.4f} rad")`; warning text `f"{bone}: two rotation channels cannot hold its mean walking rotation; reference fitted with {residual:.4f} rad left over"` when `residual > TWO_CHANNEL_WARN_RAD`.
- Consumes: `Decision`. `rig/mapping.py` imports `rig/calibration.py` at module level (mapping.py:49), so calibration cannot import mapping. Create `src/emotive_gait/rig/decision.py` holding `Source = Literal["name", "geometry", "user"]` and the frozen `Decision` dataclass (moved verbatim from mapping.py:90-100); `rig/mapping.py` imports both from it and keeps them in its namespace so `from emotive_gait.rig.mapping import Decision` (tests, cli, synthesis) keeps working; `rig/calibration.py` imports `Decision` from `rig/decision.py`.
- `load_rig` merges: `decisions=(*units, *reference.decisions, *mapping.decisions, pelvis decision, p0 decisions)` and `warnings=(*mapping.warnings, *reference.warnings)`.
- Where the fit runs: after `means` and `means[root]` are formed and BEFORE `reference_world` is built (so the ankles' stance rotations compose with fitted parents), for every bone whose rotation channel count is exactly two; and again on each `stance_rotations[name]` whose ankle bone has two rotation channels.

- [ ] **Step 1: Write the failing unit tests** in `tests/rig/test_calibration.py`

Test-module helper: `_compose(channels: Sequence[str], radians: Sequence[float]) -> np.ndarray`, the product of `rotation_about(_AXIS[c[0]], [np.degrees(a)])[0]` in channel order (`_AXIS` from `emotive_gait.mocap.readers.bvh`).

```python
def test_fit_two_channel_returns_a_representable_mean_unchanged():
    channels = ("Zrotation", "Xrotation")
    mean = rotation_about(2, [12.0])[0] @ rotation_about(0, [-7.0])[0]
    fitted, residual = fit_two_channel(mean, channels)
    assert np.allclose(fitted, mean, atol=1e-9) and residual < 1e-9

def test_fit_two_channel_projects_a_circulating_mean_and_reports_the_residual():
    # a(t) = 0.3 sin(t), b(t) = 0.3 cos(t): the chordal mean has a component about the missing axis
    ...
    fitted, residual = fit_two_channel(chordal_mean(rots), ("Zrotation", "Xrotation"))
    a, b = channel_angles(fitted, ("Zrotation", "Xrotation"))
    assert np.allclose(compose(a, b), fitted, atol=1e-9)   # representable
    assert residual > TWO_CHANNEL_WARN_RAD
    assert np.linalg.norm(fitted - mean) <= np.linalg.norm(compose(*channel_angles(mean, channels)) - mean) + 1e-12  # no worse than the plain projection
```

- [ ] **Step 2: Run; expect FAIL** (`fit_two_channel` undefined)

- [ ] **Step 3: Implement `fit_two_channel`, `TWO_CHANNEL_WARN_RAD`, the `ReferencePose` fields and the fit inside `reference_pose`**; merge in `load_rig`.

- [ ] **Step 4: Add the synthetic variant** `two_channel_limbs=True`: knees and elbows get the spec's order minus the up-axis channel (`"Yrotation"` on a y-up spec, `"Zrotation"` on z-up); `write_walk` writes their motion with `channel_angles(r, joint.channels)` (import from `emotive_gait.rig.compile`) instead of `euler_from_matrix` when a joint has fewer than three rotation channels (the walker's flexion is about the lateral axis, which is kept). Add `replace(base, two_channel_limbs=True)` to `_variants`.

- [ ] **Step 5: Write the failing integration test** in `tests/rig/test_calibration.py`

```python
def test_a_two_channel_bone_with_an_unrepresentable_mean_is_fitted_recorded_and_writes_back(tmp_path):
    spec = replace(RigSpec(), two_channel_limbs=True)
    path = walk_file(spec, tmp_path)
    f = read_bvh_file(path, scale=1.0 / spec.units_per_m, up_axis=spec.up_axis)
    # give the left upper arm two channels and a circulating swing: columns 0.3 sin, 0.3 cos (rad -> deg)
    f = _two_channel_circulating(f, FAMILIES[spec.family]["upper_arm"])
    write_bvh(f, tmp_path / "circ.bvh")
    rig = load_rig(tmp_path / "circ.bvh", scale=None, up_axis=None, mapping_overrides=None)
    bone = FAMILIES[spec.family]["upper_arm"]
    d = next(x for x in rig.mapping.decisions if x.what == f"reference:{bone}")
    assert d.chosen == "fitted" and "residual" in d.note
    assert any(bone in w and "two rotation channels" in w for w in rig.mapping.warnings)
    i = [j.name for j in f.joints].index(bone)
    channels = tuple(c for c in f.joints[i].channels if c.endswith("rotation"))
    back = compose(*channel_angles(rig.reference.local_rotations[i], channels))
    assert np.allclose(back, rig.reference.local_rotations[i], atol=1e-9)
    knee = FAMILIES[spec.family]["knee"]
    assert next(x for x in rig.mapping.decisions if x.what == f"reference:{knee}").note.endswith("rad")
    assert not any(knee in w for w in rig.mapping.warnings)  # a pure lateral swing is representable
```

(`LoadedRig` must expose the reference; check its fields and use the one that holds `ReferencePose`, else assert through `rig.driven`'s baked body quaternion of the bone.)

- [ ] **Step 6: Run the calibration, compile and synthesis suites**

Run: `.venv/Scripts/python.exe -m pytest tests/rig -q`
Expected: PASS, including the FK gate, `test_every_bound_role_satisfies_its_convention_on_the_driven_chain` and the zero-emotion round trip on the `two_channel_limbs` variant (knee and elbow residuals within tolerance; slip <= 2 mm).

- [ ] **Step 7: Commit**

```bash
git add src/emotive_gait/rig/calibration.py src/emotive_gait/rig/synthesis.py tests/rig/rigs.py tests/rig/test_calibration.py
git commit -m "feat(rig): a bone with two rotation channels gets a reference its channels can write back; fit recorded as a reference:<bone> decision, warned above 0.01 rad; two-channel synthetic variant"
```

---

### Task 6: The report's commit line

**Files:**
- Modify: `src/emotive_gait/rig/validate.py` (`_commit` lines 281-293; `_commit_line` lines 299-306; `render_report` call line 416; `report_from_spec` and `main` so the output path reaches `_commit`)
- Test: `tests/rig/test_validate.py`

**Interfaces:**
- Produces: `_commit(report_path: Path | None = None) -> str`: `git rev-parse --short HEAD` and `git status --porcelain --untracked-files=no`, both run with `cwd` = the repository root (walk up from `Path(__file__)` to the directory holding `.git`, or `git rev-parse --show-toplevel`); returns `"<hash>"` when the modified-tracked list is empty or holds only `report_path` (compared as paths relative to the root, `/`-separated), else `"<hash>|a,b,c"` (paths sorted). `"unknown"` on `OSError`/`CalledProcessError`.
- Produces: `_commit_line(commit: str) -> str`: `"Commit: `<hash>`."` or `"Commit: `<hash>`; uncommitted changes in N tracked files at generation: a, b, c."`; `"Commit: `unknown`."` for unknown.
- `render_report(..., commit: str | None = None, report_path: Path | None = None)`; `report_from_spec(..., report_path=...)`; `main` passes `Path(args.out)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_commit_line_is_the_hash_alone_when_clean_or_only_the_report_changed():
    assert _commit_line("abc1234") == "Commit: `abc1234`."

def test_commit_line_names_the_modified_tracked_files():
    assert _commit_line("abc1234|src/a.py,tests/b.py") == (
        "Commit: `abc1234`; uncommitted changes in 2 tracked files at generation: src/a.py, tests/b.py.")

def test_commit_ignores_the_report_being_written(tmp_path, monkeypatch):
    # monkeypatch subprocess.run to return "abc1234" and " M docs/reports/x.md\n"
    assert _commit(report_path=root / "docs/reports/x.md") == "abc1234"

def test_commit_line_outside_a_checkout_says_unknown(monkeypatch):
    # subprocess.run raises FileNotFoundError
    assert _commit() == "unknown" and _commit_line("unknown") == "Commit: `unknown`."
```

Keep the existing render tests; `render_report({}, not_run=...)` must still render (it calls `_commit()` on the real checkout).

- [ ] **Step 2: Run; expect FAIL**

- [ ] **Step 3: Implement**; delete the "generated on uncommitted changes" sentence; update `render_report`'s docstring ("``git rev-parse`` of this checkout and its modified tracked files when ``commit`` is ``None``").

- [ ] **Step 4: Run** `.venv/Scripts/python.exe -m pytest tests/rig/test_validate.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/emotive_gait/rig/validate.py tests/rig/test_validate.py
git commit -m "fix(rig): the validation report names the commit and the actual modified tracked files at generation"
```

---

### Task 7: Report spec, regeneration, docs, version 0.4.1

**Files:**
- Modify: `docs/reports/phase-4-6-rig-validation.yaml` (remove the two check rigs; rewrite the first two findings; the `Slip on feasible runs` finding is rewritten after the run to name only the runs that still exceed)
- Regenerate: `docs/reports/phase-4-6-rig-validation.md`
- Modify: `README.md` (lines 190-192 the scales sentence: six scales; lines 228-235 the claim table rows for Bandai and CMU and the slip paragraph; lines 246 the pelvis factors; lines 265-267 limitation 8; lines 271-275 the two findings bullets replaced by the two convention bullets)
- Modify: `CLAUDE.md` ("Facts that bite": the Bandai/translation entry lines 220-223, the units entry lines 224-225; add the two-channel rule and the commit-line rule next to the validation-report fact)
- Modify: `data/rig-manifest.yaml` comment lines 20-34 (not the entries)
- Modify: `pyproject.toml` (`version = "0.4.1"`)

- [ ] **Step 1: Edit the spec**: delete the two `(check)` rig entries (the per-rig command lines in the report are generated from the entries; `setup_commands` are fetch commands and stay). Rewrite finding 1: "CMU units: the inference now knows the CMU ASF length unit (1/0.45 inch, 0.0564444 m) and picks it for this file from its 14.33-unit rest leg (0.809 m); Phase 4.6 read the file as inches, at 0.45 of its size." Finding 2: "Bandai units: position channels replace the OFFSET axis by axis, so the constant child columns that repeat their OFFSETs leave the bones at true size and the Hips' varying columns carry the travel; Phase 4.6's reader added them and read every bone at twice its length (leg 1.848 m)." Drop the sentence about the check table in finding 4 ("at both scales" → "on every run").

- [ ] **Step 2: Commit the spec and code state first** (so the report's commit line has no file list):

```bash
git add docs/reports/phase-4-6-rig-validation.yaml
git commit -m "docs: validation report spec without the two check tables; findings state the new reading"
```

- [ ] **Step 3: Regenerate the report** (long; run in the background, from the worktree root):

Run: `.venv/Scripts/python.exe -m emotive_gait.rig.validate docs/reports/phase-4-6-rig-validation.yaml --out docs/reports/phase-4-6-rig-validation.md`
Expected: exit 0; the header reads "Commit: `<hash>`."; four rig sections; Bandai leg length about 0.924 m and pelvis factor about 0.99; CMU leg length about 0.86 m with scale `0.0564444 m per file unit (inferred)`; both 100STYLE sections identical to the committed report except the header. Diff against `git show HEAD:docs/reports/phase-4-6-rig-validation.md` and confirm only the header, the Bandai and CMU sections and the aggregate sections moved.

- [ ] **Step 4: Finish the spec's slip finding** from the regenerated numbers (name the feasible runs still above 2 mm), regenerate once more, and check the header is still a bare hash apart from the spec (`git add` the spec before regenerating, or accept the one-file list and regenerate after the commit in Step 6 as the final step).

- [ ] **Step 5: Docs**: README table rows, slip paragraph, pelvis factors, limitation 8 and the two findings bullets from the regenerated report; the scales sentence lists six scales; CLAUDE.md facts as listed above; the manifest comments; version 0.4.1. Numbers in prose come from the regenerated report only.

- [ ] **Step 6: Verify and commit**

Run: `.venv/Scripts/python.exe -m ruff check . && .venv/Scripts/python.exe -m ruff format --check . && .venv/Scripts/python.exe -m pytest -q` (the full suite; about 9 minutes; run in the background).
Expected: all pass.

```bash
git add docs/reports/phase-4-6-rig-validation.yaml docs/reports/phase-4-6-rig-validation.md README.md CLAUDE.md data/rig-manifest.yaml pyproject.toml
git commit -m "docs: Phase 4.7 rig reader fidelity; validation report regenerated at true size for Bandai and CMU; version 0.4.1"
```

If the report was regenerated before this commit and its header lists files, regenerate it once more after the commit and amend nothing: commit the regenerated report as its own small commit ("docs: regenerate the validation report on the committed tree").
