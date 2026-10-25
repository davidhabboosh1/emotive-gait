# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research prototype exploring **emotion-modulated bipedal gait**: author a walk cycle as per-joint
Bézier curves, perturb it according to emotion intensities, and keep the robot balanced in
simulation. There is no package, no build, no test suite, and no requirements file — every entry
point is a standalone script run from the repo root.

## Running things

All scripts resolve `scene.xml` and `meshes/` relative to the **current working directory**, so
always run from the repo root.

```sh
python new.py                    # LQR balance controller, recomputes gain every step (MuJoCo viewer)
python new2.py                   # Same controller, gain solved once offline — much faster
python emotive-gait.py           # Tkinter gait editor; writes clipped_spline_points.npy
python emotive-gait-mujoco.py    # Offscreen render + video (needs a notebook-like display; see caveats)
python emotive-gait-pybullet.py  # Unrelated PyBullet sanity check of stock Atlas
```

`emotive-gait-mujoco.ipynb` is a single giant cell — a working-notes transcript of the MuJoCo LQR
tutorial adapted to this model. It is the derivation behind `new.py`/`new2.py`, not a maintained
artifact.

Environment is whatever is installed globally: `mujoco` 3.2.5, `numpy` 1.24.3, `scipy` 1.10.1,
`pybullet` 3.2.6, `mediapy`. Two imports in the tree have **no installed package** — `ikpy`
(top of `emotive-gait.py`) and `bvhtoolbox` (`emotionmodel.py`); see "Known broken paths".

## The model is a deliberately crippled Atlas

`scene.xml` (floor, lights, skybox) `<include>`s `atlas_minimal_contact.xml`. That file has been
hand-edited down to a **sagittal-plane, 11-DoF, 8-actuator** robot, and the reductions are the single
most important thing to understand before touching anything:

- The pelvis `<freejoint/>` is **commented out** and replaced by three explicit DoFs
  (`world_joint_TX`, `world_joint_TY`, `world_joint_RZ`) — so `nq == nv == 11`, with no quaternion in
  `qpos`. Arms, wrists, hip roll/yaw, and ankle roll joints are commented out too.
- Most `<motor>` entries in `<actuator>` are commented out, leaving 8: `back_bky`, `neck_ay`, and
  hip-pitch / knee / ankle-pitch per leg.
- `nkey == 0` — there are no keyframes.

Why it matters: `nq == nv` lets `np.hstack((dq, data.qvel))` form a valid `2*nv` state and keeps
`solve_discrete_are` on an 8-input, 22-state problem that actually converges. Re-enabling the
freejoint or the commented motors silently invalidates the LQR code, the saved `.npy` shape, and the
curve count in the GUI. If you need the full robot, expect to rework all three.

`test.xml` is a near-duplicate of `atlas_minimal_contact.xml` with the full joint set and `<general>`
actuators. Nothing references it. `MUJOCO_LOG.TXT` is a committed append-only MuJoCo log (its
`Nan, Inf or huge value in CTRL` lines are from past unstable runs, not a live problem).

## Component 1 — LQR balance (`new.py`, `new2.py`)

Both build the same infinite-horizon discrete LQR around the model's initial pose:

1. `mjd_transitionFD` → finite-difference `A` (2nv×2nv) and `B` (2nv×nu).
2. `Qbalance = J_diff.T @ J_diff` where `J_diff = jacSubtreeCom(root) - mean(jacBodyCom(c))` over
   every body currently in `data.contact`. This penalizes CoM motion *relative to the support
   bodies*, so the cost is only meaningful when contacts exist.
3. `Qpos = BALANCE_COST * Qbalance + Qjoint` with root DoFs zero-weighted; the velocity block of `Q`
   is left at zero. `R = I`.
4. `solve_discrete_are` → `K`, then `ctrl = ctrl0 - K @ [mj_differentiatePos(qpos0, qpos); qvel]`.

The difference is **when** step 1–4 run. `new2.py` runs them once before the viewer loop (textbook
form). `new.py` re-runs the whole pipeline every single step inside the loop and swallows
`LinAlgError` from the Riccati solve by stepping and continuing — the git history marks this as the
one that actually stays upright ("new.py is best"), at the cost of being very slow. Keep both;
they are a deliberate A/B, not duplication.

`another.py` is an abandoned TVLQR sketch: it hand-builds `A`/`B` instead of using
`mjd_transitionFD`, mis-sizes them with `nq` where `nv` belongs, and tracks a zero trajectory. Don't
extend it — use `mjd_transitionFD` as in `new.py`.

## Component 2 — gait editor (`emotive-gait.py`)

A ~660-line Tkinter app, `CurveCreatorApp`. One cubic-Bézier curve per joint, discovered at startup
from the MuJoCo model, so the UI shape is driven by `scene.xml`:

- `curve_names` / `limits` / `colors` come from `model.joint(i).name` and `.range` for `i in range(nq)`.
- `curves[j]` is a list of 6-tuples `(x, y, handle1_x, handle1_y, handle2_x, handle2_y)`;
  `spline_points[j]` is the sampled polyline. First and last anchors are X-pinned to `0` and
  `canvas_width` and **share a Y value** (enforced in `on_drag`) so the cycle loops seamlessly.
- Canvas Y ↔ joint angle is a linear map of `canvas_height` onto `[-π, π]` (`y_to_angle` /
  `angle_to_y`). `update_gait` re-derives the same mapping inline rather than calling them — change
  one and you must change the other.
- Edits are rejected rather than clamped: `on_click` / `on_drag` recompute the curve and revert the
  move if it would exceed the joint's range or make the spline non-monotonic in X.
- Interaction: left-click adds/drags an anchor, Shift+left-click drags a handle, right-click deletes
  an interior anchor, "Update Gait" resamples to `total_points` (600) and saves.

**Output contract**: `clipped_spline_points.npy`, shape `(nq, total_points, 2)` — currently
`(11, 600, 2)` — where `[j, t, 0]` is canvas X and `[j, t, 1]` is the joint angle in radians, clipped
to that joint's range. This is the handoff to any playback/control code.

`emotionmodel.py` is a planning stub: an emotion→index map and the dataset shortlist (EWalk,
Emotion-Gait, Human3.6M, CMU Mocap, …) for the intended learned emotion-prediction model. Nothing
imports it.

## Known broken paths (don't assume these run)

Git history ("change to supervisor so we can check balance", then "get rid of false leads") left
working Webots and NAO code stranded mid-port to MuJoCo + Atlas:

- `emotive-gait.py` imports `ikpy`, which is not installed and not used — the app won't start until
  that import is dropped or the package added.
- `move_forward` (bound to **W** / **Up**), `get_inertial_force`, and `calculate_moments` call the
  Webots API via `self.sup`, `self.node`, `self.accelerometer`, `self.mass` — **none are ever
  assigned**. The ZMP math in `compute_zmp` is sound and worth keeping; its inputs are not wired to
  MuJoCo.
- `emotion_from` (the actual emotion→gait modulation) indexes NAO joint names — `LHipPitch`,
  `RShoulderPitch`, `HeadYaw`, … — which do not exist in the Atlas model. Its call site in
  `update_gait` is commented out for exactly that reason. Porting it means remapping those to the
  Atlas joints above, where there are no arm or head-yaw actuators at all.
- `upload_motion` parses Webots `.motion` files (`TaiChi.motion`, NAO joint columns) but appends
  2-tuples into `self.curves`, which the rest of the code reads as 6-tuples.
- `emotive-gait-mujoco.py` and the notebook call `mj_resetDataKeyframe(model, data, 1)`, but
  `scene.xml` defines no keyframes (`nkey == 0`).

The loose URDFs (`nao*.urdf`, `NAOH25V60.urdf`, `robot.urdf`, `atlas_v4_with_multisense.urdf`,
`naoV50_generated_urdf/`) are leftovers from that exploration. `emotive-gait-pybullet.py` loads Atlas
from `pybullet_data`, not the local copy. Only `scene.xml` + `atlas_minimal_contact.xml` + `meshes/`
are live.
