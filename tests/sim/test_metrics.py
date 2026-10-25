import mujoco
import numpy as np
import pytest

from emotive_gait.robot.loader import load_robot
from emotive_gait.sim.metrics import (
    _foot_contacts,
    center_of_pressure,
    is_upright,
    support_polygon,
    uprightness,
    zero_moment_point,
)


def _settle(r, steps=200):
    d = mujoco.MjData(r.model)
    d.qpos[:] = r.stand_qpos
    for _ in range(steps):
        d.ctrl[:] = [
            r.stand_qpos[r.model.jnt_qposadr[r.model.actuator_trnid[i, 0]]]
            for i in range(r.model.nu)
        ]
        mujoco.mj_step(r.model, d)
    return d


def test_uprightness_is_one_when_standing_and_drops_when_tipped():
    r = load_robot()
    d = mujoco.MjData(r.model)
    d.qpos[:] = r.stand_qpos
    mujoco.mj_forward(r.model, d)
    assert uprightness(r.model, d) == pytest.approx(1.0, abs=1e-3)
    assert is_upright(r.model, d)
    d.qpos[3:7] = [0.707, 0.0, 0.707, 0.0]  # 90 deg pitch about y
    mujoco.mj_forward(r.model, d)
    assert not is_upright(r.model, d)


def test_unknown_body_raises_value_error_naming_it():
    r = load_robot()
    d = mujoco.MjData(r.model)
    mujoco.mj_forward(r.model, d)
    with pytest.raises(ValueError, match="no_such_body"):
        uprightness(r.model, d, body="no_such_body")


def test_cop_is_none_without_contact_and_inside_support_when_standing():
    r = load_robot()
    d = mujoco.MjData(r.model)
    d.qpos[:] = r.stand_qpos
    d.qpos[2] += 2.0  # lifted clear of the floor
    mujoco.mj_forward(r.model, d)
    assert center_of_pressure(r.model, d) is None
    assert zero_moment_point(r.model, d) is None
    assert support_polygon(r.model, d).shape == (0, 2)

    d = _settle(r)
    cop = center_of_pressure(r.model, d)
    assert cop is not None
    hull = support_polygon(r.model, d)
    assert hull[:, 0].min() <= cop[0] <= hull[:, 0].max()
    assert hull[:, 1].min() <= cop[1] <= hull[:, 1].max()


def test_zmp_agrees_with_cop_on_flat_ground_when_standing():
    r = load_robot()
    d = _settle(r)
    cop = center_of_pressure(r.model, d)
    zmp = zero_moment_point(r.model, d)
    assert cop is not None and zmp is not None
    assert np.linalg.norm(cop - zmp) < 2e-3


def test_zmp_equals_cop_under_horizontal_load():
    """A sideways push makes feet carry friction; ground-origin ZMP must still equal CoP."""
    r = load_robot()
    d = mujoco.MjData(r.model)
    d.qpos[:] = r.stand_qpos
    torso = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    d.xfrc_applied[torso, 0] = 40.0
    for _ in range(300):
        d.ctrl[:] = [
            r.stand_qpos[r.model.jnt_qposadr[r.model.actuator_trnid[i, 0]]]
            for i in range(r.model.nu)
        ]
        mujoco.mj_step(r.model, d)
    fx = sum(f[0] for _, f, _ in _foot_contacts(r.model, d))
    print(f"tangential Fx={fx:.2f}")
    assert abs(fx) > 10.0
    cop = center_of_pressure(r.model, d)
    zmp = zero_moment_point(r.model, d)
    print(f"cop={cop} zmp={zmp}")
    assert zmp == pytest.approx(cop, abs=2e-3)


def test_support_polygon_is_ordered_hull_of_foot_contacts():
    r = load_robot()
    d = _settle(r)
    hull = support_polygon(r.model, d)
    assert hull.ndim == 2 and hull.shape[1] == 2 and len(hull) >= 3
    # counter-clockwise hull has positive signed area
    x, y = hull[:, 0], hull[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    assert area > 0
