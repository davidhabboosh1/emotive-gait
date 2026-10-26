"""Role sign conventions, probed on the G1 (the reference rig, spec 4.5).

Each probe starts from the standing pose of the welded G1, adds ``PROBE_RAD * sign(role)`` at
the role's joint and checks that the named landmark moves in the convention's direction.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from mocap.g1_landmarks import G1_LANDMARKS, HEAD_TOP, HEAD_TOP_OFFSET

from emotive_gait.gait.ik import weld_robot_for
from emotive_gait.robot.conventions import PROBE_RAD, ROLE_CONVENTIONS, convention_holds
from emotive_gait.robot.loader import RobotModel
from emotive_gait.robot.profiles.g1 import G1_PROFILE_WITH_NECK
from emotive_gait.robot.roles import JointRole

X, Y, Z = np.eye(3)
G1_POINTS = {lm.value: body for lm, body in G1_LANDMARKS.items() if body != HEAD_TOP}
TOE_OFFSET = np.array([0.10, 0.0, 0.0])  # along the foot site's own x axis: the G1 has no toe


@pytest.fixture(scope="module")
def weld() -> RobotModel:
    return weld_robot_for(G1_PROFILE_WITH_NECK)


def _points(robot: RobotModel, qpos: np.ndarray) -> dict[str, np.ndarray]:
    model = robot.model
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    out = {name: data.xpos[model.body(body).id].copy() for name, body in G1_POINTS.items()}
    head = model.body("head_link").id
    out["head"] = data.xpos[head] + data.xmat[head].reshape(3, 3) @ HEAD_TOP_OFFSET
    for side, site_name in zip(("l", "r"), robot.profile.foot_sites, strict=True):
        site = model.site(site_name).id
        out[f"toe_{side}"] = data.site_xpos[site] + data.site_xmat[site].reshape(3, 3) @ TOE_OFFSET
    return out


def probe_g1(
    robot: RobotModel, role: JointRole
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    qpos = robot.stand_qpos.copy()
    before = _points(robot, qpos)
    qpos[robot.qpos_adr(role)] += PROBE_RAD * robot.profile.sign(role)
    return before, _points(robot, qpos)


_ELBOWS = (JointRole.ELBOW_L, JointRole.ELBOW_R)


@pytest.mark.parametrize("role", [r for r in ROLE_CONVENTIONS if r not in _ELBOWS])
def test_g1_positive_joint_motion_follows_the_convention(weld, role):
    before, after = probe_g1(weld, role)
    assert convention_holds(role, before, after, forward=X, left=Y, up=Z)


@pytest.mark.xfail(
    strict=True, reason="G1 elbow joint straightens when positive (Phase 4 rendering finding)"
)
@pytest.mark.parametrize("role", _ELBOWS)
def test_g1_elbow_follows_the_convention(weld, role):
    before, after = probe_g1(weld, role)
    assert convention_holds(role, before, after, forward=X, left=Y, up=Z)


def test_convention_holds_is_pure_geometry():
    before = {"knee_l": np.array([0.0, 0.1, 0.4])}
    after = {"knee_l": np.array([-0.01, 0.1, 0.4])}
    assert convention_holds(JointRole.HIP_PITCH_L, before, after, forward=X, left=Y, up=Z)
    assert not convention_holds(JointRole.HIP_PITCH_L, after, before, forward=X, left=Y, up=Z)
