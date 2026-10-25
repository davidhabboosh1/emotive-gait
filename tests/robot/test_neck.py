import mujoco
import pytest

from emotive_gait.robot.loader import load_robot
from emotive_gait.robot.profiles.g1 import G1_PROFILE, G1_PROFILE_WITH_NECK
from emotive_gait.robot.roles import JointRole

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
    d = mujoco.MjData(r.model)
    d.qpos[:] = r.stand_qpos
    mujoco.mj_forward(r.model, d)
    head = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_BODY, "head_link")
    z_up = d.xipos[head][2]  # CoM: xpos is the hinge itself and cannot move
    d.qpos[r.qpos_adr(JointRole.NECK_PITCH)] = 0.8
    mujoco.mj_forward(r.model, d)
    assert d.xipos[head][2] < z_up


def test_standing_pose_survives_the_keyframe_rebuild():
    r = load_robot(with_neck=True)
    bare = load_robot(with_neck=False)
    for role in (JointRole.ELBOW_L, JointRole.SHOULDER_PITCH_R, JointRole.SHOULDER_ROLL_R):
        assert r.stand_qpos[r.qpos_adr(role)] == pytest.approx(bare.stand_qpos[bare.qpos_adr(role)])


def test_default_load_includes_neck():
    assert load_robot().profile.has(JointRole.NECK_PITCH)


def test_neck_profile_is_g1_plus_neck_only():
    assert set(G1_PROFILE_WITH_NECK.actuator_for) == set(G1_PROFILE.actuator_for) | {
        JointRole.NECK_PITCH
    }
    assert G1_PROFILE_WITH_NECK.actuator(JointRole.NECK_PITCH) == "neck_pitch_joint"
    assert G1_PROFILE_WITH_NECK.sign(JointRole.NECK_PITCH) == 1.0
    assert not G1_PROFILE.has(JointRole.NECK_PITCH)
