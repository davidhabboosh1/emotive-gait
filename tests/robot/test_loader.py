import mujoco
import pytest

from emotive_gait.robot.loader import ModelUnavailableError, load_robot
from emotive_gait.robot.profiles.g1 import G1_PROFILE
from emotive_gait.robot.roles import JointRole


def test_every_mapped_role_resolves_to_its_named_joint():
    r = load_robot(with_neck=False)
    for role in G1_PROFILE.actuator_for:
        aid = r.ctrl_index(role)
        assert aid >= 0
        jid = r.model.actuator_trnid[aid, 0]
        joint_name = mujoco.mj_id2name(r.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        assert joint_name == G1_PROFILE.actuator(role)


def test_g1_has_no_neck_without_injection():
    assert G1_PROFILE.has(JointRole.NECK_PITCH) is False


def test_roll_signs_are_mirrored_not_copied():
    assert G1_PROFILE.sign_for[JointRole.SHOULDER_ROLL_L] == 1.0
    assert G1_PROFILE.sign_for[JointRole.SHOULDER_ROLL_R] == -1.0


def test_stand_qpos_matches_model_and_is_upright():
    r = load_robot(with_neck=False)
    assert r.stand_qpos.shape == (r.model.nq,)
    d = mujoco.MjData(r.model)
    d.qpos[:] = r.stand_qpos
    mujoco.mj_forward(r.model, d)
    tid = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    assert d.xmat[tid].reshape(3, 3)[2, 2] > 0.99


def test_known_ranges_match_the_spec():
    r = load_robot(with_neck=False)
    assert r.joint_range(JointRole.WAIST_PITCH) == pytest.approx((-0.52, 0.52))
    assert r.joint_range(JointRole.ANKLE_PITCH_L) == pytest.approx((-0.87267, 0.5236), abs=1e-4)


def test_qpos_adr_matches_joint_address_for_every_role():
    r = load_robot(with_neck=False)
    for role in G1_PROFILE.actuator_for:
        expected = r.model.jnt_qposadr[r.model.actuator_trnid[r.ctrl_index(role), 0]]
        assert r.qpos_adr(role) == expected


def test_mirrored_roll_ranges_are_read_per_side():
    r = load_robot(with_neck=False)
    lo_l, _ = r.joint_range(JointRole.SHOULDER_ROLL_L)
    _, hi_r = r.joint_range(JointRole.SHOULDER_ROLL_R)
    assert lo_l == pytest.approx(-1.5882, abs=1e-4)
    assert hi_r == pytest.approx(1.5882, abs=1e-4)


@pytest.mark.parametrize("exc", [OSError("no network"), TypeError("filter")])
def test_unavailable_model_raises_actionable_error(monkeypatch, exc):
    import mujoco_menagerie

    def boom(name):
        raise exc

    monkeypatch.setattr(mujoco_menagerie, "get", boom)
    with pytest.raises(ModelUnavailableError) as e:
        load_robot()
    msg = str(e.value)
    assert "python -m mujoco_menagerie prefetch unitree_g1" in msg
    assert "3.11.4" in msg
    assert "sparse-checkout" in msg
