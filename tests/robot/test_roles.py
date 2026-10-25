import pytest

from emotive_gait.robot.roles import JointRole, MissingRoleError, RobotProfile


def _profile(**kw):
    return RobotProfile(
        name="t",
        scene_entry="scene",
        actuator_for={JointRole.ELBOW_L: "left_elbow_joint"},
        sign_for={JointRole.ELBOW_L: 1.0},
        **kw,
    )


def test_actuator_resolves_known_role():
    assert _profile().actuator(JointRole.ELBOW_L) == "left_elbow_joint"


def test_missing_role_raises_with_context():
    with pytest.raises(MissingRoleError) as e:
        _profile().actuator(JointRole.NECK_PITCH)
    assert e.value.role is JointRole.NECK_PITCH
    assert e.value.profile_name == "t"


def test_missing_role_error_is_keyerror():
    with pytest.raises(KeyError):
        _profile().actuator(JointRole.NECK_PITCH)


# Degradation is explicit and queryable, not an exception deep in a loop
def test_missing_reports_absent_roles_without_raising():
    p = _profile()
    assert p.has(JointRole.ELBOW_L) is True
    assert p.has(JointRole.NECK_PITCH) is False
    assert p.missing([JointRole.ELBOW_L, JointRole.NECK_PITCH]) == (JointRole.NECK_PITCH,)


def test_mirror_swaps_sides_and_fixes_midline():
    assert JointRole.ELBOW_L.mirror() is JointRole.ELBOW_R
    assert JointRole.ELBOW_R.mirror() is JointRole.ELBOW_L
    assert JointRole.WAIST_PITCH.mirror() is JointRole.WAIST_PITCH


def test_mirror_is_involution_and_midline_roles_are_fixed():
    for role in JointRole:
        assert role.mirror().mirror() is role
    for role in (
        JointRole.WAIST_YAW,
        JointRole.WAIST_ROLL,
        JointRole.WAIST_PITCH,
        JointRole.NECK_PITCH,
    ):
        assert role.mirror() is role
    assert JointRole.HIP_ROLL_L.mirror() is JointRole.HIP_ROLL_R
