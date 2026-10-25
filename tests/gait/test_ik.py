import mujoco
import numpy as np
import pytest

from emotive_gait.gait.footsteps import foot_travel
from emotive_gait.gait.ik import LEG_ROLES, solve_leg_cycle, weld_robot_for
from emotive_gait.gait.params import GaitParameters
from emotive_gait.robot.loader import load_robot
from emotive_gait.robot.profiles.g1 import G1_PROFILE_WITH_NECK
from emotive_gait.robot.roles import JointRole


@pytest.fixture(scope="module")
def weld():
    return weld_robot_for(G1_PROFILE_WITH_NECK)


@pytest.fixture(scope="module")
def neutral(weld):
    return solve_leg_cycle(GaitParameters(), weld)


def test_requires_a_welded_robot():
    with pytest.raises(ValueError, match="WELD"):
        solve_leg_cycle(GaitParameters(), load_robot())


@pytest.mark.xfail(
    strict=True,
    reason="measured: the neutral early swing (foot behind the pelvis, lifted, level sole) needs "
    "ankle dorsiflexion past the -0.873 rad limit; 7.27 mm residual at phase 0.142/0.642. "
    "Awaiting a ruling (pelvis height, step_height or ankle seed), not a solver bug.",
)
def test_neutral_cycle_is_feasible_with_negligible_residual(neutral):
    c = neutral
    assert c.report.feasible
    assert c.report.max_foot_position_error_m < 1e-4
    assert set(c.angles) == set(LEG_ROLES) and c.phases.shape == (120,)


def test_absurd_stride_is_flagged_at_the_ankle(weld):
    c = solve_leg_cycle(GaitParameters(step_length=0.75), weld)  # 0.9 m separation
    assert not c.report.feasible
    assert c.report.max_foot_position_error_m > 0.01
    assert {JointRole.ANKLE_PITCH_L, JointRole.ANKLE_PITCH_R} & set(c.report.roles_at_limit)


def test_solve_is_deterministic(weld, neutral):
    a, b = neutral, solve_leg_cycle(GaitParameters(), weld)
    assert all(np.array_equal(a.angles[r], b.angles[r]) for r in LEG_ROLES)


def test_wrap_step_is_no_larger_than_the_largest_intra_cycle_step(neutral):
    for r in LEG_ROLES:
        q = neutral.angles[r]
        assert abs(q[0] - q[-1]) <= np.abs(np.diff(q)).max() + 1e-9


def test_left_foot_starts_its_swing_half_a_travel_behind_the_pelvis(weld, neutral):
    d = mujoco.MjData(weld.model)
    d.qpos[:] = weld.stand_qpos
    for r in LEG_ROLES:
        d.qpos[weld.qpos_adr(r)] = neutral.angles[r][0]
    mujoco.mj_forward(weld.model, d)
    site = mujoco.mj_name2id(weld.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    pelvis = mujoco.mj_name2id(weld.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    assert d.site_xpos[site][0] - d.xpos[pelvis][0] == pytest.approx(
        -foot_travel(GaitParameters()) / 2, abs=2e-3
    )


def test_legs_are_half_a_cycle_apart(neutral):
    n = len(neutral.phases)
    assert np.allclose(
        neutral.angles[JointRole.HIP_PITCH_L],
        np.roll(neutral.angles[JointRole.HIP_PITCH_R], n // 2),
        atol=2e-3,
    )


def test_in_place_cycle_lifts_the_feet_without_travel(weld):
    c = solve_leg_cycle(GaitParameters(step_length=0.0), weld)
    assert c.report.feasible
    assert np.ptp(c.angles[JointRole.KNEE_L]) > 0.2
