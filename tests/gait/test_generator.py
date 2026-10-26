import math

import numpy as np
import pytest
from conftest import PHASE0_PRESETS

from emotive_gait.gait.footsteps import swing_fraction
from emotive_gait.gait.generator import generate
from emotive_gait.gait.ik import N_SAMPLES
from emotive_gait.gait.params import GaitParameters
from emotive_gait.gait.spline import PeriodicCurve
from emotive_gait.robot.loader import load_robot
from emotive_gait.robot.roles import JointRole


def test_legs_are_antiphase_and_arms_contralateral():
    """Solved hip curves are not zero-mean sinusoids, so antiphase is a half-cycle shift.

    Contralateral (each arm follows the opposite hip) is ``test_arms_follow_the_opposite_hip``.
    """
    t = generate(GaitParameters(), load_robot())
    for ph in np.linspace(0.0, 0.5, 16, endpoint=False):
        a, b = t.angles_at(float(ph)), t.angles_at(float(ph) + 0.5)
        assert a[JointRole.HIP_PITCH_L] == pytest.approx(b[JointRole.HIP_PITCH_R], abs=1e-3)
        assert a[JointRole.SHOULDER_PITCH_L] == pytest.approx(
            b[JointRole.SHOULDER_PITCH_R], abs=1e-3
        )


def test_cycle_is_continuous_across_the_phase_wrap():
    t = generate(GaitParameters(), load_robot())
    for role, v in t.angles_at(1.0 - 1e-6).items():
        assert v == pytest.approx(t.angles_at(0.0)[role], abs=1e-4)


def test_cycle_is_continuous_at_the_stance_to_swing_handoff():
    """The easier boundary to get wrong than the phase wrap."""
    t = generate(GaitParameters(), load_robot())
    sf = swing_fraction(GaitParameters())
    for ph in (sf, 0.5 + sf, 0.0, 0.5):
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

    def amp(t, role):
        v = [t.angles_at(p)[role] for p in np.linspace(0, 1, 64)]
        return max(v) - min(v)

    assert amp(big, JointRole.SHOULDER_PITCH_L) > amp(base, JointRole.SHOULDER_PITCH_L)
    assert generate(GaitParameters(expansiveness=1.5), r).params.step_length == (
        GaitParameters().step_length
    )


def test_offsets_are_additive_and_overrides_replace():
    r, p = load_robot(), GaitParameters()
    flat = PeriodicCurve.from_knots([0.0, 0.5], [0.2, 0.2])
    base = generate(p, r).angles_at(0.3)[JointRole.ELBOW_L]
    assert generate(p, r, offsets={JointRole.ELBOW_L: flat}).angles_at(0.3)[
        JointRole.ELBOW_L
    ] == pytest.approx(base + 0.2)
    assert generate(p, r, overrides={JointRole.ELBOW_L: flat}).angles_at(0.3)[
        JointRole.ELBOW_L
    ] == pytest.approx(0.2)


def test_expansiveness_does_not_rescale_hand_authored_offsets():
    r = load_robot()
    flat = PeriodicCurve.from_knots([0.0, 0.5], [0.2, 0.2])

    def d(e):
        with_offset = generate(
            GaitParameters(expansiveness=e), r, offsets={JointRole.ELBOW_L: flat}
        )
        without = generate(GaitParameters(expansiveness=e), r)
        return (
            with_offset.angles_at(0.3)[JointRole.ELBOW_L]
            - without.angles_at(0.3)[JointRole.ELBOW_L]
        )

    assert d(1.0) == pytest.approx(d(1.5))


def test_absent_role_is_skipped_not_raised():
    r = load_robot(with_neck=False)
    assert JointRole.NECK_PITCH not in generate(GaitParameters(neck_pitch=0.5), r).angles_at(0.0)


@pytest.mark.parametrize(
    "params",
    [
        GaitParameters(trunk_pitch=3.0),
        GaitParameters(trunk_pitch=-3.0),
        GaitParameters(arm_swing_amp=9.0),
        GaitParameters(step_length=0.75),
        GaitParameters(expansiveness=1.5, arm_swing_amp=3.0),
    ],
)
def test_extreme_parameters_stay_inside_joint_limits(params):
    r = load_robot()
    t = generate(params, r)
    for ph in np.linspace(0, 1, 64):
        for role, v in t.angles_at(float(ph)).items():
            lo, hi = r.joint_range(role)
            assert lo - 1e-9 <= v <= hi + 1e-9, f"{role}={v} outside [{lo},{hi}] at phase {ph}"


def test_qpos_and_ctrl_use_robot_layout_without_mutating_stand_pose():
    r = load_robot()
    stand = r.stand_qpos.copy()
    t = generate(GaitParameters(), r)
    q = t.qpos_at(r, 0.25)
    c = t.ctrl_at(r, 0.25)
    a = t.angles_at(0.25)
    assert np.array_equal(r.stand_qpos, stand)
    assert q.shape == stand.shape
    assert c.shape == (r.model.nu,)
    for role, v in a.items():
        assert q[r.qpos_adr(role)] == pytest.approx(v)
        assert c[r.ctrl_index(role)] == pytest.approx(v)


def _ptp(r, role, **kw):
    t = generate(GaitParameters(**kw), r)
    v = [t.angles_at(float(p))[role] for p in np.linspace(0, 1, 32, endpoint=False)]
    return max(v) - min(v)


def _elbow_ptp(r, **kw):
    return _ptp(r, JointRole.ELBOW_L, **kw)


def test_elbow_swing_ratio_zero_holds_the_elbow_constant():
    assert _elbow_ptp(load_robot(), elbow_swing_ratio=0.0) < 1e-9


def test_elbow_swing_ratio_scales_elbow_oscillation():
    r = load_robot()
    kw = {"arm_swing_amp": 0.4, "elbow_flexion": 0.8}
    p0, p5, p1 = (_elbow_ptp(r, elbow_swing_ratio=x, **kw) for x in (0.0, 0.5, 1.0))
    # Elbow swing / shoulder swing = elbow_swing_ratio. The shoulder peak-to-peak is not
    # 2 * arm_swing_amp: the solved hip curve it follows is asymmetric about its mean.
    shoulder = _ptp(r, JointRole.SHOULDER_PITCH_L, elbow_swing_ratio=1.0, **kw)
    assert p1 == pytest.approx(shoulder, abs=1e-3)
    assert p1 > p5 > p0


def test_expansiveness_scales_elbow_swing_too():
    r = load_robot()
    assert _elbow_ptp(r, elbow_swing_ratio=1.0, expansiveness=1.4) > _elbow_ptp(
        r, elbow_swing_ratio=1.0, expansiveness=1.0
    )
    assert generate(GaitParameters(expansiveness=1.5), r).params.step_length == (
        GaitParameters().step_length
    )


def test_elbow_oscillates_in_phase_with_its_own_shoulder():
    r = load_robot()
    p = GaitParameters(elbow_swing_ratio=1.0)
    t = generate(p, r)
    stand = r.stand_qpos[r.qpos_adr(JointRole.SHOULDER_PITCH_L)]
    for ph in (0.25, 0.75):
        a = t.angles_at(ph)
        assert (a[JointRole.ELBOW_L] - p.elbow_flexion) * (
            a[JointRole.SHOULDER_PITCH_L] - stand
        ) > 0


def _flat(v):
    return PeriodicCurve.from_knots([0.0, 0.5], [v, v])


def test_trajectories_with_different_overrides_are_not_equal():
    r, p = load_robot(), GaitParameters()
    flat = _flat(0.2)
    assert generate(p, r) != generate(p, r, overrides={JointRole.ELBOW_L: flat})
    assert generate(p, r) != generate(p, r, offsets={JointRole.ELBOW_L: flat})


def test_identical_generation_is_equal_and_hash_equal():
    r, p = load_robot(), GaitParameters()
    a = generate(p, r, offsets={JointRole.ELBOW_L: _flat(0.2)})
    b = generate(p, r, offsets={JointRole.ELBOW_L: _flat(0.2)})
    assert a == b
    assert hash(a) == hash(b)


def test_trajectory_captured_state_is_immutable():
    t = generate(GaitParameters(), load_robot(), offsets={JointRole.ELBOW_L: _flat(0.2)})
    for name in ("_ranges", "_stand", "_signs", "_offsets", "_overrides"):
        assert isinstance(getattr(t, name), tuple)


def test_override_past_range_is_still_clamped():
    t = generate(GaitParameters(), load_robot(), overrides={JointRole.WAIST_PITCH: _flat(2.0)})
    assert t.angles_at(0.3)[JointRole.WAIST_PITCH] == pytest.approx(0.52)


def test_offset_past_range_is_still_clamped():
    t = generate(GaitParameters(), load_robot(), offsets={JointRole.WAIST_PITCH: _flat(2.0)})
    assert t.angles_at(0.3)[JointRole.WAIST_PITCH] == pytest.approx(0.52)


def test_every_preset_is_feasible():
    from emotive_gait.emotion.evidence import Emotion
    from emotive_gait.emotion.model import PRESET_PARAMS, modulate
    from emotive_gait.robot.profiles.g1 import G1_PROFILE_WITH_NECK

    r = load_robot()
    cases = dict(PRESET_PARAMS)
    cases |= {f"phase0_{k}": v for k, v in PHASE0_PRESETS.items()}
    cases |= {e.value: modulate(GaitParameters(), {e: 1.0}, G1_PROFILE_WITH_NECK) for e in Emotion}
    # Analytic presets only. The learned presets are flagged and rendered, not gated (spec 4.5,
    # decision 2026-10-26): their residuals are documented in docs/reports/phase-4-5-fit-report.md
    # and pinned by tests/emotion/test_artefact_committed.py.
    for name, p in cases.items():
        rep = generate(p, r).feasibility
        assert rep.feasible, f"{name}: {rep}"


def test_infeasible_stride_is_flagged_not_rescaled():
    r = load_robot()
    t = generate(GaitParameters(step_length=0.75), r)
    assert not t.feasibility.feasible and t.params.step_length == 0.75


def test_arms_follow_the_opposite_hip():
    t = generate(GaitParameters(), load_robot())
    # The solver's own sample grid: the peak deviation is taken over the solved samples, and a
    # coarser grid misses a sharp hip peak by a few tenths of a percent.
    ph = np.arange(N_SAMPLES) / N_SAMPLES
    hip_r = np.array([t.angles_at(p)[JointRole.HIP_PITCH_R] for p in ph])
    hip_r -= hip_r.mean()
    arm_l = np.array([t.angles_at(p)[JointRole.SHOULDER_PITCH_L] for p in ph])
    arm_l -= arm_l.mean()
    assert np.corrcoef(hip_r, arm_l)[0, 1] > 0.99
    assert np.abs(arm_l).max() == pytest.approx(GaitParameters().arm_swing_amp, rel=1e-3)


def test_static_gait_has_no_arm_swing():
    """The hip curve of a static gait is solver noise; it must not drive a full arm swing."""
    static = GaitParameters(
        step_length=0.0, step_height=0.0, com_bounce_amp=0.0, lateral_sway_amp=0.0
    )
    t = generate(static, load_robot())
    v = [
        t.angles_at(float(p))[JointRole.SHOULDER_PITCH_L]
        for p in np.linspace(0, 1, 32, endpoint=False)
    ]
    assert all(math.isfinite(x) for x in v)
    assert max(v) - min(v) < 1e-3


def test_in_place_gait_still_swings_the_arms():
    t = generate(GaitParameters(step_length=0.0), load_robot())
    v = [
        t.angles_at(float(p))[JointRole.SHOULDER_PITCH_L]
        for p in np.linspace(0, 1, 32, endpoint=False)
    ]
    assert max(v) - min(v) > 0.1 and all(math.isfinite(x) for x in v)


def test_root_path_carries_crouch_bounce_and_forward_travel():
    r = load_robot()
    t = generate(GaitParameters(com_height_scale=0.9, com_bounce_amp=0.02), r)
    assert t.root_offset_at(0.0)[2] == pytest.approx(0.68 * 0.9 + 0.02)
    assert t.root_x_at(2.0) == pytest.approx(2.0 * t.params.speed)


def test_every_preset_keeps_full_arm_swing_under_every_projection():
    from emotive_gait.emotion.model import PRESET_PARAMS
    from emotive_gait.robot.root import RootMode
    from emotive_gait.sim.physical import project_for_root

    r = load_robot()
    for name, p in PRESET_PARAMS.items():
        for mode in RootMode:
            t = generate(project_for_root(p, mode), r)
            ph = np.arange(N_SAMPLES) / N_SAMPLES
            sh = np.array([t.angles_at(float(x))[JointRole.SHOULDER_PITCH_L] for x in ph])
            peak = float(np.abs(sh - sh.mean()).max())
            print(f"{name} {mode.value}: shoulder peak {peak:.4f}")
            want = p.arm_swing_amp * p.expansiveness
            assert peak == pytest.approx(want, abs=1e-3), f"{name} {mode.value}"
