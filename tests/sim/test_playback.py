import math

import mujoco
import numpy as np
import pytest

from emotive_gait.gait.footsteps import Side, foot_point, phase_at
from emotive_gait.gait.generator import generate
from emotive_gait.gait.params import GaitParameters
from emotive_gait.robot.loader import load_robot
from emotive_gait.robot.roles import JointRole
from emotive_gait.robot.root import RootMode, to_root_mode
from emotive_gait.sim import physical
from emotive_gait.sim.kinematic import play_kinematic
from emotive_gait.sim.physical import project_for_root, simulate, summarize


def test_kinematic_playback_never_integrates_physics():
    r = load_robot()
    t = generate(GaitParameters(), r)
    frames = list(play_kinematic(r, t, duration=1.0, fps=30))
    assert len(frames) == 30
    assert all(f.time == 0.0 for f in frames)  # mj_forward does not advance time


def test_kinematic_playback_reaches_commanded_angles_exactly():
    r = load_robot()
    t = generate(GaitParameters(trunk_pitch=0.4), r)
    f = next(iter(play_kinematic(r, t, duration=0.1, fps=10)))
    assert f.qpos[r.qpos_adr(JointRole.WAIST_PITCH)] == pytest.approx(0.4, abs=1e-9)


def test_sim_physical_re_exports_the_root_api():
    """The brief documents ``sim.physical`` as an import path; robot.root is the home."""
    assert physical.RootMode is RootMode
    assert physical.to_root_mode is to_root_mode


def test_z_rail_root_has_no_quaternion():
    r = load_robot(root=RootMode.Z_RAIL)
    assert r.model.nq == r.model.nv


def test_weld_root_has_no_root_dof():
    r = load_robot(root=RootMode.WELD)
    assert r.model.nq == r.model.nv == load_robot(root=RootMode.Z_RAIL).model.nq - 1


def test_root_mode_keeps_the_offscreen_framebuffer():
    r = load_robot(root=RootMode.Z_RAIL)
    assert (r.model.vis.global_.offwidth, r.model.vis.global_.offheight) == (960, 720)


# Local literals, not emotion.model.PRESET_PARAMS: that is Task 14 and this is Task 10.
# These mirror the Phase 0 spike's crude presets, minus the retired knee bias.
_PHASE0_PRESETS = {
    "neutral": GaitParameters(
        cadence=1.8,
        arm_swing_amp=0.30,
        elbow_flexion=1.28,
        trunk_pitch=0.00,
        neck_pitch=0.00,
    ),
    "sad": GaitParameters(
        cadence=1.0,
        arm_swing_amp=0.08,
        elbow_flexion=1.05,
        trunk_pitch=0.32,
        neck_pitch=0.70,
    ),
    "angry": GaitParameters(
        cadence=2.7,
        arm_swing_amp=0.60,
        elbow_flexion=1.85,
        trunk_pitch=0.18,
        neck_pitch=-0.10,
    ),
    "happy": GaitParameters(
        cadence=2.3,
        arm_swing_amp=0.62,
        elbow_flexion=0.95,
        trunk_pitch=-0.12,
        neck_pitch=-0.35,
    ),
}


@pytest.mark.parametrize("preset", list(_PHASE0_PRESETS))
def test_z_rail_survives_six_seconds_on_every_preset(preset):
    """Measured in Phase 0. The free root fell at 0.74-2.25 s; the Z-rail did not.

    On the rail the pelvis cannot rotate, so ``not fell`` comes from the constrained-root
    criteria (rail stop, foot contact, pelvis drop). The IK places the feet so that, with the
    soles on the floor, the pelvis sits at ``nominal * com_height_scale + dz``: the commanded
    pelvis path is known, so the instrument measures the deviation from it (tracking error),
    which is what a raw bob bound measured while the bounce was inert. Its peak-to-peak bounds
    uncommanded oscillation and its mean bounds servo sag under gravity. Everything is measured
    after the 0.5 s startup ramp, so the settle from the standing pose is not counted.

    Calibration (Phase 4 Task 5, nominal height 0.68 m): post-ramp peak-to-peak deviation was
    neutral 2.46, sad 2.80, angry 5.01 and happy 3.60 cm; the mean (sag) was about -1 cm on
    every preset; angry had no foot contact on 1.13% of steps, the others on none. The
    angry/happy excess over 3 cm survived a 0-120 ms lag shift of the commanded path, so it is
    real servo-compliant leg bounce at 2.3-2.7 steps/s, not measurement lag. The 0.06 m bound
    is the spec's bob bound applied to the deviation.
    """
    r = load_robot(root=RootMode.Z_RAIL)
    res = simulate(r, generate(_PHASE0_PRESETS[preset], r), duration=6.0)
    assert not res.fell, f"{preset} fell at {res.fell_at}: {res.fail_reason}"
    assert res.fell_at is None and res.fail_reason is None
    n = len(res.pelvis_z)
    assert n == round(6.0 / r.model.opt.timestep)
    assert res.uprightness.shape == (n,)
    assert res.cop.shape == (n, 2)
    post = math.ceil(0.5 / r.model.opt.timestep)
    played = generate(res.played_params, r)  # the same projection simulate played
    # pelvis_z[k] is sampled after mj_step, i.e. at time (k + 1) * dt.
    dt = r.model.opt.timestep
    expected = np.array(
        [
            played.root_offset_at(phase_at(res.played_params, (k + 1) * dt))[2]
            for k in range(post, n)
        ]
    )
    dev = res.pelvis_z[post:] - expected
    assert np.ptp(dev) < 0.06, f"{preset} uncommanded pelvis oscillation {np.ptp(dev):.4f} m"
    assert abs(dev.mean()) < 0.05, f"{preset} pelvis sag {dev.mean():.4f} m"
    no_contact = np.isnan(res.cop[post:, 0]).mean()
    assert no_contact < 0.25, f"{preset} had no foot contact on {no_contact:.1%} of steps"


def test_z_rail_detects_collapse():
    """An absurd gait drives the pelvis down the rail with the feet off the floor."""
    r = load_robot(root=RootMode.Z_RAIL)
    absurd = GaitParameters(cadence=6, step_height=0.5, trunk_pitch=3, expansiveness=2)
    res = simulate(r, generate(absurd, r), duration=6.0)
    assert res.fell
    assert res.fail_reason is not None
    assert 0.0 < res.fell_at < 6.0


def test_ramp_avoids_the_startup_yank():
    r = load_robot(root=RootMode.Z_RAIL)
    t = generate(GaitParameters(com_height_scale=0.9), r)
    assert not simulate(r, t, duration=1.0, ramp=0.5).fell


def _post_ramp_mean_pelvis_z(r, res, ramp=0.5):
    return float(res.pelvis_z[math.ceil(ramp / r.model.opt.timestep) :].mean())


def test_crouch_and_neutral_do_not_fall_on_the_z_rail():
    r = load_robot(root=RootMode.Z_RAIL)
    for scale in (0.9, 1.0):
        res = simulate(r, generate(GaitParameters(com_height_scale=scale), r), duration=2.0)
        assert not res.fell, f"com_height_scale={scale}: {res.fail_reason}"


def test_crouch_lowers_the_pelvis_at_least_3cm_below_neutral():
    r = load_robot(root=RootMode.Z_RAIL)
    crouch, neutral = (
        simulate(r, generate(GaitParameters(com_height_scale=s), r), duration=2.0)
        for s in (0.9, 1.0)
    )
    assert _post_ramp_mean_pelvis_z(r, crouch) <= _post_ramp_mean_pelvis_z(r, neutral) - 0.03


def test_free_root_open_loop_is_flagged_as_a_fall_and_stops_early():
    """The reason the Z-rail exists: with no feedback a free base tips over."""
    r = load_robot(root=RootMode.FREE)
    res = simulate(r, generate(_PHASE0_PRESETS["angry"], r), duration=6.0)
    assert res.fell
    assert 0.0 < res.fell_at < 6.0
    assert len(res.pelvis_z) < round(6.0 / r.model.opt.timestep)
    assert res.uprightness[-1] < 0.6
    assert res.fail_reason is not None and res.fail_reason.startswith("uprightness")


def test_z_rail_plays_an_in_place_projection():
    r = load_robot(root=RootMode.Z_RAIL)
    res = simulate(
        r, generate(GaitParameters(step_length=0.3, lateral_sway_amp=0.02), r), duration=0.2
    )
    assert res.played_params.step_length == 0.0 and res.played_params.lateral_sway_amp == 0.0
    assert res.played_params.com_bounce_amp == GaitParameters().com_bounce_amp


def test_weld_projection_freezes_everything_the_pelvis_would_move_for():
    p = project_for_root(
        GaitParameters(
            step_length=0.3, com_bounce_amp=0.03, com_height_scale=0.9, lateral_sway_amp=0.02
        ),
        RootMode.WELD,
    )
    assert (p.step_length, p.com_bounce_amp, p.lateral_sway_amp, p.com_height_scale) == (
        0.0,
        0.0,
        0.0,
        1.0,
    )


def test_free_root_is_not_projected():
    p = GaitParameters(step_length=0.3)
    assert project_for_root(p, RootMode.FREE) == p


def test_kinematic_stance_foot_is_stationary_in_the_world():
    r = load_robot()
    p = GaitParameters()
    t = generate(p, r)
    site = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    fps = 200
    xs = []
    for k, f in enumerate(play_kinematic(r, t, duration=2 / p.cadence, fps=fps)):
        if foot_point(p, phase_at(p, k / fps), Side.LEFT).in_stance:
            xs.append(f.site_xpos[site][:2].copy())
    xs = np.array(xs)
    assert len(xs) > 50 and np.ptp(xs, axis=0).max() < 1e-3


def test_kinematic_pelvis_advances_two_step_lengths_per_cycle():
    r = load_robot()
    p = GaitParameters(cadence=1.5)
    t = generate(p, r)
    # 2 / cadence s at 30 fps is exactly 40 frames; the yielded MjData is reused, so read inside.
    xs = [float(f.qpos[0]) for f in play_kinematic(r, t, duration=2 / p.cadence, fps=30)]
    assert len(xs) == 40
    assert xs[-1] - xs[0] == pytest.approx(2 * p.step_length * (39 / 40), abs=1e-6)


def test_kinematic_crouch_lowers_the_world_pelvis():
    r = load_robot()

    def z(scale):
        traj = generate(GaitParameters(com_height_scale=scale), r)
        return next(iter(play_kinematic(r, traj, duration=0.1, fps=10))).qpos[2]

    # the G1 nominal pelvis height is 0.68 m, so a 10% crouch is 0.068 m
    assert z(1.0) - z(0.9) == pytest.approx(0.068, abs=1e-6)


def test_kinematic_playback_rejects_a_constrained_root():
    r = load_robot(root=RootMode.Z_RAIL)
    with pytest.raises(ValueError, match="FREE"):
        next(iter(play_kinematic(r, generate(GaitParameters(), r), duration=0.1, fps=10)))


def test_xz_rail_has_two_slide_dofs():
    r = load_robot(root=RootMode.XZ_RAIL)
    assert r.model.nq == r.model.nv == load_robot(root=RootMode.Z_RAIL).model.nq + 1


def test_xz_rail_forward_stride_loses_foot_contact_open_loop():
    """Measured negative result: neutral on XZ_RAIL falls at 0.540 s on lost foot contact.

    The first no-contact step is at 0.074 s and 25.2 % of the final 0.5 s window has no foot on
    the floor. The servoed-rail variant (kp 5000-100000, ramped command) failed the same way at
    0.52-1.12 s, so forward walking stays a kinematic-tier result and the physics tier is in place.
    """
    r = load_robot(root=RootMode.XZ_RAIL)
    p = GaitParameters()
    res = simulate(r, generate(p, r), duration=4.0)
    assert res.fell and res.fail_reason.startswith("foot contact lost")
    assert 0.0 < res.fell_at < 1.0
    assert res.played_params.step_length == p.step_length
    assert res.played_params.lateral_sway_amp == 0.0
    # The prescribed x tracks exactly up to the fall.
    assert res.root_x[-1] == pytest.approx(p.speed * res.fell_at, abs=0.02)


def test_xz_rail_reports_a_fall_on_an_absurd_gait():
    r = load_robot(root=RootMode.XZ_RAIL)
    res = simulate(r, generate(GaitParameters(cadence=8.0, step_length=1.0), r), duration=3.0)
    assert res.fell and res.fail_reason.startswith(("foot contact lost", "rail stop"))


def test_report_on_the_z_rail_has_finite_numbers_and_no_slip_claim():
    r = load_robot(root=RootMode.Z_RAIL)
    res = simulate(r, generate(GaitParameters(), r), duration=2.0)
    rep = summarize(res)
    assert 0.0 <= rep.zmp_outside_frac <= 1.0 and math.isfinite(rep.zmp_mean_margin_m)
    assert all(s < 0.05 for s in rep.foot_slip_rms_m_s)  # in place: feet do not slide
    assert 0.0 <= rep.pelvis_tracking_ptp_m <= rep.pelvis_bob_m
    assert not rep.pelvis_held


def test_weld_report_is_labelled_pelvis_held():
    r = load_robot(root=RootMode.WELD)
    res = simulate(r, generate(GaitParameters(), r), duration=1.0)
    assert summarize(res).pelvis_held


def test_summarize_with_no_post_ramp_steps_is_nan_and_does_not_raise():
    r = load_robot(root=RootMode.Z_RAIL)
    res = simulate(r, generate(GaitParameters(), r), duration=0.2)
    rep = summarize(res)
    assert math.isnan(rep.zmp_outside_frac) and math.isnan(rep.pelvis_bob_m)


def test_summarize_accepts_explicit_ramp_and_timestep_overrides():
    r = load_robot(root=RootMode.Z_RAIL)
    res = simulate(r, generate(GaitParameters(), r), duration=0.6)
    assert (res.ramp, res.timestep) == (0.5, r.model.opt.timestep)
    default = summarize(res)
    late = summarize(res, ramp=10.0, timestep=r.model.opt.timestep)
    assert math.isfinite(default.pelvis_bob_m) and math.isnan(late.pelvis_bob_m)
