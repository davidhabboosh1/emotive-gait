import pytest

from emotive_gait.gait.generator import generate
from emotive_gait.gait.params import GaitParameters
from emotive_gait.robot.loader import load_robot
from emotive_gait.robot.roles import JointRole
from emotive_gait.sim.kinematic import play_kinematic
from emotive_gait.sim.physical import RootMode, simulate


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


def test_z_rail_root_has_no_quaternion():
    r = load_robot(root=RootMode.Z_RAIL)
    assert r.model.nq == r.model.nv


def test_weld_root_has_no_root_dof():
    r = load_robot(root=RootMode.WELD)
    assert r.model.nq == r.model.nv == load_robot(root=RootMode.Z_RAIL).model.nq - 1


def test_non_free_root_requires_the_neck_path():
    with pytest.raises(ValueError, match="with_neck"):
        load_robot(with_neck=False, root=RootMode.Z_RAIL)


def test_root_mode_keeps_the_offscreen_framebuffer():
    r = load_robot(root=RootMode.Z_RAIL)
    assert (r.model.vis.global_.offwidth, r.model.vis.global_.offheight) == (960, 720)


# Local literals, not emotion.model.PRESET_PARAMS: that is Task 14 and this is Task 10.
# These mirror the Phase 0 spike's crude presets; knee_flexion_bias is relative to the
# generator's 0.25 baseline knee amplitude.
_PHASE0_PRESETS = {
    "neutral": GaitParameters(cadence=1.8, arm_swing_amp=0.30, elbow_flexion=1.28,
                              trunk_pitch=0.00, neck_pitch=0.00, knee_flexion_bias=0.0),
    "sad":     GaitParameters(cadence=1.0, arm_swing_amp=0.08, elbow_flexion=1.05,
                              trunk_pitch=0.32, neck_pitch=0.70, knee_flexion_bias=-0.11),
    "angry":   GaitParameters(cadence=2.7, arm_swing_amp=0.60, elbow_flexion=1.85,
                              trunk_pitch=0.18, neck_pitch=-0.10, knee_flexion_bias=0.13),
    "happy":   GaitParameters(cadence=2.3, arm_swing_amp=0.62, elbow_flexion=0.95,
                              trunk_pitch=-0.12, neck_pitch=-0.35, knee_flexion_bias=0.11),
}


@pytest.mark.parametrize("preset", list(_PHASE0_PRESETS))
def test_z_rail_survives_six_seconds_on_every_preset(preset):
    """Measured in Phase 0. The free root fell at 0.74-2.25 s; the Z-rail did not."""
    r = load_robot(root=RootMode.Z_RAIL)
    res = simulate(r, generate(_PHASE0_PRESETS[preset], r), duration=6.0)
    assert not res.fell, f"{preset} fell at {res.fell_at}"
    assert res.fell_at is None
    n = len(res.pelvis_z)
    assert n == round(6.0 / r.model.opt.timestep)
    assert res.uprightness.shape == (n,)
    assert res.cop.shape == (n, 2)


def test_ramp_avoids_the_startup_yank():
    r = load_robot(root=RootMode.Z_RAIL)
    t = generate(GaitParameters(knee_flexion_bias=0.4), r)
    assert not simulate(r, t, duration=1.0, ramp=0.5).fell


def test_fall_is_detected_by_uprightness_not_height():
    """A crouch lowers the pelvis without falling; only tipping is a fall."""
    r = load_robot(root=RootMode.Z_RAIL)
    res = simulate(r, generate(GaitParameters(knee_flexion_bias=0.5), r), duration=2.0)
    assert not res.fell
    assert res.pelvis_z.min() < res.pelvis_z[0]


def test_free_root_open_loop_is_flagged_as_a_fall_and_stops_early():
    """The reason the Z-rail exists: with no feedback a free base tips over."""
    r = load_robot(root=RootMode.FREE)
    res = simulate(r, generate(_PHASE0_PRESETS["angry"], r), duration=6.0)
    assert res.fell
    assert 0.0 < res.fell_at < 6.0
    assert len(res.pelvis_z) < round(6.0 / r.model.opt.timestep)
    assert res.uprightness[-1] < 0.6
