"""Round trip: parameters -> G1 kinematic tier -> landmarks -> extractor -> parameters.

Module-scoped: one robot, one trajectory and one recorded clip per preset (neutral, sad, angry,
happy), since ``generate`` costs a few tenths of a second.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from emotive_gait.emotion.model import PRESET_PARAMS
from emotive_gait.gait.footsteps import Side, foot_point
from emotive_gait.gait.generator import GaitTrajectory, generate
from emotive_gait.gait.params import GaitParameters
from emotive_gait.mocap.extract import ClipSummary, summarise
from emotive_gait.robot.loader import RobotModel, load_robot
from emotive_gait.robot.roles import JointRole
from mocap.g1_landmarks import G1_LANDMARKS, record_clip

PRESETS = ("neutral", "sad", "angry", "happy")
# Absolute on the robot, so compared as deltas against neutral.
POSTURE = ("trunk_pitch", "neck_pitch", "elbow_flexion", "arm_abduction")


@dataclass(frozen=True)
class Recorded:
    params: GaitParameters
    traj: GaitTrajectory
    summary: ClipSummary


class Recordings(dict):
    robot: RobotModel


@pytest.fixture(scope="module")
def recorded() -> Recordings:
    out = Recordings()
    out.robot = load_robot()
    for name in PRESETS:
        params = PRESET_PARAMS[name]
        traj = generate(params, out.robot)
        clip = record_clip(out.robot, traj, duration=8.0, fps=120.0)
        out[name] = Recorded(params, traj, summarise(clip, G1_LANDMARKS))
    return out


def rule_double_support(params: GaitParameters, samples: int = 4000) -> float:
    """Double support that the position-extremum rule reads on the commanded foot path.

    ``gait.footsteps.foot_point`` gives the foot's x in the pelvis frame (the coordinate the
    rule uses). Its swing starts from the stance slope and overshoots +D/2 by about 11 %, so
    the maximum (strike) comes before the swing ends and the minimum (toe-off) after it starts:
    the rule's stance is longer than ``1 - swing_fraction``. Double support per half-cycle
    step is then ``2 * stance - 1``.
    """
    x = np.array([foot_point(params, k / samples, Side.LEFT).x for k in range(samples)])
    stance = ((int(np.argmin(x)) - int(np.argmax(x))) % samples) / samples
    return 2.0 * stance - 1.0


def half_ptp_shoulder_pitch(traj: GaitTrajectory) -> float:
    angles = [traj.angles_at(k / 240)[JointRole.SHOULDER_PITCH_L] for k in range(240)]
    return (max(angles) - min(angles)) / 2


@pytest.mark.parametrize("name", PRESETS)
def test_extractor_recovers_the_parameters_that_generated_the_motion(recorded, name):
    p, traj, s = recorded[name].params, recorded[name].traj, recorded[name].summary
    m = s.mean
    assert s.n_cycles >= 4
    assert m.cadence == pytest.approx(p.cadence, rel=0.02)
    assert m.step_length == pytest.approx(p.step_length, rel=0.02)
    assert m.step_height == pytest.approx(p.expansiveness * p.step_height, abs=0.003)
    assert m.com_bounce_amp == pytest.approx(p.com_bounce_amp, abs=0.003)
    assert m.lateral_sway_amp == pytest.approx(p.lateral_sway_amp, abs=0.003)
    # The extremum rule on the commanded swing, not the parameter: see rule_double_support.
    rule = rule_double_support(p)
    assert m.double_support_frac == pytest.approx(rule, abs=0.05), (
        f"rule {rule:.3f} vs parameter {p.double_support_frac:.3f}: "
        f"offset {rule - p.double_support_frac:+.3f}"
    )
    assert m.feet_spacing == pytest.approx(p.feet_spacing, abs=0.02)
    # What was rendered, after the hip-normalised drive.
    assert m.arm_swing_amp == pytest.approx(half_ptp_shoulder_pitch(traj), abs=0.02)
    assert m.elbow_swing_ratio == pytest.approx(p.elbow_swing_ratio, abs=0.05)
    assert s.leg_length == pytest.approx(0.69, abs=0.01)
    assert m.pelvis_height == pytest.approx(0.68 * p.com_height_scale, abs=0.01)


# Known G1 mismatches, measured and left to a decision outside this task (see the Task 3 report).
# Strict, so a fix that makes a case pass fails here until its mark is removed.
_TRUNK_GAIN = pytest.mark.xfail(
    strict=True,
    reason="G1 pelvis body origin sits 4.4 cm below the waist-pitch pivot, so NECK - PELVIS "
    "shows 0.88 of waist pitch (0.3 rad reads 0.264); head-on-trunk inherits the difference",
)
_ELBOW_SIGN = pytest.mark.xfail(
    strict=True,
    reason="the generator writes elbow_flexion straight to the G1 elbow joint, which extends "
    "as it increases (geometric flexion = 1.385 - joint): every delta reads with its sign flipped",
)
_POSTURE_MARKS = {
    ("sad", "trunk_pitch"): _TRUNK_GAIN,
    ("happy", "trunk_pitch"): _TRUNK_GAIN,
    ("sad", "neck_pitch"): _TRUNK_GAIN,
    ("happy", "neck_pitch"): _TRUNK_GAIN,
    **{(name, "elbow_flexion"): _ELBOW_SIGN for name in ("sad", "angry", "happy")},
}


@pytest.mark.parametrize(
    ("name", "field"),
    [
        pytest.param(name, field, marks=_POSTURE_MARKS.get((name, field), ()))
        for field in POSTURE
        for name in ("sad", "angry", "happy")
    ],
)
def test_posture_deltas_against_neutral_match_the_parameter_deltas(recorded, name, field):
    # Geometric angles differ from joint angles by rig constants; deltas do not. This also pins
    # the sign conventions.
    neutral = recorded["neutral"]
    measured = getattr(recorded[name].summary.mean, field) - getattr(neutral.summary.mean, field)
    expected = getattr(recorded[name].params, field) - getattr(neutral.params, field)
    if field == "arm_abduction":
        p = recorded[name].params
        expected = p.expansiveness * p.arm_abduction - neutral.params.arm_abduction
    assert measured == pytest.approx(expected, abs=0.03), field


def test_neutral_trunk_and_head_read_level(recorded):
    m = recorded["neutral"].summary.mean
    assert m.trunk_pitch == pytest.approx(0.0, abs=0.03)
    assert m.neck_pitch == pytest.approx(0.0, abs=0.03)


@pytest.mark.parametrize("name", ["neutral", "sad"])
def test_noisy_sixty_hertz_clip_stays_within_margin(recorded, name):
    p, traj = recorded[name].params, recorded[name].traj
    clip = record_clip(recorded.robot, traj, duration=8.0, fps=60.0)
    rng = np.random.default_rng(0)
    noisy = replace(clip, positions=clip.positions + rng.normal(0, 0.003, clip.positions.shape))
    m = summarise(noisy, G1_LANDMARKS).mean
    assert m.cadence == pytest.approx(p.cadence, rel=0.03)
    assert m.step_length == pytest.approx(p.step_length, rel=0.03)
    assert m.step_height == pytest.approx(p.expansiveness * p.step_height, abs=0.006)
