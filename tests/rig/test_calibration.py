"""Reference pose, role bindings by probe, and the conventions on the driven chain."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from types import MappingProxyType

import mujoco
import numpy as np
import pytest

from emotive_gait.gait.generator import generate
from emotive_gait.gait.ik import LEG_ROLES
from emotive_gait.gait.params import GaitParameters
from emotive_gait.mocap.readers.bvh import (
    _AXIS,
    BvhFile,
    BvhJoint,
    euler_from_matrix,
    read_bvh_file,
    rotation_about,
)
from emotive_gait.mocap.skeleton import Landmark
from emotive_gait.mocap.writers.bvh import write_bvh
from emotive_gait.rig.calibration import (
    TWO_CHANNEL_WARN_RAD,
    ReferencePose,
    chordal_mean,
    fit_two_channel,
    landmark_site_names,
    posture_map,
    reference_pose,
)
from emotive_gait.rig.compile import channel_angles, compile_driven, compile_full, z_up_matrix
from emotive_gait.rig.mapping import RigError, bind_roles, map_landmarks
from emotive_gait.rig.profile import rig_profile, rig_robot
from emotive_gait.rig.ranges import RANGES
from emotive_gait.rig.synthesis import load_rig
from emotive_gait.robot.conventions import ROLE_CONVENTIONS, convention_holds
from emotive_gait.robot.profiles.g1 import G1_PROFILE_WITH_NECK
from emotive_gait.robot.roles import POSTURE_PARAMETERS, JointRole
from rig.driven import (
    X,
    Y,
    Z,
    calibrated_rig,
    driven_rig,
    hinges_from_posture,
    landmark_points,
    probe_driven,
    spec_id,
    static_posture,
)
from rig.rigs import (
    FAMILIES,
    FAMILY_VARIANTS,
    RigSpec,
    build_rig,
    toe_rest_height_m,
    walk_and_clip,
    walk_file,
)


def test_reference_pose_heading_is_the_walk_direction_and_floor_is_the_toe_stance_height(
    tmp_path,
):
    f, clip = walk_and_clip(RigSpec(), tmp_path)
    lm = map_landmarks(f, clip).landmarks
    ref = reference_pose(f, clip, lm)
    assert np.allclose(ref.heading, [1.0, 0.0, 0.0], atol=0.05)
    assert abs(ref.floor_z - toe_rest_height_m(RigSpec())) < 0.01
    assert 0.7 < ref.pelvis_height < 1.0
    assert ref.local_rotations.shape == (len(f.joints), 3, 3)
    for r in ref.local_rotations:
        assert np.allclose(r.T @ r, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(r), 1.0)
    assert set(ref.foot_stance_rotations) == {"mixamorig:LeftFoot", "mixamorig:RightFoot"}
    left, right = ref.stance_frames
    assert left.dtype == bool and left.shape == (clip.frames(),) and left.any() and right.any()


def test_reference_pose_refuses_a_standing_clip():
    f = build_rig(RigSpec())
    f = f.with_motion(np.repeat(f.motion, 120, axis=0))
    clip = f.to_clip(end_sites=True)
    with pytest.raises(RigError, match="no usable cycle"):
        reference_pose(f, clip, map_landmarks(f, clip).landmarks)


@pytest.mark.parametrize("spec", FAMILY_VARIANTS, ids=spec_id)
def test_every_bound_role_satisfies_its_convention_on_the_driven_chain(tmp_path, spec):
    robot, mapping, _, driven, _ = driven_rig(spec, tmp_path)
    assert set(mapping.roles) >= set(LEG_ROLES)
    probed = 0
    for role in ROLE_CONVENTIONS:
        if role in mapping.roles:  # every bound row, the elbow included (its sign was chosen)
            before, after = probe_driven(robot, driven, mapping, role)
            assert convention_holds(role, before, after, forward=X, left=Y, up=Z), role
            probed += 1
    assert probed == len(ROLE_CONVENTIONS)  # every synthetic rig binds every convention row


def test_xyz_and_zyx_orders_bind_the_same_anatomical_hinges(tmp_path):
    hinges = {}
    for order in ("XYZ", "ZYX"):
        _, mapping, _, _, _ = driven_rig(replace(RigSpec(), order=order), tmp_path / order)
        assert mapping.roles[JointRole.KNEE_L].axis_purity > 0.95 and not mapping.skewed
        hinges[order] = {role: b.hinge for role, b in mapping.roles.items()}
    # The synthetic rig is y-up: lateral is file Z, forward file X, up file Y, whatever the order.
    for order in ("XYZ", "ZYX"):
        assert hinges[order][JointRole.KNEE_L] == "mixamorig:LeftLeg/Zrotation"
        assert hinges[order][JointRole.HIP_ROLL_R] == "mixamorig:RightUpLeg/Xrotation"
        assert hinges[order][JointRole.HIP_YAW_L] == "mixamorig:LeftUpLeg/Yrotation"
        assert hinges[order][JointRole.ANKLE_ROLL_L] == "mixamorig:LeftFoot/Xrotation"


def test_bound_hinges_get_the_committed_ranges(tmp_path):
    robot, mapping, _, _, _ = driven_rig(RigSpec(), tmp_path)
    for role, binding in mapping.roles.items():
        shift = binding.zero_offset
        if role.name.startswith(("KNEE", "ELBOW")):
            assert robot.joint_range(role) == pytest.approx(np.add(RANGES[role], shift))
        else:
            assert shift == 0.0 and robot.joint_range(role) == pytest.approx(RANGES[role])
    assert robot.profile.sign_for == {role: 1.0 for role in mapping.roles}
    assert robot.profile.elbow_swing_sign == -1.0
    assert np.all(robot.stand_qpos == 0.0) and robot.stand_qpos.shape == (robot.model.nq,)
    assert robot.model.nq == len(mapping.roles)  # undriven channels are not hinges


def _bound(spec: RigSpec, directory, file=None):
    f, clip = walk_and_clip(spec, directory)
    if file is not None:
        f = file(f)
        clip = f.to_clip(end_sites=True)
    landmarks = map_landmarks(f, clip)
    full = compile_full(f, pelvis=landmarks.landmarks[Landmark.PELVIS])
    return full, reference_pose(f, clip, landmarks.landmarks), landmarks


def test_a_role_override_binds_the_named_channel_as_user(tmp_path):
    full, ref, landmarks = _bound(RigSpec(), tmp_path)
    hinge = "mixamorig:Spine1/Zrotation"
    mapping = bind_roles(full, ref, landmarks, overrides={"waist_pitch": hinge})
    assert mapping.roles[JointRole.WAIST_PITCH].hinge == hinge
    assert any(d.what == "role:waist_pitch" and d.source == "user" for d in mapping.decisions)
    assert any(d.what == "waist_pitch" for d in mapping.decisions) is False  # landmark rows
    assert any(d.what == "role:knee_l" for d in mapping.decisions)
    assert any(d.what == "knee_l" for d in mapping.decisions)  # the landmark row, distinct
    with pytest.raises(RigError, match="NoSuchBone/Xrotation"):
        bind_roles(full, ref, landmarks, overrides={"waist_pitch": "NoSuchBone/Xrotation"})
    with pytest.raises(RigError, match="not_a_role"):
        bind_roles(full, ref, landmarks, overrides={"not_a_role": hinge})
    # Waist roll has no RANGES row: undriven on a rig, so an override cannot bind it.
    with pytest.raises(RigError, match="waist_roll"):
        bind_roles(full, ref, landmarks, overrides={"waist_roll": "mixamorig:Spine/Xrotation"})


def _without_channel(f: BvhFile, bone: str, channel: str) -> BvhFile:
    """``f`` with ``bone``'s ``channel`` removed (its motion column dropped)."""
    column = f.channel_index(bone, channel)
    joints = tuple(
        replace(j, channels=tuple(c for c in j.channels if c != channel)) if j.name == bone else j
        for j in f.joints
    )
    return replace(f, joints=joints, motion=np.delete(f.motion, column, axis=1))


def test_a_rig_without_hip_yaw_is_refused_naming_the_missing_leg_roles(tmp_path):
    bone = "mixamorig:LeftUpLeg"
    full, ref, landmarks = _bound(
        RigSpec(), tmp_path, file=lambda f: _without_channel(f, bone, "Yrotation")
    )
    with pytest.raises(RigError, match="hip_yaw_l"):
        bind_roles(full, ref, landmarks)


def test_a_tilted_reference_root_above_the_pelvis_is_absorbed_into_the_pelvis(tmp_path):
    spec = replace(RigSpec(), root_above_pelvis=True)
    f, _ = walk_and_clip(spec, tmp_path)
    motion = f.motion.copy()
    motion[:, f.channel_index("reference", "Zrotation")] = 6.0  # file Z is lateral (y-up)
    f = f.with_motion(motion)
    clip = f.to_clip(end_sites=True)
    ref = reference_pose(f, clip, map_landmarks(f, clip).landmarks)
    pelvis = [j.name for j in f.joints].index("mixamorig:Hips")
    assert np.allclose(ref.local_rotations[pelvis], rotation_about(2, np.array([6.0]))[0])


def _foot_pitch(ankle: np.ndarray, toe: np.ndarray) -> np.ndarray:
    """Toe-down pitch (rad) of the ankle-to-toe vector(s) against the horizontal."""
    d = np.atleast_2d(toe - ankle)
    return np.arctan2(-d[:, 2], np.hypot(d[:, 0], d[:, 1]))


@pytest.mark.parametrize("spec", [RigSpec(), replace(RigSpec(), toes=False)], ids=spec_id)
def test_the_driven_foot_at_zero_has_the_clip_stance_world_pitch(tmp_path, spec):
    _, mapping, ref, driven, _ = driven_rig(spec, tmp_path)
    _, clip = walk_and_clip(spec, tmp_path)  # the same clip, written again
    tracks = clip.landmark_positions(mapping.landmarks)
    data = mujoco.MjData(driven.model)
    mujoco.mj_forward(driven.model, data)
    for side, mask in zip("lr", ref.stance_frames, strict=True):
        ankle, toe = Landmark(f"ankle_{side}"), Landmark(f"toe_{side}")
        stance = _foot_pitch(tracks[ankle][mask], tracks[toe][mask]).mean()
        site = {
            lm: data.site_xpos[driven.model.site(driven.landmark_sites[mapping.landmarks[lm]]).id]
            for lm in (ankle, toe)
        }
        at_zero = _foot_pitch(site[ankle], site[toe])[0]
        assert abs(at_zero - stance) < 0.01, (side, at_zero, stance)


def _with_elbow(ref: ReferencePose, f: BvhFile, bone: str, flexion_deg: float) -> ReferencePose:
    """``ref`` with ``bone``'s reference turned by ``flexion_deg`` of anatomical flexion: about
    the lateral axis (z-up +y is file -Z on the y-up rig), swinging the hanging forearm forward."""
    p_file = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # z-up -> file axes (y-up)
    rotations = np.array(ref.local_rotations)
    i = [j.name for j in f.joints].index(bone)
    rotations[i] = p_file @ rotation_about(1, np.array([-flexion_deg]))[0] @ p_file.T
    return replace(ref, local_rotations=rotations)


def _signed_flexion(points: dict, side: str) -> float:
    """Elbow bend about the lateral axis (rad), positive with the forearm forward of the
    upper arm's line (flexion), negative hyperextended."""
    upper = points[f"elbow_{side}"] - points[f"shoulder_{side}"]
    fore = points[f"wrist_{side}"] - points[f"elbow_{side}"]
    return float(np.arctan2(-np.cross(upper, fore) @ Y, upper @ fore))


@pytest.mark.parametrize("flexion_deg", [-20.0, 20.0], ids=["hyperextended", "bent"])
def test_the_elbow_binds_anatomical_flexion_positive_whatever_the_reference_bend(
    tmp_path, flexion_deg
):
    full, ref, landmarks = _bound(RigSpec(), tmp_path)
    f = full.file
    for side in "lr":
        bone = landmarks.landmarks[Landmark(f"elbow_{side}")]
        ref = _with_elbow(ref, f, bone, flexion_deg)
    mapping = bind_roles(full, ref, landmarks)
    driven = compile_driven(f, mapping, ref)
    robot = rig_robot(driven, rig_profile("elbow", mapping, ref, posture_map={}))
    for side, role in (("l", JointRole.ELBOW_L), ("r", JointRole.ELBOW_R)):
        before, after = probe_driven(robot, driven, mapping, role)
        assert np.sign(_signed_flexion(before, side)) == np.sign(flexion_deg)
        assert _signed_flexion(after, side) > _signed_flexion(before, side), (side, flexion_deg)
        assert convention_holds(role, before, after, forward=X, left=Y, up=Z)


def test_a_bound_hinge_writes_back_as_its_own_euler_channel_through_a_three_axis_reference(
    tmp_path,
):
    """R_post != I: the hip's reference is Rz(10) Rx(15) Ry(20) in its ZXY order; turning the
    bound hinges moves exactly their own channels by sign * delta (the BVH Euler semantics)."""
    _, mapping, ref, _, full = driven_rig(RigSpec(), tmp_path)
    f = full.file
    names = [j.name for j in f.joints]
    hip = mapping.landmarks[Landmark.HIP_L]
    i = names.index(hip)
    channels = tuple(c for c in f.joints[i].channels if c.endswith("rotation"))
    assert channels == ("Zrotation", "Xrotation", "Yrotation")
    theta = np.array([10.0, 15.0, 20.0])
    rotations = np.array(ref.local_rotations)
    rotations[i] = (
        rotation_about(2, theta[:1])[0]
        @ rotation_about(0, theta[1:2])[0]
        @ rotation_about(1, theta[2:])[0]
    )
    driven = compile_driven(f, mapping, replace(ref, local_rotations=rotations))
    p = z_up_matrix(f.up_axis)
    data = mujoco.MjData(driven.model)
    parent = names[f.joints[i].parent]

    def channel_degrees() -> np.ndarray:
        mujoco.mj_forward(driven.model, data)
        frame = data.xmat[driven.model.body(driven.frame_of[hip]).id].reshape(3, 3)
        above = data.xmat[driven.model.body(driven.frame_of[parent]).id].reshape(3, 3)
        return euler_from_matrix(p.T @ above.T @ frame @ p, channels)

    hips = (JointRole.HIP_PITCH_L, JointRole.HIP_ROLL_L, JointRole.HIP_YAW_L)
    for role in hips:
        b = mapping.roles[role]
        data.qpos[:] = 0.0
        data.qpos[driven.model.joint(b.hinge).qposadr[0]] = 0.3
        expected = theta.copy()
        expected[channels.index(b.hinge.rsplit("/", 1)[1])] += b.sign * np.degrees(0.3)
        assert np.allclose(channel_degrees(), expected, atol=1e-9), role
    data.qpos[:] = 0.0
    expected = theta.copy()
    for role, delta in zip(hips, (0.2, -0.1, 0.25), strict=True):
        b = mapping.roles[role]
        data.qpos[driven.model.joint(b.hinge).qposadr[0]] = delta
        expected[channels.index(b.hinge.rsplit("/", 1)[1])] += b.sign * np.degrees(delta)
    # Several bound channels on one bone telescope: each hinge's axis is its channel's axis
    # seen through the later channels, so the composition is the Euler update of every channel.
    assert np.allclose(channel_degrees(), expected, atol=1e-9)


POSTURE_SPECS = [
    RigSpec(),
    replace(RigSpec(), spine_bones=1),
    replace(RigSpec(), family="style100"),
]


@pytest.mark.parametrize("spec", POSTURE_SPECS, ids=spec_id)
def test_posture_map_makes_commanded_posture_read_back(tmp_path, spec):
    robot, mapping, _, pm, p_ref = calibrated_rig(spec, tmp_path / spec_id(spec))
    assert tuple(p_ref) == POSTURE_PARAMETERS
    commands = (
        {},
        {"trunk_pitch": 0.3},
        {"neck_pitch": 0.3},
        {"elbow_flexion": 0.5},
        {"arm_abduction": 0.3},
    )
    for command in commands:
        target = {k: p_ref[k] + command.get(k, 0.0) for k in POSTURE_PARAMETERS}
        qpos = hinges_from_posture(robot, pm, target)
        read = static_posture(robot, qpos, mapping)
        for k in POSTURE_PARAMETERS:
            assert abs(read[k] - target[k]) < 0.02, (k, command, read[k] - target[k])
    assert pm[JointRole.ELBOW_L] == pm[JointRole.ELBOW_R]
    assert pm[JointRole.SHOULDER_ROLL_L] == pm[JointRole.SHOULDER_ROLL_R]


def test_posture_map_commands_the_reference_pose_at_p_ref(tmp_path):
    robot, _, _, pm, p_ref = calibrated_rig(RigSpec(), tmp_path)
    assert set(pm) == {
        JointRole.WAIST_PITCH,
        JointRole.NECK_PITCH,
        JointRole.ELBOW_L,
        JointRole.ELBOW_R,
        JointRole.SHOULDER_ROLL_L,
        JointRole.SHOULDER_ROLL_R,
    }
    assert pm[JointRole.ELBOW_L] == pm[JointRole.ELBOW_R]
    assert pm[JointRole.SHOULDER_ROLL_L] == pm[JointRole.SHOULDER_ROLL_R]
    assert np.allclose(hinges_from_posture(robot, pm, p_ref), 0.0, atol=1e-12)


def test_identity_posture_map_on_the_g1_is_the_default():
    assert G1_PROFILE_WITH_NECK.posture_map == {}


def _calibrate(full, ref, landmarks, drop=()):
    """``(robot, mapping, driven, rows, p_ref)`` with the roles in ``drop`` left unbound."""
    mapping = bind_roles(full, ref, landmarks)
    roles = {r: b for r, b in mapping.roles.items() if r not in drop}
    mapping = replace(mapping, roles=MappingProxyType(roles))
    driven = compile_driven(full.file, mapping, ref)
    uncalibrated = rig_robot(driven, rig_profile("calib", mapping, ref, posture_map={}))
    sites = landmark_site_names(driven, mapping.landmarks)
    rows, p_ref = posture_map(uncalibrated, mapping, landmark_sites=sites)
    robot = rig_robot(driven, rig_profile("calib", mapping, ref, posture_map=rows))
    return robot, mapping, driven, rows, p_ref


def _bend(points: dict, side: str) -> float:
    """Elbow bend (rad) from the sites: the angle between the segments, signed by
    ``_signed_flexion`` (negative hyperextended)."""
    upper = points[f"elbow_{side}"] - points[f"shoulder_{side}"]
    fore = points[f"wrist_{side}"] - points[f"elbow_{side}"]
    angle = np.arccos(np.clip(upper @ fore / np.linalg.norm(upper) / np.linalg.norm(fore), -1, 1))
    return float(np.sign(_signed_flexion(points, side)) * angle)


def _bends(robot, driven, mapping, qpos) -> tuple[float, float]:
    data = mujoco.MjData(robot.model)
    data.qpos[:] = qpos
    mujoco.mj_forward(robot.model, data)
    points = landmark_points(driven, mapping, data)
    return _bend(points, "l"), _bend(points, "r")


def test_posture_rows_carry_every_parameter_key_without_a_neck_and_generate_runs(tmp_path):
    full, ref, landmarks = _bound(RigSpec(), tmp_path)
    robot, _, _, rows, _ = _calibrate(full, ref, landmarks, drop=(JointRole.NECK_PITCH,))
    assert JointRole.NECK_PITCH not in rows and JointRole.WAIST_PITCH in rows
    for row in rows.values():
        assert set(row) == {*POSTURE_PARAMETERS, "offset"}
        assert row["neck_pitch"] == 0.0
    angles = generate(GaitParameters(), robot, welded=robot).angles_at(0.3)
    assert JointRole.NECK_PITCH not in angles and JointRole.WAIST_PITCH in angles


def test_a_hyperextended_reference_elbow_keeps_the_gain_on_the_flexion_side(tmp_path):
    full, ref, landmarks = _bound(RigSpec(), tmp_path)
    for side in "lr":
        ref = _with_elbow(ref, full.file, landmarks.landmarks[Landmark(f"elbow_{side}")], -20.0)
    robot, mapping, driven, rows, p_ref = _calibrate(full, ref, landmarks)
    assert rows[JointRole.ELBOW_L]["elbow_flexion"] > 0.0
    zero = _bends(robot, driven, mapping, np.zeros(robot.model.nq))
    assert all(b < 0.0 for b in zero)  # hyperextended at the reference
    target = {**p_ref, "elbow_flexion": p_ref["elbow_flexion"] + 0.5}
    bent = _bends(robot, driven, mapping, hinges_from_posture(robot, rows, target))
    for before, after in zip(zero, bent, strict=True):
        assert after - before > 0.4, (before, after)  # signed flexion grows


def test_a_half_bound_elbow_group_moves_the_bound_arm_by_the_command(tmp_path):
    full, ref, landmarks = _bound(RigSpec(), tmp_path)
    robot, mapping, driven, rows, p_ref = _calibrate(
        full, ref, landmarks, drop=(JointRole.ELBOW_R,)
    )
    assert JointRole.ELBOW_R not in rows
    target = {**p_ref, "elbow_flexion": p_ref["elbow_flexion"] + 0.5}
    zero = _bends(robot, driven, mapping, np.zeros(robot.model.nq))
    left, right = _bends(robot, driven, mapping, hinges_from_posture(robot, rows, target))
    assert abs(left - zero[0] - 0.5) < 0.02, (zero, left)
    assert abs(right - zero[1]) < 1e-9 and abs(zero[1]) < 1e-6


def _limb_bend(points: dict, a: str, b: str, c: str) -> float:
    """Unsigned angle (rad) between ``a -> b`` and ``b -> c``: 0 when straight."""
    u, v = points[b] - points[a], points[c] - points[b]
    return float(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1.0, 1.0)))


_LIMBS = {
    JointRole.KNEE_L: ("hip_l", "knee_l", "ankle_l"),
    JointRole.KNEE_R: ("hip_r", "knee_r", "ankle_r"),
    JointRole.ELBOW_L: ("shoulder_l", "elbow_l", "wrist_l"),
    JointRole.ELBOW_R: ("shoulder_r", "elbow_r", "wrist_r"),
}


def test_knee_and_elbow_ranges_start_at_straight_not_at_the_walking_mean(tmp_path):
    robot, mapping, _, driven, _ = driven_rig(RigSpec(), tmp_path)
    data = mujoco.MjData(robot.model)
    for role, limb in _LIMBS.items():
        data.qpos[:] = 0.0
        mujoco.mj_forward(robot.model, data)
        bend = _limb_bend(landmark_points(driven, mapping, data), *limb)
        lo, hi = robot.joint_range(role)
        assert lo == pytest.approx(-bend, abs=0.01), role
        assert hi - lo == pytest.approx(RANGES[role][1] - RANGES[role][0])
        if role.name.startswith("KNEE"):
            assert bend > 0.1  # the walking mean is bent, so the old range clamped
        data.qpos[robot.qpos_adr(role)] = lo  # the lower bound is the straight limb
        mujoco.mj_forward(robot.model, data)
        assert _limb_bend(landmark_points(driven, mapping, data), *limb) < 0.01, role


def test_a_hyperextended_reference_elbow_is_straight_above_its_zero(tmp_path):
    full, ref, landmarks = _bound(RigSpec(), tmp_path)
    for side in "lr":
        ref = _with_elbow(ref, full.file, landmarks.landmarks[Landmark(f"elbow_{side}")], -20.0)
    mapping = bind_roles(full, ref, landmarks)
    driven = compile_driven(full.file, mapping, ref)
    robot = rig_robot(driven, rig_profile("hyper", mapping, ref, posture_map={}))
    data = mujoco.MjData(robot.model)
    for role in (JointRole.ELBOW_L, JointRole.ELBOW_R):
        offset = mapping.roles[role].zero_offset
        data.qpos[:] = 0.0
        mujoco.mj_forward(robot.model, data)
        bend = _limb_bend(landmark_points(driven, mapping, data), *_LIMBS[role])
        assert offset == pytest.approx(bend, abs=1e-9) and offset > 0.2  # straight is ABOVE 0
        data.qpos[robot.qpos_adr(role)] = offset
        mujoco.mj_forward(robot.model, data)
        # The test's elbow tilt is not about the hinge axis, so a little bend stays off-plane.
        assert _limb_bend(landmark_points(driven, mapping, data), *_LIMBS[role]) < 0.03


def _compose(channels: Sequence[str], radians: Sequence[float]) -> np.ndarray:
    """``R_c1 @ R_c2 ...`` for rotation ``channels`` at ``radians`` (the reader's order)."""
    out = np.eye(3)
    for c, a in zip(channels, radians, strict=True):
        out = out @ rotation_about(_AXIS[c[0]], [np.degrees(a)])[0]
    return out


def test_fit_two_channel_returns_a_representable_mean_unchanged():
    channels = ("Zrotation", "Xrotation")
    mean = rotation_about(2, [12.0])[0] @ rotation_about(0, [-7.0])[0]
    fitted, residual = fit_two_channel(mean, channels)
    assert np.allclose(fitted, mean, atol=1e-9) and residual < 1e-9


def test_fit_two_channel_projects_a_circulating_mean_and_reports_the_residual():
    # a(t) = 0.3 sin t, b(t) = 0.3 sin(t + pi/4): an elliptic circulation whose angles
    # correlate, so the chordal mean has a component about the missing (Y) axis. (A pure
    # sin/cos circle is uncorrelated over a period: its mean is the identity, representable.)
    channels = ("Zrotation", "Xrotation")
    t = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    rots = np.stack([_compose(channels, (0.3 * np.sin(x), 0.3 * np.sin(x + np.pi / 4))) for x in t])
    mean = chordal_mean(rots)
    fitted, residual = fit_two_channel(mean, channels)
    assert np.allclose(_compose(channels, channel_angles(fitted, channels)), fitted, atol=1e-9)
    assert residual > TWO_CHANNEL_WARN_RAD
    angle = np.arccos(np.clip((np.trace(mean.T @ fitted) - 1.0) / 2.0, -1.0, 1.0))
    assert np.isclose(residual, angle)
    plain = _compose(channels, channel_angles(mean, channels))
    assert np.linalg.norm(fitted - mean) <= np.linalg.norm(plain - mean) + 1e-12


def test_fit_two_channel_holds_a_representable_mean_past_ninety_degrees_on_its_branch():
    # b = 120 deg: the three-axis Euler branch with |middle| <= 90 deg is a different two-angle
    # rotation (dropped angle pi); channel_angles must take the branch whose dropped angle is 0.
    channels = ("Zrotation", "Xrotation")
    mean = _compose(channels, (0.3, np.radians(120.0)))
    assert np.allclose(channel_angles(mean, channels), [0.3, np.radians(120.0)], atol=1e-12)
    fitted, residual = fit_two_channel(mean, channels)
    assert np.allclose(fitted, mean, atol=1e-9) and residual < 1e-9


def test_fit_two_channel_beats_the_plain_projection_on_an_asymmetric_mean():
    # An offset elliptic circulation: the mean's representable part is not the identity, and
    # the least-squares fit is strictly nearer (Frobenius) than dropping the third angle.
    channels = ("Zrotation", "Xrotation")
    t = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    rots = np.stack(
        [_compose(channels, (0.6 + 0.5 * np.sin(x), 0.8 + 0.5 * np.sin(x + np.pi / 4))) for x in t]
    )
    mean = chordal_mean(rots)
    fitted, _ = fit_two_channel(mean, channels)
    plain = _compose(channels, channel_angles(mean, channels))
    assert np.linalg.norm(fitted - mean) < np.linalg.norm(plain - mean) - 0.01


def _two_channel_circulating(f: BvhFile, bone: str, *, period_s: float = 3.0) -> BvhFile:
    """``f`` with ``bone``'s rotation channels cut to two (the up-axis one dropped) moving
    0.3 sin(t) and 0.3 sin(t + pi/4) rad (written in degrees), t = 2 pi time / ``period_s``."""
    i = [j.name for j in f.joints].index(bone)
    joint = f.joints[i]
    up = "Y" if f.up_axis == "y" else "Z"
    drop = f.channel_index(bone, f"{up}rotation")
    kept = tuple(c for c in joint.channels if c != f"{up}rotation")
    joints = tuple(
        BvhJoint(j.name, j.parent, j.offset, kept, j.end_site) if k == i else j
        for k, j in enumerate(f.joints)
    )
    motion = np.delete(f.motion, drop, axis=1)
    t = 2.0 * np.pi * np.arange(motion.shape[0]) * f.frame_time / period_s
    start = sum(len(j.channels) for j in joints[:i])
    rotations = [c for c in kept if c.endswith("rotation")]
    for c, column in zip(rotations, (np.sin(t), np.sin(t + np.pi / 4)), strict=True):
        motion[:, start + kept.index(c)] = np.degrees(0.3 * column)
    return replace(f, joints=joints, motion=motion)


def test_a_two_channel_bone_with_an_unrepresentable_mean_is_fitted_recorded_and_writes_back(
    tmp_path,
):
    spec = replace(RigSpec(), two_channel_limbs=True)
    path = walk_file(spec, tmp_path)
    f = read_bvh_file(path, scale=1.0 / spec.units_per_m, up_axis=spec.up_axis)
    bone = FAMILIES[spec.family]["upper_arm"]
    f = _two_channel_circulating(f, bone)
    write_bvh(f, tmp_path / "circ.bvh")
    rig = load_rig(tmp_path / "circ.bvh", scale=None, up_axis=None, mapping_overrides=None)
    d = next(x for x in rig.mapping.decisions if x.what == f"reference:{bone}")
    assert d.chosen == "fitted" and d.source == "geometry" and "residual" in d.note
    assert any(bone in w and "two rotation channels" in w for w in rig.mapping.warnings)
    i = [j.name for j in f.joints].index(bone)
    channels = tuple(c for c in f.joints[i].channels if c.endswith("rotation"))
    assert len(channels) == 2
    reference = rig.reference.local_rotations[i]
    back = _compose(channels, channel_angles(reference, channels))
    assert np.allclose(back, reference, atol=1e-9)
    knee = FAMILIES[spec.family]["knee"]
    kd = next(x for x in rig.mapping.decisions if x.what == f"reference:{knee}")
    assert kd.note.endswith("rad")
    assert not any(knee in w for w in rig.mapping.warnings)  # a pure lateral swing fits exactly
