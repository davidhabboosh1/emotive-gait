"""Reference pose, role bindings by probe, and the conventions on the driven chain."""

from __future__ import annotations

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from emotive_gait.gait.ik import LEG_ROLES
from emotive_gait.mocap.readers.bvh import BvhFile, euler_from_matrix, rotation_about
from emotive_gait.mocap.skeleton import Landmark
from emotive_gait.rig.calibration import ReferencePose, reference_pose
from emotive_gait.rig.compile import _z_up_matrix, compile_driven, compile_full
from emotive_gait.rig.mapping import RigError, bind_roles, map_landmarks
from emotive_gait.rig.profile import rig_profile, rig_robot
from emotive_gait.rig.ranges import RANGES
from emotive_gait.robot.conventions import ROLE_CONVENTIONS, convention_holds
from emotive_gait.robot.roles import JointRole
from rig.driven import X, Y, Z, driven_rig, probe_driven, spec_id
from rig.rigs import FAMILY_VARIANTS, RigSpec, build_rig, toe_rest_height_m, walk_and_clip


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
    for role in mapping.roles:
        assert robot.joint_range(role) == pytest.approx(RANGES[role])
    assert robot.profile.sign_for == {role: 1.0 for role in mapping.roles}
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
    p = _z_up_matrix(f.up_axis)
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
