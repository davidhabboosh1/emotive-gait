"""Reference pose, role bindings by probe, and the conventions on the driven chain."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from emotive_gait.gait.ik import LEG_ROLES
from emotive_gait.mocap.readers.bvh import BvhFile, rotation_about
from emotive_gait.mocap.skeleton import Landmark
from emotive_gait.rig.calibration import reference_pose
from emotive_gait.rig.compile import compile_full
from emotive_gait.rig.mapping import RigError, bind_roles, map_landmarks
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
    assert any(d.what == "waist_pitch" and d.source == "user" for d in mapping.decisions)
    with pytest.raises(RigError, match="NoSuchBone/Xrotation"):
        bind_roles(full, ref, landmarks, overrides={"waist_pitch": "NoSuchBone/Xrotation"})
    with pytest.raises(RigError, match="not_a_role"):
        bind_roles(full, ref, landmarks, overrides={"not_a_role": hinge})


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
