"""FK-equivalence gate: the compiled MuJoCo chain reproduces the BVH reader's positions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from emotive_gait.mocap.readers.bvh import BvhFile, BvhJoint, euler_from_matrix, read_bvh_file
from emotive_gait.mocap.skeleton import Landmark
from emotive_gait.rig.compile import PELVIS_CARRIER, _z_up_matrix, compile_full, heading_matrix
from emotive_gait.rig.errors import RigError
from rig.driven import driven_rig
from rig.rigs import RigSpec, build_rig, pelvis_name, write_walk

FIX = Path(__file__).resolve().parents[1] / "mocap" / "fixtures"


def _assert_chain_matches_reader(f, pelvis: str, frames: tuple[int, ...]) -> None:
    rig = compile_full(f, pelvis=pelvis)
    clip = f.to_clip(end_sites=True)
    data = mujoco.MjData(rig.model)
    pelvis_index = clip.joint_names.index(pelvis)
    assert rig.landmark_sites
    for frame in frames:
        data.qpos[:] = rig.qpos_from_motion(frame)
        mujoco.mj_forward(rig.model, data)
        for name, site in rig.landmark_sites.items():
            expected = (
                clip.positions[frame, clip.joint_names.index(name)]
                - clip.positions[frame, pelvis_index]
            )
            got = data.site_xpos[rig.model.site(site).id] - data.xpos[rig.model.body("pelvis").id]
            assert np.allclose(got, expected, atol=1e-6), (frame, name, got - expected)


@pytest.mark.parametrize(
    "spec",
    [
        RigSpec(),
        replace(RigSpec(), order="XYZ"),
        replace(RigSpec(), order="ZYX"),
        replace(RigSpec(), up_axis="z"),
        replace(RigSpec(), root_above_pelvis=True),
    ],
    ids=["ZXY", "XYZ", "ZYX", "z-up", "root-above-pelvis"],
)
def test_full_chain_reproduces_the_reader_positions_relative_to_the_pelvis(tmp_path, spec):
    write_walk(spec, tmp_path / "w.bvh")
    f = read_bvh_file(tmp_path / "w.bvh", scale=1.0 / spec.units_per_m, up_axis=spec.up_axis)
    _assert_chain_matches_reader(f, pelvis_name(spec), (0, 17, 150))


def test_full_chain_reproduces_style100():
    f = read_bvh_file(FIX / "style100_neutral_4s.bvh", scale=0.01, up_axis="y")
    _assert_chain_matches_reader(f, "Hips", (0, 100, 239))


def test_full_chain_names_bodies_hinges_and_sites():
    spec = replace(RigSpec(), root_above_pelvis=True)

    f = build_rig(spec)
    pelvis = pelvis_name(spec)
    rig = compile_full(f, pelvis=pelvis, pelvis_height=0.8)
    assert rig.body_of[pelvis] == "pelvis"
    assert rig.frame_of[pelvis] == PELVIS_CARRIER  # the pelvis bone's rotated frame
    assert all(rig.frame_of[b] == rig.body_of[b] for b in rig.body_of if b != pelvis)
    assert "reference" not in rig.body_of  # ancestors of the pelvis are not compiled
    assert np.allclose(rig.model.body("pelvis").pos, [0.0, 0.0, 0.8])
    assert rig.model.body("pelvis").jntnum == 0
    assert rig.model.nq == rig.model.nu == len(rig.hinges)
    for i, hinge in enumerate(rig.hinges):
        assert rig.model.joint(hinge).qposadr[0] == i
        assert rig.model.actuator(hinge).trnid[0] == rig.model.joint(hinge).id
    clip = f.to_clip(end_sites=True)
    below = [n for i, n in enumerate(clip.joint_names) if _descends(clip.parents, i, pelvis, clip)]
    assert set(rig.landmark_sites) == set(below)
    assert all(site == name for name, site in rig.landmark_sites.items())
    assert f"{pelvis}_End" not in below and "mixamorig:Head_End" in rig.landmark_sites
    assert f"{pelvis}/Zrotation" == rig.hinges[0]
    assert "mixamorig:LeftLeg/Xrotation" in rig.hinges


def _descends(parents, i: int, ancestor: str, clip) -> bool:
    while i >= 0:
        if clip.joint_names[i] == ancestor:
            return True
        i = parents[i]
    return False


def _with_translation(f: BvhFile, bone: str, values: float) -> BvhFile:
    """``f`` with ``Xposition Yposition Zposition`` prepended to ``bone``, all set to ``values``."""
    start = f.channel_slice(bone).start
    joints = tuple(
        BvhJoint(
            j.name,
            j.parent,
            j.offset,
            ("Xposition", "Yposition", "Zposition") + j.channels,
            j.end_site,
        )
        if j.name == bone
        else j
        for j in f.joints
    )
    extra = np.full((f.motion.shape[0], 3), values)
    motion = np.concatenate([f.motion[:, :start], extra, f.motion[:, start:]], axis=1)
    return replace(f, joints=joints, motion=motion)


def test_translation_channels_below_the_pelvis_are_refused_unless_zero():
    f = build_rig(RigSpec())
    bone = "mixamorig:LeftLeg"
    with pytest.raises(RigError, match="mixamorig:LeftLeg"):
        compile_full(_with_translation(f, bone, 1.5), pelvis=pelvis_name(RigSpec()))
    rig = compile_full(_with_translation(f, bone, 0.0), pelvis=pelvis_name(RigSpec()))
    assert f"{bone}/Xposition" not in rig.hinges


@pytest.mark.parametrize("forward", [[1.0, 0.0, 0.0], [0.0, 2.0, 0.3], [-1.0, -1.0, 0.0]])
def test_heading_matrix_takes_x_to_the_horizontal_forward_and_keeps_z(forward):
    r = heading_matrix(np.array(forward))
    flat = np.array([forward[0], forward[1], 0.0])
    assert np.allclose(r @ [1.0, 0.0, 0.0], flat / np.linalg.norm(flat))
    assert np.allclose(r @ [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    assert np.allclose(r.T @ r, np.eye(3)) and np.isclose(np.linalg.det(r), 1.0)


def test_a_pelvis_quaternion_goes_on_the_carrier_and_the_root_stays_world_aligned():
    from emotive_gait.rig.compile import _build

    f = build_rig(RigSpec())
    pelvis = pelvis_name(RigSpec())
    tilt = np.array([np.cos(0.1), 0.0, np.sin(0.1), 0.0])
    rig = _build(
        f,
        pelvis=pelvis,
        pelvis_height=1.0,
        hinges_for=lambda bone: [],
        quat_for=lambda bone: tilt if f.joints[bone].name == pelvis else None,
        hinge_range=lambda name: None,
    )
    assert np.allclose(rig.model.body("pelvis").quat, [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(rig.model.body(PELVIS_CARRIER).quat, tilt)
    assert rig.frame_of[pelvis] == PELVIS_CARRIER and rig.model.nq == 0


def _site_points(rig, data) -> dict[str, np.ndarray]:
    root = data.xpos[rig.model.body("pelvis").id]
    return {
        name: data.site_xpos[rig.model.site(site).id] - root
        for name, site in rig.landmark_sites.items()
    }


@pytest.mark.parametrize(
    "spec",
    [
        RigSpec(),
        replace(RigSpec(), order="XYZ"),
        replace(RigSpec(), order="ZYX"),
        replace(RigSpec(), up_axis="z"),
        replace(RigSpec(), root_above_pelvis=True),
        replace(RigSpec(), toes=False),
    ],
    ids=["ZXY", "XYZ", "ZYX", "z-up", "root-above-pelvis", "no-toes"],
)
def test_driven_chain_at_zero_equals_full_chain_at_the_reference(tmp_path, spec):
    robot, mapping, ref, driven, full = driven_rig(spec, tmp_path)
    names = [j.name for j in full.file.joints]
    pelvis = names.index(full.pelvis)
    p = _z_up_matrix(full.file.up_axis)
    heading = heading_matrix(ref.heading)
    # The full chain at the reference channel values, the pelvis with the heading put back.
    rotations = ref.baked_rotations(names)
    rotations[pelvis] = p.T @ heading @ p @ rotations[pelvis]
    angles = {
        bone: np.radians(euler_from_matrix(rotations[i], full.file.joints[i].channels))
        for i, bone in enumerate(names)
        if len([c for c in full.file.joints[i].channels if c.endswith("rotation")]) == 3
    }
    qpos = np.zeros(full.model.nq)
    for i, hinge in enumerate(full.hinges):
        bone, channel = hinge.rsplit("/", 1)
        rot = [c for c in full.file.joints[names.index(bone)].channels if c.endswith("rotation")]
        qpos[i] = angles[bone][rot.index(channel)]
    full_data, driven_data = mujoco.MjData(full.model), mujoco.MjData(driven.model)
    full_data.qpos[:] = qpos
    mujoco.mj_forward(full.model, full_data)
    mujoco.mj_forward(driven.model, driven_data)  # qpos zero
    expected = {k: heading.T @ v for k, v in _site_points(full, full_data).items()}
    got = _site_points(driven, driven_data)
    assert set(got) == set(expected)
    for name in got:
        assert np.allclose(got[name], expected[name], atol=1e-9), (name, got[name] - expected[name])
    # Root at the reference pelvis height; the foot sites sit at the profile's floor height.
    assert np.allclose(driven.model.body("pelvis").pos, [0.0, 0.0, ref.pelvis_height])
    feet = [driven_data.site_xpos[driven.model.site(s).id] for s in robot.profile.foot_sites]
    assert np.mean([f[2] for f in feet]) == pytest.approx(
        robot.profile.foot_site_floor_height, abs=1e-12
    )
    # The left foot site is the toe landmark (toe bone, or the foot's End Site without toes).
    point = driven.landmark_sites[mapping.landmarks[Landmark.TOE_L]]
    assert np.allclose(feet[0], driven_data.site_xpos[driven.model.site(point).id], atol=1e-12)


def test_driven_chain_has_hinges_only_for_bound_roles(tmp_path):
    robot, mapping, ref, driven, _ = driven_rig(RigSpec(), tmp_path)
    assert set(driven.hinges) == {b.hinge for b in mapping.roles.values()}
    assert driven.model.nu == driven.model.nq == len(driven.hinges)
    assert driven.model.body("pelvis").jntnum == 0  # welded, what gait.ik looks for
    profile = robot.profile
    assert profile.foot_sites == ("foot_l", "foot_r") and profile.scene_entry == ""
    assert profile.nominal_pelvis_height == ref.pelvis_height
    assert profile.leg_length == mapping.leg_length and 0.7 < mapping.leg_length < 1.0
    assert {r: profile.actuator(r) for r in mapping.roles} == {
        r: b.hinge for r, b in mapping.roles.items()
    }
