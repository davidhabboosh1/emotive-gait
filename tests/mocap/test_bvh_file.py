"""BvhFile: everything the file says, a writer that reads back, and the Euler decomposition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emotive_gait.mocap.readers.bvh import (
    _AXIS,
    euler_from_matrix,
    read_bvh,
    read_bvh_file,
    rotation_about,
)
from emotive_gait.mocap.writers.bvh import write_bvh

FIX = Path(__file__).parent / "fixtures"


def compose(channels: tuple[str, ...], degrees: np.ndarray) -> np.ndarray:
    r = np.eye(3)
    for name, d in zip(channels, degrees, strict=True):
        r = r @ rotation_about(_AXIS[name[0]], np.array([d]))[0]
    return r


def test_read_bvh_file_keeps_hierarchy_channels_and_motion():
    f = read_bvh_file(FIX / "two_bone.bvh", scale=0.01, up_axis="y")
    assert [j.name for j in f.joints] == ["Hips", "Knee"]
    assert f.joints[1].channels == ("Zrotation", "Xrotation", "Yrotation")
    assert f.motion.shape == (3, 9) and f.channel_index("Knee", "Xrotation") == 7
    assert f.joints[1].end_site == pytest.approx([0.0, -40.0, 0.0])


def test_to_clip_matches_read_bvh_bit_for_bit():
    f = read_bvh_file(FIX / "two_bone.bvh", scale=0.01, up_axis="y")
    clip = read_bvh(FIX / "two_bone.bvh", scale=0.01, up_axis="y")
    assert np.array_equal(f.to_clip().positions, clip.positions)


def test_to_clip_can_append_end_sites_as_joints():
    clip = read_bvh_file(FIX / "two_bone.bvh", scale=0.01, up_axis="y").to_clip(end_sites=True)
    assert clip.joint_names[-1] == "Knee_End" and clip.parents[-1] == 1
    assert clip.positions[0, -1] == pytest.approx([0.0, 0.0, 0.2], abs=1e-6)


@pytest.mark.parametrize("order", ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"])
def test_euler_from_matrix_inverts_the_reader_composition(order):
    channels = tuple(f"{a}rotation" for a in order)
    rng = np.random.default_rng(0)
    for deg in rng.uniform(-170, 170, size=(50, 3)):
        r = compose(channels, deg)
        assert np.allclose(compose(channels, euler_from_matrix(r, channels)), r, atol=1e-9)


@pytest.mark.parametrize("order", ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"])
@pytest.mark.parametrize("middle", [90.0, -90.0])
def test_euler_from_matrix_resolves_gimbal_lock_with_a_zero_last_channel(order, middle):
    channels = tuple(f"{a}rotation" for a in order)
    r = compose(channels, np.array([37.0, middle, 21.0]))
    out = euler_from_matrix(r, channels)
    assert out[2] == 0.0 and np.allclose(compose(channels, out), r, atol=1e-9)


def test_world_rotations_of_a_rest_pose_are_identity():
    f = read_bvh_file(FIX / "two_bone.bvh", scale=0.01, up_axis="y")
    assert np.allclose(f.world_rotations(0), np.tile(np.eye(3), (2, 1, 1)))


def test_write_then_read_reproduces_the_file(tmp_path):
    f = read_bvh_file(FIX / "two_bone.bvh", scale=0.01, up_axis="y")
    write_bvh(f, tmp_path / "copy.bvh")
    g = read_bvh_file(tmp_path / "copy.bvh", scale=0.01, up_axis="y")
    assert [j.name for j in g.joints] == [j.name for j in f.joints]
    assert g.joints[1].channels == f.joints[1].channels
    assert np.allclose(g.motion, f.motion, atol=1e-6)
    assert g.frame_time == pytest.approx(f.frame_time)
    assert np.allclose(g.to_clip().positions, f.to_clip().positions, atol=1e-9)


def test_style100_excerpt_reads_and_has_240_frames():
    f = read_bvh_file(FIX / "style100_neutral_4s.bvh", scale=0.01, up_axis="y")
    assert f.motion.shape[0] == 240 and f.fps == pytest.approx(60.0, abs=0.01)
    assert "LeftToe" in [j.name for j in f.joints]
