"""Readers: forward kinematics, unit scaling, landmark lookup and error reporting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emotive_gait.mocap.readers.asf_amc import read_asf_amc
from emotive_gait.mocap.readers.bvh import read_bvh
from emotive_gait.mocap.readers.ptd import read_ptd
from emotive_gait.mocap.skeleton import Landmark, MocapFormatError

FIX = Path(__file__).parent / "fixtures"


def test_bvh_forward_kinematics_rotates_the_child_about_the_parent():
    clip = read_bvh(FIX / "two_bone.bvh", scale=0.01, up_axis="y")
    assert clip.fps == pytest.approx(30.0, abs=0.01) and clip.frames() == 3
    assert clip.joint_names == ("Hips", "Knee") and clip.parents == (-1, 0)
    knee0 = clip.positions[0, 1]  # straight down: (0, 0, 0.6) after y-up -> z-up
    assert knee0 == pytest.approx([0.0, 0.0, 0.6], abs=1e-6)
    knee1 = clip.positions[1, 1]  # Hips 90 deg about x swings the bone to +/-y
    assert abs(knee1[2] - 1.0) < 1e-6 and abs(abs(knee1[1]) - 0.4) < 1e-6
    assert clip.positions[2, 0, 2] == pytest.approx(1.1, abs=1e-6)  # root translated


def test_bvh_bone_length_is_preserved_in_every_frame():
    clip = read_bvh(FIX / "two_bone.bvh", scale=0.01, up_axis="y")
    lengths = np.linalg.norm(clip.positions[:, 1] - clip.positions[:, 0], axis=1)
    assert lengths == pytest.approx(0.4, abs=1e-9)


def test_asf_amc_scales_cmu_units_to_metres_and_rotates():
    clip = read_asf_amc(FIX / "two_bone.asf", FIX / "two_bone.amc")
    length = np.linalg.norm(clip.positions[0, 1] - clip.positions[0, 0])
    assert length == pytest.approx(20 * (1 / 0.45) * 0.0254, rel=1e-6)
    assert clip.joint_names == ("root", "lfemur") and clip.parents == (-1, 0)
    assert clip.positions[1, 1, 2] != pytest.approx(clip.positions[0, 1, 2])  # rotated frame


def test_ptd_columns_follow_the_given_joint_order():
    clip = read_ptd(FIX / "three_point.ptd", joint_names=("a", "b", "c"), scale=0.001, up_axis="z")
    assert clip.joint_names == ("a", "b", "c") and clip.parents is None and clip.frames() == 2
    assert clip.positions[1, 2] == pytest.approx([0.0, 0.021, 0.0])


def test_landmark_positions_omit_missing_joints_and_average_tuples():
    clip = read_ptd(FIX / "three_point.ptd", joint_names=("a", "b", "c"), scale=0.001, up_axis="z")
    got = clip.landmark_positions({Landmark.PELVIS: ("a", "b"), Landmark.HEAD: "zzz"})
    assert Landmark.HEAD not in got
    expected = (clip.positions[0, 0] + clip.positions[0, 1]) / 2
    assert got[Landmark.PELVIS][0] == pytest.approx(expected)
    assert got[Landmark.PELVIS].shape == (2, 3)


def test_malformed_bvh_names_the_line(tmp_path):
    bad = tmp_path / "bad.bvh"
    bad.write_text("HIERARCHY\nROOT Hips\n{\nOFFSET 0 0\n")
    with pytest.raises(MocapFormatError, match="line 4"):
        read_bvh(bad, scale=1.0, up_axis="y")


def test_ptd_real_layout_has_a_frame_count_header_and_no_frame_column(tmp_path):
    f = tmp_path / "real.ptd"
    f.write_text("2\n1 2 3 4 5 6 7 8 9\n\n10 20 30 40 50 60 70 80 90\n")
    clip = read_ptd(f, joint_names=("a", "b", "c"), scale=1.0, up_axis="z")
    assert clip.frames() == 2
    assert clip.positions[1, 2] == pytest.approx([70, 80, 90])
    assert clip.positions[0, 0] == pytest.approx([1, 2, 3])


def test_ptd_header_frame_count_mismatch_and_bad_width_name_the_file(tmp_path):
    f = tmp_path / "bad.ptd"
    f.write_text("3\n1 2 3 4 5 6 7 8 9\n")
    with pytest.raises(MocapFormatError, match=r"bad\.ptd.*line 1"):
        read_ptd(f, joint_names=("a", "b", "c"), scale=1.0, up_axis="z")
    f.write_text("1 2 3 4 5\n")
    with pytest.raises(MocapFormatError, match="line 1"):
        read_ptd(f, joint_names=("a", "b", "c"), scale=1.0, up_axis="z")
