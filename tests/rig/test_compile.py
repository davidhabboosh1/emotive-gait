"""FK-equivalence gate: the compiled MuJoCo chain reproduces the BVH reader's positions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from emotive_gait.mocap.readers.bvh import read_bvh_file
from emotive_gait.rig.compile import compile_full, heading_matrix
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
    assert "reference" not in rig.body_of  # ancestors of the pelvis are not compiled
    assert np.allclose(rig.model.body("pelvis").pos, [0.0, 0.0, 0.8])
    assert rig.model.body("pelvis").jntnum == 0
    assert rig.model.nq == rig.model.nu == len(rig.hinges)
    for i, hinge in enumerate(rig.hinges):
        assert rig.model.joint(hinge).qposadr[0] == i
        assert rig.model.actuator(hinge).trnid[0] == rig.model.joint(hinge).id
    assert any(name.endswith("_End") for name in rig.landmark_sites)


@pytest.mark.parametrize("forward", [[1.0, 0.0, 0.0], [0.0, 2.0, 0.3], [-1.0, -1.0, 0.0]])
def test_heading_matrix_takes_x_to_the_horizontal_forward_and_keeps_z(forward):
    r = heading_matrix(np.array(forward))
    flat = np.array([forward[0], forward[1], 0.0])
    assert np.allclose(r @ [1.0, 0.0, 0.0], flat / np.linalg.norm(flat))
    assert np.allclose(r @ [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    assert np.allclose(r.T @ r, np.eye(3)) and np.isclose(np.linalg.det(r), 1.0)
