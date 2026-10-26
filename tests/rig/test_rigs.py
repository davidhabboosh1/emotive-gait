from __future__ import annotations

from dataclasses import replace

import pytest

from emotive_gait.mocap.readers.bvh import read_bvh_file
from rig.rigs import FAMILY_VARIANTS, RigSpec, build_rig, hip_to_ankle_rest_m, write_walk


@pytest.mark.parametrize(
    "spec",
    FAMILY_VARIANTS,
    ids=lambda s: s.family + ("-" + s.order) + ("-z" if s.up_axis == "z" else ""),
)
def test_every_variant_reads_back_with_a_human_leg_and_cycles(tmp_path, spec):
    write_walk(spec, tmp_path / "walk.bvh")
    g = read_bvh_file(tmp_path / "walk.bvh", scale=1.0 / spec.units_per_m, up_axis=spec.up_axis)
    clip = g.to_clip(end_sites=True)
    assert abs(hip_to_ankle_rest_m(g) - spec.leg_length_m * 0.9) < 0.05
    assert clip.positions[-1, 0, 0] - clip.positions[0, 0, 0] > 3.0  # walked forward along +x


def test_root_above_pelvis_variant_has_a_reference_root():
    f = build_rig(replace(RigSpec(), root_above_pelvis=True))
    assert f.joints[0].name == "reference" and f.joints[1].parent == 0
