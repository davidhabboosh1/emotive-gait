"""Landmark mapping: name lexicon first, rest-pose geometry as referee; units and up-axis."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from emotive_gait.mocap.landmarks import STYLE100
from emotive_gait.mocap.readers.bvh import BvhFile, read_bvh_file
from emotive_gait.mocap.skeleton import Landmark
from emotive_gait.rig.mapping import (
    RigError,
    infer_scale_and_up,
    map_landmarks,
    required_landmarks_present,
)
from rig.rigs import FAMILIES, FAMILY_VARIANTS, RigSpec, walk_and_clip, write_walk

FIX = Path(__file__).resolve().parents[1] / "mocap" / "fixtures"


def _spec_id(spec: RigSpec) -> str:
    default = RigSpec()
    changed = [f"{k}={v}" for k, v in vars(spec).items() if getattr(default, k) != v]
    return "-".join(changed) or "default"


def drop_bones(file: BvhFile, names: tuple[str, ...]) -> BvhFile:
    """Splice ``names`` out: children re-parent to the dropped bone's parent with the offsets
    summed, and the dropped bones' channel columns are removed."""
    drop = {file.joints.index(j) for j in file.joints if j.name in names}
    assert len(drop) == len(names), "every bone to drop must be in the file"
    keep = [i for i in range(len(file.joints)) if i not in drop]
    new_index = {old: new for new, old in enumerate(keep)}
    joints = []
    for i in keep:
        joint = file.joints[i]
        parent, offset = joint.parent, np.array(joint.offset, dtype=float)
        while parent in drop:
            offset = offset + file.joints[parent].offset
            parent = file.joints[parent].parent
        joints.append(replace(joint, parent=new_index.get(parent, -1), offset=offset))
    columns = np.concatenate(
        [
            np.arange(len(file.joints[i].channels)) + file.channel_slice(file.joints[i].name).start
            for i in keep
        ]
    ).astype(int)
    return replace(file, joints=tuple(joints), motion=file.motion[:, columns])


@pytest.mark.parametrize("spec", FAMILY_VARIANTS, ids=_spec_id)
def test_every_family_maps_the_seventeen_landmarks(tmp_path, spec):
    f, clip = walk_and_clip(spec, tmp_path)
    m = map_landmarks(f, clip)
    assert set(m.landmarks) >= {
        Landmark.PELVIS,
        Landmark.HIP_L,
        Landmark.KNEE_R,
        Landmark.ANKLE_L,
        Landmark.NECK,
        Landmark.HEAD,
        Landmark.WRIST_R,
    }
    assert set(m.landmarks) == set(Landmark)
    assert (Landmark.TOE_L in m.landmarks) and m.landmarks[Landmark.TOE_L].endswith("_End") == (
        not spec.toes
    )
    assert m.family == spec.family
    assert not required_landmarks_present(m.landmarks)


def test_style100_fixture_maps_to_the_phase_4_5_landmark_map():
    f = read_bvh_file(FIX / "style100_neutral_4s.bvh", scale=0.01, up_axis="y")
    assert dict(map_landmarks(f, f.to_clip(end_sites=True)).landmarks) == dict(STYLE100)


def test_swapped_rest_sides_are_corrected_by_geometry_with_a_warning(tmp_path):
    f, clip = walk_and_clip(replace(RigSpec(), swap_rest_sides=True), tmp_path)
    m = map_landmarks(f, clip)
    right_hip = FAMILIES["mixamo"]["hip"].replace("Left", "Right")
    assert m.landmarks[Landmark.HIP_L] == right_hip and any("side" in w for w in m.warnings)


def test_override_with_an_unknown_bone_is_a_rig_error(tmp_path):
    f, clip = walk_and_clip(RigSpec(), tmp_path)
    with pytest.raises(RigError, match="NoSuchBone"):
        map_landmarks(f, clip, overrides={"head": "NoSuchBone"})


def test_override_is_recorded_as_a_user_decision(tmp_path):
    f, clip = walk_and_clip(RigSpec(), tmp_path)
    m = map_landmarks(f, clip, overrides={"head": "mixamorig:Head_End"})
    assert m.landmarks[Landmark.HEAD] == "mixamorig:Head_End"
    assert any(d.what == "head" and d.source == "user" for d in m.decisions)


def test_a_rig_without_knees_is_refused_naming_them(tmp_path):
    f, _ = walk_and_clip(RigSpec(), tmp_path)
    f = drop_bones(f, ("mixamorig:LeftLeg", "mixamorig:RightLeg"))
    with pytest.raises(RigError, match="knee_l"):
        map_landmarks(f, f.to_clip(end_sites=True))


def test_unknown_names_fall_back_to_tokens_and_geometry(tmp_path):
    """Lower-cased, namespaced, side-less-prefix names: no family matches, tokens do."""
    f, _ = walk_and_clip(RigSpec(), tmp_path)
    renamed = tuple(
        replace(
            j,
            name="rig:" + j.name.split(":")[1].lower().replace("left", "l_").replace("right", "r_"),
        )
        for j in f.joints
    )
    f = replace(f, joints=renamed)
    m = map_landmarks(f, f.to_clip(end_sites=True))
    assert m.family is None
    assert m.landmarks[Landmark.HIP_L] == "rig:l_upleg"
    assert m.landmarks[Landmark.KNEE_R] == "rig:r_leg"
    assert m.landmarks[Landmark.TOE_L] == "rig:l_toebase"
    assert m.landmarks[Landmark.ELBOW_L] == "rig:l_forearm"
    assert m.landmarks[Landmark.SHOULDER_R] == "rig:r_arm"
    assert m.landmarks[Landmark.WRIST_L] == "rig:l_hand"
    assert m.landmarks[Landmark.NECK] == "rig:neck"
    assert any(d.what == "trunk_root" and d.chosen == "rig:spine" for d in m.decisions)


def test_required_landmarks_present_lists_the_missing_minimum():
    assert required_landmarks_present({Landmark.PELVIS: "Hips"}) == (
        Landmark.HIP_L,
        Landmark.HIP_R,
        Landmark.KNEE_L,
        Landmark.KNEE_R,
        Landmark.ANKLE_L,
        Landmark.ANKLE_R,
    )


@pytest.mark.parametrize(
    "spec",
    [
        RigSpec(),
        replace(RigSpec(), up_axis="z", units_per_m=1.0),
        replace(RigSpec(), units_per_m=39.37),
        replace(RigSpec(), units_per_m=10.0),
    ],
    ids=_spec_id,
)
def test_scale_and_up_axis_are_inferred(tmp_path, spec):
    write_walk(spec, tmp_path / "w.bvh")
    raw = read_bvh_file(tmp_path / "w.bvh", scale=1.0, up_axis="z")
    scale, up, decisions = infer_scale_and_up(raw)
    assert up == spec.up_axis and scale == pytest.approx(1.0 / spec.units_per_m, rel=0.02)
    assert {d.what for d in decisions} >= {"scale", "up_axis"}


def test_uninferable_units_are_a_rig_error(tmp_path):
    write_walk(replace(RigSpec(), units_per_m=3.0), tmp_path / "w.bvh")  # 2.43 or 0.24 m legs
    with pytest.raises(RigError, match="--scale"):
        infer_scale_and_up(read_bvh_file(tmp_path / "w.bvh", scale=1.0, up_axis="z"))
