"""Synthetic BVH rigs in many naming, channel-order, axis and unit variants, plus a walker.

Geometry is built in metres, +z up, +x forward, +y left, then converted to the spec's up-axis
(y-up files take the inverse of ``to_z_up``: ``(x, y, z) -> (x, z, -y)``) and units. Rest pose:
legs straight, arms in an A-pose 20 degrees from the trunk, every rest rotation zero, one frame.

Channels: every joint has three rotation channels in ``spec.order``; the root has
``Xposition Yposition Zposition`` first. With ``root_above_pelvis`` the root is a ``reference``
bone at the floor (position + rotation channels) and the pelvis is its child with three rotation
channels only; without it the pelvis is the root and sits at ``leg_length_m`` above the floor.

Leg proportions (fractions of ``leg_length_m`` L): pelvis origin at L, hip joint 0.06 L below it,
thigh 0.45 L, shin 0.45 L, ankle 0.04 L above the floor, toe joint 0.01 L above the floor. The
hip-to-ankle distance is thus 0.9 L and ``toe_rest_height_m`` is 0.01 L.

The walker writes joint-space motion only (hip pitch sinusoid, half-wave knee flexion in swing,
counter-phase arm swing, root translating along +x); the feet slide, which the extractor's
position-extremum rule tolerates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from emotive_gait.mocap.readers.bvh import (
    BvhFile,
    BvhJoint,
    euler_from_matrix,
    read_bvh_file,
    rotation_about,
)
from emotive_gait.mocap.skeleton import SkeletonClip
from emotive_gait.mocap.writers.bvh import write_bvh


@dataclass(frozen=True)
class RigSpec:
    family: str = "mixamo"
    order: str = "ZXY"
    up_axis: str = "y"
    units_per_m: float = 100.0
    toes: bool = True
    spine_bones: int = 3
    fingers: bool = True
    clavicles: bool = True
    root_above_pelvis: bool = False
    swap_rest_sides: bool = False
    leg_length_m: float = 0.9


# Canonical part -> bone name. Sided parts hold the LEFT bone's name; the right bone's name is
# derived by ``_mirror``. "spineN" are the spine bones bottom to top; "neck2", "hip_joint" and
# "finger" exist only where the family has them.
FAMILIES: Mapping[str, Mapping[str, str]] = {
    # Mixamo: the documented "mixamorig:" prefix on every bone.
    "mixamo": {
        "pelvis": "mixamorig:Hips",
        "hip": "mixamorig:LeftUpLeg",
        "knee": "mixamorig:LeftLeg",
        "ankle": "mixamorig:LeftFoot",
        "toe": "mixamorig:LeftToeBase",
        "spine1": "mixamorig:Spine",
        "spine2": "mixamorig:Spine1",
        "spine3": "mixamorig:Spine2",
        "neck": "mixamorig:Neck",
        "head": "mixamorig:Head",
        "clavicle": "mixamorig:LeftShoulder",
        "upper_arm": "mixamorig:LeftArm",
        "elbow": "mixamorig:LeftForeArm",
        "wrist": "mixamorig:LeftHand",
        "finger": "mixamorig:LeftHandIndex1",
    },
    # cgspeed's BVH conversion of the CMU data. The CMU ASF names on disk (lhipjoint, lfemur,
    # ...) differ; these are the cgspeed BVH names, not checkable against a file here.
    "motionbuilder": {
        "pelvis": "Hips",
        "hip_joint": "LHipJoint",
        "hip": "LeftUpLeg",
        "knee": "LeftLeg",
        "ankle": "LeftFoot",
        "toe": "LeftToeBase",
        "spine1": "LowerBack",
        "spine2": "Spine",
        "spine3": "Spine1",
        "neck": "Neck",
        "neck2": "Neck1",
        "head": "Head",
        "clavicle": "LeftShoulder",
        "upper_arm": "LeftArm",
        "elbow": "LeftForeArm",
        "wrist": "LeftHand",
        "finger": "LeftFingerBase",  # unverified spelling
    },
    # 100STYLE: verified against tests/mocap/fixtures/style100_neutral_4s.bvh (no finger bones).
    "style100": {
        "pelvis": "Hips",
        "hip": "LeftHip",
        "knee": "LeftKnee",
        "ankle": "LeftAnkle",
        "toe": "LeftToe",
        "spine1": "Chest",
        "spine2": "Chest2",
        "spine3": "Chest3",
        "spine4": "Chest4",
        "neck": "Neck",
        "head": "Head",
        "clavicle": "LeftCollar",
        "upper_arm": "LeftShoulder",
        "elbow": "LeftElbow",
        "wrist": "LeftWrist",
    },
    # Rokoko: whole family unverified (taken from the brief).
    "rokoko": {
        "pelvis": "Hips",  # unverified spelling
        "hip": "LeftThigh",  # unverified spelling
        "knee": "LeftShin",  # unverified spelling
        "ankle": "LeftFoot",  # unverified spelling
        "toe": "LeftToe",  # unverified spelling
        "spine1": "Spine",  # unverified spelling
        "spine2": "Spine1",  # unverified spelling
        "spine3": "Spine2",  # unverified spelling
        "neck": "Neck",  # unverified spelling
        "head": "Head",  # unverified spelling
        "clavicle": "LeftShoulder",  # unverified spelling
        "upper_arm": "LeftArm",  # unverified spelling
        "elbow": "LeftForeArm",  # unverified spelling
        "wrist": "LeftHand",  # unverified spelling
        "finger": "LeftHandIndex1",  # unverified spelling
    },
    # Xsens: whole family unverified (taken from the brief).
    "xsens": {
        "pelvis": "Hips",  # unverified spelling
        "hip": "LeftUpperLeg",  # unverified spelling
        "knee": "LeftLowerLeg",  # unverified spelling
        "ankle": "LeftFoot",  # unverified spelling
        "toe": "LeftToe",  # unverified spelling
        "spine1": "Spine",  # unverified spelling
        "spine2": "Spine1",  # unverified spelling
        "spine3": "Spine2",  # unverified spelling
        "spine4": "Spine3",  # unverified spelling
        "neck": "Neck",  # unverified spelling
        "head": "Head",  # unverified spelling
        "clavicle": "LeftShoulder",  # unverified spelling
        "upper_arm": "LeftArm",  # unverified spelling
        "elbow": "LeftForeArm",  # unverified spelling
        "wrist": "LeftHand",  # unverified spelling
        "finger": "LeftHandIndex1",  # unverified spelling
    },
    # SMPL joint names (the documented 24-joint skeleton; L_Foot is the toe, L_Hand the end bone).
    "smpl": {
        "pelvis": "Pelvis",
        "hip": "L_Hip",
        "knee": "L_Knee",
        "ankle": "L_Ankle",
        "toe": "L_Foot",
        "spine1": "Spine1",
        "spine2": "Spine2",
        "spine3": "Spine3",
        "neck": "Neck",
        "head": "Head",
        "clavicle": "L_Collar",
        "upper_arm": "L_Shoulder",
        "elbow": "L_Elbow",
        "wrist": "L_Wrist",
        "finger": "L_Hand",
    },
    # Rigify deform bones: DEF-spine is the pelvis, .001-.003 the spine, .004 the neck and
    # .006 the head (documented Rigify naming).
    "rigify": {
        "pelvis": "DEF-spine",
        "hip": "DEF-thigh.L",
        "knee": "DEF-shin.L",
        "ankle": "DEF-foot.L",
        "toe": "DEF-toe.L",
        "spine1": "DEF-spine.001",
        "spine2": "DEF-spine.002",
        "spine3": "DEF-spine.003",
        "neck": "DEF-spine.004",
        "head": "DEF-spine.006",
        "clavicle": "DEF-shoulder.L",
        "upper_arm": "DEF-upper_arm.L",
        "elbow": "DEF-forearm.L",
        "wrist": "DEF-hand.L",
        "finger": "DEF-f_index.01.L",  # unverified spelling
    },
    # Unity humanoid (HumanBodyBones) names, from the Unity documentation.
    "unity": {
        "pelvis": "Hips",
        "hip": "LeftUpperLeg",
        "knee": "LeftLowerLeg",
        "ankle": "LeftFoot",
        "toe": "LeftToes",
        "spine1": "Spine",
        "spine2": "Chest",
        "spine3": "UpperChest",
        "neck": "Neck",
        "head": "Head",
        "clavicle": "LeftShoulder",
        "upper_arm": "LeftUpperArm",
        "elbow": "LeftLowerArm",
        "wrist": "LeftHand",
        "finger": "LeftIndexProximal",
    },
}


def _mirror(name: str) -> str:
    if "Left" in name:
        return name.replace("Left", "Right")
    if name.startswith("L_"):
        return "R_" + name[2:]
    if name.startswith("LHip"):
        return "R" + name[1:]
    if name.endswith(".L"):
        return name[:-2] + ".R"
    raise ValueError(f"no right-hand spelling known for {name!r}")


def _sided(names: Mapping[str, str], part: str, left: bool) -> str:
    return names[part] if left else _mirror(names[part])


def pelvis_name(spec: RigSpec) -> str:
    return FAMILIES[spec.family]["pelvis"]


def toe_rest_height_m(spec: RigSpec) -> float:
    """Height of the toe joint (or, without toes, the foot's end site) above the floor."""
    return 0.01 * spec.leg_length_m


def _to_file(v: np.ndarray, up_axis: str) -> np.ndarray:
    """z-up (x forward, y left) vector to the file's axes: the inverse of ``to_z_up``."""
    v = np.asarray(v, dtype=float)
    if up_axis == "z":
        return v
    return np.array([v[0], v[2], -v[1]])


def _channels(spec: RigSpec, *, position: bool) -> tuple[str, ...]:
    rot = tuple(f"{a}rotation" for a in spec.order)
    return (("Xposition", "Yposition", "Zposition") + rot) if position else rot


def build_rig(spec: RigSpec) -> BvhFile:
    """The rest-pose rig: hierarchy plus one zero frame."""
    names = FAMILIES[spec.family]
    leg = spec.leg_length_m
    rows: list[tuple[str, int, np.ndarray, np.ndarray | None]] = []

    def add(name: str, parent: int, offset, end=None) -> int:
        end_site = None if end is None else np.asarray(end, float) * leg
        rows.append((name, parent, np.asarray(offset, float) * leg, end_site))
        return len(rows) - 1

    # Offsets are fractions of leg_length_m, z-up. Lateral sign +1 puts the bone named Left on
    # the left; swap_rest_sides flips it, so the Left bone sits on the right of the pelvis.
    flip = -1.0 if spec.swap_rest_sides else 1.0
    if spec.root_above_pelvis:
        add("reference", -1, (0, 0, 0))
        pelvis = add(names["pelvis"], 0, (0, 0, 1.0))
    else:
        pelvis = add(names["pelvis"], -1, (0, 0, 1.0))

    for left, sign in ((True, 1.0), (False, -1.0)):
        y = sign * flip
        if "hip_joint" in names:
            joint = add(_sided(names, "hip_joint", left), pelvis, (0, 0.04 * y, -0.03))
            hip = add(_sided(names, "hip", left), joint, (0, 0.05 * y, -0.03))
        else:
            hip = add(_sided(names, "hip", left), pelvis, (0, 0.09 * y, -0.06))
        knee = add(_sided(names, "knee", left), hip, (0, 0, -0.45))
        if spec.toes:
            ankle = add(_sided(names, "ankle", left), knee, (0, 0, -0.45))
            # ankle 0.04 above the floor, toe joint 0.01 above it
            add(_sided(names, "toe", left), ankle, (0.12, 0, -0.03), end=(0.06, 0, 0))
        else:
            add(_sided(names, "ankle", left), knee, (0, 0, -0.45), end=(0.15, 0, -0.03))

    top = pelvis
    segment = 0.30 / spec.spine_bones
    for i in range(1, spec.spine_bones + 1):
        top = add(names[f"spine{i}"], top, (0, 0, segment if i > 1 else 0.06))
    neck = add(names["neck"], top, (0, 0, 0.08))
    if "neck2" in names:
        neck = add(names["neck2"], neck, (0, 0, 0.04))
    add(names["head"], neck, (0, 0, 0.08), end=(0, 0, 0.12))

    angle = np.radians(20.0)
    arm_dir = np.array([0.0, np.sin(angle), -np.cos(angle)])
    for left, sign in ((True, 1.0), (False, -1.0)):
        y = sign * flip
        direction = arm_dir * np.array([1.0, y, 1.0])
        if spec.clavicles:
            clavicle = add(_sided(names, "clavicle", left), top, (0, 0.02 * y, 0.07))
            shoulder = add(_sided(names, "upper_arm", left), clavicle, (0, 0.13 * y, 0))
        else:
            shoulder = add(_sided(names, "upper_arm", left), top, (0, 0.15 * y, 0.07))
        elbow = add(_sided(names, "elbow", left), shoulder, direction * 0.20)
        has_finger = spec.fingers and "finger" in names
        wrist = add(
            _sided(names, "wrist", left),
            elbow,
            direction * 0.18,
            end=None if has_finger else direction * 0.08,
        )
        if has_finger:
            add(_sided(names, "finger", left), wrist, direction * 0.08, end=direction * 0.04)

    # The root carries position channels; with root_above_pelvis the pelvis has rotations only.
    joints = tuple(
        BvhJoint(
            name,
            parent,
            _to_file(offset, spec.up_axis) * spec.units_per_m,
            _channels(spec, position=parent < 0),
            None if end is None else _to_file(end, spec.up_axis) * spec.units_per_m,
        )
        for name, parent, offset, end in rows
    )
    return BvhFile(
        joints=joints,
        motion=np.zeros((1, sum(len(j.channels) for j in joints))),
        frame_time=1.0 / 60.0,
        scale=1.0 / spec.units_per_m,
        up_axis=spec.up_axis,
        source=f"synthetic:{spec.family}",
    )


def hip_to_ankle_rest_m(file: BvhFile) -> float:
    """Length of the straight rest leg (thigh + shin offsets) in metres, from the hierarchy."""
    index = {j.name: i for i, j in enumerate(file.joints)}
    for names in FAMILIES.values():
        hip, knee, ankle = names["hip"], names["knee"], names["ankle"]
        if hip in index and knee in index and ankle in index:
            thigh = np.linalg.norm(file.joints[index[knee]].offset)
            shin = np.linalg.norm(file.joints[index[ankle]].offset)
            return float((thigh + shin) * file.scale)
    raise KeyError("no known family's leg bones are in this file")


def write_walk(
    spec: RigSpec,
    path: Path,
    *,
    seconds: float = 6.0,
    cadence: float = 1.8,
    step_length: float = 0.5,
    fps: float = 60.0,
) -> BvhFile:
    """Write a joint-space walking clip for ``spec`` and return it (file units and degrees)."""
    rig = build_rig(spec)
    names = FAMILIES[spec.family]
    t = np.arange(round(seconds * fps)) / fps
    phase = 2.0 * np.pi * (cadence / 2.0) * t  # one cycle is two steps
    hip_amp = np.arcsin(min(0.9, step_length / (1.8 * spec.leg_length_m)))
    knee_amp = np.radians(40.0)
    arm_amp = np.radians(20.0)

    # Flexion (radians, forward positive) per joint name: left leg leads at phase 0, right at pi,
    # each arm in counter-phase with its own-side leg; the knee flexes in the swing half-wave.
    flexion: dict[str, np.ndarray] = {}
    for left, ph in ((True, phase), (False, phase + np.pi)):
        flexion[_sided(names, "hip", left)] = hip_amp * np.sin(ph)
        flexion[_sided(names, "upper_arm", left)] = -arm_amp * np.sin(ph)
        flexion[_sided(names, "knee", left)] = -knee_amp * np.maximum(0.0, np.cos(ph))

    # z-up to file axes, so that R_file = P R_zup P^T.
    p = np.eye(3) if spec.up_axis == "z" else np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
    motion = np.zeros((len(t), rig.motion.shape[1]))
    for joint in rig.joints:
        if joint.name not in flexion:
            continue
        # Rotation about the lateral (+y, z-up) axis by -flexion swings a hanging bone forward.
        r_zup = rotation_about(1, np.degrees(-flexion[joint.name]))
        r_file = p @ r_zup @ p.T
        motion[:, rig.channel_slice(joint.name)] = np.array(
            [euler_from_matrix(r, joint.channels) for r in r_file]
        )
    # Forward is +x in z-up terms and in a y-up file alike (the inverse of to_z_up keeps x).
    motion[:, 0] = step_length * cadence * t * spec.units_per_m
    walked = replace(rig.with_motion(motion), frame_time=1.0 / fps)
    write_bvh(walked, Path(path))
    return walked


def _variants() -> tuple[RigSpec, ...]:
    base = RigSpec()
    out = [base]
    out += [replace(base, family=f) for f in FAMILIES if f != base.family]
    out += [replace(base, order=o) for o in ("XYZ", "ZYX")]
    out += [replace(base, up_axis="z")]
    out += [replace(base, units_per_m=u) for u in (1.0, 39.3701, 10.0)]
    out += [
        replace(base, toes=False),
        replace(base, spine_bones=1),
        replace(base, fingers=False),
        replace(base, clavicles=False),
        replace(base, root_above_pelvis=True),
        replace(base, swap_rest_sides=True),
        replace(base, leg_length_m=0.6),
        replace(base, leg_length_m=1.1),
    ]
    return tuple(out)


FAMILY_VARIANTS: tuple[RigSpec, ...] = _variants()


def walk_file(spec: RigSpec, directory: Path, **walk_kwargs) -> Path:
    path = Path(directory) / f"{spec.family}.bvh"
    write_walk(spec, path, **walk_kwargs)
    return path


def standing_file(spec: RigSpec, directory: Path) -> Path:
    rig = build_rig(spec)
    path = Path(directory) / f"{spec.family}_standing.bvh"
    write_bvh(rig.with_motion(np.zeros((120, rig.motion.shape[1]))), path)
    return path


def walk_and_clip(spec: RigSpec, directory: Path) -> tuple[BvhFile, SkeletonClip]:
    path = walk_file(spec, directory)
    file = read_bvh_file(path, scale=1.0 / spec.units_per_m, up_axis=spec.up_axis)
    return file, file.to_clip(end_sites=True)


def _with_position_columns(f: BvhFile, bone: str, vector) -> BvhFile:
    """``f`` with constant position channels (``vector``, file units and axes) prepended to
    ``bone``'s channels; the OFFSET is left as is and ``f.motion``'s other columns are kept."""
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
    extra = np.tile(np.asarray(vector, dtype=float), (f.motion.shape[0], 1))
    motion = np.concatenate([f.motion[:, :start], extra, f.motion[:, start:]], axis=1)
    return replace(f, joints=joints, motion=motion)


def duplicate_offsets(f: BvhFile, bone: str) -> BvhFile:
    """``f`` with position channels on ``bone`` equal to its OFFSET (the Bandai pattern):
    under the column-replaces-axis rule the reader's positions are unchanged."""
    joint = next(j for j in f.joints if j.name == bone)
    return _with_position_columns(f, bone, joint.offset)


def move_offset_into_channels(f: BvhFile, bone: str, vector) -> BvhFile:
    """``f`` with position channels on ``bone`` set to ``vector`` (file units and axes, length
    3); the OFFSET stays, so the column replaces it and the bone vector becomes ``vector``."""
    return _with_position_columns(f, bone, vector)
