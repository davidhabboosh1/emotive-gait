"""Per-dataset landmark maps (``Landmark`` -> joint name in the reader's ``SkeletonClip``).

100STYLE: BVH bones as read from the files' headers (joint = bone origin).
CMU: the ASF/AMC reader returns bone TIPS, so the knee is ``lfemur``, the ankle ``ltibia``, the
toe ``lfoot``, the elbow ``lhumerus`` and the wrist ``lradius``; the hip is the end of
``lhipjoint``.
BML: the 15 columns of the Point-Tracked Data files, in file order (no toe markers).
"""

from __future__ import annotations

from collections.abc import Mapping

from emotive_gait.mocap.skeleton import Landmark as L
from emotive_gait.mocap.skeleton import LandmarkMap

STYLE100: LandmarkMap = {
    L.PELVIS: "Hips",
    L.HIP_L: "LeftHip",
    L.HIP_R: "RightHip",
    L.KNEE_L: "LeftKnee",
    L.KNEE_R: "RightKnee",
    L.ANKLE_L: "LeftAnkle",
    L.ANKLE_R: "RightAnkle",
    L.TOE_L: "LeftToe",
    L.TOE_R: "RightToe",
    L.SHOULDER_L: "LeftShoulder",
    L.SHOULDER_R: "RightShoulder",
    L.ELBOW_L: "LeftElbow",
    L.ELBOW_R: "RightElbow",
    L.WRIST_L: "LeftWrist",
    L.WRIST_R: "RightWrist",
    L.NECK: "Neck",
    L.HEAD: "Head",
}

CMU: LandmarkMap = {
    L.PELVIS: "root",
    L.HIP_L: "lhipjoint",
    L.HIP_R: "rhipjoint",
    L.KNEE_L: "lfemur",
    L.KNEE_R: "rfemur",
    L.ANKLE_L: "ltibia",
    L.ANKLE_R: "rtibia",
    L.TOE_L: "lfoot",
    L.TOE_R: "rfoot",
    L.SHOULDER_L: "lclavicle",
    L.SHOULDER_R: "rclavicle",
    L.ELBOW_L: "lhumerus",
    L.ELBOW_R: "rhumerus",
    L.WRIST_L: "lradius",
    L.WRIST_R: "rradius",
    L.NECK: "thorax",
    L.HEAD: "head",
}

BML_PTD_JOINTS: tuple[str, ...] = (
    "Head",
    "Neck",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RShoulder",
    "RElbow",
    "RWrist",
    "Pelvis",
    "LHip",
    "LKnee",
    "LAnkle",
    "RHip",
    "RKnee",
    "RAnkle",
)

BML_PTD: LandmarkMap = {
    L.PELVIS: "Pelvis",
    L.HIP_L: "LHip",
    L.HIP_R: "RHip",
    L.KNEE_L: "LKnee",
    L.KNEE_R: "RKnee",
    L.ANKLE_L: "LAnkle",
    L.ANKLE_R: "RAnkle",
    L.SHOULDER_L: "LShoulder",
    L.SHOULDER_R: "RShoulder",
    L.ELBOW_L: "LElbow",
    L.ELBOW_R: "RElbow",
    L.WRIST_L: "LWrist",
    L.WRIST_R: "RWrist",
    L.NECK: "Neck",
    L.HEAD: "Head",
}

LANDMARKS: Mapping[str, LandmarkMap] = {"100style": STYLE100, "cmu": CMU, "bml": BML_PTD}
