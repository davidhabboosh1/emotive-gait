"""A compiled rig as a ``RobotProfile`` and ``RobotModel`` the leg IK and the generator accept.

Every role's sign is baked into its hinge axis (``compile_driven``), so ``sign_for`` is all
+1.0; the stand pose is all zeros, which is the reference pose. The floor is ``floor_z`` of the
reference, the median stance height of the very point each foot site sits on, so the sites'
stance height above the floor (``foot_site_floor_height``) is zero by construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from emotive_gait.rig.calibration import ReferencePose
from emotive_gait.rig.compile import CompiledRig
from emotive_gait.rig.mapping import RigMapping
from emotive_gait.robot.loader import RobotModel
from emotive_gait.robot.roles import JointRole, RobotProfile

FOOT_SITES = ("foot_l", "foot_r")


def rig_profile(
    name: str,
    mapping: RigMapping,
    reference: ReferencePose,
    *,
    posture_map: Mapping[JointRole, Mapping[str, float]],
) -> RobotProfile:
    """The profile of a ``driven`` chain compiled from ``mapping`` and ``reference``."""
    return RobotProfile(
        name=name,
        scene_entry="",
        actuator_for=MappingProxyType({role: b.hinge for role, b in mapping.roles.items()}),
        sign_for=MappingProxyType({role: 1.0 for role in mapping.roles}),
        nominal_pelvis_height=float(reference.pelvis_height),
        foot_site_floor_height=0.0,  # the floor is the foot point's own stance height
        foot_sites=FOOT_SITES,
        leg_length=float(mapping.leg_length),
        posture_map=MappingProxyType(dict(posture_map)),
    )


def rig_robot(driven: CompiledRig, profile: RobotProfile) -> RobotModel:
    """``driven`` wrapped for role lookups; the stand pose is qpos zero (the reference)."""
    return RobotModel(model=driven.model, profile=profile, stand_qpos=np.zeros(driven.model.nq))
