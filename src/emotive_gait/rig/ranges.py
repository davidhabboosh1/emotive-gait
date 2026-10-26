"""Joint ranges for the driven hinges of a compiled rig (spec 4.3).

BVH joints have no limits, so every driven hinge gets a rig-independent, human-like range:
radians about the calibrated zero (the actor's reference pose), in the positive direction of
the role's convention (``robot.conventions``; the sign is baked into the hinge axis). These are
reachability guards for the IK and the generator's clamp, not evidence claims. A role without a
row here is not driven on a rig.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from emotive_gait.robot.roles import JointRole

R = JointRole

_SIDED: dict[str, tuple[float, float]] = {
    "HIP_PITCH": (-1.2, 2.0),
    "HIP_ROLL": (-0.8, 0.8),
    "HIP_YAW": (-0.8, 0.8),
    "KNEE": (0.0, 2.6),  # flexion only, no hyperextension
    "ANKLE_PITCH": (-1.0, 1.0),
    "ANKLE_ROLL": (-0.6, 0.6),
    "SHOULDER_PITCH": (-3.0, 3.0),
    "SHOULDER_ROLL": (-0.3, 2.5),  # abduction
    "ELBOW": (0.0, 2.6),  # flexion
}

RANGES: Mapping[JointRole, tuple[float, float]] = MappingProxyType(
    {
        **{R[f"{stem}_{side}"]: limits for stem, limits in _SIDED.items() for side in "LR"},
        R.WAIST_PITCH: (-0.8, 0.8),
        R.NECK_PITCH: (-0.8, 0.8),
    }
)
