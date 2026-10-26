"""What "positive" means for each joint role, as a landmark motion.

``ROLE_CONVENTIONS`` maps a role to ``(landmark value, direction)``: a small positive rotation of
the role's joint (times the profile's ``sign``) moves that landmark in that direction. The G1 is
the reference rig (spec 4.5); a rig whose joint disagrees needs a negative ``sign``. Roles that
are IK-solved or unused with symmetric ranges (``HIP_YAW_*``, ``ANKLE_ROLL_*``, ``WAIST_YAW``,
``WAIST_ROLL``) have no row and no probe.

Pure geometry on NumPy arrays: this module imports nothing from mujoco or the other packages.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from emotive_gait.robot.roles import JointRole

# Probe rotation (rad) used to read a joint's convention from a rig.
PROBE_RAD = 0.05

_LEFT: dict[JointRole, tuple[str, str]] = {
    JointRole.HIP_PITCH_L: ("knee_l", "backward"),
    JointRole.KNEE_L: ("ankle_l", "backward"),
    JointRole.ANKLE_PITCH_L: ("toe_l", "down"),
    JointRole.HIP_ROLL_L: ("knee_l", "outward"),
    JointRole.SHOULDER_PITCH_L: ("elbow_l", "backward"),
    JointRole.SHOULDER_ROLL_L: ("elbow_l", "outward"),
    JointRole.ELBOW_L: ("wrist_l", "toward_shoulder"),
}

ROLE_CONVENTIONS: Mapping[JointRole, tuple[str, str]] = {
    **_LEFT,
    **{role.mirror(): (lm[:-2] + "_r", d) for role, (lm, d) in _LEFT.items()},
    JointRole.WAIST_PITCH: ("neck", "forward"),
    JointRole.NECK_PITCH: ("head", "forward"),
}


def convention_holds(
    role: JointRole,
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
    *,
    forward: np.ndarray,
    left: np.ndarray,
    up: np.ndarray,
) -> bool:
    """Whether the landmark named for ``role`` moved from ``before`` to ``after`` as the
    convention says.

    ``before``/``after`` map landmark values to world points. The displacement is projected on
    the direction's unit vector and must be positive; ``"outward"`` is ``+left`` for ``_l`` roles
    and ``-left`` for ``_r`` roles; ``"toward_shoulder"`` requires a shorter distance to
    ``shoulder_<side>``.
    """
    landmark, direction = ROLE_CONVENTIONS[role]
    a, b = np.asarray(before[landmark], float), np.asarray(after[landmark], float)
    if direction == "toward_shoulder":
        shoulder = f"shoulder_{landmark[-1]}"
        return bool(
            np.linalg.norm(b - np.asarray(after[shoulder], float))
            < np.linalg.norm(a - np.asarray(before[shoulder], float))
        )
    if direction == "outward":
        axis = np.asarray(left, float) * (1.0 if role.value.endswith("_l") else -1.0)
    else:
        axis = {
            "forward": np.asarray(forward, float),
            "backward": -np.asarray(forward, float),
            "down": -np.asarray(up, float),
        }[direction]
    return bool(np.dot(b - a, axis / np.linalg.norm(axis)) > 0.0)
