"""Root (pelvis) attachment modes for physical playback.

Open-loop stepping on a free floating base is an unstable equilibrium with no feedback path, so
the physics tier constrains the root. ``Z_RAIL`` leaves only vertical translation; ``WELD`` fixes
the pelvis. Both remove the pelvis freejoint, so ``nq == nv`` with no quaternion in ``qpos``.
"""

from __future__ import annotations

from enum import Enum

import mujoco

ROOT_SLIDE_JOINT = "root_slide_z"
_PELVIS = "pelvis"
_SLIDE_RANGE = (-0.5, 0.5)


class RootMode(str, Enum):
    FREE = "free"
    Z_RAIL = "z_rail"
    WELD = "weld"


def to_root_mode(spec: mujoco.MjSpec, mode: RootMode) -> mujoco.MjSpec:
    """Constrain the pelvis of ``spec`` per ``mode``, in place. ``FREE`` is a no-op.

    Keyframes are not touched here: removing the freejoint invalidates them, so the caller
    deletes them before compiling.
    """
    if mode is RootMode.FREE:
        return spec
    pelvis = spec.body(_PELVIS)
    if pelvis is None:
        raise KeyError(f"body {_PELVIS!r} not found")
    for joint in list(pelvis.joints):
        spec.delete(joint)
    if mode is RootMode.Z_RAIL:
        pelvis.add_joint(
            name=ROOT_SLIDE_JOINT,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0.0, 0.0, 1.0],
            range=list(_SLIDE_RANGE),
        )
    return spec
