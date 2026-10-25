"""Root (pelvis) attachment modes for physical playback.

Open-loop stepping on a free floating base is an unstable equilibrium with no feedback path, so
the physics tier constrains the root. ``Z_RAIL`` leaves only vertical translation; ``XZ_RAIL``
adds a forward slide that playback prescribes; ``WELD`` fixes the pelvis. All remove the pelvis freejoint, so ``nq == nv`` with no quaternion in ``qpos``.
"""

from __future__ import annotations

from enum import Enum

import mujoco
import numpy as np

ROOT_SLIDE_JOINT = "root_slide_z"
ROOT_SLIDE_X_JOINT = "root_slide_x"
_PELVIS = "pelvis"
_SLIDE_RANGE = (-0.5, 0.5)
_X_RANGE = (-1.0, 20.0)


class RootMode(str, Enum):
    """``XZ_RAIL`` is experimental: a prescribed forward slide, and the open-loop legs lose floor
    contact within 0.5 s at neutral; it is kept as a measured negative result."""

    FREE = "free"
    Z_RAIL = "z_rail"
    XZ_RAIL = "xz_rail"
    WELD = "weld"


def to_root_mode(
    spec: mujoco.MjSpec, mode: RootMode, *, pelvis_z: float | None = None
) -> mujoco.MjSpec:
    """Constrain the pelvis of ``spec`` per ``mode``, in place. ``FREE`` is a no-op.

    ``pelvis_z`` re-heights the pelvis, and only on ``WELD`` (the rails keep the keyframe
    height: the rail is free in z and the playback ramp lowers it).

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
    if mode is RootMode.WELD and pelvis_z is not None:
        pos = np.array(pelvis.pos, dtype=float)
        pos[2] = pelvis_z
        pelvis.pos = pos
    if mode is RootMode.XZ_RAIL:
        # Added before the Z slide so qpos[0] is x and qpos[1] is z.
        pelvis.add_joint(
            name=ROOT_SLIDE_X_JOINT,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[1.0, 0.0, 0.0],
            range=list(_X_RANGE),
        )
    if mode in (RootMode.Z_RAIL, RootMode.XZ_RAIL):
        pelvis.add_joint(
            name=ROOT_SLIDE_JOINT,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0.0, 0.0, 1.0],
            range=list(_SLIDE_RANGE),
        )
    return spec
