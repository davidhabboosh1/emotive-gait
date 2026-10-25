"""Balance metrics read from an already-computed ``MjData``.

These are instrumentation, not control: nothing here steps or forwards the simulation.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
_FLOOR_GEOM = "floor"
_EPS = 1e-9
_UPRIGHT_THRESHOLD = 0.6


def _body_id(model: mujoco.MjModel, name: str) -> int:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if idx < 0:
        raise ValueError(f"body {name!r} not found in model")
    return idx


def _foot_contacts(model: mujoco.MjModel, data: mujoco.MjData):
    """Yield ``(position, world_force, normal_force)`` for every floor-foot contact."""
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _FLOOR_GEOM)
    if floor < 0:
        raise ValueError(f"geom {_FLOOR_GEOM!r} not found in model")
    feet = {_body_id(model, n) for n in _FOOT_BODIES}
    wrench = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if floor not in (c.geom1, c.geom2):
            continue
        # The contact frame's normal points from geom1 to geom2, so the force below is the
        # force on geom2. That is the force the floor exerts on the foot only if the floor is
        # geom1 (true while the floor is geom id 0); fail loudly if a model orders it otherwise.
        assert c.geom1 == floor, "floor must be geom1 of its contacts"
        other = c.geom2
        if int(model.geom_bodyid[other]) not in feet:
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        # Contact-frame rows are (normal, tangent1, tangent2) in world coordinates.
        force_world = np.asarray(c.frame).reshape(3, 3).T @ wrench[:3]
        yield np.array(c.pos, dtype=float), force_world, float(wrench[0])


def center_of_pressure(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray | None:
    """Normal-force-weighted mean foot contact position (world xy), or ``None``."""
    total = 0.0
    weighted = np.zeros(2)
    for pos, _, normal in _foot_contacts(model, data):
        total += normal
        weighted += normal * pos[:2]
    if total < _EPS:
        return None
    return weighted / total


def zero_moment_point(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray | None:
    """Ground point about which the horizontal ground-reaction moment vanishes.

    Moments are taken about the ground origin, which stays exact under friction:
    ``zmp = (-M_y / F_z, M_x / F_z)`` with ``M = sum(pos x f)``. The original project's
    ``compute_zmp`` (moments about the CoM, ``x_com - M_y/F_z``) is the special case with no
    horizontal ground force. Returns ``None`` without foot contact or with ``|F_z| ~ 0``.
    """
    moment = np.zeros(3)
    f_z = 0.0
    n = 0
    for pos, force, _ in _foot_contacts(model, data):
        moment += np.cross(pos, force)
        f_z += force[2]
        n += 1
    if n == 0 or abs(f_z) < _EPS:
        return None
    return np.array([-moment[1] / f_z, moment[0] / f_z])


def support_polygon(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Convex hull (ordered vertices, ``(N, 2)``) of foot contact points.

    With fewer than three distinct points the points are returned as-is.
    """
    pts = np.array([pos[:2] for pos, _, _ in _foot_contacts(model, data)], dtype=float).reshape(-1, 2)
    if len(np.unique(pts, axis=0)) < 3:
        return pts
    try:
        return pts[ConvexHull(pts).vertices]
    except QhullError:  # collinear points: hull is degenerate
        return pts


def uprightness(model: mujoco.MjModel, data: mujoco.MjData, *, body: str = "torso_link") -> float:
    """World-z component of ``body``'s own z-axis: 1 upright, 0 horizontal, -1 inverted."""
    return float(data.xmat[_body_id(model, body)].reshape(3, 3)[2, 2])


def is_upright(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    threshold: float = _UPRIGHT_THRESHOLD,
    body: str = "torso_link",
) -> bool:
    return uprightness(model, data, body=body) >= threshold
