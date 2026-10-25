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
        if c.geom1 == floor:
            other = c.geom2
        elif c.geom2 == floor:
            other = c.geom1
        else:
            continue
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
    """Point on the ground about which the horizontal ground-reaction moment vanishes."""
    com = np.array(data.subtree_com[0], dtype=float)
    moment = np.zeros(3)
    f_z = 0.0
    n = 0
    for pos, force, _ in _foot_contacts(model, data):
        moment += np.cross(pos - com, force)
        f_z += force[2]
        n += 1
    if n == 0 or abs(f_z) < _EPS:
        return None
    zmp_x = com[0] - moment[1] / f_z
    zmp_y = com[1] + moment[0] / f_z
    return np.array([zmp_x, zmp_y])


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
