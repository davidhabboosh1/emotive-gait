"""Inject a pitch-only neck into a robot that has none (the Unitree G1).

The G1 head is a rigid mesh welded to ``torso_link``. We re-parent it onto a new ``head_link``
body hung from a hinge joint so head inclination, the strongest posture cue for sadness, becomes
controllable. The parent's inertia is explicit and already includes the head, so the split below
keeps total robot mass unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from emotive_gait.robot.root import RootMode, to_root_mode


@dataclass(frozen=True)
class NeckConfig:
    """Where and how to hang the neck.

    Provenance: ``mass``, ``diaginertia`` and ``centroid`` came from compiling a probe body
    carrying the ``head_link`` mesh at density 1000 under menagerie revision
    ``unitree_g1-ad24af3fa9bc1a99`` and reading its body_mass / body_inertia / body_ipos. They
    are ESTIMATES from the visual mesh at uniform density, not Unitree's CAD values. ``pivot``
    z=0.32 was chosen because the head centroid sits at z~0.385 in torso frame, placing the
    hinge roughly at the neck base. ``parent_mass_after`` = 7.818 - 1.8025; it is only a
    cross-check that the model revision still matches these constants.
    """

    parent_body: str = "torso_link"
    mesh_match: str = "head"
    pivot: tuple[float, float, float] = (0.0, 0.0, 0.32)
    mass: float = 1.8025
    diaginertia: tuple[float, float, float] = (0.008112, 0.007909, 0.002908)
    # Head mesh vertex centroid, measured from the mesh origin.
    centroid: tuple[float, float, float] = (0.003687, 0.0000248, 0.429132)
    parent_mass_after: float = 6.0155
    joint_range: tuple[float, float] = (-0.5, 0.8)
    kp: float = 300.0
    kv: float = 30.0
    joint_name: str = "neck_pitch_joint"


DEFAULT_NECK = NeckConfig()


def _head_geoms(body: mujoco.MjsBody, mesh_match: str) -> list[mujoco.MjsGeom]:
    found = []
    g = body.first_geom()
    while g is not None:
        if mesh_match in (g.meshname or ""):
            found.append(g)
        g = body.next_geom(g)
    return found


def _subtract_child_inertia(
    parent: mujoco.MjsBody, child_mass: float, child_com: np.ndarray, child_diag: np.ndarray
) -> float:
    """Remove a child's mass, CoM and inertia from ``parent``'s explicit inertial, in place.

    ``child_com`` is in the parent frame and ``child_diag`` is assumed parent-axis-aligned.
    Returns the parent's new mass.
    """
    eye = np.eye(3)

    def shift(mass: float, com: np.ndarray) -> np.ndarray:
        return mass * (com @ com * eye - np.outer(com, com))

    big_m = float(parent.mass)
    c_big = np.asarray(parent.ipos, dtype=float)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(parent.iquat, dtype=float))
    rot = rot.reshape(3, 3)
    i_big_o = rot @ np.diag(parent.inertia) @ rot.T + shift(big_m, c_big)
    i_child_o = np.diag(child_diag) + shift(child_mass, child_com)

    m_r = big_m - child_mass
    c_r = (big_m * c_big - child_mass * child_com) / m_r
    i_r = i_big_o - i_child_o - shift(m_r, c_r)
    i_r = (i_r + i_r.T) / 2
    d_r, r_r = np.linalg.eigh(i_r)
    if np.any(d_r <= 0):
        raise ValueError(
            f"head estimate exceeds the torso's recorded inertia (remainder eigenvalues {d_r}); "
            "the split is physically impossible for this model revision"
        )
    if np.linalg.det(r_r) < 0:
        r_r[:, -1] *= -1
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, r_r.reshape(9))
    parent.mass = m_r
    parent.ipos = c_r
    parent.iquat = quat
    parent.inertia = d_r
    return m_r


def inject_neck(spec: mujoco.MjSpec, cfg: NeckConfig = DEFAULT_NECK) -> mujoco.MjSpec:
    """Add a ``head_link`` body on a pitch hinge under ``cfg.parent_body``, in place."""
    torso = spec.body(cfg.parent_body)
    if torso is None:
        raise KeyError(f"body {cfg.parent_body!r} not found")
    if not torso.explicitinertial:
        raise ValueError(
            f"{cfg.parent_body!r} has no explicit inertial; setting its mass would be ignored"
        )
    sources = _head_geoms(torso, cfg.mesh_match)
    if not sources:
        raise KeyError(f"no geom on {cfg.parent_body!r} uses a mesh matching {cfg.mesh_match!r}")

    pivot = np.asarray(cfg.pivot, dtype=float)
    head = torso.add_body(name="head_link", pos=pivot)
    head.explicitinertial = True
    head.mass = cfg.mass
    head.inertia = list(cfg.diaginertia)
    head_com = np.asarray(sources[0].pos, dtype=float) + np.asarray(cfg.centroid)  # torso frame
    head.ipos = head_com - pivot

    for src in sources:
        head.add_geom(
            type=src.type,
            meshname=src.meshname,
            pos=np.asarray(src.pos, dtype=float) - pivot,
            quat=src.quat,
            group=src.group,
            contype=src.contype,
            conaffinity=src.conaffinity,
            rgba=src.rgba,
            material=src.material,
            density=0.0,  # inertia is explicit; do not double count
        )
        spec.delete(src)

    m_r = _subtract_child_inertia(torso, cfg.mass, head_com, np.asarray(cfg.diaginertia))
    if not math.isclose(m_r, cfg.parent_mass_after, abs_tol=1e-6):
        raise ValueError(
            f"torso mass remainder {m_r} != expected {cfg.parent_mass_after}: this model "
            "revision's torso mass no longer matches the one these constants were derived against"
        )

    # Inherit joint conventions from a neighbouring joint so the neck is dynamically like the rest.
    ref = next((j for j in spec.joints if j.name == "waist_pitch_joint"), None)
    conv = (
        {}
        if ref is None
        else {
            "armature": ref.armature,
            "frictionloss": ref.frictionloss,
            "actfrcrange": ref.actfrcrange,
            "actfrclimited": ref.actfrclimited,
        }
    )
    head.add_joint(
        name=cfg.joint_name,
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0.0, 1.0, 0.0],
        range=list(cfg.joint_range),
        limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        **conv,
    )
    act = spec.add_actuator(name=cfg.joint_name, target=cfg.joint_name)
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    act.gainprm = [cfg.kp] + [0.0] * 9
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    act.biasprm = [0.0, -cfg.kp, -cfg.kv] + [0.0] * 7
    act.ctrlrange = list(cfg.joint_range)
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE

    # Adding a joint changes nq, which invalidates every stored keyframe.
    for key in list(spec.keys):
        spec.delete(key)
    return spec


def build_model(
    entry_xml: str,
    *,
    neck: NeckConfig | None,
    root: RootMode,
    pelvis_z: float | None,
    offwidth: int = 960,
    offheight: int = 720,
) -> mujoco.MjModel:
    """Compile ``entry_xml`` with an optional neck, a root mode and an offscreen framebuffer.

    Neck and root are applied to one spec compiled once, so the framebuffer size survives.
    ``pelvis_z`` re-heights a welded pelvis (see ``to_root_mode``).
    """
    spec = mujoco.MjSpec.from_file(entry_xml)
    visual_global = getattr(spec.visual, "global")
    visual_global.offwidth = offwidth
    visual_global.offheight = offheight
    if neck is not None:
        inject_neck(spec, neck)
    to_root_mode(spec, root, pelvis_z=pelvis_z)
    for key in list(spec.keys):  # a changed root or joint set invalidates every stored keyframe
        spec.delete(key)
    return spec.compile()


def build_with_neck(
    entry_xml: str,
    cfg: NeckConfig = DEFAULT_NECK,
    *,
    offwidth: int = 960,
    offheight: int = 720,
    root: RootMode = RootMode.FREE,
    pelvis_z: float | None = None,
) -> mujoco.MjModel:
    """Compile ``entry_xml`` with a neck; a thin wrapper over :func:`build_model`."""
    return build_model(
        entry_xml,
        neck=cfg,
        root=root,
        pelvis_z=pelvis_z,
        offwidth=offwidth,
        offheight=offheight,
    )


def standing_qpos(src_model: mujoco.MjModel, new_model: mujoco.MjModel) -> np.ndarray:
    """Map ``src_model``'s ``stand`` keyframe onto ``new_model`` by joint name."""
    key = mujoco.mj_name2id(src_model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key < 0:
        raise KeyError("source model has no 'stand' keyframe")
    src_qpos = src_model.key_qpos[key]
    out = np.array(new_model.qpos0, dtype=float)
    if new_model.njnt and new_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        out[0:7] = src_qpos[0:7]  # free root: position + quaternion
    # Otherwise the root is a slide or welded: its slide joint stays 0, the pelvis at spec pos.
    for j in range(src_model.njnt):
        if src_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(src_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        new_j = mujoco.mj_name2id(new_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if new_j < 0:
            raise KeyError(f"joint {name!r} missing from the new model")
        out[new_model.jnt_qposadr[new_j]] = src_qpos[src_model.jnt_qposadr[j]]
    return out
