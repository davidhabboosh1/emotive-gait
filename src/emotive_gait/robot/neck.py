"""Inject a pitch-only neck into a robot that has none (the Unitree G1).

The G1 head is a rigid mesh welded to ``torso_link``. We re-parent it onto a new ``head_link``
body hung from a hinge joint so head inclination, the strongest posture cue for sadness, becomes
controllable. The parent's inertia is explicit and already includes the head, so the split below
keeps total robot mass unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class NeckConfig:
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


def inject_neck(spec: mujoco.MjSpec, cfg: NeckConfig = DEFAULT_NECK) -> mujoco.MjSpec:
    """Add a ``head_link`` body on a pitch hinge under ``cfg.parent_body``, in place."""
    torso = spec.find_body(cfg.parent_body)
    if torso is None:
        raise KeyError(f"body {cfg.parent_body!r} not found")
    sources = _head_geoms(torso, cfg.mesh_match)
    if not sources:
        raise KeyError(f"no geom on {cfg.parent_body!r} uses a mesh matching {cfg.mesh_match!r}")

    pivot = np.asarray(cfg.pivot, dtype=float)
    head = torso.add_body(name="head_link", pos=pivot)
    head.explicitinertial = True
    head.mass = cfg.mass
    head.inertia = list(cfg.diaginertia)
    head.ipos = np.asarray(sources[0].pos, dtype=float) + np.asarray(cfg.centroid) - pivot

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
            density=0.0,  # inertia is explicit; do not double count
        )
        src.delete()

    torso.mass = cfg.parent_mass_after

    head.add_joint(
        name=cfg.joint_name,
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0.0, 1.0, 0.0],
        range=list(cfg.joint_range),
        limited=mujoco.mjtLimited.mjLIMITED_TRUE,
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
        key.delete()
    return spec


def build_with_neck(
    entry_xml: str, cfg: NeckConfig = DEFAULT_NECK, *, offwidth: int = 960, offheight: int = 720
) -> mujoco.MjModel:
    """Compile ``entry_xml`` with a neck injected and an offscreen framebuffer of the given size."""
    spec = mujoco.MjSpec.from_file(entry_xml)
    visual_global = getattr(spec.visual, "global")
    visual_global.offwidth = offwidth
    visual_global.offheight = offheight
    inject_neck(spec, cfg)
    return spec.compile()


def standing_qpos(src_model: mujoco.MjModel, new_model: mujoco.MjModel) -> np.ndarray:
    """Map ``src_model``'s ``stand`` keyframe onto ``new_model`` by joint name."""
    key = mujoco.mj_name2id(src_model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key < 0:
        raise KeyError("source model has no 'stand' keyframe")
    src_qpos = src_model.key_qpos[key]
    out = np.array(new_model.qpos0, dtype=float)
    out[0:7] = src_qpos[0:7]  # free root: position + quaternion
    for j in range(src_model.njnt):
        if src_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(src_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        new_j = mujoco.mj_name2id(new_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if new_j < 0:
            raise KeyError(f"joint {name!r} missing from the new model")
        out[new_model.jnt_qposadr[new_j]] = src_qpos[src_model.jnt_qposadr[j]]
    return out
