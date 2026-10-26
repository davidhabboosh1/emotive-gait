"""Bind a BVH file's bones to the seventeen :class:`Landmark` s, offline and deterministically.

Names first, geometry as referee:

1. Family: the family in ``data/bone_names.yaml`` with the largest overlap of its bone names
   with the file (at least ``MIN_FAMILY_OVERLAP``) is chosen and its exact names win.
2. Tokens, only for landmarks the family leaves unbound: a name is normalised (namespace up to
   the last ``:`` and a ``DEF-`` prefix stripped, lower-cased) and classified into a part and a
   side by the lexicon's tokens in their listed order. Picks follow the hierarchy: the knee is
   the shallowest knee bone (below the hip when the hip is bound), the hip its nearest hip
   ancestor, the ankle the shallowest ankle bone below the knee, the toe the shallowest toe
   bone below the ankle; the arm likewise (shoulder = upper-arm bone, else the elbow's parent).
   A sided bone needs a side token; one without is not bound.
3. Structure: ``PELVIS`` falls back to the hips' lowest common ancestor; the lowest spine bone
   is the trunk root; ``NECK`` is the neck bone else the highest spine bone; ``HEAD`` the head
   bone else the neck's End Site; ``TOE`` the toe bone else the foot's End Site. Hip, knee and
   ankle must be an ancestor chain.
4. Side referee: the sign of each limb root's lateral coordinate relative to the pelvis along
   ``path_frame(pelvis, fps).left``, averaged over the clip. Limbs whose names disagree with
   geometry are swapped, with a warning that says "side". A clip without forward travel has
   no path frame: the referee is skipped and the names are kept, with a warning.
5. ``overrides`` (landmark value -> bone name) are applied last, ``source="user"``.

Every choice is a :class:`Decision`, so ``inspect`` can print an audit.

Roles (``bind_roles``, after ``calibration.reference_pose``): each bone that owns a landmark
has its hinges probed on the ``full`` chain at the reference pose and classified as pitch, roll
or yaw by the probe's rotation axis in the path frame, with a sign that makes the role's
``ROLE_CONVENTIONS`` row hold; the probe machinery lives in ``rig.calibration``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from types import MappingProxyType
from typing import Literal

import numpy as np
import yaml

from emotive_gait.gait.ik import LEG_ROLES
from emotive_gait.mocap.events import path_frame
from emotive_gait.mocap.extract import leg_length
from emotive_gait.mocap.readers.bvh import BvhFile
from emotive_gait.mocap.skeleton import Landmark, LandmarkMap, SkeletonClip
from emotive_gait.rig.calibration import HingeProbe, ReferencePose, probe_hinges
from emotive_gait.rig.compile import CompiledRig
from emotive_gait.rig.errors import RigError
from emotive_gait.rig.ranges import RANGES
from emotive_gait.robot.conventions import ROLE_CONVENTIONS, convention_holds
from emotive_gait.robot.roles import POSTURE_ROLES, JointRole

__all__ = [
    "Decision",
    "LandmarkMapping",
    "RigError",  # re-exported from rig.errors
    "RigMapping",
    "RoleBinding",
    "bind_roles",
    "infer_scale_and_up",
    "map_landmarks",
    "required_landmarks_present",
]

L = Landmark
MIN_FAMILY_OVERLAP = 6
# Frozen inference constants: candidate file-unit-to-metre scales, the target hip-to-ankle rest
# distance, and the window a chosen scale must land the leg inside.
SCALES = (1.0, 0.01, 0.0254, 0.1, 0.001)
LEG_TARGET_M = 0.8
LEG_WINDOW_M = (0.3, 1.3)

REQUIRED: tuple[Landmark, ...] = (
    L.PELVIS,
    L.HIP_L,
    L.HIP_R,
    L.KNEE_L,
    L.KNEE_R,
    L.ANKLE_L,
    L.ANKLE_R,
)
# landmark stem -> token part
_LEG = (("hip", "hip"), ("knee", "knee"), ("ankle", "ankle"), ("toe", "toe"))
_ARM = (("shoulder", "upper_arm"), ("elbow", "forearm"), ("wrist", "hand"))
_SIDES = ("l", "r")

Source = Literal["name", "geometry", "user"]


@dataclass(frozen=True)
class Decision:
    """One mapping choice and where it came from."""

    what: str
    chosen: str
    source: Source
    note: str = ""


@dataclass(frozen=True)
class LandmarkMapping:
    """The bound landmarks, the matched family (``None``: tokens only) and the audit trail."""

    landmarks: LandmarkMap
    family: str | None
    decisions: tuple[Decision, ...]
    warnings: tuple[str, ...]
    trunk_root: str | None = None  # the lowest spine bone; None without a spine


def required_landmarks_present(landmarks: LandmarkMap) -> tuple[Landmark, ...]:
    """The missing ones from PELVIS, HIP, KNEE, ANKLE both sides (in that order)."""
    return tuple(lm for lm in REQUIRED if lm not in landmarks)


@cache
def _lexicon() -> dict:
    text = (resources.files("emotive_gait.rig") / "data" / "bone_names.yaml").read_text("utf-8")
    return yaml.safe_load(text)


def _normalise(name: str) -> str:
    name = name.rsplit(":", 1)[-1]
    return name.removeprefix("DEF-").lower()


def _side_of(norm: str) -> str | None:
    tokens = _lexicon()["tokens"]
    for side, key in (("l", "side_left"), ("r", "side_right")):
        for token in tokens[key]:
            if token[0] in "_.":
                hit = norm.endswith(token)
            elif token.endswith("_"):
                hit = norm.startswith(token)
            else:
                hit = token in norm
            if hit:
                return side
    return None


def _part_of(norm: str) -> str | None:
    for part, tokens in _lexicon()["tokens"].items():
        if part.startswith("side_"):
            continue
        if any(token in norm for token in tokens):
            return part
    return None


def _landmark(stem: str, side: str | None = None) -> Landmark:
    return Landmark(stem if side is None else f"{stem}_{side}")


@dataclass
class _Tree:
    names: tuple[str, ...]
    parents: tuple[int, ...]
    end_sites: frozenset[str]  # bones that carry an End Site
    index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.index = {n: i for i, n in enumerate(self.names)}

    @classmethod
    def of(cls, file: BvhFile) -> _Tree:
        return cls(
            tuple(j.name for j in file.joints),
            tuple(j.parent for j in file.joints),
            frozenset(j.name for j in file.joints if j.end_site is not None),
        )

    def ancestors(self, name: str) -> list[str]:
        out, i = [], self.parents[self.index[name]]
        while i >= 0:
            out.append(self.names[i])
            i = self.parents[i]
        return out

    def depth(self, name: str) -> int:
        return len(self.ancestors(name))

    def below(self, ancestor: str, name: str) -> bool:
        return ancestor in self.ancestors(name)

    def end(self, name: str) -> str | None:
        return f"{name}_End" if name in self.end_sites else None


@dataclass
class _Binding:
    tree: _Tree
    family: str | None = None
    bound: dict[Landmark, str] = field(default_factory=dict)
    why: dict[Landmark, tuple[Source, str]] = field(default_factory=dict)
    spine: tuple[str, ...] = ()
    trunk_root: str | None = None
    decisions: list[Decision] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def bind(self, lm: Landmark, bone: str | None, source: Source, note: str) -> None:
        if bone is not None and lm not in self.bound:
            self.bound[lm] = bone
            self.why[lm] = (source, note)


def _choose_family(names: Sequence[str]) -> tuple[str | None, str]:
    present = set(names)
    best: tuple[int, float, str] | None = None
    for family, bones in _lexicon()["families"].items():
        spelled = {v for k, v in bones.items() if k != "spine"} | set(bones.get("spine", ()))
        overlap = len(spelled & present)
        key = (overlap, overlap / len(spelled), family)
        if overlap >= MIN_FAMILY_OVERLAP and (best is None or key[:2] > best[:2]):
            best = key
    if best is None:
        return None, f"no family shares at least {MIN_FAMILY_OVERLAP} bone names with the file"
    return best[2], f"{best[0]} family bone names in the file"


def _bind_by_name(tree: _Tree) -> _Binding:
    """Family names, then tokens, then the structural fallbacks; no motion is looked at."""
    b = _Binding(tree)
    b.family, note = _choose_family(tree.names)
    b.decisions.append(Decision("family", b.family or "none", "name", note))
    if b.family is not None:
        bones = _lexicon()["families"][b.family]
        for key, bone in bones.items():
            if key != "spine" and bone in tree.index:
                b.bind(Landmark(key), bone, "name", f"{b.family} family name")
        b.spine = tuple(s for s in bones.get("spine", ()) if s in tree.index)

    taken = set(b.bound.values()) | set(b.spine)
    classes: dict[tuple[str, str | None], list[str]] = {}
    for name in tree.names:
        if name in taken:
            continue
        norm = _normalise(name)
        part = _part_of(norm)
        if part is not None:
            classes.setdefault((part, _side_of(norm)), []).append(name)

    def pick(part, side=None, *, below=None, above=None, nearest=False) -> str | None:
        if side is None:
            pool = [n for (p, _), names in classes.items() if p == part for n in names]
        else:
            pool = list(classes.get((part, side), ()))
        pool = [n for n in pool if n not in b.bound.values()]
        if below is not None:
            pool = [n for n in pool if tree.below(below, n)]
        if above is not None:
            pool = [n for n in pool if tree.below(n, above)]
        if not pool:
            return None
        ordered = sorted(pool, key=lambda n: (tree.depth(n), tree.index[n]))
        return ordered[-1] if nearest else ordered[0]

    token = "lexicon token"
    for s in _SIDES:
        hip, knee, ankle, toe = (_landmark(stem, s) for stem, _ in _LEG)
        b.bind(knee, pick("knee", s, below=b.bound.get(hip)), "name", token)
        if b.bound.get(knee):
            b.bind(hip, pick("hip", s, above=b.bound[knee], nearest=True), "name", token)
        b.bind(hip, pick("hip", s), "name", token)
        b.bind(ankle, pick("ankle", s, below=b.bound.get(knee, b.bound.get(hip))), "name", token)
        b.bind(toe, pick("toe", s, below=b.bound.get(ankle)), "name", token)
        if ankle in b.bound:
            b.bind(toe, tree.end(b.bound[ankle]), "geometry", "the foot's End Site")

        shoulder, elbow, wrist = (_landmark(stem, s) for stem, _ in _ARM)
        b.bind(elbow, pick("forearm", s, below=b.bound.get(shoulder)), "name", token)
        if elbow in b.bound:
            up = pick("upper_arm", s, above=b.bound[elbow], nearest=True)
            b.bind(shoulder, up, "name", token)
            parent = tree.parents[tree.index[b.bound[elbow]]]
            if parent >= 0:
                b.bind(shoulder, tree.names[parent], "geometry", "the elbow bone's parent")
            b.bind(wrist, pick("hand", s, below=b.bound[elbow]), "name", token)

    b.bind(L.PELVIS, pick("pelvis"), "name", token)
    if L.PELVIS not in b.bound and L.HIP_L in b.bound and L.HIP_R in b.bound:
        common = [
            a for a in tree.ancestors(b.bound[L.HIP_L]) if a in tree.ancestors(b.bound[L.HIP_R])
        ]
        if common:
            b.bind(L.PELVIS, common[0], "geometry", "lowest common ancestor of the hips")

    if not b.spine:
        pool = [n for (p, _), names in classes.items() if p == "spine" for n in names]
        b.spine = tuple(n for n in pool if n not in b.bound.values())
    spine = sorted(b.spine, key=lambda n: (tree.depth(n), tree.index[n]))
    if spine:
        b.trunk_root = spine[0]
        b.decisions.append(Decision("trunk_root", spine[0], "geometry", "lowest spine bone"))
    b.bind(L.NECK, pick("neck"), "name", token)
    if spine:
        b.bind(L.NECK, spine[-1], "geometry", "highest spine bone (no neck bone)")
    b.bind(L.HEAD, pick("head", below=b.bound.get(L.NECK)), "name", token)
    if L.NECK in b.bound:
        b.bind(L.HEAD, tree.end(b.bound[L.NECK]), "geometry", "the neck's End Site")

    for s in _SIDES:
        chain = [b.bound.get(_landmark(stem, s)) for stem in ("hip", "knee", "ankle")]
        if None in chain:
            continue
        if not (tree.below(chain[0], chain[1]) and tree.below(chain[1], chain[2])):
            raise RigError(
                f"hip_{s}, knee_{s}, ankle_{s} ({', '.join(chain)}) are not an ancestor chain"
            )
    return b


def _referee_sides(b: _Binding, clip: SkeletonClip) -> None:
    """Swap limbs whose side names disagree with the rest-pose lateral coordinate.

    Without forward travel there is no path frame, so no "left" to measure against: the
    referee is skipped and the names are kept unverified (with a warning), never guessed.
    """
    index = {n: i for i, n in enumerate(clip.joint_names)}
    pelvis = clip.positions[:, index[b.bound[L.PELVIS]]]
    try:
        left = path_frame(pelvis, clip.fps).left
    except ValueError:
        b.warnings.append(
            "side referee skipped: the clip has no forward travel, so there is no path frame; "
            "side names kept unverified"
        )
        return

    def lateral(bone: str) -> float:
        rel = clip.positions[:, index[bone]] - pelvis
        return float(np.mean(np.sum(rel * left, axis=1)))

    for limb, stems in (("leg", [s for s, _ in _LEG]), ("arm", [s for s, _ in _ARM])):
        root = stems[0]
        sides = {s: b.bound.get(_landmark(root, s)) for s in _SIDES}
        geo = {s: (lateral(bone) > 0) for s, bone in sides.items() if bone is not None}
        wrong = [s for s, is_left in geo.items() if is_left != (s == "l")]
        if not wrong:
            continue
        if len(geo) == 2 and len(wrong) == 1:
            b.warnings.append(f"{limb}: both {root} bones sit on the same side; names kept")
            continue
        for stem in stems:
            pair = [_landmark(stem, s) for s in _SIDES]
            bones = [b.bound.pop(lm, None) for lm in pair]
            whys = [b.why.pop(lm, None) for lm in pair]
            for lm, bone, why in zip(pair, reversed(bones), reversed(whys), strict=True):
                if bone is not None:
                    b.bound[lm] = bone
                    b.why[lm] = ("geometry", f"side corrected by geometry ({why[1]})")
        message = f"{limb}: names disagree with the rest-pose side of the pelvis; sides swapped"
        b.warnings.append(message)
        b.decisions.append(Decision(f"{limb}_side", "swapped", "geometry", message))


def _warn_on_broken_leg_chains(b: _Binding, clip: SkeletonClip, overridden: set[Landmark]) -> None:
    """An override is the user's call: a leg chain it breaks is warned about, not refused."""

    def ancestors(name: str) -> set[str]:
        out, i = set(), clip.parents[clip.joint_names.index(name)] if clip.parents else -1
        while i >= 0:
            out.add(clip.joint_names[i])
            i = clip.parents[i]
        return out

    for s in _SIDES:
        lms = [_landmark(stem, s) for stem in ("hip", "knee", "ankle")]
        chain = [b.bound.get(lm) for lm in lms]
        if None in chain or clip.parents is None or not overridden.intersection(lms):
            continue
        if not (chain[0] in ancestors(chain[1]) and chain[1] in ancestors(chain[2])):
            b.warnings.append(
                f"override: hip_{s}, knee_{s}, ankle_{s} ({', '.join(chain)}) are not an "
                "ancestor chain"
            )


def map_landmarks(
    file: BvhFile, clip: SkeletonClip, *, overrides: Mapping[str, str] | None = None
) -> LandmarkMapping:
    """Bind landmarks by name, referee sides by geometry, then apply ``overrides``."""
    tree = _Tree.of(file)
    b = _bind_by_name(tree)
    if L.PELVIS in b.bound:
        _referee_sides(b, clip)
    for key, bone in (overrides or {}).items():
        try:
            lm = Landmark(key)
        except ValueError:
            raise RigError(f"override names an unknown landmark {key!r}") from None
        if bone not in clip.joint_names:
            raise RigError(f"override {key}={bone}: no bone {bone!r} in {file.source or 'file'}")
        b.bound[lm] = bone
        b.why[lm] = ("user", "override")
    if overrides:
        _warn_on_broken_leg_chains(b, clip, {Landmark(k) for k in overrides})
    missing = required_landmarks_present(b.bound)
    if missing:
        names = ", ".join(lm.value for lm in missing)
        raise RigError(f"no bone found for {names}; map them with an override")
    order = list(Landmark)
    landmarks = {lm: b.bound[lm] for lm in order if lm in b.bound}
    per_landmark = [Decision(lm.value, bone, *b.why[lm]) for lm, bone in landmarks.items()]
    return LandmarkMapping(
        landmarks=landmarks,
        family=b.family,
        decisions=tuple(b.decisions + per_landmark),
        warnings=tuple(b.warnings),
        trunk_root=b.trunk_root,
    )


def infer_scale_and_up(
    file_at_unit_scale: BvhFile,
) -> tuple[float, Literal["y", "z"], tuple[Decision, ...]]:
    """Scale and up-axis from the rest hierarchy of a file read at scale 1.0, up_axis "z"."""
    tree = _Tree.of(file_at_unit_scale)
    b = _bind_by_name(tree)
    rest: dict[str, np.ndarray] = {}
    for joint in file_at_unit_scale.joints:
        base = rest[tree.names[joint.parent]] if joint.parent >= 0 else np.zeros(3)
        rest[joint.name] = base + np.asarray(joint.offset, dtype=float)
        if joint.end_site is not None:
            rest[f"{joint.name}_End"] = rest[joint.name] + np.asarray(joint.end_site, dtype=float)

    top = b.bound.get(L.HEAD, b.bound.get(L.NECK))
    if L.PELVIS not in b.bound or top is None:
        raise RigError("cannot find the pelvis and head to infer the up-axis; pass --up-axis")
    extent = np.abs(rest[top] - rest[b.bound[L.PELVIS]])
    up: Literal["y", "z"] = "y" if extent[1] > extent[2] else "z"
    decisions = [
        Decision(
            "up_axis",
            up,
            "geometry",
            f"pelvis-to-head rest extent y {extent[1]:.4g}, z {extent[2]:.4g} file units",
        )
    ]

    for s in _SIDES:
        hip, ankle = b.bound.get(_landmark("hip", s)), b.bound.get(_landmark("ankle", s))
        if hip is not None and ankle is not None:
            break
    else:
        raise RigError("cannot find a hip and ankle to infer the units; pass --scale")
    leg = float(np.linalg.norm(rest[ankle] - rest[hip]))
    scale = min(SCALES, key=lambda s: abs(s * leg - LEG_TARGET_M))
    metres = scale * leg
    if not LEG_WINDOW_M[0] <= metres <= LEG_WINDOW_M[1]:
        raise RigError(
            f"hip-to-ankle rest distance {leg:.4g} file units fits no known unit "
            f"(nearest {metres:.3f} m at scale {scale:g}); pass --scale"
        )
    decisions.append(
        Decision(
            "scale", f"{scale:g}", "geometry", f"hip-to-ankle {leg:.4g} units = {metres:.3f} m"
        )
    )
    return scale, up, tuple(decisions)


# --- Role bindings by probe (spec 4.4) -----------------------------------------------------

AXIS_PURITY_MIN = 0.8
_CLASS_AXIS = {"pitch": 1, "roll": 0, "yaw": 2}  # rotation about left, forward, up
_PATH = {"forward": np.eye(3)[0], "left": np.eye(3)[1], "up": np.eye(3)[2]}


@dataclass(frozen=True)
class RoleBinding:
    """The hinge (``"Bone/Channel"``) a role drives, its sign and the probe's axis purity."""

    hinge: str
    sign: float
    axis_purity: float


@dataclass(frozen=True)
class RigMapping:
    """Landmarks and role bindings of a rig, with its units and the whole audit trail."""

    landmarks: LandmarkMap
    roles: Mapping[JointRole, RoleBinding]
    leg_length: float
    scale: float
    up_axis: str
    family: str | None
    decisions: tuple[Decision, ...]
    warnings: tuple[str, ...]
    skewed: tuple[JointRole, ...]


def _role_groups(
    trunk_root: str | None, landmarks: LandmarkMap
) -> list[tuple[str | None, dict[str, JointRole]]]:
    """``(bone, {axis class: role})`` for each bone that owns a landmark."""
    R = JointRole
    groups: list[tuple[str | None, dict[str, JointRole]]] = []
    for s in _SIDES:
        S = s.upper()
        groups += [
            (
                landmarks.get(_landmark("hip", s)),
                {
                    "pitch": R[f"HIP_PITCH_{S}"],
                    "roll": R[f"HIP_ROLL_{S}"],
                    "yaw": R[f"HIP_YAW_{S}"],
                },
            ),
            (landmarks.get(_landmark("knee", s)), {"pitch": R[f"KNEE_{S}"]}),
            (
                landmarks.get(_landmark("ankle", s)),
                {"pitch": R[f"ANKLE_PITCH_{S}"], "roll": R[f"ANKLE_ROLL_{S}"]},
            ),
            (
                landmarks.get(_landmark("shoulder", s)),
                {
                    "pitch": R[f"SHOULDER_PITCH_{S}"],
                    "roll": R[f"SHOULDER_ROLL_{S}"],
                    "yaw": R[f"SHOULDER_YAW_{S}"],
                },
            ),
            (landmarks.get(_landmark("elbow", s)), {"pitch": R[f"ELBOW_{S}"]}),
        ]
    groups += [
        (trunk_root, {"pitch": R.WAIST_PITCH, "roll": R.WAIST_ROLL, "yaw": R.WAIST_YAW}),
        (landmarks.get(L.NECK), {"pitch": R.NECK_PITCH}),
    ]
    return groups


def _axis_class(role: JointRole) -> str:
    name = role.name
    if "ROLL" in name:
        return "roll"
    if "YAW" in name:
        return "yaw"
    return "pitch"  # *_PITCH, KNEE_*, ELBOW_*


def _sign(role: JointRole, probe: HingeProbe) -> tuple[float, str]:
    """+1 when a positive probe follows the role's convention, else -1; and a note."""
    if role not in ROLE_CONVENTIONS:
        return 1.0, "no convention row, sign +1"
    landmark, direction = ROLE_CONVENTIONS[role]
    needed = [landmark]
    if direction == "toward_shoulder":
        needed.append(f"shoulder_{landmark[-1]}")
    if any(n not in probe.before for n in needed):
        return 1.0, f"no {' or '.join(needed)} landmark to probe, sign +1"
    plus = convention_holds(role, probe.before, probe.after, **_PATH)
    if direction == "toward_shoulder":
        minus = convention_holds(role, probe.before, probe.minus, **_PATH)
        if plus == minus:
            # A straight elbow at the reference: both directions shorten the shoulder-wrist
            # distance (to second order). Flexion is taken as the wrist moving forward.
            moved = probe.after[landmark] - probe.before[landmark]
            sign = 1.0 if float(moved @ _PATH["forward"]) > 0.0 else -1.0
            return sign, "elbow straight at the reference, flexion taken as the wrist forward"
    return (1.0, "probe follows the convention") if plus else (-1.0, "probe opposes it, flipped")


def bind_roles(
    full: CompiledRig,
    reference: ReferencePose,
    landmarks: LandmarkMapping,
    *,
    overrides: Mapping[str, str] | None = None,
) -> RigMapping:
    """Bind each ``JointRole`` to a hinge of ``full`` by probing at the reference pose.

    Per bone that owns a landmark (hip, knee and ankle bones, the trunk root, the neck bone,
    the upper-arm and forearm bones), each rotation hinge is turned by ``PROBE_RAD`` on the
    ``full`` chain posed at the reference, whose world frame is the reference path frame
    (forward +x, left +y, up +z). The probe's rotation axis is projected on (forward, left,
    up): the hinge turning most about ``left`` (its child landmark moves in the sagittal plane)
    is pitch, of the rest the one turning most about ``forward`` (lateral motion) is roll, the
    last is yaw; ``axis_purity`` is the winning share. The sign is +1 when the probe moves the
    child landmark as ``ROLE_CONVENTIONS`` says, else -1 (+1 for roles without a row). Roles
    without a ``RANGES`` row (waist roll and yaw, shoulder yaw) stay undriven. A role whose
    purity is below ``AXIS_PURITY_MIN`` is ``skewed``: leg roles are bound anyway, a skewed
    posture role or shoulder pitch is left unbound with a warning. ``overrides`` (role value ->
    ``"Bone/Xrotation"``) win, ``source="user"``. Missing ``LEG_ROLES`` raise ``RigError``.
    """
    lm = landmarks.landmarks
    decisions = list(landmarks.decisions)
    warnings = list(landmarks.warnings)
    by_bone: dict[str, list[str]] = {}
    for hinge in full.hinges:
        by_bone.setdefault(hinge.rsplit("/", 1)[0], []).append(hinge)

    parsed: dict[JointRole, str] = {}
    for key, hinge in (overrides or {}).items():
        try:
            role = JointRole(key)
        except ValueError:
            raise RigError(f"role override names an unknown role {key!r}") from None
        if hinge not in full.hinges:
            source = full.file.source or "the file"
            raise RigError(f"role override {key}={hinge}: no rotation channel {hinge} in {source}")
        parsed[role] = hinge

    groups = [(b, c) for b, c in _role_groups(landmarks.trunk_root, lm) if b in by_bone]
    probed = sorted({h for b, _ in groups for h in by_bone[b]} | set(parsed.values()))
    probes = probe_hinges(full, reference, lm, probed)

    def share(hinge: str, cls: str) -> float:
        return float(abs(probes[hinge].axis[_CLASS_AXIS[cls]]))

    roles: dict[JointRole, RoleBinding] = {}
    skewed: list[JointRole] = []
    for bone, classes in groups:
        free = list(by_bone[bone])
        chosen: dict[str, str] = {}
        for cls in ("pitch", "roll", "yaw"):
            if free:
                best = max(free, key=lambda h, c=cls: share(h, c))
                free.remove(best)
                chosen[cls] = best
        for cls, role in classes.items():
            if cls not in chosen or role in parsed:
                continue
            hinge, purity = chosen[cls], share(chosen[cls], cls)
            if role not in RANGES:
                decisions.append(Decision(role.value, "undriven", "geometry", f"{hinge}: no range"))
                continue
            if purity < AXIS_PURITY_MIN:
                skewed.append(role)
                if role in POSTURE_ROLES or role.name.startswith("SHOULDER_PITCH"):
                    warnings.append(
                        f"{role.value}: {hinge} axis purity {purity:.2f} is below "
                        f"{AXIS_PURITY_MIN} (skewed); left unbound"
                    )
                    continue
                warnings.append(f"{role.value}: {hinge} axis purity {purity:.2f} (skewed)")
            sign, note = _sign(role, probes[hinge])
            roles[role] = RoleBinding(hinge=hinge, sign=sign, axis_purity=purity)
            decisions.append(
                Decision(role.value, hinge, "geometry", f"purity {purity:.3f}, {note}")
            )

    for role, hinge in parsed.items():
        purity = share(hinge, _axis_class(role))
        sign, note = _sign(role, probes[hinge])
        roles[role] = RoleBinding(hinge=hinge, sign=sign, axis_purity=purity)
        decisions.append(Decision(role.value, hinge, "user", f"purity {purity:.3f}, {note}"))
        if purity < AXIS_PURITY_MIN:
            skewed.append(role)
            warnings.append(f"{role.value}: override {hinge} axis purity {purity:.2f} (skewed)")

    owners: dict[str, JointRole] = {}
    for role, binding in roles.items():
        if binding.hinge in owners:
            raise RigError(
                f"hinge {binding.hinge} is bound to both {owners[binding.hinge].value} and "
                f"{role.value}; fix the role overrides"
            )
        owners[binding.hinge] = role
    missing = [r.value for r in LEG_ROLES if r not in roles]
    if missing:
        raise RigError(
            "the legs need hip pitch, roll and yaw, knee pitch and ankle pitch and roll "
            f"channels; no hinge found for {', '.join(missing)}"
        )
    order = list(JointRole)
    return RigMapping(
        landmarks=lm,
        roles=MappingProxyType(dict(sorted(roles.items(), key=lambda kv: order.index(kv[0])))),
        leg_length=leg_length(full.file.to_clip(end_sites=True), lm),
        scale=full.file.scale,
        up_axis=full.file.up_axis,
        family=landmarks.family,
        decisions=tuple(decisions),
        warnings=tuple(warnings),
        skewed=tuple(dict.fromkeys(skewed)),
    )
