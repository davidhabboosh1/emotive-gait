"""An on-disk synthetic dataset: the shared walker written as BVH plus its manifest.

``write_bvh`` is a tiny BVH writer (one zero-channel root, every joint a child with three
position channels holding its world position), reusable by later tasks' on-disk fixtures
(``from learn.synthetic import ...``). ``write_toy_dataset`` writes two actors, each with a
neutral and an "angry" (cadence 2.4: the same walk played 1.2 times faster) clip, plus one clip
too short to hold a cycle, and returns the manifest path.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from mocap.walker import full_walker

from emotive_gait.mocap.skeleton import SkeletonClip
from learn.manifests import clip_dict, dataset_dict, write_manifest

SECONDS = 10.0
SHORT_SECONDS = 1.0
ANGRY_SPEEDUP = 1.2


def write_bvh(path: Path, clip: SkeletonClip) -> bytes:
    """Write ``clip`` (metres, +z up) as a BVH and return the file bytes."""
    lines = ["HIERARCHY", "ROOT world", "{", "  OFFSET 0 0 0", "  CHANNELS 0"]
    for name in clip.joint_names:
        lines += [
            f"  JOINT {name}",
            "  {",
            "    OFFSET 0 0 0",
            "    CHANNELS 3 Xposition Yposition Zposition",
            "  }",
        ]
    lines += ["}", "MOTION", f"Frames: {clip.frames()}", f"Frame Time: {1.0 / clip.fps:.9f}"]
    flat = clip.positions.reshape(clip.frames(), -1)
    lines += [" ".join(f"{v:.6f}" for v in row) for row in flat]
    data = ("\n".join(lines) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def walker_clip(seconds: float = SECONDS, speedup: float = 1.0) -> SkeletonClip:
    clip = full_walker(seconds).clip
    return replace(clip, fps=clip.fps * speedup, positions=np.asarray(clip.positions))


def write_toy_dataset(directory: Path) -> Path:
    """Write ``directory/raw/toy/*.bvh`` and ``directory/manifest.yaml``; return the manifest."""
    root = directory / "raw" / "toy"
    clips = []
    plan = [
        ("a1", "neutral", SECONDS, 1.0),
        ("a1", "anger", SECONDS, ANGRY_SPEEDUP),
        ("a2", "neutral", SECONDS, 1.0),
        ("a2", "anger", SECONDS, ANGRY_SPEEDUP),
        ("a2", "sadness", SHORT_SECONDS, 1.0),
    ]
    for actor, emotion, seconds, speedup in plan:
        name = f"{actor}_{emotion}.bvh"
        data = write_bvh(root / name, walker_clip(seconds, speedup))
        kind = "exact" if emotion != "neutral" else "mapped"
        clips.append(
            clip_dict(
                name,
                data,
                subject=actor,
                actor=actor,
                emotion=emotion,
                label=emotion.title(),
                label_kind=kind,
            )
        )
    return write_manifest(directory, [dataset_dict(files=clips)])
