"""Writer for BVH files: the hierarchy in file order, channels never reordered."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from emotive_gait.mocap.readers.bvh import BvhFile


def _vec(values: Iterable[float]) -> str:
    return " ".join(f"{float(v):.6f}" for v in values)


def write_bvh(file: BvhFile, path: Path) -> None:
    """Write ``file`` (file units, degrees) as a BVH; ``read_bvh_file`` reads it back."""
    children: dict[int, list[int]] = {}
    roots: list[int] = []
    for i, joint in enumerate(file.joints):
        (roots if joint.parent < 0 else children.setdefault(joint.parent, [])).append(i)

    out = ["HIERARCHY"]

    def emit(index: int, depth: int) -> None:
        joint = file.joints[index]
        pad = "\t" * depth
        out.append(f"{pad}{'ROOT' if joint.parent < 0 else 'JOINT'} {joint.name}")
        out.append(f"{pad}{{")
        out.append(f"{pad}\tOFFSET {_vec(joint.offset)}")
        out.append(f"{pad}\tCHANNELS {len(joint.channels)} {' '.join(joint.channels)}".rstrip())
        for child in children.get(index, []):
            emit(child, depth + 1)
        if joint.end_site is not None:
            out.extend(
                [
                    f"{pad}\tEnd Site",
                    f"{pad}\t{{",
                    f"{pad}\t\tOFFSET {_vec(joint.end_site)}",
                    f"{pad}\t}}",
                ]
            )
        out.append(f"{pad}}}")

    for root in roots:
        emit(root, 0)
    out += ["MOTION", f"Frames: {file.motion.shape[0]}", f"Frame Time: {file.frame_time:.7g}"]
    out += [" ".join(f"{v:.6f}" for v in row) for row in file.motion]
    Path(path).write_text("\n".join(out) + "\n", newline="\n")
