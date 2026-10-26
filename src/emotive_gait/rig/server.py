"""The MCP server behind ``emotive-gait serve``: emotion for any rig over stdio (spec 4.9).

Tools: ``inspect_rig``, ``measure_gait``, ``emotional_walk``, ``parse_emotion_prompt``,
``list_emotions``; resources ``emotive-gait://evidence`` and
``emotive-gait://rig-mapping-schema``; prompt ``emotional_walk_request``. Paths are local file
paths and the server keeps no state between calls. The package never calls an LLM: the prompt
tells the host model what to do.

The ``mcp`` package is imported only inside ``build_server``, ``serve`` and the error helper, so
importing this module works without the ``mcp`` extra. Nothing here writes to stdout (the stdio
transport owns it); a failure in the pipeline becomes a tool error, never a crash of the process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MAPPING_SCHEMA = """\
# Overrides for the automatic bone mapping. Every key is optional.
# `landmarks` maps a landmark name to the rig's bone name; `roles` maps a G1 joint role to the
# bone that drives it. Pass this as the `mapping` argument (a dict of the same shape).
landmarks:
  pelvis: Hips
  ankle_l: LeftFoot
  ankle_r: RightFoot
  toe_l: LeftToeBase
  toe_r: RightToeBase
  neck: Neck
roles:
  knee_l: LeftLeg
  knee_r: RightLeg
# optional: the file's unit and up axis (explicit tool arguments win)
scale: 0.01      # metres per file unit
up_axis: y       # y or z
"""

PROMPT = """\
You are driving the emotive-gait tools to make the rig in {bvh_path} walk with an emotion.

The user's description: {description}

Do this, in order:
1. Call `inspect_rig` with bvh_path={bvh_path}. Read the mapping decisions and warnings and the
   source gait (P0). If it fails, tell the user why in plain words and stop.
2. Map the description to emotion intensities in [0, 1] over anger, disgust, fear, happiness,
   sadness and surprise. You may call `parse_emotion_prompt` to get a lexicon reading, or give
   `intensities` yourself. Use `list_emotions` for the grounded emotions.
3. Call `emotional_walk` with bvh_path={bvh_path}, an `out_path` for the new BVH, and your
   intensities (or `prompt`, not both).
4. Report to the user in words: whether the walk is feasible, the foot slip (max travel and
   height error), the cues the rig could not express (`unexpressible`) and where the round trip
   deviates. Say which emotions were used.
5. Disgust and surprise have no literature source: never present them as grounded. Say so
   whenever they were requested.
"""


def _call(fn, *args):
    """Run a pipeline call; ``ValueError`` and ``OSError`` become a tool error."""
    from mcp.server.mcpserver.exceptions import ToolError

    try:
        return fn(*args)
    except (ValueError, OSError) as exc:  # RigError, MocapFormatError, IntensityError, ...
        raise ToolError(str(exc)) from exc


def _prepare(bvh_path: str, scale: float | None, up_axis: str | None, mapping: dict | None):
    """``(source, scale, up_axis, overrides, ignored)`` from the tool arguments; ``ignored`` lists
    the informational mapping keys the pipeline does not use."""
    from emotive_gait.cli_common import check_positive_finite
    from emotive_gait.rig.cli import mapping_overrides_from, resolve_scale_up
    from emotive_gait.rig.errors import RigError

    source = Path(bvh_path)
    if not source.is_file():
        raise RigError(f"{source}: no such file")
    overrides, info = (None, {}) if mapping is None else mapping_overrides_from(mapping)
    if scale is not None:
        scale = check_positive_finite(scale, "scale")
    if up_axis is not None and up_axis not in ("y", "z"):
        raise RigError(f"up_axis must be 'y' or 'z', got {up_axis!r}")
    scale, up_axis = resolve_scale_up(info, scale, up_axis)
    ignored = [k for k in info if k not in ("scale", "up_axis")]
    return source, scale, up_axis, overrides, ignored


def _with_ignored(data: dict, ignored: list[str]) -> dict:
    if ignored:
        data["ignored_mapping_keys"] = ignored
    return data


def _load(bvh_path: str, scale: float | None, up_axis: str | None, mapping: dict | None):
    """The ``LoadedRig`` and the ignored informational mapping keys."""
    from emotive_gait.rig.synthesis import load_rig

    source, scale, up_axis, overrides, ignored = _prepare(bvh_path, scale, up_axis, mapping)
    rig = load_rig(source, scale=scale, up_axis=up_axis, mapping_overrides=overrides)
    return rig, ignored


def _inspect(bvh_path, scale, up_axis, mapping) -> dict:
    from emotive_gait.cli_common import clean
    from emotive_gait.rig.cli import inspect_data

    rig, ignored = _load(bvh_path, scale, up_axis, mapping)
    return _with_ignored(clean(inspect_data(rig)), ignored)


def _measure(bvh_path, scale, up_axis, mapping) -> dict:
    import dataclasses

    from emotive_gait.cli_common import clean
    from emotive_gait.rig.synthesis import measure

    source, scale, up_axis, overrides, ignored = _prepare(bvh_path, scale, up_axis, mapping)
    s = measure(source, scale=scale, up_axis=up_axis, mapping_overrides=overrides)
    return _with_ignored(
        clean(
            {
                "mean": dataclasses.asdict(s.mean),
                "cycles": {
                    "used": s.n_cycles,
                    "discarded": s.n_discarded,
                    "discard_reasons": dict(s.discard_reasons),
                },
                "leg_length": s.leg_length,
            }
        ),
        ignored,
    )


def _walk(
    bvh_path, out_path, intensities, prompt, model, scaling, duration_s, mapping, scale, up_axis
) -> dict:
    from emotive_gait.cli_common import check_positive_finite, clean
    from emotive_gait.emotion.evidence import Emotion
    from emotive_gait.emotion.learned import (
        ArtefactMissingError,
        ArtefactSchemaError,
        LearnedEmotionModel,
    )
    from emotive_gait.emotion.model import ANALYTIC
    from emotive_gait.emotion.prompt import parse_emotion_prompt
    from emotive_gait.mocap.writers.bvh import write_bvh
    from emotive_gait.rig.cli import walk_data
    from emotive_gait.rig.synthesis import walk

    if (intensities is None) == (prompt is None):
        raise ValueError("give exactly one of intensities and prompt")
    if model not in ("analytic", "learned"):
        raise ValueError(f"model must be 'analytic' or 'learned', got {model!r}")
    if scaling not in ("froude", "ratio"):
        raise ValueError(f"scaling must be 'froude' or 'ratio', got {scaling!r}")
    if duration_s is not None:
        duration_s = check_positive_finite(duration_s, "duration_s")
    parsed = None
    if prompt is not None:
        parsed = parse_emotion_prompt(prompt)
        if not parsed.matched:
            raise ValueError(
                f"no emotion word found in the prompt {prompt!r}; unmatched: "
                + (", ".join(parsed.unmatched) or "(empty)")
            )
        values, source = parsed.intensities, "lexicon"
    else:
        names = ", ".join(e.value for e in Emotion)
        values, source = {}, "caller"
        for name, value in intensities.items():
            try:
                values[Emotion(str(name).strip().lower())] = float(value)
            except ValueError:
                raise ValueError(f"unknown emotion {name!r}; choose from {names}") from None
    emotion_model = ANALYTIC
    if model == "learned":
        try:
            emotion_model = LearnedEmotionModel.default(scaling=scaling)
        except (ArtefactMissingError, ArtefactSchemaError) as exc:
            raise ValueError(str(exc)) from exc
    if Path(out_path).resolve() == Path(bvh_path).resolve():
        raise ValueError("out_path must differ from bvh_path (the source would be overwritten)")
    rig, ignored = _load(bvh_path, scale, up_axis, mapping)
    result = walk(
        rig,
        intensities=values,
        intensities_source=source,
        model=emotion_model,
        model_name=model,
        scaling=scaling,
        duration_s=duration_s,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_bvh(result.output, out)
    data = clean(walk_data(result, None if parsed is None else (prompt, parsed), str(out)))
    return _with_ignored(data, ignored)


def _parse(text: str) -> dict:
    from emotive_gait.cli_common import clean
    from emotive_gait.emotion.prompt import parse_emotion_prompt

    r = parse_emotion_prompt(text)
    return clean(
        {
            "intensities": {e.value: v for e, v in r.intensities.items()},
            "matched": [
                {"term": t, "emotion": e.value, "contribution": c} for t, e, c in r.matched
            ],
            "unmatched": list(r.unmatched),
            "grounded": r.grounded,
        }
    )


def _emotions() -> dict:
    from emotive_gait.emotion.evidence import UNSOURCED_EMOTIONS, Emotion
    from emotive_gait.emotion.learned import (
        ArtefactMissingError,
        ArtefactSchemaError,
        LearnedEmotionModel,
    )

    out: dict[str, Any] = {
        "emotions": [
            {"emotion": e.value, "grounded": e not in UNSOURCED_EMOTIONS} for e in Emotion
        ],
        "thin_cells": [],
    }
    try:
        out["thin_cells"] = [
            {"emotion": e, "parameter": p} for e, p in LearnedEmotionModel.default().thin_cells
        ]
    except (ArtefactMissingError, ArtefactSchemaError) as exc:
        out["note"] = f"the learned model is unavailable, so no thin cells are listed: {exc}"
    return out


def build_server():
    """The ``MCPServer`` named ``emotive-gait`` with its tools, resources and prompt."""
    from mcp.server import MCPServer

    server = MCPServer(
        name="emotive-gait",
        instructions=(
            "Make a humanoid rig's BVH walk with an emotion. Start with inspect_rig, then "
            "emotional_walk. Disgust and surprise are unsourced: never present them as grounded."
        ),
    )

    @server.tool()
    def inspect_rig(
        bvh_path: str,
        scale: float | None = None,
        up_axis: str | None = None,
        mapping: dict | None = None,
    ) -> dict[str, Any]:
        """Map the rig in a BVH walk: decisions, warnings, leg length and the neutral gait P0."""
        return _call(_inspect, bvh_path, scale, up_axis, mapping)

    @server.tool()
    def measure_gait(
        bvh_path: str,
        scale: float | None = None,
        up_axis: str | None = None,
        mapping: dict | None = None,
    ) -> dict[str, Any]:
        """Cycle-mean gait measurements of a BVH walk, with cycle counts and discard reasons."""
        return _call(_measure, bvh_path, scale, up_axis, mapping)

    @server.tool()
    def emotional_walk(
        bvh_path: str,
        out_path: str,
        intensities: dict[str, float] | None = None,
        prompt: str | None = None,
        model: str = "analytic",
        scaling: str = "froude",
        duration_s: float | None = None,
        mapping: dict | None = None,
        scale: float | None = None,
        up_axis: str | None = None,
    ) -> dict[str, Any]:
        """Write the rig's walk moved by an emotion to out_path and report feasibility, slip and
        the round trip. Give exactly one of intensities (emotion name to [0, 1]) and prompt."""
        return _call(
            _walk,
            bvh_path,
            out_path,
            intensities,
            prompt,
            model,
            scaling,
            duration_s,
            mapping,
            scale,
            up_axis,
        )

    @server.tool(name="parse_emotion_prompt")
    def parse_emotion_prompt_tool(text: str) -> dict[str, Any]:
        """Read free text through the emotion lexicon: intensities, matched and unmatched words."""
        return _call(_parse, text)

    @server.tool()
    def list_emotions() -> dict[str, Any]:
        """The six emotions, which are grounded, and the learned model's thin cells."""
        return _call(_emotions)

    @server.resource("emotive-gait://evidence")
    def evidence() -> str:
        """The claim table with its grounding, as the evidence command prints it."""
        from emotive_gait.cli_common import evidence_lines

        return "\n".join(evidence_lines()) + "\n"

    @server.resource("emotive-gait://rig-mapping-schema")
    def rig_mapping_schema() -> str:
        """The mapping-override YAML's shape, with an example."""
        return MAPPING_SCHEMA

    @server.prompt()
    def emotional_walk_request(bvh_path: str, description: str) -> str:
        """Instructions for the host model: inspect, map the description, walk, report."""
        return PROMPT.format(bvh_path=bvh_path, description=description)

    return server


def serve() -> None:
    """Run the server over stdio until the client disconnects."""
    build_server().run("stdio")


if __name__ == "__main__":
    serve()
