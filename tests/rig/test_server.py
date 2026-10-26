"""The MCP server: tools, resources and prompt over the SDK's in-memory client, plus one real
stdio subprocess. Skipped when the ``mcp`` extra is not installed."""

from __future__ import annotations

import asyncio
import sys

import pytest

pytest.importorskip("mcp")

from mcp.client import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

from emotive_gait.rig.server import build_server
from rig.rigs import RigSpec, standing_file, walk_file

TOOLS = {"inspect_rig", "measure_gait", "emotional_walk", "parse_emotion_prompt", "list_emotions"}


@pytest.fixture(scope="module")
def server():
    return build_server()


def _run(server, action):
    async def go():
        async with Client(server) as client:
            return await action(client)

    return asyncio.run(asyncio.wait_for(go(), 120))


def call_tool(server, name, arguments):
    result = _run(server, lambda c: c.call_tool(name, arguments))
    assert not result.is_error, [getattr(b, "text", b) for b in result.content]
    return result.structured_content


def call_tool_expecting_error(server, name, arguments):
    result = _run(server, lambda c: c.call_tool(name, arguments))
    assert result.is_error
    return " ".join(getattr(b, "text", "") for b in result.content)


def test_server_lists_the_five_tools_two_resources_and_the_prompt(server):
    async def action(c):
        return await c.list_tools(), await c.list_resources(), await c.list_prompts()

    tools, resources, prompts = _run(server, action)
    assert {t.name for t in tools.tools} == TOOLS
    assert {str(r.uri) for r in resources.resources} == {
        "emotive-gait://evidence",
        "emotive-gait://rig-mapping-schema",
    }
    assert [p.name for p in prompts.prompts] == ["emotional_walk_request"]


def test_resources_return_the_claim_table_and_the_mapping_schema(server):
    async def action(c):
        return (
            await c.read_resource("emotive-gait://evidence"),
            await c.read_resource("emotive-gait://rig-mapping-schema"),
        )

    evidence, schema = _run(server, action)
    assert "GROUNDING" in evidence.contents[0].text
    assert "landmarks:" in schema.contents[0].text and "roles:" in schema.contents[0].text


def test_prompt_tells_the_host_what_to_do(server):
    prompt = _run(
        server,
        lambda c: c.get_prompt(
            "emotional_walk_request", {"bvh_path": "a.bvh", "description": "sad"}
        ),
    )
    text = prompt.messages[0].content.text
    assert "a.bvh" in text and "inspect_rig" in text and "emotional_walk" in text
    assert "disgust" in text and "surprise" in text


def test_parse_emotion_prompt_tool_returns_intensities(server):
    result = call_tool(server, "parse_emotion_prompt", {"text": "very gloomy"})
    assert result["intensities"]["sadness"] == pytest.approx(0.6)


def test_list_emotions_names_six_grounded_flags(server):
    result = call_tool(server, "list_emotions", {})
    by_name = {e["emotion"]: e["grounded"] for e in result["emotions"]}
    assert len(by_name) == 6 and not by_name["disgust"] and by_name["sadness"]
    assert "thin_cells" in result


def test_inspect_and_measure_report_the_source_gait(server, tmp_path):
    path = str(walk_file(RigSpec(), tmp_path))
    assert call_tool(server, "inspect_rig", {"bvh_path": path})["p0"]["cadence"] > 0
    measured = call_tool(server, "measure_gait", {"bvh_path": path})
    assert measured["cycles"]["used"] >= 1 and measured["mean"]["cadence"] > 0


def test_emotional_walk_tool_writes_the_file_and_reports(server, tmp_path):
    path = walk_file(RigSpec(), tmp_path)
    result = call_tool(
        server,
        "emotional_walk",
        {
            "bvh_path": str(path),
            "out_path": str(tmp_path / "o.bvh"),
            "intensities": {"sadness": 0.8},
        },
    )
    assert (tmp_path / "o.bvh").is_file()
    assert result["intensities_source"] == "caller" and "slip" in result


def test_emotional_walk_from_a_prompt_is_lexicon_sourced(server, tmp_path):
    path = walk_file(RigSpec(), tmp_path)
    result = call_tool(
        server,
        "emotional_walk",
        {"bvh_path": str(path), "out_path": str(tmp_path / "p.bvh"), "prompt": "very gloomy"},
    )
    assert result["intensities_source"] == "lexicon" and result["prompt"]["matched"]


@pytest.mark.parametrize(
    ("extra", "needle"),
    [
        ({}, "intensities"),
        ({"prompt": "sad", "intensities": {"sadness": 0.5}}, "exactly one"),
        ({"prompt": "zzz qqq"}, "unmatched"),
        ({"intensities": {"sadness": 2.0}}, "sadness"),
        ({"intensities": {"joy": 0.5}}, "joy"),
    ],
)
def test_emotional_walk_bad_requests_are_tool_errors(server, tmp_path, extra, needle):
    args = {"bvh_path": str(walk_file(RigSpec(), tmp_path)), "out_path": str(tmp_path / "x.bvh")}
    err = call_tool_expecting_error(server, "emotional_walk", {**args, **extra})
    assert needle in err
    assert not (tmp_path / "x.bvh").exists()


def test_missing_file_and_bad_mapping_are_tool_errors(server, tmp_path):
    assert "no such file" in call_tool_expecting_error(
        server, "inspect_rig", {"bvh_path": str(tmp_path / "nope.bvh")}
    )
    path = str(walk_file(RigSpec(), tmp_path))
    err = call_tool_expecting_error(
        server, "inspect_rig", {"bvh_path": path, "mapping": {"bogus": {}}}
    )
    assert "bogus" in err


def test_standing_clip_is_a_tool_error_not_a_crash(server, tmp_path):
    path = standing_file(RigSpec(), tmp_path)
    err = call_tool_expecting_error(server, "inspect_rig", {"bvh_path": str(path)})
    assert "no usable cycle" in err


def test_stdio_server_starts_and_answers_list_tools():
    params = StdioServerParameters(command=sys.executable, args=["-m", "emotive_gait.rig.server"])

    async def go():
        async with stdio_client(params) as (read, write):
            from mcp import ClientSession

            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                parsed = await session.call_tool("parse_emotion_prompt", {"text": "very gloomy"})
                return {t.name for t in tools.tools}, parsed.structured_content

    names, parsed = asyncio.run(asyncio.wait_for(go(), 20))
    assert names == TOOLS and parsed["intensities"]["sadness"] == pytest.approx(0.6)


def test_mapping_schema_example_uses_real_landmark_and_role_names():
    import yaml

    from emotive_gait.rig.cli import mapping_overrides_from
    from emotive_gait.rig.server import MAPPING_SCHEMA
    from emotive_gait.rig.synthesis import _split_overrides

    data = yaml.safe_load(MAPPING_SCHEMA)
    overrides, info = mapping_overrides_from(data)
    lm_over, role_over = _split_overrides(overrides)  # raises on an unknown landmark or role
    assert lm_over and role_over and set(info) == {"scale", "up_axis"}


def test_mapping_schema_example_is_accepted_by_the_tool(server, tmp_path):
    import yaml

    from emotive_gait.rig.server import MAPPING_SCHEMA

    mapping = yaml.safe_load(MAPPING_SCHEMA)  # unblanked: the example names a Mixamo rig's bones
    path = str(walk_file(RigSpec(), tmp_path))  # the synthetic Mixamo rig (y-up, centimetres)
    result = call_tool(server, "inspect_rig", {"bvh_path": path, "mapping": mapping})
    decisions = {d["what"]: d for d in result["mapping"]["decisions"]}
    asked = [*mapping["landmarks"], *(f"role:{r}" for r in mapping["roles"])]
    assert all(decisions[w]["source"] == "user" for w in asked), asked
    assert decisions["role:knee_l"]["chosen"] == "mixamorig:LeftLeg/Zrotation"
    assert result["mapping"]["roles"]["knee_r"] == "mixamorig:RightLeg/Zrotation"
    assert result["cycles"]["used"] >= 1


def test_measure_gait_on_a_standing_clip_names_the_discard_reasons(server, tmp_path):
    path = str(standing_file(RigSpec(), tmp_path))
    err = call_tool_expecting_error(server, "measure_gait", {"bvh_path": path})
    assert "no usable cycle" in err and "no forward travel" in err


def test_measure_gait_never_loads_the_rig(server, tmp_path, monkeypatch):
    from emotive_gait.rig import synthesis

    def boom(*args, **kwargs):
        raise AssertionError("measure_gait must not call load_rig or bind roles")

    monkeypatch.setattr(synthesis, "load_rig", boom)
    monkeypatch.setattr(synthesis, "bind_roles", boom)
    result = call_tool(server, "measure_gait", {"bvh_path": str(walk_file(RigSpec(), tmp_path))})
    assert result["cycles"]["used"] >= 1 and result["leg_length"] > 0


def test_measure_gait_without_knees_fails_on_the_landmark_minimum(server, tmp_path):
    from emotive_gait.mocap.writers.bvh import write_bvh
    from rig.rigs import walk_and_clip
    from rig.test_mapping import drop_bones

    f, _ = walk_and_clip(RigSpec(), tmp_path)
    path = tmp_path / "noknee.bvh"
    write_bvh(drop_bones(f, ("mixamorig:LeftLeg", "mixamorig:RightLeg")), path)
    assert "knee_l" in call_tool_expecting_error(server, "measure_gait", {"bvh_path": str(path)})


def test_out_path_equal_to_the_source_is_a_tool_error(server, tmp_path):
    path = walk_file(RigSpec(), tmp_path)
    before = path.read_bytes()
    err = call_tool_expecting_error(
        server,
        "emotional_walk",
        {"bvh_path": str(path), "out_path": str(path), "intensities": {"sadness": 0.5}},
    )
    assert "overwritten" in err and path.read_bytes() == before


def test_informational_mapping_keys_are_reported(server, tmp_path):
    path = str(walk_file(RigSpec(), tmp_path))
    result = call_tool(
        server, "inspect_rig", {"bvh_path": path, "mapping": {"family": "mixamo", "warnings": []}}
    )
    assert result["ignored_mapping_keys"] == ["family", "warnings"]
