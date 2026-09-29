"""Guards on the text a calling agent reads at tool-selection time.

Tools are registered on a bare server, so this loads no model and stays fast.
"""

import re

import pytest
from mcp.server.mcpserver import MCPServer

from fusion_vision_mcp.tools import register_all

# A caller cannot open CLAUDE.md, the benchmark tree or a spec/sprint number, and a
# version stamp describes history rather than current behaviour.
_INTERNAL_REFERENCE = re.compile(r"\bSpec \d+|CLAUDE\.md|benchmarks/|Sprint \d+|\bv\d+\.\d+\.\d+\b")

# EasyOCR replaced the head this advice was written about; routing text away from
# `ocr` sends it to a model that paraphrases.
_STALE_OCR_ROUTING = re.compile(r"ocr\W{0,3}\s*(tool\s+)?misreads", re.IGNORECASE)


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


async def _descriptions() -> list[tuple[str, str]]:
    mcp = MCPServer("descriptions")
    register_all(mcp)
    found = []
    for tool in await mcp.list_tools():
        found.append((tool.name, tool.description or ""))
        for name, prop in tool.input_schema["properties"].items():
            found.append((f"{tool.name}.{name}", prop.get("description", "")))
    return found


@pytest.mark.anyio
async def test_every_tool_and_parameter_is_described() -> None:
    missing = [where for where, text in await _descriptions() if not text.strip()]
    assert missing == []


@pytest.mark.anyio
async def test_descriptions_carry_no_internal_references() -> None:
    leaks = [(where, _INTERNAL_REFERENCE.findall(text)) for where, text in await _descriptions()]
    assert [leak for leak in leaks if leak[1]] == []


@pytest.mark.anyio
async def test_no_description_routes_text_reading_away_from_ocr() -> None:
    stale = [where for where, text in await _descriptions() if _STALE_OCR_ROUTING.search(text)]
    assert stale == []
