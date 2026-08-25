#  test_server.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
import json
import os
import socketserver
import threading
from collections.abc import AsyncGenerator, Generator
from functools import partial
from http import server
from pathlib import Path
from typing import cast

import pytest
from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp.types import TextContent
from PIL import Image

TEST_DIR = Path(__file__).resolve().parent
SAMPLE_IMAGE_FILEPATH = str(TEST_DIR / "sample.jpg")
SAMPLE_PDF_FILEPATH = str(TEST_DIR / "sample.pdf")

SERVER_PARAMS = StdioServerParameters(
    command="uv",
    args=["run", "fusion-vision-mcp", "--cache-model", "--model", "florence-community/Florence-2-base"],
)


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module")
async def mcp_client_session() -> AsyncGenerator[ClientSession, None]:
    async with stdio_client(SERVER_PARAMS) as streams, ClientSession(streams[0], streams[1]) as session:
        await session.initialize()
        yield session


@pytest.fixture(scope="module")
def static_file_server() -> Generator[str, None, None]:
    with socketserver.TCPServer(
        ("", 0),
        partial(server.SimpleHTTPRequestHandler, directory=os.path.dirname(__file__)),
    ) as httpd:
        port = httpd.server_address[1]
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.start()

        try:
            yield f"http://localhost:{port}"
        finally:
            httpd.shutdown()
            httpd.server_close()
            server_thread.join()


@pytest.mark.anyio
async def test_list_tools(mcp_client_session: ClientSession) -> None:
    res = await mcp_client_session.list_tools()
    tools = {tool.name for tool in res.tools}

    assert tools == {
        "caption",
        "ocr",
        "detect_objects",
        "count_objects",
        "dense_region_caption",
        "query_image",
        "batch_analyze_images",
        "process",
        "spatial_relations",
        "score_aesthetics",
        "critique_composition",
    }


@pytest.mark.anyio
async def test_caption(mcp_client_session: ClientSession) -> None:
    res = await mcp_client_session.call_tool(
        "caption",
        arguments={"src": SAMPLE_IMAGE_FILEPATH},
    )
    text = "\n".join(cast(TextContent, c).text for c in res.content)

    assert len(text) > 0
    assert not res.is_error


@pytest.mark.anyio
async def test_caption_url(mcp_client_session: ClientSession, static_file_server: str) -> None:
    res = await mcp_client_session.call_tool(
        "caption",
        arguments={"src": static_file_server + "/sample.jpg"},
    )
    text = "\n".join(cast(TextContent, c).text for c in res.content)

    assert len(text) > 0
    assert not res.is_error


@pytest.mark.anyio
async def test_caption_pdf(mcp_client_session: ClientSession) -> None:
    res = await mcp_client_session.call_tool(
        "caption",
        arguments={"src": SAMPLE_PDF_FILEPATH},
    )
    text = "\n".join(cast(TextContent, c).text for c in res.content)

    assert len(text) > 0
    assert not res.is_error


@pytest.mark.anyio
async def test_caption_pdf_from_web(mcp_client_session: ClientSession, static_file_server: str) -> None:
    res = await mcp_client_session.call_tool(
        "caption",
        arguments={"src": static_file_server + "/sample.pdf"},
    )
    text = "\n".join(cast(TextContent, c).text for c in res.content)

    assert len(text) > 0
    assert not res.is_error


@pytest.mark.anyio
async def test_ocr(mcp_client_session: ClientSession) -> None:
    res = await mcp_client_session.call_tool(
        "ocr",
        arguments={"src": SAMPLE_IMAGE_FILEPATH},
    )
    text = "\n".join(cast(TextContent, c).text for c in res.content)

    assert len(text) > 0
    assert not res.is_error


@pytest.mark.anyio
async def test_ocr_url(mcp_client_session: ClientSession, static_file_server: str) -> None:
    res = await mcp_client_session.call_tool(
        "ocr",
        arguments={"src": static_file_server + "/sample.jpg"},
    )
    text = "\n".join(cast(TextContent, c).text for c in res.content)

    assert len(text) > 0
    assert not res.is_error


@pytest.mark.anyio
async def test_ocr_pdf(mcp_client_session: ClientSession) -> None:
    res = await mcp_client_session.call_tool(
        "ocr",
        arguments={"src": SAMPLE_PDF_FILEPATH},
    )
    text = "\n".join(cast(TextContent, c).text for c in res.content)

    assert len(text) > 0
    assert not res.is_error


@pytest.mark.anyio
async def test_ocr_pdf_from_web(mcp_client_session: ClientSession, static_file_server: str) -> None:
    res = await mcp_client_session.call_tool(
        "ocr",
        arguments={"src": static_file_server + "/sample.pdf"},
    )
    text = "\n".join(cast(TextContent, c).text for c in res.content)

    assert len(text) > 0
    assert not res.is_error


@pytest.mark.anyio
async def test_detect_objects(mcp_client_session: ClientSession) -> None:
    res = await mcp_client_session.call_tool(
        "detect_objects",
        arguments={"src": SAMPLE_IMAGE_FILEPATH, "object_name": "person"},
    )
    regions = json.loads("\n".join(cast(TextContent, c).text for c in res.content))

    assert not res.is_error
    assert "bboxes" in regions
    assert "labels" in regions


@pytest.mark.anyio
async def test_detect_objects_returns_center_points_alongside_boxes(mcp_client_session: ClientSession) -> None:
    """Center points used to be their own tool; they now ride along with the boxes."""
    res = await mcp_client_session.call_tool(
        "detect_objects",
        arguments={"src": SAMPLE_IMAGE_FILEPATH, "object_name": "person"},
    )
    regions = json.loads("\n".join(cast(TextContent, c).text for c in res.content))

    assert not res.is_error
    assert "points" in regions
    # Points are centres, so each one carries an x and a y, one per box.
    assert len(regions["points"]) == len(regions["bboxes"])
    assert all(len(point) == 2 for point in regions["points"])
    for (x1, y1, x2, y2), (px, py) in zip(regions["bboxes"], regions["points"], strict=True):
        assert px == pytest.approx((x1 + x2) / 2)
        assert py == pytest.approx((y1 + y2) / 2)


@pytest.mark.anyio
async def test_count_objects(mcp_client_session: ClientSession) -> None:
    res = await mcp_client_session.call_tool(
        "count_objects",
        arguments={"src": SAMPLE_IMAGE_FILEPATH, "object_name": "petal"},
    )
    counted = json.loads("\n".join(cast(TextContent, c).text for c in res.content))

    assert not res.is_error
    assert isinstance(counted["count"], int)
    # The count IS the number of detections kept -- that equality is the whole point of
    # the tool, since `detect_objects`' region count is explicitly not a tally.
    assert counted["count"] == len(counted["bboxes"])
    assert len(counted["labels"]) == len(counted["bboxes"])
    assert len(counted["scores"]) == len(counted["bboxes"])
    assert all(0.0 <= score <= 1.0 for score in counted["scores"])
    assert isinstance(counted["group_boxes_dropped"], int)


@pytest.mark.anyio
async def test_count_objects_returns_pixel_space_boxes(mcp_client_session: ClientSession) -> None:
    """Boxes must land on the image's own pixel grid, matching detect_objects."""
    with Image.open(SAMPLE_IMAGE_FILEPATH) as img:
        width, height = img.size

    res = await mcp_client_session.call_tool(
        "count_objects",
        arguments={"src": SAMPLE_IMAGE_FILEPATH, "object_name": "petal"},
    )
    counted = json.loads("\n".join(cast(TextContent, c).text for c in res.content))

    assert not res.is_error
    # A box still in normalized space would sit inside 0-1 and fail this on any real image.
    for x1, y1, x2, y2 in counted["bboxes"]:
        assert 0 <= x1 <= width and 0 <= x2 <= width
        assert 0 <= y1 <= height and 0 <= y2 <= height
    for (x1, y1, x2, y2), (px, py) in zip(counted["bboxes"], counted["points"], strict=True):
        assert px == pytest.approx((x1 + x2) / 2)
        assert py == pytest.approx((y1 + y2) / 2)


@pytest.mark.anyio
async def test_count_objects_adds_a_silhouette_when_it_finds_only_one(mcp_client_session: ClientSession) -> None:
    """A lone detection is the collapse case, so it gets a geometric second opinion."""
    res = await mcp_client_session.call_tool(
        "count_objects",
        arguments={"src": SAMPLE_IMAGE_FILEPATH, "object_name": "petal"},
    )
    counted = json.loads("\n".join(cast(TextContent, c).text for c in res.content))

    assert not res.is_error
    if counted["count"] == 1:
        # `count` is never rewritten -- the estimate arrives beside it, not instead.
        assert counted["count"] == 1
        assert "lobes" in counted["silhouette"]
        assert counted["silhouette"]["box"] == [int(v) for v in counted["bboxes"][0]]
    else:
        assert "silhouette" not in counted


@pytest.mark.anyio
async def test_count_objects_can_skip_the_silhouette_check(mcp_client_session: ClientSession) -> None:
    res = await mcp_client_session.call_tool(
        "count_objects",
        arguments={"src": SAMPLE_IMAGE_FILEPATH, "object_name": "petal", "verify_silhouette": False},
    )
    counted = json.loads("\n".join(cast(TextContent, c).text for c in res.content))

    assert not res.is_error
    assert "silhouette" not in counted


@pytest.mark.anyio
async def test_batch_analyze_images_requires_object_name_for_count(mcp_client_session: ClientSession) -> None:
    res = await mcp_client_session.call_tool(
        "batch_analyze_images",
        arguments={"srcs": [SAMPLE_IMAGE_FILEPATH], "operation": "count"},
    )
    results = [json.loads(cast(TextContent, c).text) for c in res.content]

    assert not res.is_error
    assert not results[0]["success"]
    assert "object_name" in results[0]["error"]


@pytest.mark.anyio
async def test_dense_region_caption(mcp_client_session: ClientSession) -> None:
    res = await mcp_client_session.call_tool(
        "dense_region_caption",
        arguments={"src": SAMPLE_IMAGE_FILEPATH},
    )
    regions = json.loads("\n".join(cast(TextContent, c).text for c in res.content))

    assert not res.is_error
    assert "bboxes" in regions
    assert "labels" in regions


@pytest.mark.anyio
async def test_batch_analyze_images_rejects_unknown_operation(mcp_client_session: ClientSession) -> None:
    res = await mcp_client_session.call_tool(
        "batch_analyze_images",
        arguments={"srcs": [SAMPLE_IMAGE_FILEPATH], "operation": "translate"},
    )
    results = [json.loads(cast(TextContent, c).text) for c in res.content]

    # Per-image isolation means a bad operation surfaces as a failed entry, not a raised call.
    assert not res.is_error
    assert not results[0]["success"]


@pytest.mark.anyio
async def test_batch_analyze_images_requires_object_name_for_detect(mcp_client_session: ClientSession) -> None:
    res = await mcp_client_session.call_tool(
        "batch_analyze_images",
        arguments={"srcs": [SAMPLE_IMAGE_FILEPATH], "operation": "detect"},
    )
    results = [json.loads(cast(TextContent, c).text) for c in res.content]

    assert not res.is_error
    assert not results[0]["success"]
    assert "object_name" in results[0]["error"]


@pytest.mark.anyio
async def test_batch_analyze_images(mcp_client_session: ClientSession) -> None:
    res = await mcp_client_session.call_tool(
        "batch_analyze_images",
        arguments={"srcs": [SAMPLE_IMAGE_FILEPATH, SAMPLE_IMAGE_FILEPATH], "operation": "ocr"},
    )
    results = [json.loads(cast(TextContent, c).text) for c in res.content]

    assert not res.is_error
    assert len(results) == 2
    assert all(item["success"] for item in results)


@pytest.mark.anyio
async def test_batch_analyze_images_reports_failures_per_image(mcp_client_session: ClientSession) -> None:
    missing = str(TEST_DIR / "does-not-exist.jpg")
    res = await mcp_client_session.call_tool(
        "batch_analyze_images",
        arguments={"srcs": [SAMPLE_IMAGE_FILEPATH, missing], "operation": "ocr"},
    )
    results = [json.loads(cast(TextContent, c).text) for c in res.content]

    # A bad image must not abort the batch: the good one still reports a result.
    assert not res.is_error
    assert results[0]["success"]
    assert not results[1]["success"]
    assert results[1]["error"]


@pytest.mark.anyio
async def test_process(mcp_client_session: ClientSession) -> None:
    res = await mcp_client_session.call_tool(
        "process",
        arguments={"src": SAMPLE_IMAGE_FILEPATH, "prompt": "<CAPTION>"},
    )
    text = "\n".join(cast(TextContent, c).text for c in res.content)

    assert len(text) > 0
    assert not res.is_error


@pytest.mark.anyio
async def test_process_url(mcp_client_session: ClientSession, static_file_server: str) -> None:
    res = await mcp_client_session.call_tool(
        "process",
        arguments={"src": static_file_server + "/sample.jpg", "prompt": "<CAPTION>"},
    )
    text = "\n".join(cast(TextContent, c).text for c in res.content)

    assert len(text) > 0
    assert not res.is_error


@pytest.mark.anyio
async def test_process_pdf(mcp_client_session: ClientSession) -> None:
    res = await mcp_client_session.call_tool(
        "process",
        arguments={"src": SAMPLE_PDF_FILEPATH, "prompt": "<CAPTION>"},
    )
    text = "\n".join(cast(TextContent, c).text for c in res.content)

    assert len(text) > 0
    assert not res.is_error


@pytest.mark.anyio
async def test_process_pdf_from_web(mcp_client_session: ClientSession, static_file_server: str) -> None:
    res = await mcp_client_session.call_tool(
        "process",
        arguments={"src": static_file_server + "/sample.pdf", "prompt": "<CAPTION>"},
    )
    text = "\n".join(cast(TextContent, c).text for c in res.content)

    assert len(text) > 0
    assert not res.is_error


@pytest.mark.anyio
async def test_score_aesthetics(mcp_client_session: ClientSession) -> None:
    res = await mcp_client_session.call_tool(
        "score_aesthetics",
        arguments={"src": SAMPLE_IMAGE_FILEPATH},
    )
    results = [json.loads(cast(TextContent, c).text) for c in res.content]

    assert not res.is_error
    assert len(results) == 1
    assert "score" in results[0]
    assert "rating" in results[0]


@pytest.mark.anyio
async def test_score_aesthetics_pdf(mcp_client_session: ClientSession) -> None:
    res = await mcp_client_session.call_tool(
        "score_aesthetics",
        arguments={"src": SAMPLE_PDF_FILEPATH},
    )
    results = [json.loads(cast(TextContent, c).text) for c in res.content]

    assert not res.is_error
    assert len(results) >= 1
    assert all("score" in item for item in results)


@pytest.mark.anyio
async def test_critique_composition(mcp_client_session: ClientSession) -> None:
    res = await mcp_client_session.call_tool(
        "critique_composition",
        arguments={"src": SAMPLE_IMAGE_FILEPATH, "target_subject": "person"},
    )
    result = json.loads("\n".join(cast(TextContent, c).text for c in res.content))

    assert not res.is_error
    assert "aesthetics" in result
    assert "framing" in result
    assert "subject_box" in result


@pytest.mark.anyio
async def test_critique_composition_auto_detects_subject(mcp_client_session: ClientSession) -> None:
    res = await mcp_client_session.call_tool(
        "critique_composition",
        arguments={"src": SAMPLE_IMAGE_FILEPATH},
    )
    result = json.loads("\n".join(cast(TextContent, c).text for c in res.content))

    assert not res.is_error
    assert "aesthetics" in result


@pytest.mark.anyio
async def test_critique_composition_reports_note_when_subject_not_found(
    mcp_client_session: ClientSession, tmp_path: Path
) -> None:
    """Florence-2's grounding is generous even for a nonsense phrase (it still finds *something*
    in a real photo, per detect_objects's own ambiguous-class caveat), so the reliable way to
    trigger the "nothing found" path is a genuinely featureless image, not an implausible phrase.
    """
    blank_filepath = tmp_path / "blank.jpg"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(blank_filepath)

    res = await mcp_client_session.call_tool(
        "critique_composition",
        arguments={"src": str(blank_filepath)},
    )
    result = json.loads("\n".join(cast(TextContent, c).text for c in res.content))

    assert not res.is_error
    assert "note" in result
