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
SPATIAL_TOUCH_SEPARATE_FILEPATH = str(TEST_DIR / "spatial_touch_separate.png")
SPATIAL_CONTAINMENT_FILEPATH = str(TEST_DIR / "spatial_containment.png")
LAYOUT_TWO_COLUMN_FILEPATH = str(TEST_DIR / "layout_two_column.png")

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
async def test_caption_verify_text_returns_caption_and_text_regions(mcp_client_session: ClientSession) -> None:
    """`verify_text=true` returns the caption plus verbatim OCR spans to cross-check it."""
    res = await mcp_client_session.call_tool(
        "caption",
        arguments={"src": SAMPLE_IMAGE_FILEPATH, "verify_text": True},
    )
    pages = [json.loads(cast(TextContent, c).text) for c in res.content]

    assert not res.is_error
    assert len(pages) == 1
    page = pages[0]
    assert "caption" in page and "text_regions" in page
    assert page["caption"]
    # Each text region is a {text, box} with a 4-int box.
    for region in page["text_regions"]:
        assert "text" in region and "box" in region
        assert len(region["box"]) == 4




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
async def test_ocr_with_regions_returns_text_and_boxes(mcp_client_session: ClientSession) -> None:
    """`with_regions=true` returns each text span alongside its page-coordinate box."""
    res = await mcp_client_session.call_tool(
        "ocr",
        arguments={"src": LAYOUT_TWO_COLUMN_FILEPATH, "with_regions": True},
    )
    pages = [json.loads(cast(TextContent, c).text) for c in res.content]

    assert not res.is_error
    assert len(pages) == 1
    page = pages[0]
    assert "text" in page and "text_regions" in page
    assert len(page["text_regions"]) > 0
    # Every region carries a non-empty text and a 4-int box in page coordinates.
    for region in page["text_regions"]:
        assert region["text"]
        assert len(region["box"]) == 4
        assert all(isinstance(v, int) for v in region["box"])
    # The joined text is the concatenation of the per-region texts.
    assert page["text"] == "\n".join(r["text"] for r in page["text_regions"])


@pytest.mark.anyio
async def test_ocr_with_regions_false_keeps_string_shape(mcp_client_session: ClientSession) -> None:
    """`with_regions=false` (default) keeps the original list[str] return shape."""
    res = await mcp_client_session.call_tool(
        "ocr",
        arguments={"src": LAYOUT_TWO_COLUMN_FILEPATH},
    )
    text = "\n".join(cast(TextContent, c).text for c in res.content)

    assert not res.is_error
    assert len(text) > 0




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
async def test_count_objects_adds_consensus_and_separability(mcp_client_session: ClientSession) -> None:
    """`consensus=true` (default) adds a second-opinion count and a separability flag.

    sample.jpg + "petal" is the canonical collapse case: count=1, by_distance=1,
    by_radial=8, agreement=False.  The tool's own docstring names this as the
    overlapping-petal example nothing local can count honestly, so separable
    must be "no" here, not "yes".
    """
    res = await mcp_client_session.call_tool(
        "count_objects",
        arguments={"src": SAMPLE_IMAGE_FILEPATH, "object_name": "petal"},
    )
    counted = json.loads("\n".join(cast(TextContent, c).text for c in res.content))

    assert not res.is_error
    assert "consensus" in counted
    assert "separable" in counted
    assert counted["consensus"]["grounding_dino_count"] == counted["count"]
    assert isinstance(counted["consensus"]["region_label_count"], int)
    assert isinstance(counted["consensus"]["agree"], bool)

    if counted["count"] == 1 and counted.get("silhouette") and counted["silhouette"].get("by_radial", 0) > 1:
        # This is the canonical rosette collapse: distance says 1, radial says N,
        # separable must reflect the ambiguity, not assert a clean count.
        assert counted["separable"] == "no", (
            f"Expected separable='no' on the rosette collapse case "
            f"(count=1, by_radial={counted['silhouette'].get('by_radial')}), "
            f"got '{counted['separable']}'"
        )
    assert counted["separable"] in ("yes", "no", "unknown")


@pytest.mark.anyio
async def test_count_objects_can_skip_consensus(mcp_client_session: ClientSession) -> None:
    """`consensus=false` omits the second-opinion and separability fields."""
    res = await mcp_client_session.call_tool(
        "count_objects",
        arguments={
            "src": SAMPLE_IMAGE_FILEPATH,
            "object_name": "petal",
            "verify_silhouette": False,
            "consensus": False,
        },
    )
    counted = json.loads("\n".join(cast(TextContent, c).text for c in res.content))

    assert not res.is_error
    assert "consensus" not in counted
    assert "separable" not in counted




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
@pytest.mark.parametrize("prompt", ["<OD>", "<REGION_PROPOSAL>", "<OCR_WITH_REGION>"])
async def test_process_handles_structured_task_tokens(mcp_client_session: ClientSession, prompt: str) -> None:
    """These three are the exact examples process's own docstring names as valid uses, but a
    structured task token decodes to a dict rather than a string, and calling `.strip()` on
    that crashed here until `Florence2.generate` learned to JSON-encode a non-string result.
    """
    res = await mcp_client_session.call_tool(
        "process",
        arguments={"src": SAMPLE_IMAGE_FILEPATH, "prompt": prompt},
    )
    text = "\n".join(cast(TextContent, c).text for c in res.content)

    assert not res.is_error
    parsed = json.loads(text)
    assert isinstance(parsed, dict)


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
async def test_score_aesthetics_with_style_context(mcp_client_session: ClientSession) -> None:
    """`style_context=true` adds the image's medium (from CLIP zero-shot) to the score."""
    res = await mcp_client_session.call_tool(
        "score_aesthetics",
        arguments={"src": SAMPLE_IMAGE_FILEPATH, "style_context": True},
    )
    results = [json.loads(cast(TextContent, c).text) for c in res.content]

    assert not res.is_error
    assert len(results) == 1
    assert "style" in results[0]
    assert "style_distribution" in results[0]
    assert len(results[0]["style_distribution"]) == 16


@pytest.mark.anyio
async def test_score_aesthetics_style_context_false_omits_style(mcp_client_session: ClientSession) -> None:
    """`style_context=false` (default) keeps the original {score, rating} shape."""
    res = await mcp_client_session.call_tool(
        "score_aesthetics",
        arguments={"src": SAMPLE_IMAGE_FILEPATH},
    )
    results = [json.loads(cast(TextContent, c).text) for c in res.content]

    assert not res.is_error
    assert "style" not in results[0]


@pytest.mark.anyio
async def test_query_image_check_consistency_returns_agreement_fields(
    mcp_client_session: ClientSession,
) -> None:
    """`check_consistency=true` returns answer/control/consistent/confidence per image."""
    res = await mcp_client_session.call_tool(
        "query_image",
        arguments={"src": SAMPLE_IMAGE_FILEPATH, "question": "What is in this image?", "check_consistency": True},
    )
    results = [json.loads(cast(TextContent, c).text) for c in res.content]

    assert not res.is_error
    assert len(results) == 1
    r = results[0]
    assert {"answer", "control_answer", "consistent", "confidence"} <= set(r.keys())
    assert r["confidence"] in ("low", "normal")


@pytest.mark.anyio
async def test_query_image_check_consistency_false_keeps_string_shape(
    mcp_client_session: ClientSession,
) -> None:
    """`check_consistency=false` (default) keeps the original list[str] return shape."""
    res = await mcp_client_session.call_tool(
        "query_image",
        arguments={"src": SAMPLE_IMAGE_FILEPATH, "question": "What is in this image?"},
    )
    text = "\n".join(cast(TextContent, c).text for c in res.content)

    assert not res.is_error
    assert len(text) > 0




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


@pytest.mark.anyio
async def test_spatial_relations_discriminates_same_shaped_objects_by_color(
    mcp_client_session: ClientSession,
) -> None:
    """Locating by color used to fail silently: asking for 'red circle'/'blue circle'/
    'green circle' on a scene with one of each returned the same three boxes for every
    query, so a caller comparing "red circle" to "blue circle" was really comparing one
    object to itself. Taking the single best-scoring match per name fixed it -- this
    pins the three distinct locations and the geometrically correct relations between them.
    """
    res = await mcp_client_session.call_tool(
        "spatial_relations",
        arguments={
            "src": SPATIAL_TOUCH_SEPARATE_FILEPATH,
            "objects": ["red circle", "blue circle", "green circle"],
        },
    )
    data = json.loads("\n".join(cast(TextContent, c).text for c in res.content))

    assert not res.is_error
    assert len(data["objects"]) == 3
    boxes = [tuple(obj["box"]) for obj in data["objects"]]
    assert len(set(boxes)) == 3  # each color must land on its own, distinct box

    by_pair = {frozenset((r["a"].split("#")[0], r["b"].split("#")[0])): r for r in data["relations"]}
    red_blue = by_pair[frozenset(("red circle", "blue circle"))]
    red_green = by_pair[frozenset(("red circle", "green circle"))]
    blue_green = by_pair[frozenset(("blue circle", "green circle"))]

    # Constructed to just touch: two circles of radius 60 with centres 120px apart.
    assert red_blue["state"] in ("touching", "overlapping")
    assert red_blue["gap"] < 10
    # Constructed far apart on a 600px-wide canvas.
    assert red_green["state"] == "separate"
    assert red_green["gap"] > 100
    assert blue_green["state"] == "separate"
    assert blue_green["gap"] > 100


@pytest.mark.anyio
async def test_spatial_relations_measures_containment(mcp_client_session: ClientSession) -> None:
    res = await mcp_client_session.call_tool(
        "spatial_relations",
        arguments={"src": SPATIAL_CONTAINMENT_FILEPATH, "objects": ["yellow circle", "purple circle"]},
    )
    data = json.loads("\n".join(cast(TextContent, c).text for c in res.content))

    assert not res.is_error
    assert len(data["objects"]) == 2
    relation = data["relations"][0]
    # The purple circle is constructed entirely inside the yellow one.
    assert relation["b_inside_a"] > 0.9 or relation["a_inside_b"] > 0.9

# ---------------------------------------------------------------------------
# Unit tests for the consensus/agreement helpers.
# These use documented fixture data directly, no model required.
# ---------------------------------------------------------------------------

from fusion_vision_mcp import _separability, _vqa_consistency  # noqa: E402


class TestSeparability:
    """Pins _separability against the exact geometry data from the documented failure cases."""

    def test_rosette_collapse_returns_no(self) -> None:
        """The documented flower case: count=1, lobes(by_distance)=1, by_radial=8, agreement=False.

        By_distance found no saddle between overlapping petals (lobes=1); by_radial
        correctly counted 8 lobes from the angular notch pattern. The estimator
        disagreement is itself the signal the count is a structural collapse.
        """
        result = {
            "count": 1,
            "silhouette": {
                "lobes": 1,
                "by_radial": 8,
                "agreement": False,
            },
        }
        assert _separability(result) == "no"

    def test_genuine_singleton_agreement_returns_yes(self) -> None:
        """Count=1 and both estimators agree the outline is one lobe → "yes"."""
        result = {
            "count": 1,
            "silhouette": {
                "lobes": 1,
                "by_radial": 1,
                "agreement": True,
            },
        }
        assert _separability(result) == "yes"

    def test_agreed_multi_lobe_returns_no(self) -> None:
        """Count=1 but both estimators agree on 4 lobes → "no" (collapsed)."""
        result = {
            "count": 1,
            "silhouette": {
                "lobes": 4,
                "by_radial": 4,
                "agreement": True,
            },
        }
        assert _separability(result) == "no"

    def test_multi_detection_returns_yes(self) -> None:
        """Count>1 means the detector separated things → "yes" regardless of silhouette."""
        assert _separability({"count": 5}) == "yes"

    def test_no_silhouette_returns_unknown(self) -> None:
        assert _separability({"count": 1}) == "unknown"

    def test_shattered_returns_unknown(self) -> None:
        result = {"count": 1, "silhouette": {"lobes": 1, "shattered": True}}
        assert _separability(result) == "unknown"

    def test_by_radial_zero_means_not_measured_trusts_by_distance(self) -> None:
        """by_radial=0: a non-rosette shape — by_distance is the only signal."""
        assert _separability({"count": 1, "silhouette": {"lobes": 1, "by_radial": 0, "agreement": False}}) == "yes"
        assert _separability({"count": 1, "silhouette": {"lobes": 3, "by_radial": 0, "agreement": False}}) == "no"


class TestVqaConsistency:
    """Pins _vqa_consistency against the documented failure modes."""

    def test_agreed_flat_default_returns_low(self) -> None:
        """"None"/"None" — same default token on two phrasings → low confidence."""
        r = _vqa_consistency("None", "None")
        assert r["consistent"] is True
        assert r["confidence"] == "low"

    def test_different_flat_defaults_returns_low(self) -> None:
        """"None"/"Nothing" — both are defaults but they disagree → low confidence."""
        r = _vqa_consistency("None", "Nothing")
        assert r["consistent"] is False
        assert r["confidence"] == "low"

    def test_contradictory_substantive_answers_returns_low(self) -> None:
        """The documented "pette" probe case: one answer says something's wrong,
        the control says nothing's wrong. Both substantive, both contradict.
        Confidence must be "low" — the model is contradicting itself.
        """
        r = _vqa_consistency(
            "The image is missing a centerpiece and there is no visible stem or base",
            "Nothing is wrong with this image; it is a beautiful, symmetrical flower arrangement",
        )
        assert r["consistent"] is False
        assert r["confidence"] == "low"

    def test_agreed_substantive_answer_returns_normal(self) -> None:
        """Substantive answers that genuinely agree → normal confidence."""
        r = _vqa_consistency(
            "a red car",
            "I see a red car in the image",
        )
        assert r["consistent"] is True
        assert r["confidence"] == "normal"

    def test_agreed_semantic_opposite_defaults_returns_low(self) -> None:
        """"Yes"/"No" are both flat defaults, and they disagree → low confidence."""
        r = _vqa_consistency("Yes", "No")
        assert r["consistent"] is False
        assert r["confidence"] == "low"
