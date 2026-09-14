"""Pure unit tests for routing.py.  No model, no server, no network -- runs in milliseconds.

Live routing accuracy against the clip-art scene fixture is covered in ``test_server.py``
(the ``test_count_objects_auto_routes_clipart_scene`` integration test), where the full
MCP server with Florence-2 is available.  That test is excluded from the fast suite by the
``-k "not test_server"`` filter.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fusion_vision_mcp.routing import (
    COLOR_DENSITY_PHOTO_THRESHOLD,
    LUMINANCE_STDDEV_PHOTO_THRESHOLD,
    MEAN_SATURATION_GRAYSCALE_THRESHOLD,
    RoutingDecision,
    choose_count_backend,
    count_unique_colors_per_1k_px,
    luminance_stddev,
    mean_saturation,
)

TESTS_DIR = Path(__file__).parent
SAMPLE_JPG = TESTS_DIR / "sample.jpg"
CLIPART_SCENE = TESTS_DIR / "clipart_scene.png"


# ---------------------------------------------------------------------------
# Threshold constant sanity
# ---------------------------------------------------------------------------


def test_color_density_threshold_is_positive_and_in_gap() -> None:
    """COLOR_DENSITY_PHOTO_THRESHOLD must lie in the measured gap between synthetic (<3) and photo (>=65)."""
    assert COLOR_DENSITY_PHOTO_THRESHOLD > 3
    assert COLOR_DENSITY_PHOTO_THRESHOLD < 65


def test_mean_saturation_threshold_is_positive() -> None:
    """MEAN_SATURATION_GRAYSCALE_THRESHOLD must be positive and below the clip-art scene's measured saturation."""
    assert MEAN_SATURATION_GRAYSCALE_THRESHOLD > 0
    # Clip-art scene measured at 87.2; threshold must leave a wide margin below that.
    assert MEAN_SATURATION_GRAYSCALE_THRESHOLD < 50


def test_luminance_stddev_threshold_is_positive_and_below_photo_range() -> None:
    """LUMINANCE_STDDEV_PHOTO_THRESHOLD must be positive and below the expected greyscale-photo range (40-80+)."""
    assert LUMINANCE_STDDEV_PHOTO_THRESHOLD > 0
    assert LUMINANCE_STDDEV_PHOTO_THRESHOLD < 40


# ---------------------------------------------------------------------------
# count_unique_colors_per_1k_px
# ---------------------------------------------------------------------------


def test_all_white_image_has_near_zero_density() -> None:
    """An all-white 512x512 image has exactly 1 unique color; density is ~0.004/1k px."""
    img = Image.new("RGB", (512, 512), (255, 255, 255))
    density = count_unique_colors_per_1k_px(img)
    assert density < 0.01
    assert density > 0


def test_clipart_scene_has_low_density() -> None:
    """The flat-color clip-art scene fixture scores well below the photo threshold.

    Measured: 13 unique colors over 262144 pixels = ~0.050/1k px.
    """
    with Image.open(CLIPART_SCENE) as img:
        density = count_unique_colors_per_1k_px(img)
    assert density < COLOR_DENSITY_PHOTO_THRESHOLD
    # Confirm it is in the synthetic-art range observed across all benchmarks (< 3)
    assert density < 3.0


def test_real_photo_has_high_density() -> None:
    """sample.jpg (the flower photo) should score well above the photo threshold.

    Sprint 15 measured this image at ~384/1k px.  Even at a different internal resolution
    it should exceed 65 (the Sprint 15 minimum across all 35 COCO photos).
    """
    with Image.open(SAMPLE_JPG) as img:
        density = count_unique_colors_per_1k_px(img)
    assert density >= 65.0


def test_1x1_image_does_not_raise() -> None:
    """A 1x1 image is the smallest possible input; ensure no divide-by-zero."""
    img = Image.new("RGB", (1, 1), (128, 64, 32))
    density = count_unique_colors_per_1k_px(img)
    # 1 unique color, 1 pixel: 1/1 * 1000 = 1000.0
    assert density == pytest.approx(1000.0)


def test_rgba_input_is_handled() -> None:
    """RGBA images must not raise; the alpha channel is stripped before counting."""
    img = Image.new("RGBA", (64, 64), (100, 150, 200, 128))
    density = count_unique_colors_per_1k_px(img)
    # Single (R,G,B) colour after stripping alpha -> near-zero density
    assert density < COLOR_DENSITY_PHOTO_THRESHOLD


# ---------------------------------------------------------------------------
# mean_saturation
# ---------------------------------------------------------------------------


def test_pure_greyscale_image_has_zero_saturation() -> None:
    """A pure greyscale image (R=G=B for every pixel) has mean saturation 0."""
    arr = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    img = Image.fromarray(np.stack([arr, arr, arr], axis=2))
    sat = mean_saturation(img)
    assert sat == pytest.approx(0.0, abs=0.01)


def test_clipart_scene_has_high_saturation() -> None:
    """The flat clip-art scene uses saturated hues; mean saturation is well above the greyscale gate.

    Measured: 87.2/255 for the 13-fill tree/house scene.
    """
    with Image.open(CLIPART_SCENE) as img:
        sat = mean_saturation(img)
    assert sat >= MEAN_SATURATION_GRAYSCALE_THRESHOLD
    # Confirm it is comfortably above the threshold (not borderline)
    assert sat > 50


def test_uniform_white_has_zero_saturation() -> None:
    """White (255,255,255) has R=G=B, so saturation is 0."""
    img = Image.new("RGB", (256, 256), (255, 255, 255))
    sat = mean_saturation(img)
    assert sat == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# luminance_stddev
# ---------------------------------------------------------------------------


def test_uniform_white_image_has_zero_luminance_stddev() -> None:
    """A completely uniform image has zero V-channel variance."""
    img = Image.new("RGB", (256, 256), (255, 255, 255))
    std = luminance_stddev(img)
    assert std == pytest.approx(0.0, abs=0.1)


def test_gradient_image_has_high_luminance_stddev() -> None:
    """A smooth gradient from black to white has high V-channel variance (greyscale-photo proxy).

    Std of 0..255 uniform distribution is ~73.6, which should comfortably exceed threshold.
    """
    arr = np.tile(np.arange(256, dtype=np.uint8), (256, 1))  # shape (256,256)
    img = Image.fromarray(np.stack([arr, arr, arr], axis=2))
    std = luminance_stddev(img)
    assert std >= LUMINANCE_STDDEV_PHOTO_THRESHOLD


# ---------------------------------------------------------------------------
# choose_count_backend
# ---------------------------------------------------------------------------


def test_real_photo_routes_to_grounding_dino() -> None:
    """sample.jpg must route to Grounding DINO (photograph path, caught at color density step 1)."""
    with Image.open(SAMPLE_JPG) as img:
        decision = choose_count_backend(img)
    assert decision.backend == "grounding_dino"


def test_clipart_scene_routes_to_florence2() -> None:
    """The flat-color clip-art fixture must route to Florence-2 (caught at saturation step 2)."""
    with Image.open(CLIPART_SCENE) as img:
        decision = choose_count_backend(img)
    assert decision.backend == "florence2"
    # Confirm the saturation path fired (not the luminance path)
    assert "saturated" in decision.reason


def test_blank_canvas_routes_to_florence2() -> None:
    """A blank white canvas routes to Florence-2 (0 colors, 0 saturation, 0 luminance variance)."""
    img = Image.new("RGB", (512, 512), (255, 255, 255))
    decision = choose_count_backend(img)
    assert decision.backend == "florence2"


def test_grayscale_gradient_routes_to_grounding_dino() -> None:
    """A greyscale gradient is rescued by the luminance-stddev check at step 3.

    This represents the Sprint 15 edge case: coco_bird_3 had only 0.94 unique colors/1k px
    but is a real greyscale photograph.  Low saturation (0) passes the greyscale gate;
    high luminance std (~73) rescues it as a photograph.
    """
    arr = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    img = Image.fromarray(np.stack([arr, arr, arr], axis=2))
    decision = choose_count_backend(img)
    assert decision.backend == "grounding_dino"
    assert "greyscale photograph" in decision.reason


def test_routing_decision_is_frozen() -> None:
    """RoutingDecision must be immutable (frozen dataclass)."""
    d = RoutingDecision(
        backend="florence2",
        unique_colors_per_1k_px=0.05,
        mean_saturation=87.2,
        luminance_stddev=0.0,
        reason="test",
    )
    with pytest.raises((AttributeError, TypeError)):
        d.backend = "grounding_dino"  # type: ignore[misc]


def test_routing_decision_fields_are_accessible() -> None:
    """RoutingDecision exposes all five expected fields."""
    d = RoutingDecision(
        backend="grounding_dino",
        unique_colors_per_1k_px=384.0,
        mean_saturation=0.0,
        luminance_stddev=0.0,
        reason="color density 384.00 >= threshold 10.0 -> photograph",
    )
    assert d.backend == "grounding_dino"
    assert d.unique_colors_per_1k_px == 384.0
    assert d.mean_saturation == 0.0
    assert d.luminance_stddev == 0.0
    assert d.reason != ""


def test_routing_reason_is_non_empty_string() -> None:
    """Every routing path must return a non-empty reason string."""
    with Image.open(CLIPART_SCENE) as img:
        d = choose_count_backend(img)
    assert isinstance(d.reason, str)
    assert len(d.reason) > 0

    with Image.open(SAMPLE_JPG) as img:
        d2 = choose_count_backend(img)
    assert isinstance(d2.reason, str)
    assert len(d2.reason) > 0
