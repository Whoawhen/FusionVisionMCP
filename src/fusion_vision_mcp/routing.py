#  routing.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Automatic back-end selection for `count_objects` based on model-free pixel statistics.

`choose_count_backend` decides whether to route a counting request through Florence-2
(better on flat vector / clip-art imagery) or Grounding DINO (better on real photographs)
without calling any neural network.  The decision uses two cheap pixel statistics measured
in a three-step cascade:

1. **Unique RGB colors per 1,000 pixels.**  Flat synthetic art is built from a handful of
   solid fills -- the ring fixtures score under 3 colors / 1k px, while every real COCO
   photograph in the Sprint 15 benchmark scored between 65 and 686 (mean 260).  A threshold
   of 10 sits roughly seven times above the highest synthetic score and seven times below the
   lowest real-photo score, giving a wide empirical margin in both directions.

2. **Mean HSV saturation** (grayscale gate).  When color density is below the photo
   threshold, the image could be either flat synthetic art OR a genuine grayscale photograph
   (Sprint 15 found one: coco_bird_3, a real photo scoring 0.94 unique colors/1k px because
   the camera image was greyscale).  Mean HSV saturation distinguishes the two: a truly
   greyscale image has saturation ~= 0 for every pixel, whereas flat synthetic art with
   diverse solid fills spans a wide range of hues and scores high saturation.  Empirically
   measured: clip-art scene with 13 distinct fills scores 87.2 saturation; a pure greyscale
   gradient scores 0.0.  A real colour photo scores 39.5 but is already caught by step 1.
   Threshold: below `MEAN_SATURATION_GRAYSCALE_THRESHOLD` (15.0) qualifies as potentially
   greyscale and advances to step 3; otherwise -- saturated fills -> synthetic art -> Florence-2.

3. **Luminance standard deviation** (grayscale-photo rescue).  An image that is both low
   color density AND low saturation is either a flat nearly-uniform grey synthetic image or a
   genuine greyscale photograph.  A real photograph -- even greyscale -- has strong tonal
   variation from camera noise and natural lighting; a flat-fill synthetic image has near-zero
   luminance variance.  At or above `LUMINANCE_STDDEV_PHOTO_THRESHOLD` (20.0) -> greyscale
   photograph -> Grounding DINO.  Below -> flat synthetic art -> Florence-2.

SigLIP2's domain label is intentionally absent from the routing decision.  Sprint 4's
empirical finding was that SigLIP2 confidently misclassified the one real photograph tested
as `clip_art` (0.60 softmax, margin 0.36 -- no margin threshold could have caught it), which
makes it unsafe as the primary routing signal for exactly the photograph vs. clip_art split
this module gates.  Pixel statistics are model-free, deterministic, and directly measurable
against the project's existing fixture corpus.  The domain-router result can still be
attached as descriptive metadata by the caller, but it does not influence the routing
outcome here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL.Image import Image

# ---------------------------------------------------------------------------
# Tunable thresholds -- defined as module-level constants so a benchmark sweep
# can import and override them without changing this file, and so the unit
# tests can verify they are > 0 and plausible.
# ---------------------------------------------------------------------------

#: Unique RGB colors per 1,000 pixels at or above which an image is routed as a
#: photograph.  Derived from Sprint 15: lowest real-photo score was 65.02 (an airplane
#: photo with a large uniform sky region); highest synthetic-art score was < 3; the
#: threshold at 10 keeps a >= 6x gap to each boundary.
COLOR_DENSITY_PHOTO_THRESHOLD: float = 10.0

#: Mean HSV saturation (0-255 scale) below which a low-color-count image is considered
#: potentially greyscale and advances to the luminance-stddev rescue check.
#: A pure greyscale image scores 0.0; the flat clip-art scene (diverse hues) scores 87.2;
#: threshold of 15 leaves a comfortable margin below the lowest expected coloured-art score.
MEAN_SATURATION_GRAYSCALE_THRESHOLD: float = 15.0

#: Standard deviation of the V channel in HSV space (0-255 scale) at or above which a
#: low-color-count, low-saturation image is still routed as a photograph (greyscale rescue).
#: A completely uniform grey fill scores ~0; a real greyscale photograph with tonal gradation
#: typically scores 40-80+.  Threshold of 20 sits well below the observed greyscale-photo
#: range while comfortably exceeding flat near-uniform fills.
LUMINANCE_STDDEV_PHOTO_THRESHOLD: float = 20.0

Backend = Literal["florence2", "grounding_dino"]


@dataclass(frozen=True)
class RoutingDecision:
    """Result of `choose_count_backend`: the selected back-end and the statistics that drove it.

    `backend` is either ``"grounding_dino"`` (real photograph, incl. greyscale) or
    ``"florence2"`` (flat synthetic / clip-art).  The three statistics fields hold the
    raw computed values (``luminance_stddev`` and ``mean_saturation`` are 0.0 when the
    cascade short-circuits before computing them).  `reason` is a human-readable
    explanation of which branch fired, surfaced in ``_routing`` diagnostic metadata
    attached to ``count_objects`` results.
    """

    backend: Backend
    unique_colors_per_1k_px: float
    mean_saturation: float
    luminance_stddev: float
    reason: str


def count_unique_colors_per_1k_px(image: Image) -> float:
    """Count distinct (R, G, B) tuples in `image`, normalized to per 1,000 pixels.

    The image is converted to RGB before sampling, so palette-mode and RGBA inputs are
    handled cleanly.  A 1x1 image returns 1000.0 (one unique color, one pixel:
    1 / 1 * 1000 = 1000).  Flat art with one fill color returns (1 / total_pixels) * 1000,
    which rounds to near-zero for any reasonably sized image.

    The normalization lets scores be compared across images of different resolutions without
    the raw unique-color count growing simply because the image has more pixels.
    """
    with image.convert("RGB") as rgb:
        arr = np.asarray(rgb, dtype=np.uint8)  # shape: (H, W, 3)
    h, w = arr.shape[:2]
    total_pixels = int(h) * int(w)
    # Pack three uint8 channels into one uint32 for fast unique-counting.
    flat = arr.reshape(-1, 3)
    packed = flat[:, 0].astype(np.uint32) << 16 | flat[:, 1].astype(np.uint32) << 8 | flat[:, 2].astype(np.uint32)
    n_unique = int(np.unique(packed).size)
    return float(n_unique / total_pixels * 1_000)


def mean_saturation(image: Image) -> float:
    """Mean HSV saturation across all pixels, on a 0-255 scale.

    A truly greyscale image (all pixels have R=G=B) has saturation exactly 0.
    A flat-colour synthetic image with diverse hue fills has saturation proportional
    to the fill colours' intensity -- measured at 87.2/255 for a 13-fill clip-art scene.
    A colour photograph has intermediate saturation driven by scene content.

    Used in the routing cascade as a gate before the luminance-variance rescue check:
    only images with mean saturation below `MEAN_SATURATION_GRAYSCALE_THRESHOLD` are
    considered potentially greyscale and worth checking for tonal variance.

    The saturation S in HSV is (max(R,G,B) - min(R,G,B)) / max(R,G,B) per pixel,
    which is 0 for any pixel where all channels are equal (grey), and 1 for fully
    saturated primary/secondary colours.  Multiplying by 255 puts it on the same
    integer scale as the threshold constant.

    Images where max(R,G,B)=0 (pure black pixels) have their saturation clamped to 0
    to avoid a divide-by-zero -- consistent with the conventional HSV definition where
    black has undefined but conventionally 0 saturation.
    """
    with image.convert("RGB") as rgb:
        arr = np.asarray(rgb, dtype=np.float32) / 255.0  # shape: (H, W, 3), 0-1
    v = arr.max(axis=2)  # V channel: max(R,G,B), shape (H,W)
    chroma = v - arr.min(axis=2)  # max - min, shape (H,W)
    # S = chroma / V where V > 0, else 0.  np.where evaluates both branches before
    # selecting, so the division produces a NaN for black pixels (V=0) before the
    # mask suppresses it; errstate silences that transient invalid-value warning.
    with np.errstate(invalid="ignore"):
        s = np.where(v > 0, chroma / v, 0.0)  # shape (H,W), 0-1
    return float(s.mean() * 255.0)


def luminance_stddev(image: Image) -> float:
    """Standard deviation of the V (value/brightness) channel in HSV space, on a 0-255 scale.

    Used as the final greyscale-photo rescue: a real photograph in greyscale carries tonal
    variation from camera noise and natural lighting.  Flat near-uniform grey synthetic art
    has near-zero luminance variance.

    The image is converted to RGB then the V channel is computed as max(R,G,B) per pixel;
    no external HSV conversion library is needed.  Returning a value on the 0-255 scale
    keeps `LUMINANCE_STDDEV_PHOTO_THRESHOLD` in an intuitive integer range.
    """
    with image.convert("RGB") as rgb:
        arr = np.asarray(rgb, dtype=np.float32) / 255.0  # shape: (H, W, 3), 0-1
    v_channel = arr.max(axis=2)  # shape: (H, W), 0-1
    return float(np.std(v_channel) * 255.0)


def choose_count_backend(image: Image) -> RoutingDecision:
    """Route a `count_objects` call to Florence-2 or Grounding DINO using pixel statistics.

    The decision cascade is:

    1. Compute unique RGB colors per 1,000 pixels.  At or above
       `COLOR_DENSITY_PHOTO_THRESHOLD` (10.0) -> photograph -> Grounding DINO.

    2. Compute mean HSV saturation.  At or above
       `MEAN_SATURATION_GRAYSCALE_THRESHOLD` (15.0) -> saturated fills -> synthetic art
       -> Florence-2.  Below threshold -> potentially greyscale -> continue.

    3. Compute V-channel luminance standard deviation.  At or above
       `LUMINANCE_STDDEV_PHOTO_THRESHOLD` (20.0) -> greyscale photograph -> Grounding DINO.
       Below -> flat near-uniform grey synthetic art -> Florence-2.

    The three-step cascade correctly handles the key edge cases:
    - Flat multi-colour clip art: caught by high saturation at step 2 (not rescued).
    - Greyscale photograph (Sprint 15 coco_bird_3 case): passes saturation gate, then
      rescued by luminance variance at step 3.
    - All-white blank canvas: very low saturation (0), very low luminance stddev (0)
      -> Florence-2 (correct: no real objects to count).

    The function is pure: same image always yields the same result, no global state, no
    network calls, no model loading.  It can be called from tests without any server
    infrastructure.

    When `count_objects` receives a multi-page PDF, the caller uses the first page's
    statistics for the whole batch -- consistent with the assumption that a document type
    is uniform across pages.
    """
    colors = count_unique_colors_per_1k_px(image)

    if colors >= COLOR_DENSITY_PHOTO_THRESHOLD:
        return RoutingDecision(
            backend="grounding_dino",
            unique_colors_per_1k_px=colors,
            mean_saturation=0.0,  # not computed; short-circuit on color density
            luminance_stddev=0.0,
            reason=f"color density {colors:.2f} >= threshold {COLOR_DENSITY_PHOTO_THRESHOLD} -> photograph",
        )

    sat = mean_saturation(image)

    if sat >= MEAN_SATURATION_GRAYSCALE_THRESHOLD:
        # Saturated solid fills -> synthetic art, even if luminance variance is high
        # (diverse hues span a wide brightness range, but that is structure, not tonal noise).
        return RoutingDecision(
            backend="florence2",
            unique_colors_per_1k_px=colors,
            mean_saturation=sat,
            luminance_stddev=0.0,  # not computed; saturation gate fires first
            reason=(
                f"color density {colors:.2f} < threshold {COLOR_DENSITY_PHOTO_THRESHOLD} and "
                f"mean saturation {sat:.1f} >= threshold {MEAN_SATURATION_GRAYSCALE_THRESHOLD} "
                "-> saturated flat fills -> synthetic / clip-art"
            ),
        )

    lum_std = luminance_stddev(image)

    if lum_std >= LUMINANCE_STDDEV_PHOTO_THRESHOLD:
        return RoutingDecision(
            backend="grounding_dino",
            unique_colors_per_1k_px=colors,
            mean_saturation=sat,
            luminance_stddev=lum_std,
            reason=(
                f"color density {colors:.2f} < threshold {COLOR_DENSITY_PHOTO_THRESHOLD}, "
                f"mean saturation {sat:.1f} < threshold {MEAN_SATURATION_GRAYSCALE_THRESHOLD} (greyscale gate), "
                f"luminance stddev {lum_std:.1f} >= threshold {LUMINANCE_STDDEV_PHOTO_THRESHOLD} "
                "-> greyscale photograph"
            ),
        )

    return RoutingDecision(
        backend="florence2",
        unique_colors_per_1k_px=colors,
        mean_saturation=sat,
        luminance_stddev=lum_std,
        reason=(
            f"color density {colors:.2f} < threshold {COLOR_DENSITY_PHOTO_THRESHOLD}, "
            f"mean saturation {sat:.1f} < threshold {MEAN_SATURATION_GRAYSCALE_THRESHOLD}, "
            f"luminance stddev {lum_std:.1f} < threshold {LUMINANCE_STDDEV_PHOTO_THRESHOLD} "
            "-> flat near-uniform synthetic / clip-art"
        ),
    )


__all__ = [
    "COLOR_DENSITY_PHOTO_THRESHOLD",
    "LUMINANCE_STDDEV_PHOTO_THRESHOLD",
    "MEAN_SATURATION_GRAYSCALE_THRESHOLD",
    "Backend",
    "RoutingDecision",
    "choose_count_backend",
    "count_unique_colors_per_1k_px",
    "luminance_stddev",
    "mean_saturation",
]
