#  fixtures.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
"""Counting fixtures with known ground truth.

Synthetic cases are drawn programmatically, so their instance count is exact by
construction rather than by anyone's judgement -- which is what makes a regression in
the negative controls detectable. Real photographs are included too, but any fixture
whose truth is not established by construction carries ``truth=None`` and is reported
rather than scored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_IMAGE = REPO_ROOT / "tests" / "sample.jpg"

CANVAS = 512
PETAL_FILL = (235, 120, 150)
PETAL_EDGE = (170, 70, 100)

Kind = Literal["positive", "negative", "prompt_stability"]


@dataclass
class Fixture:
    """One benchmark image, its prompt, and what the answer should be."""

    name: str
    image: Image.Image
    prompt: str
    #: Exact count where it is known by construction; None where it is not established.
    truth: int | None
    kind: Kind
    notes: str = ""
    #: Extra prompts that must yield the same answer as `prompt`.
    equivalent_prompts: list[str] = field(default_factory=list)


def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (CANVAS, CANVAS), "white")
    return img, ImageDraw.Draw(img)


def _disc(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, fill=PETAL_FILL) -> None:
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=PETAL_EDGE, width=3)


def ring(n: int, overlap: float, radius: int = 70) -> Image.Image:
    """`n` discs evenly spaced on a circle, overlapping by `overlap` of their width.

    `overlap` is the fraction of a disc's *diameter* that its neighbour intrudes on, so
    0.0 is just-touching-free, 0.5 means neighbouring centres sit one radius apart.
    """
    img, draw = _canvas()
    spacing = 2 * radius * (1 - overlap)
    ring_radius = spacing / (2 * math.sin(math.pi / n)) if n > 1 else 0.0
    for i in range(n):
        angle = 2 * math.pi * i / n
        _disc(draw, CANVAS / 2 + ring_radius * math.cos(angle), CANVAS / 2 + ring_radius * math.sin(angle), radius)
    return img


def row(n: int, overlap: float, radius: int = 45) -> Image.Image:
    img, draw = _canvas()
    spacing = 2 * radius * (1 - overlap)
    start = CANVAS / 2 - spacing * (n - 1) / 2
    for i in range(n):
        _disc(draw, start + i * spacing, CANVAS / 2, radius)
    return img


def unequal_ring(n: int = 6) -> Image.Image:
    """Repeated instances at differing sizes -- the case that undercounts, not over."""
    img, draw = _canvas()
    for i in range(n):
        angle = 2 * math.pi * i / n
        radius = 40 + 30 * (i % 3)
        _disc(draw, CANVAS / 2 + 140 * math.cos(angle), CANVAS / 2 + 140 * math.sin(angle), radius)
    return img


def clipped_row(n: int = 4, radius: int = 60) -> Image.Image:
    """Instances running off the frame edge; the count is still n visible."""
    img, draw = _canvas()
    for i in range(n):
        _disc(draw, -20 + i * (2 * radius + 10), CANVAS / 2, radius)
    return img


def dense_grid(rows: int = 5, cols: int = 6, radius: int = 22) -> Image.Image:
    img, draw = _canvas()
    for r in range(rows):
        for c in range(cols):
            _disc(draw, 70 + c * 75, 90 + r * 75, radius)
    return img


# --- negative controls: each of these is exactly ONE object -------------------------


def solid_disc() -> Image.Image:
    img, draw = _canvas()
    _disc(draw, CANVAS / 2, CANVAS / 2, 160)
    return img


def gradient_disc() -> Image.Image:
    """One object whose interior varies smoothly -- lighting must not become instances."""
    ys, xs = np.mgrid[0:CANVAS, 0:CANVAS]
    inside = ((xs - CANVAS / 2) ** 2 + (ys - CANVAS / 2) ** 2) <= 160**2
    shade = np.clip(120 + (xs + ys) / (2 * CANVAS) * 135, 0, 255)
    arr = np.full((CANVAS, CANVAS, 3), 255, dtype=np.uint8)
    for channel, base in enumerate((235, 120, 150)):
        arr[..., channel] = np.where(inside, np.clip(shade * base / 235, 0, 255), 255).astype(np.uint8)
    return Image.fromarray(arr)


def striped_ball() -> Image.Image:
    """One object with strong interior boundaries. The key interior-structure control."""
    img, draw = _canvas()
    _disc(draw, CANVAS / 2, CANVAS / 2, 160)
    mask = Image.new("L", (CANVAS, CANVAS), 0)
    ImageDraw.Draw(mask).ellipse([CANVAS / 2 - 160, CANVAS / 2 - 160, CANVAS / 2 + 160, CANVAS / 2 + 160], fill=255)
    stripes, sdraw = Image.new("RGB", (CANVAS, CANVAS), PETAL_FILL), None
    sdraw = ImageDraw.Draw(stripes)
    for i in range(-CANVAS, CANVAS, 56):
        sdraw.rectangle([i, 0, i + 28, CANVAS], fill=(60, 90, 190))
    img.paste(stripes, (0, 0), mask)
    draw.ellipse([CANVAS / 2 - 160, CANVAS / 2 - 160, CANVAS / 2 + 160, CANVAS / 2 + 160], outline=PETAL_EDGE, width=3)
    return img


def spotted_ball() -> Image.Image:
    """One object covered in high-contrast spots -- texture, not instances."""
    img, draw = _canvas()
    _disc(draw, CANVAS / 2, CANVAS / 2, 160)
    rng = np.random.default_rng(7)
    for _ in range(40):
        angle, dist = rng.uniform(0, 2 * math.pi), rng.uniform(0, 135)
        cx, cy = CANVAS / 2 + dist * math.cos(angle), CANVAS / 2 + dist * math.sin(angle)
        draw.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], fill=(50, 60, 130))
    return img


def multicolour_logo() -> Image.Image:
    """One object made of differently coloured parts -- the hardest colour-based control."""
    img, draw = _canvas()
    colours = [(220, 60, 60), (60, 160, 90), (60, 100, 200), (240, 190, 60)]
    for i, colour in enumerate(colours):
        start, end = i * 90, (i + 1) * 90
        draw.pieslice([CANVAS / 2 - 150, CANVAS / 2 - 150, CANVAS / 2 + 150, CANVAS / 2 + 150], start, end, fill=colour)
    return img


def noisy_rod() -> Image.Image:
    """The photographic-roughness control that broke an earlier silhouette metric."""
    rng = np.random.default_rng(0)
    img, draw = _canvas()
    for x in range(60, CANVAS - 60):
        top = CANVAS / 2 - 22 + rng.integers(-4, 5)
        draw.line([(x, top), (x, top + 44 + rng.integers(-4, 5))], fill=PETAL_FILL)
    return img.filter(ImageFilter.GaussianBlur(0.6))


def textured_natural() -> Image.Image:
    """A single blob with fine multi-scale texture, standing in for bark or fur."""
    rng = np.random.default_rng(3)
    ys, xs = np.mgrid[0:CANVAS, 0:CANVAS]
    inside = ((xs - CANVAS / 2) ** 2 + (ys - CANVAS / 2) ** 2) <= 160**2
    # scipy rather than PIL: PIL cannot blur a float-mode image, and the texture has to
    # be correlated across a few pixels to stand in for a real surface.
    noise = gaussian_filter(rng.normal(0, 34, (CANVAS, CANVAS)), sigma=2.0)
    arr = np.full((CANVAS, CANVAS, 3), 255, dtype=np.uint8)
    for channel, base in enumerate((150, 110, 70)):
        arr[..., channel] = np.where(inside, np.clip(base + noise, 0, 255), 255).astype(np.uint8)
    return Image.fromarray(arr)


def distractors() -> Image.Image:
    """Three target discs plus four square distractors that must not be counted."""
    img, draw = _canvas()
    for cx, cy in ((120, 130), (256, 300), (390, 140)):
        _disc(draw, cx, cy, 50)
    for cx, cy in ((120, 380), (256, 90), (390, 380), (200, 200)):
        draw.rectangle([cx - 40, cy - 40, cx + 40, cy + 40], fill=(90, 170, 220), outline=(40, 100, 150), width=3)
    return img


def flower(size: int = 384) -> Image.Image:
    with Image.open(SAMPLE_IMAGE) as im:
        return im.convert("RGB").resize((size, size), Image.LANCZOS)


def all_fixtures() -> list[Fixture]:
    """Every benchmark case, in a stable order."""
    petal_alts = ["petals", "a petal", "pink circle", "round pink shape"]
    return [
        # --- positives, truth exact by construction -------------------------------
        Fixture("ring8_separated", ring(8, -0.15), "petal", 8, "positive", "clear gaps between instances"),
        Fixture("ring8_touching", ring(8, 0.0), "petal", 8, "positive", "just touching"),
        Fixture("ring8_overlap40", ring(8, 0.40), "petal", 8, "positive", "40% of width overlapped"),
        Fixture("ring8_overlap67", ring(8, 0.67), "petal", 8, "positive", "the documented hard case"),
        Fixture("row2_touching", row(2, 0.0), "petal", 2, "positive"),
        Fixture("row5_overlap30", row(5, 0.30), "petal", 5, "positive", "non-radial arrangement"),
        Fixture("unequal_ring6", unequal_ring(6), "petal", 6, "positive", "differing instance sizes"),
        Fixture("clipped_row4", clipped_row(4), "petal", 4, "positive", "instances run off the frame"),
        Fixture("dense_grid30", dense_grid(), "petal", 30, "positive", "many small repeated objects"),
        # --- negative controls: every one of these is exactly ONE object ----------
        Fixture("neg_solid_disc", solid_disc(), "petal", 1, "negative"),
        Fixture("neg_gradient_disc", gradient_disc(), "petal", 1, "negative", "lighting gradient interior"),
        Fixture("neg_striped_ball", striped_ball(), "petal", 1, "negative", "strong interior edges"),
        Fixture("neg_spotted_ball", spotted_ball(), "petal", 1, "negative", "high-contrast spots"),
        Fixture("neg_multicolour_logo", multicolour_logo(), "shape", 1, "negative", "differently coloured parts"),
        Fixture("neg_noisy_rod", noisy_rod(), "petal", 1, "negative", "photographic edge roughness"),
        Fixture("neg_textured_natural", textured_natural(), "blob", 1, "negative", "fine multi-scale texture"),
        Fixture("neg_two_overlapping", row(2, 0.45), "petal", 2, "positive", "exactly two, heavily overlapped"),
        Fixture("neg_distractors", distractors(), "pink circle", 3, "negative", "4 squares must not count"),
        # --- real photograph: truth NOT established by construction ---------------
        Fixture(
            "flower_128",
            flower(128),
            "petal",
            None,
            "positive",
            "truth unestablished; ~10 in BRIEFING.md is Moondream's self-report, not a hand count",
        ),
        Fixture("flower_384", flower(384), "petal", None, "positive", "same image, upscaled", petal_alts),
        Fixture("flower_768", flower(768), "petal", None, "positive", "same image, upscaled"),
        # --- prompt stability: same image, equivalent wordings --------------------
        Fixture("prompt_ring8_sep", ring(8, -0.15), "petal", 8, "prompt_stability", "", petal_alts),
        Fixture("prompt_ring8_overlap40", ring(8, 0.40), "petal", 8, "prompt_stability", "", petal_alts),
    ]
