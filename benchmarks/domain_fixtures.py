"""Fixtures for domain-router testing, covering all five `domain_router.py` groups.

Kept separate from `fixtures.py`'s `all_fixtures()`, which is the *counting* benchmark's
fixture list -- reusing it here would mix an unrelated concern into it. This module reuses
`fixtures.py`'s existing photograph (`flower()`) and clip-art-style shapes
(`ring()`/`distractors()`), and adds new minimal fixtures for the three domains nothing
here previously covered: painting, document, screenshot.

`true_group` is honest ground truth, not a label reverse-engineered to match whatever the
model outputs -- an earlier version of this sweep set the one real photograph's expected
label to "clip_art" because that's what SigLIP2 predicted for it, which measures nothing.
That mistake is why `photo_flower`'s recorded truth is "photograph" even though, as
documented below and in `CHANGELOG.md`, the router gets it wrong.

Painting and screenshot are documented, measured negative results, not fixtures still being
tuned to pass. Six materially different constructions were tried for painting (posterize,
saturation+blur+grain, colour quantization+blur, mode-filter smear, heavy blur alone, a
wholly synthetic gradient/blob landscape, and hundreds of angled brush-dab strokes) -- not
one ever placed "oil painting" or "watercolor painting" even in the top 3 predicted labels.
Two materially different UI mockups were tried for screenshot (a native-app settings panel,
a browser chrome with tab/address bar and page content) -- both were read as "document or
scanned page" instead of "computer screenshot". This is the same posture as the flower/petal
counting negative result elsewhere in this project: a real, convergent finding worth keeping
rather than a fixture worth discarding because it "failed". `painting_stylized` and
`screenshot_ui` below are the most structurally representative of the attempts made, kept for
reproducibility; a real (not procedurally generated) painting or screen capture might score
differently, and that gap is exactly what's recorded as untested in `CHANGELOG.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from fixtures import CANVAS, distractors, flower, ring
from PIL import Image, ImageDraw, ImageFont


@dataclass
class DomainFixture:
    """One domain-router test case: an image and the domain group it actually belongs to."""

    name: str
    image: Image.Image
    #: One of "photograph", "clip_art", "painting", "document", "screenshot" -- ground
    #: truth, established by what the fixture actually is, not by what the model predicts.
    true_group: str
    notes: str = ""


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A real scalable font where available; PIL's bitmap default otherwise.

    Falls back rather than failing outright -- this only affects how convincing the
    document/screenshot fixtures look, not whether they can be generated at all.
    """
    for candidate in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def document_page() -> Image.Image:
    """A page of printed paragraph text -- correctly classified (0.86 confidence)."""
    img = Image.new("RGB", (CANVAS, CANVAS), (250, 249, 245))
    draw = ImageDraw.Draw(img)
    header, body = _font(26), _font(15)
    draw.text((40, 36), "Quarterly Report", fill=(20, 20, 20), font=header)
    draw.line([(40, 78), (CANVAS - 40, 78)], fill=(120, 120, 120), width=1)

    words = [
        "revenue",
        "increased",
        "across",
        "every",
        "region",
        "this",
        "quarter",
        "as",
        "demand",
        "for",
        "the",
        "product",
        "line",
        "continued",
        "to",
        "grow",
        "steadily",
        "and",
        "operating",
        "costs",
        "remained",
        "within",
        "the",
        "range",
        "forecast",
        "at",
        "the",
        "start",
        "of",
        "the",
    ]
    rng = np.random.default_rng(11)
    y = 110
    while y < CANVAS - 60:
        line = " ".join(rng.choice(words, size=int(rng.integers(8, 13))))
        draw.text((40, y), line, fill=(35, 35, 35), font=body)
        y += 26
    draw.text((CANVAS - 70, CANVAS - 34), "- 1 -", fill=(120, 120, 120), font=_font(13))
    return img


def screenshot_ui() -> Image.Image:
    """A browser-chrome mockup -- tab bar, address bar, page content with cards.

    Documented negative result: classified as "document or scanned page" (0.56), not
    "computer screenshot", despite browser chrome being a much stronger signal than the
    native-app settings panel tried first. See the module docstring.
    """
    img = Image.new("RGB", (CANVAS, CANVAS), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, CANVAS, 32], fill=(223, 225, 230))
    draw.rectangle([8, 4, 180, 30], fill=(255, 255, 255), outline=(200, 200, 205))
    draw.text((16, 10), "Example Site", fill=(60, 60, 65), font=_font(13))
    draw.rectangle([0, 32, CANVAS, 64], fill=(245, 245, 247))
    draw.ellipse([12, 42, 24, 54], outline=(150, 150, 155), width=2)
    draw.rectangle([40, 40, CANVAS - 16, 56], fill=(255, 255, 255), outline=(210, 210, 215))
    draw.text((50, 44), "https://www.example.com/dashboard", fill=(90, 90, 95), font=_font(13))
    draw.rectangle([0, 64, CANVAS, 104], fill=(255, 255, 255), outline=(225, 225, 228))
    for i, label in enumerate(["Home", "Products", "Pricing", "About"]):
        draw.text((30 + i * 90, 78), label, fill=(50, 50, 55), font=_font(14))
    draw.text((30, 130), "Dashboard", fill=(20, 20, 20), font=_font(24))
    for i in range(3):
        x = 30 + i * 165
        draw.rectangle([x, 175, x + 145, 290], fill=(248, 248, 250), outline=(220, 220, 224))
        draw.rectangle([x + 16, 191, x + 70, 215], fill=(70, 130, 240))
        draw.text((x + 16, 228), f"Metric {i + 1}", fill=(40, 40, 45), font=_font(14))
        draw.text((x + 16, 250), "128", fill=(20, 20, 20), font=_font(20))
    return img


def painting_stylized() -> Image.Image:
    """Hundreds of angled, partially-transparent brush-dab strokes over a base wash.

    Documented negative result: the most painting-technique-representative of six attempts
    (see the module docstring), and the only one that at least knocked "flat vector clip
    art" down to a genuine toss-up with "a photorealistic image" (0.37 vs 0.36) -- but
    neither "oil painting" nor "watercolor painting" was ever the top prediction.
    """
    rng = np.random.default_rng(13)
    base = Image.new("RGBA", (CANVAS, CANVAS), (210, 220, 235, 255))
    palette = [
        (90, 130, 80),
        (60, 100, 60),
        (255, 230, 150),
        (150, 180, 220),
        (200, 150, 90),
        (70, 90, 140),
        (230, 240, 250),
    ]
    overlay = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(2200):
        cx, cy = rng.uniform(0, CANVAS), rng.uniform(0, CANVAS)
        length, angle = rng.uniform(10, 28), rng.uniform(0, math.pi)
        dx, dy = math.cos(angle) * length / 2, math.sin(angle) * length / 2
        width = int(rng.uniform(3, 7))
        colour = palette[rng.integers(0, len(palette))]
        alpha = int(rng.uniform(140, 220))
        draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=(*colour, alpha), width=width)
    return Image.alpha_composite(base, overlay).convert("RGB")


def domain_fixtures() -> list[DomainFixture]:
    """One or more representative cases per `domain_router.py` group, honest ground truth."""
    return [
        DomainFixture("photo_flower", flower(384), "photograph", "the real tests/sample.jpg photo"),
        DomainFixture("clipart_ring8", ring(8, -0.15), "clip_art", "flat-coloured discs, no texture"),
        DomainFixture("clipart_distractors", distractors(), "clip_art", "flat discs and squares"),
        DomainFixture("painting_stylized", painting_stylized(), "painting", "brush-dab strokes; documented miss"),
        DomainFixture("document_page", document_page(), "document", "rendered paragraph text on a page"),
        DomainFixture("screenshot_ui", screenshot_ui(), "screenshot", "browser chrome + content; documented miss"),
    ]
