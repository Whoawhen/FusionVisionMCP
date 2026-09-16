import string
from pathlib import Path

from fusion_vision_mcp.granite_docling import GraniteDoclingOCR
from PIL import Image

TEST_DIR = Path("tests")


def extract_words(s):
    s = s.replace("-", " ")
    return [w.strip(string.punctuation).lower() for w in s.split() if w.strip(string.punctuation)]


expected = (
    "This paragraph is rendered in a small eleven point font against a nine hundred pixel wide canvas to test whether verbatim transcription holds up at reduced text scale, "
    "independent of the layout and column-splitting cases already covered. The exact sentence is reproduced here as ground truth so any dropped or substituted word is directly "
    "checkable against this source string without needing a separate human read of the rendered image."
)

img_path = TEST_DIR / "layout_small_text_paragraph.png"
img = Image.open(img_path).convert("RGB")

print("\n--- Granite Docling ---")
granite_model = GraniteDoclingOCR()
granite_text = granite_model.ocr(img)
print("Text:", granite_text)

expected_words = extract_words(expected)
granite_words = extract_words(granite_text)
gr_matches = sum(1 for a, b in zip(expected_words, granite_words) if a == b)

print(f"\nGranite Docling Accuracy: {gr_matches / len(expected_words):.2%} ({gr_matches}/{len(expected_words)})")
