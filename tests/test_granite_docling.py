from pathlib import Path

from PIL import Image

from fusion_vision_mcp.granite_docling import GraniteDoclingOCR

TEST_DIR = Path(__file__).parent


def test_granite_docling_accuracy():
    img_path = TEST_DIR / "layout_small_text_paragraph.png"
    img = Image.open(img_path)

    ocr_model = GraniteDoclingOCR()
    result = ocr_model.ocr(img)
    print("\n[Granite Docling Output]:", result)

    # Let's also print expected
    expected = (
        "This paragraph is rendered in a small eleven point font against a nine hundred pixel wide canvas to test whether verbatim transcription holds up at reduced text scale, "
        "independent of the layout and column-splitting cases already covered. The exact sentence is reproduced here as ground truth so any dropped or substituted word is directly "
        "checkable against this source string without needing a separate human read of the rendered image."
    )
    print("\n[Expected]:", expected)

    # Calculate word accuracy
    def extract_words(s):
        import string

        s = s.replace("-", " ")
        return [w.strip(string.punctuation).lower() for w in s.split() if w.strip(string.punctuation)]

    expected_words = extract_words(expected)
    actual_words = extract_words(result)

    import difflib
    matcher = difflib.SequenceMatcher(None, expected_words, actual_words)
    matches = sum(triple.size for triple in matcher.get_matching_blocks())
    accuracy = matches / len(expected_words)
    print(f"\nWord Accuracy: {accuracy:.2%} ({matches}/{len(expected_words)})")

    # It should be reasonably high for Granite Docling!
    assert accuracy > 0.90, f"Granite Docling word accuracy too low: {accuracy:.2%}"


if __name__ == "__main__":
    test_granite_docling_accuracy()
