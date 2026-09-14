import torch
from PIL import Image
from fusion_vision_mcp.granite_docling import GraniteDoclingOCR, DEFAULT_GRANITE_DOCLING_MODEL

def test_granite():
    # Hardcode cpu for the test to bypass resolve_device string parsing issues
    device = torch.device("cpu")
    print(f"Loading Granite-Docling on {device}...")
    ocr_model = GraniteDoclingOCR(DEFAULT_GRANITE_DOCLING_MODEL, device)
    
    fixtures = [
        "tests/defect_test6.jpg",
        "tests/layout_two_column_ruled.png"
    ]
    
    for path in fixtures:
        print(f"\n--- Testing {path} ---")
        try:
            img = Image.open(path)
            # Granite processes in markdown
            text = ocr_model.ocr(img, max_new_tokens=20)
            print("Result:")
            print(text)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_granite()
