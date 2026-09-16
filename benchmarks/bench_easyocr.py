import time

import easyocr
import torch


def test_easyocr():
    print("Loading EasyOCR reader...")
    # Use GPU if available, else CPU
    has_gpu = torch.cuda.is_available()
    reader = easyocr.Reader(["en"], gpu=has_gpu)

    fixtures = ["tests/defect_test6.jpg", "tests/layout_two_column_ruled.png"]

    for path in fixtures:
        print(f"\n--- Testing {path} ---")
        t0 = time.time()
        result = reader.readtext(path)
        t1 = time.time()

        print(f"Time: {t1 - t0:.2f}s")
        print("Result:")
        for bbox, text, prob in result:
            print(f"  [{prob:.2f}] {text}")


if __name__ == "__main__":
    test_easyocr()
