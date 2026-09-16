import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, "src")
from fusion_vision_mcp.florence2 import Florence2


def main():
    img_path = "tests/defect_test6.jpg"
    if not Path(img_path).exists():
        return
    img = Image.open(img_path)

    florence = Florence2()
    print("\n=== Florence-2 Dense Region Caption ===")
    results = florence.dense_region_caption([img])
    for r in results:
        print(f"Labels found: {', '.join(set(r.get('labels', [])))}")


if __name__ == "__main__":
    main()
