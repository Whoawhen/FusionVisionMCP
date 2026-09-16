import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, "src")
from fusion_vision_mcp.florence2 import Florence2
from fusion_vision_mcp.grounding_dino import GroundingDino
from fusion_vision_mcp.moondream import Moondream


def main():
    img_path = "tests/defect_test6.jpg"
    if not Path(img_path).exists():
        print(f"Error: {img_path} not found.")
        return

    img = Image.open(img_path)

    print("=== Moondream2 Observations ===")
    vqa = Moondream()
    prompts = [
        "Describe this image in detail.",
        "Does the person have any extra or missing body parts?",
        "How many fingers are visible on the hands?",
        "Are there any anatomical anomalies, AI generation artifacts, or structural defects in this image? Describe them.",
    ]
    for prompt in prompts:
        ans = vqa.query([img], prompt)
        print(f"Q: {prompt}\nA: {ans[0]}\n")

    print("\n=== Grounding DINO Detections ===")
    dino = GroundingDino()
    for obj in ["hand", "finger", "arm", "leg", "person"]:
        results = dino.detect_objects([img], obj, threshold=0.15, adaptive_threshold=True)
        print(f"Target '{obj}': found {results[0]['count']} instances.")

    print("\n=== Florence-2 Detections ===")
    florence = Florence2()
    for obj in ["hand", "finger", "arm", "leg", "person"]:
        results = florence.detect_objects([img], obj)
        print(f"Target '{obj}': found {results[0]['count']} instances.")


if __name__ == "__main__":
    main()
