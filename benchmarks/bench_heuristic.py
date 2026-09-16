import sys

from PIL import Image

sys.path.insert(0, "src")
from fusion_vision_mcp.grounding_dino import GroundingDino


def test_anomaly():
    img = Image.open("tests/defect_test6.jpg")
    dino = GroundingDino()

    parts = ["person", "arm", "leg", "hand", "head"]
    counts = {}
    for p in parts:
        res = dino.detect_objects([img], p, adaptive_threshold=True)[0]
        counts[p] = res["count"]
        print(f"{p}: {counts[p]}")

    persons = max(1, counts.get("person", 1))

    anomalies = []
    if counts.get("arm", 0) > persons * 2 + 1:  # +1 for margin of error/occlusion
        anomalies.append(f"Excessive arms detected: {counts['arm']} arms for {persons} people.")
    if counts.get("leg", 0) > persons * 2 + 1:
        anomalies.append(f"Excessive legs detected: {counts['leg']} legs for {persons} people.")
    if counts.get("head", 0) > persons + 1:
        anomalies.append(f"Excessive heads detected: {counts['head']} heads for {persons} people.")

    print("Anomalies:", anomalies)


if __name__ == "__main__":
    test_anomaly()
