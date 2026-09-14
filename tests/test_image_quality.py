import pytest
from PIL import Image

try:
    from fusion_vision_mcp.image_quality import HAS_ONNX, ImageQuality
except ImportError:
    HAS_ONNX = False


@pytest.mark.skipif(not HAS_ONNX, reason="onnxruntime or huggingface_hub not installed")
def test_image_quality_musiq() -> None:
    iqa = ImageQuality()

    # Create a dummy image
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))

    result = iqa.score(img)

    assert "technical_quality" in result
    assert "model" in result
    assert result["model"] == "86Cao/IQA-ONNX-Models/musiq_model"
    assert isinstance(result["technical_quality"], float)
