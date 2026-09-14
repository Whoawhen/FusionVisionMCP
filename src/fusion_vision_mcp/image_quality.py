from typing import Any

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort  # type: ignore
    from huggingface_hub import hf_hub_download

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class ImageQuality:
    """Technical Image Quality Assessment (IQA) using MUSIQ via ONNX."""

    def __init__(self, model_id: str = "86Cao/IQA-ONNX-Models"):
        if not HAS_ONNX:
            raise ImportError(
                "Technical IQA requires `onnxruntime` and `huggingface_hub`. "
                "Install them with: pip install onnxruntime huggingface-hub"
            )
        self.model_id = model_id
        self._session: Any = None
        self._input_name: str = ""

    def _load_model(self) -> None:
        if self._session is not None:
            return

        model_path = hf_hub_download(repo_id=self.model_id, filename="musiq_model.onnx")
        # Ensure external data is downloaded
        hf_hub_download(repo_id=self.model_id, filename="musiq_model.onnx.data")

        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        assert self._session is not None
        self._input_name = self._session.get_inputs()[0].name

    def _preprocess(self, image: Image.Image, size: int = 224) -> np.ndarray:
        img = image.convert("RGB")
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        img_data = np.array(img).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_data = (img_data - mean) / std

        img_data = np.transpose(img_data, (2, 0, 1))
        return np.expand_dims(img_data, axis=0).astype(np.float32)

    def score(self, image: Image.Image) -> dict[str, Any]:
        """
        Evaluate the technical quality of the image.
        Returns a score typically between 0 and 100 (higher is better).
        """
        self._load_model()

        input_data = self._preprocess(image)
        output = self._session.run(None, {self._input_name: input_data})

        score_val = float(output[0][0][0])

        return {
            "technical_quality": score_val,
            "model": f"{self.model_id}/musiq_model",
        }
