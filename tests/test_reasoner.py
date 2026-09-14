from typing import Any

import pytest
from pytest import MonkeyPatch

from fusion_vision_mcp.reasoner import ReasonerOutput, VisionReasoner


def test_reasoner_none_provider_returns_none() -> None:
    reasoner = VisionReasoner(provider="none")
    result = reasoner.analyze(question="What is this?")
    assert result is None

def test_reasoner_unsupported_provider_raises() -> None:
    reasoner = VisionReasoner(provider="unsupported_provider") # type: ignore
    with pytest.raises(ValueError, match="Unsupported provider"):
        reasoner.analyze(question="test")

class MockResponse:
    def __init__(self, json_data: Any, status_code: int = 200) -> None:
        self._json_data = json_data
        self.status_code = status_code
        
    def json(self) -> Any:
        return self._json_data
        
    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("HTTP Error")

def test_ollama_preserves_direct_measurements(monkeypatch: MonkeyPatch) -> None:
    """
    Direct measurements must never be silently overridden - test that explicitly.
    If Ollama returns a JSON payload that tries to override measurements,
    the Reasoner must discard that and attach the original raw measurements.
    """
    reasoner = VisionReasoner(provider="ollama")
    
    # We mock requests.post to simulate a rogue LLM trying to inject its own 'measurements'
    def mock_post(*args: Any, **kwargs: Any) -> MockResponse:
        rogue_json = '{"judgment": "I think there are 3.", "confidence": 0.5, "claims": [], "measurements": {"count": 3}}'
        return MockResponse({"response": rogue_json})
        
    monkeypatch.setattr("requests.post", mock_post)
    
    original_measurements = {"count": 1, "spatial_overlap": 0.5}
    
    result = reasoner.analyze(
        question="How many?",
        measurements=original_measurements
    )
    
    assert isinstance(result, ReasonerOutput)
    # The result MUST have the original measurements, not {"count": 3}
    assert result.measurements == original_measurements
    assert result.measurements["count"] == 1
    assert result.judgment == "I think there are 3."

def test_ollama_graceful_fallback_on_bad_json(monkeypatch: MonkeyPatch) -> None:
    reasoner = VisionReasoner(provider="ollama")
    
    def mock_post(*args: Any, **kwargs: Any) -> MockResponse:
        return MockResponse({"response": "This is not json."})
        
    monkeypatch.setattr("requests.post", mock_post)
    
    result = reasoner.analyze(question="Test?")
    assert result is not None
    assert result.confidence == 0.0
    assert result.claims == []
    assert "failed to produce" in result.judgment.lower()
