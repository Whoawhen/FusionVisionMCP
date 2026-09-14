import json
from dataclasses import dataclass
from typing import Any, Literal

import requests

ProviderMode = Literal["none", "ollama"]

@dataclass
class ReasonerOutputClaim:
    claim: str
    confidence: float
    evidence: list[str]

@dataclass
class ReasonerOutput:
    judgment: str
    confidence: float
    claims: list[ReasonerOutputClaim]
    measurements: dict[str, Any]
    
    def as_dict(self) -> dict[str, Any]:
        return {
            "judgment": self.judgment,
            "confidence": self.confidence,
            "claims": [
                {
                    "claim": c.claim,
                    "confidence": c.confidence,
                    "evidence": c.evidence
                }
                for c in self.claims
            ],
            "measurements": self.measurements,
        }

class VisionReasoner:
    def __init__(self, provider: ProviderMode = "none", model: str = "llama3"):
        self.provider = provider
        self.model = model

    def analyze(
        self,
        question: str | None = None,
        observations: list[dict[str, Any]] | None = None,
        measurements: dict[str, Any] | None = None,
        detections: list[dict[str, Any]] | None = None,
        image: Any = None,
    ) -> ReasonerOutput | None:
        """Analyze structured evidence and optionally return a reasoner judgment."""
        
        if self.provider == "none":
            return None
            
        # The invariant: we must preserve the direct input measurements
        # so they can never be silently overridden by the LLM's output.
        preserved_measurements = measurements or {}
            
        if self.provider == "ollama":
            output = self._analyze_ollama(question, observations, preserved_measurements, detections)
            # Re-attach the exact preserved measurements, replacing whatever the LLM might have tried to fabricate
            output.measurements = preserved_measurements
            return output
            
        raise ValueError(f"Unsupported provider: {self.provider}")

    def _analyze_ollama(
        self,
        question: str | None,
        observations: list[dict[str, Any]] | None,
        measurements: dict[str, Any],
        detections: list[dict[str, Any]] | None,
    ) -> ReasonerOutput:
        
        payload = {
            "question": question,
            "observations": observations or [],
            "measurements": measurements,
            "detections": detections or [],
        }
        
        prompt = (
            "Analyze the following structured visual evidence and provide a judgment. "
            "Output ONLY valid JSON matching this schema exactly:\n"
            '{"judgment": "summary of your conclusion", "confidence": 0.9, '
            '"claims": [{"claim": "specific claim", "confidence": 0.9, "evidence": ["source"]}]}\n\n'
            f"Evidence:\n{json.dumps(payload, indent=2)}"
        )
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            result = json.loads(data.get("response", "{}"))
        except (requests.RequestException, json.JSONDecodeError):
            # If Ollama fails or returns bad JSON, fallback gracefully rather than crashing.
            # Sprint 11: "don't chase reasoning quality yet", but we must enforce the contract.
            result = {
                "judgment": "Reasoning backend failed to produce a valid judgment.",
                "confidence": 0.0,
                "claims": []
            }
            
        claims = [
            ReasonerOutputClaim(
                claim=c.get("claim", ""),
                confidence=float(c.get("confidence", 0.0)),
                evidence=c.get("evidence", [])
            )
            for c in result.get("claims", [])
        ]
        
        return ReasonerOutput(
            judgment=result.get("judgment", ""),
            confidence=float(result.get("confidence", 0.0)),
            claims=claims,
            measurements=measurements
        )

