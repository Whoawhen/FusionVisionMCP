import json
from dataclasses import dataclass
from typing import Any, Final, Literal

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

#: Ollama's default local endpoint. Overridable per instance so a non-default host
#: or port does not require editing this module.
DEFAULT_OLLAMA_URL: Final[str] = "http://localhost:11434/api/generate"

#: (connect, read) seconds. Generation is the slow half, hence the asymmetry.
DEFAULT_REASONER_TIMEOUT: Final[tuple[float, float]] = (5.0, 60.0)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """A confidence the model wrote as prose, or omitted, must not raise."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class VisionReasoner:
    def __init__(
        self,
        provider: ProviderMode = "none",
        model: str = "llama3",
        url: str = DEFAULT_OLLAMA_URL,
        timeout: tuple[float, float] = DEFAULT_REASONER_TIMEOUT,
    ):
        self.provider = provider
        self.model = model
        self.url = url
        self.timeout = timeout

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
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            result = json.loads(data.get("response", "{}"))
        except (requests.RequestException, json.JSONDecodeError, ValueError):
            # If Ollama fails or returns bad JSON, fallback gracefully rather than crashing.
            # Sprint 11: "don't chase reasoning quality yet", but we must enforce the contract.
            result = {}

        # An LLM asked for JSON can return well-formed JSON of the wrong *shape* -- a list,
        # a bare string, claims as strings rather than objects. That parses fine and then
        # raises on the first `.get`, so every field is shape-checked rather than trusted.
        if not isinstance(result, dict):
            result = {}

        raw_claims = result.get("claims", [])
        if not isinstance(raw_claims, list):
            raw_claims = []

        claims = []
        for c in raw_claims:
            if not isinstance(c, dict):
                continue
            evidence = c.get("evidence", [])
            if not isinstance(evidence, list):
                evidence = [str(evidence)]
            claims.append(
                ReasonerOutputClaim(
                    claim=str(c.get("claim", "")),
                    confidence=_coerce_float(c.get("confidence")),
                    evidence=[str(e) for e in evidence],
                )
            )

        judgment = result.get("judgment")
        if not isinstance(judgment, str):
            judgment = "Reasoning backend failed to produce a valid judgment." if not result else ""

        return ReasonerOutput(
            judgment=judgment,
            confidence=_coerce_float(result.get("confidence")),
            claims=claims,
            measurements=measurements
        )

