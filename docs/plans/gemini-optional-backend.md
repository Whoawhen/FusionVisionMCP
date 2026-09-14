# Add Gemini as an optional, env-var-gated fast path for `caption` / `query_image`

## Context

FusionVisionMCP's whole pitch (per its own `CLAUDE.md` and `README.md`) is that it runs Florence-2,
Moondream2, Grounding DINO and SAM2 fully locally — no image leaves the machine. That's a deliberate
tradeoff: it's free and private, but Moondream2 is a small VLM with documented reliability gaps
(flat "None" on real defects, text paraphrasing instead of transcription, weak open-ended judgment).

The user wants an *optional* escape hatch: when they choose to spend Gemini quota and accept sending
an image to Google, Gemini should be used as a faster, better first-pass for the two tools where a
strong general VLM is a straight upgrade over Moondream2/Florence-2 — `caption` and `query_image`
(VQA). Everything else (detection, counting, spatial measurement, aesthetics) stays local-only for
now; those are the tools where this repo's own local pipeline (Grounding DINO, SAM2, geometry-based
measurement) is the whole value proposition, not a gap Gemini needs to fill.

Two things the user was explicit about after discussion:
- **Two independent env-var gates**, not one: an API key alone isn't enough to turn Gemini on. A
  separate enable flag lets the key live permanently in a shell profile while still defaulting to
  local-only, and lets it be flipped off for a session (e.g. quota exhausted) without touching the key.
- **Gemini shortcuts local logic, it doesn't sit alongside it.** When enabled and successful, Gemini's
  answer *is* the tool's answer — Florence-2/Moondream2 aren't also run for the plain case. Local
  models remain solely to (a) serve every other tool, and (b) provide the richer opt-in analysis
  modes (`verify_text`, `auto_verify_text`, `check_consistency`, `structured_analysis`) that Gemini
  has no equivalent for in this codebase.
- **On a Gemini failure, fail loudly and fall back**, don't silently swallow the error. The call must
  still succeed (fall back to the existing local model), but the failure must be visible to the
  calling agent in the tool's own response, not just a server log line.

## Design

### Env vars / config surface

Two new independently-gated pieces of config, following the codebase's existing convention that
*all* config flows through `cli.py`'s `click` options into `server()` → `app_lifespan()` as explicit
kwargs (no bare `os.getenv` scattered in the package) — click options can read from an env var via
`envvar=` while still being visible in `--help`, which is the best fit for a codebase that currently
has zero env-var precedent but a very established CLI-flag-config precedent (see `--reasoner-provider`
in [cli.py](../../src/fusion_vision_mcp/cli.py)).

Add to `cli.py`:
```python
@click.option("--gemini-api-key", envvar="GEMINI_API_KEY", default=None,
              help="API key for the optional Gemini backend (also read from GEMINI_API_KEY).")
@click.option("--gemini-enabled/--no-gemini-enabled", envvar="FUSIONVISION_ENABLE_GEMINI",
              default=False, show_default=True,
              help="Turns the optional Gemini backend on. Both this and --gemini-api-key must be "
                   "set for Gemini to be used; off by default so no image data leaves the machine "
                   "unless explicitly opted in.")
@click.option("--gemini-model", envvar="GEMINI_MODEL", default=DEFAULT_GEMINI_MODEL,
              show_default=True, help="Gemini model ID used for the caption/query_image fast path.")
```
Forwarded into `server(...)` exactly like every other model-id kwarg is today.

`DEFAULT_GEMINI_MODEL` goes in [constants.py](../../src/fusion_vision_mcp/constants.py) alongside the
other `DEFAULT_*` model constants, set to `"gemini-3.8-flash"` per the user's snippet — flagging here
that this model ID is unverified against a live API and is fully overridable via
`--gemini-model`/`GEMINI_MODEL` specifically so a rename/deprecation doesn't require a code change.

### New module: `src/fusion_vision_mcp/gemini.py`

A thin wrapper class, matching the shape of the other model wrappers so it can be dropped into
`AppContext`:
```python
from google import genai

class Gemini:
    def __init__(self, api_key: str, model_id: str):
        self._client = genai.Client(api_key=api_key)
        self._model_id = model_id

    def caption(self, images: list[Image]) -> list[str]:
        """One Gemini call per image, prompted to produce a single detailed prose caption."""

    def query(self, images: list[Image], question: str) -> list[str]:
        """One Gemini call per image, passing `question` straight through."""
```
Images are passed to `contents` as the `PIL.Image.Image` objects `get_images()` already produces
(the `google-genai` SDK accepts PIL images directly) — not as a raw path/URL string, which is what
the user's original snippet did and would not actually work for a local file.

### Wiring: `__init__.py`

- Import guard at the top, matching the existing `ImageQuality`/`iqa`-extra precedent exactly (same
  file, lines ~55–58):
  ```python
  try:
      from fusion_vision_mcp.gemini import Gemini
  except ImportError:
      Gemini = None  # type: ignore
  ```
  This keeps `google-genai` a genuinely optional dependency (new `[project.optional-dependencies]`
  extra in `pyproject.toml`, `gemini = ["google-genai>=1.0.0"]`, same shape as the existing `iqa`
  and `ocr-specialist` extras) and costs nothing extra at import time beyond what `iqa` already costs.
- `AppContext` gets one new field: `gemini: Any | None = None`.
- `app_lifespan` gains `gemini_api_key: str | None`, `gemini_enabled: bool`, `gemini_model: str`
  params, and constructs it only when both gates are satisfied and the extra is installed:
  ```python
  gemini = None
  if gemini_enabled and gemini_api_key and Gemini is not None:
      gemini = Gemini(gemini_api_key, gemini_model)
  elif gemini_enabled:
      logger.warning("Gemini enabled but not usable (%s); continuing local-only.",
                      "no GEMINI_API_KEY" if not gemini_api_key else "google-genai not installed")
  ```
  (`logging`/`logger = logging.getLogger(__name__)` don't currently exist in `__init__.py` — add them.)
  No `IdleReleased`/`IdleProxy` wrapping: unlike the local models, a Gemini client holds no meaningful
  memory to reclaim, so idle-release buys nothing here (see
  [idle.py](../../src/fusion_vision_mcp/idle.py)'s own docstring, which frames the mechanism
  specifically around local model VRAM/RAM cost).

### Routing inside `caption` and `query_image`

The shortcut/fallback logic is factored into two small private helpers next to the existing
`_vqa_consistency`/`_vqa_cross_check` helpers (~line 1437+), so it's directly unit-testable without a
running server, matching that existing convention:

```python
def _caption_with_gemini_fallback(app: AppContext, images: list[Image]) -> list[Any]:
    if app.gemini is None:
        return app.processor.caption(images, CaptionLevel.MORE_DETAILED)
    try:
        return app.gemini.caption(images)
    except Exception as e:
        logger.error("Gemini caption call failed, falling back to local: %s", e)
        local = app.processor.caption(images, CaptionLevel.MORE_DETAILED)
        return [{"caption": c, "gemini_error": str(e)} for c in local]
```
(`_query_with_gemini_fallback` is the same shape for `app.vqa.query`, returning
`{"answer": a, "gemini_error": str(e)}` per page on failure.)

Call sites:
- `caption`'s plain path (`not verify_text and not auto_verify_text`) calls
  `_caption_with_gemini_fallback` instead of `app.processor.caption(...)` directly. **`verify_text`
  and `auto_verify_text` are left untouched, always local** — they depend on Florence-2's
  OCR-with-region head and Granite-Docling, which Gemini has no equivalent for here; this matches the
  user's framing that local fills gaps Gemini can't cover.
- `query_image`'s plain path (`not check_consistency and not structured_analysis`) calls
  `_query_with_gemini_fallback` instead of `app.vqa.query(...)` directly, for the same reason:
  `check_consistency`'s rephrased-control-question routing and `structured_analysis`'s Grounding-DINO
  corroboration are local-only value-adds, unaffected by Gemini being enabled.

Return-shape contract: when Gemini is off, or on and succeeds, both tools return exactly what they
return today (`list[str]`) — zero behavior change for existing callers/tests. **Only on an actual
Gemini failure** does a page's entry become a dict with a `gemini_error` field alongside the local
fallback text, so the failure is visible directly in the content the calling agent reads (not just a
server-side log line, which a client isn't guaranteed to surface back into the model's context) while
never breaking the call itself. Both tool docstrings get a short paragraph documenting this: "When
Gemini is configured, this shortcuts to it for a faster/better answer; on Gemini failure it falls back
to \<Florence-2 caption / Moondream2 VQA\> and reports the error in `gemini_error`."

## Files touched

- `pyproject.toml` — new `gemini` entry under `[project.optional-dependencies]`.
- `src/fusion_vision_mcp/constants.py` — `DEFAULT_GEMINI_MODEL`.
- `src/fusion_vision_mcp/gemini.py` — new file, the `Gemini` wrapper class.
- `src/fusion_vision_mcp/__init__.py` — import guard, `AppContext.gemini` field, `app_lifespan` params
  + construction, `logging` import/`logger`, `_caption_with_gemini_fallback` /
  `_query_with_gemini_fallback` helpers, call-site changes in `caption` and `query_image`, docstring
  updates.
- `src/fusion_vision_mcp/cli.py` — three new `click.option`s, forwarded into `server(...)`.
- `README.md` — one line in the `## Tools` table for `caption`/`query_image` noting the optional
  Gemini fast path; a caveat in `## Architecture` next to the existing "no image data leaves the
  machine" claim, since this is the one path where that's no longer true if opted in.
- `README_DETAILED.md` — a bullet or two under `## Options` documenting the three new flags/env vars
  and the fallback behavior, following the existing `iqa`/`reasoner` documentation sentence template.
- `tests/test_gemini.py` — new file (see Verification).
- `tests/test_server.py` — a couple of additive regression tests (see Verification).

## Verification

Gemini calls a real, metered external API (it has a free tier, but one with rate/token limits that
this feature exists partly to let the user stay under or avoid entirely by flipping the enable flag
off), so per the project's own documented testing philosophy (real subprocess + real MCP protocol, no
HTTP-mocking library anywhere in the repo) the plan is to test the *routing/fallback logic* directly
and thoroughly, and gate any real-API test behind key presence rather than spending quota in CI:

1. **Pure unit tests, no server, no network** (`tests/test_gemini.py`), mirroring
   `TestSeparability`/`TestVqaConsistency` in `tests/test_server.py`: construct a fake `gemini` stub
   object (`.caption()`/`.query()` that either returns canned text or raises) and call
   `_caption_with_gemini_fallback`/`_query_with_gemini_fallback` directly with a real `Florence2`/
   `Moondream`-less stand-in `AppContext`. Assert: Gemini-off returns local unchanged; Gemini-success
   returns Gemini's text with no local call made (spy/counter on the local stub to prove it); Gemini-
   raises returns the local fallback text plus `gemini_error`.
2. **CLI/config resolution unit test**, mirroring `tests/test_cli.py`'s `resolve_memory_mode` tests:
   confirm Gemini is only constructed when both `--gemini-enabled` and `--gemini-api-key` (or their
   env vars) are present, and that enabling without a key logs a warning and leaves `app.gemini` as
   `None` rather than raising.
3. **Regression coverage in `tests/test_server.py`**: with no Gemini env vars set (the existing
   `SERVER_PARAMS` subprocess as-is), assert `caption` and `query_image`'s plain-path responses are
   byte-for-byte the same shape as before this change — proving the default (off) experience is
   unaffected.
4. **Manual smoke test with a real key** (not part of the automated suite, run by hand once
   implemented): export `GEMINI_API_KEY` and `FUSIONVISION_ENABLE_GEMINI=1`, run the server via
   `uv run fusion-vision-mcp`, call `caption` and `query_image` on `tests/sample.jpg` through an MCP
   client, and confirm a real Gemini response comes back; then temporarily use an invalid key and
   confirm the `gemini_error` fallback path fires and the local caption/answer still comes back.
5. `uvx ruff@0.16.1 check src tests`, `uvx ruff@0.16.1 format src tests`, and the mypy command from
   `CLAUDE.md` after implementation, since `google.genai` needs a `types.py` stub check or an
   `ignore_missing_imports` allowance if it ships no type stubs.

## Suggested build sequence and model choice

This repo's own git history chunks feature work into numbered sprints (`Sprint 12`, `Sprint 13`,
`Sprint 14` in the recent log), each landing as its own commit(s) with a working, tested state at the
end. This plan is a good fit for that same shape — three sprints, each independently testable:

- **Sprint A — plumbing, no tool behavior change yet.** `pyproject.toml` extra, `constants.py`,
  new `gemini.py` wrapper, the `__init__.py` import guard + `AppContext.gemini` field + `app_lifespan`
  construction/warning logic, and the three `cli.py` options. Done when the server starts cleanly in
  all three states (extra not installed / installed-but-disabled / installed-and-enabled-with-key) and
  `app.gemini` is populated or `None` correctly in each — no tool yet reads it.
- **Sprint B — routing.** `_caption_with_gemini_fallback` / `_query_with_gemini_fallback`, the
  call-site swaps in `caption` and `query_image`, docstring updates. Done when the unit tests in
  Verification items 1–2 pass.
- **Sprint C — regression safety, docs, real-key smoke test.** `test_server.py` regression additions,
  README/README_DETAILED updates, the manual smoke test, ruff/mypy. Done when Verification items 3–5
  all pass.

**Model recommendation: Sonnet 5 (Sonnet is fine — no need to reach for Opus here).** The reasoning:
this plan is already fully specified — exact function signatures, call sites, return-shape rules, and
file list are all pinned down above, so the implementation work is precise execution against a spec
rather than open-ended design under ambiguity, which is where Opus's extra reasoning budget earns its
cost. The trickiest single piece — getting the `google-genai` SDK call shape right (passing a PIL
`Image` in `contents`, reading `response.text`) — is a small, self-contained, easily-verified-by-running
piece of code, not a decision with wide blast radius. Sonnet 5 (the model running this session) is
well matched to sprint-sized, well-scoped implementation tasks like A/B/C above; save Opus for a
future task in this repo that's closer to open design work (e.g. the next unsolved-counting-style
research spike in `benchmarks/`), where CLAUDE.md's own history shows the payoff came from exploring
several rejected approaches rather than executing one clear spec.
