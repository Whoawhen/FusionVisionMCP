# Sprint 20: Garbled-Text Artifact Detection - Negative Result

## Objective
Measure nonsense rendered text using Florence-2 OCR, Granite-Docling specialist OCR, and `textmatch.py`.
The signal was defined as: strong disagreement between the two OCR engines on the same region, **plus** tokens that fail a lexicon check.

## Findings
The negative controls cannot be held, specifically the **COCO photo carrying real signage**.

While the "no text" and "clean rendered text" (`FusionVisionMCP-Dark.jpg`) controls pass (because Florence-2 and Granite-Docling agree perfectly on clean text, and no boxes are found on the no-text image), real-world signage breaks the heuristic entirely:

1. **OCR Disagreement on Real Signage**: In real photographs (like the COCO signage control), text is often blurry, angled, or stylized. Florence-2 and Granite-Docling frequently disagree strongly (ratio < 0.6) on these regions because they are inherently difficult to read.
2. **The Lexicon Trap**: As anticipated in the sprint plan, a naive dictionary check flags brand names, logos, legitimate acronyms, and foreign words. If we require the text to fail a lexicon check to be considered garbled, real brand names will fail it.
3. **The False Positive Intersection**: When a real, stylized brand name or foreign word is present (e.g., on a storefront), it will fail the lexicon check. Because it is stylized or blurry, the two OCR engines will also disagree on its transcription. This perfectly matches the "garbled text" heuristic (strong disagreement + lexicon failure), causing a **false positive** on real, unaltered photographs.

To fix this, the lexicon would need to understand all global brand names, multiple languages, and account for OCR transcription errors without accepting actual garbled synthetic text. This is infeasible with a simple dictionary approach like `pyspellchecker` or word lists.

## Conclusion
Per the sprint plan's instructions for this exact scenario ("If the negative controls cannot be held, ship nothing and record the negative result"):
**No code changes have been shipped for Sprint 20.** The feature is structurally unsound because the intersection of OCR unreliability and dictionary incompleteness creates unavoidable false positives on real images.
