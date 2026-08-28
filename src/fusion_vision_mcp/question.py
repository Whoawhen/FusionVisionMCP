"""Classify a VQA question and pick the measurable tool that answers it.

Moondream2 (``query_image``) is unreliable for open-ended judgment; when
``check_consistency`` flags a low-confidence answer, the project's "measure,
don't judge" rule says to route to the measurement that actually answers the
question instead. This module decides, from the question's wording, which
measurement applies and best-effort extracts the object names that measurement
needs. It is pure-Python and model-free so it is unit-testable in isolation.
"""

from __future__ import annotations

import re

#: Question categories with a measurable fallback tool.
SPATIAL = "spatial"  # -> spatial_relations (contact/gap/containment)
COUNT = "count"  # -> count_objects
OCR = "ocr"  # -> ocr (verbatim transcription)

_COUNT_KEYWORDS = ("how many", "count the", "number of", "a total of", "total number")
_SPATIAL_KEYWORDS = (
    "touch",
    "touching",
    "inside",
    "contain",
    "containing",
    "overlap",
    "overlapping",
    "in front of",
    "behind",
    "below",
    "above",
    "under",
    "beside",
    "next to",
    "gap",
    "distance",
    "between",
)
_OCR_KEYWORDS = (
    "text",
    "watermark",
    "logo",
    "sign",
    "signage",
    "label",
    "what does it say",
    "what does the",
    "read",
    "spell",
    "transcribe",
)

# Words that are not object names even when capitalized at the start of a question.
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "does",
        "is",
        "are",
        "do",
        "what",
        "how",
        "many",
        "in",
        "on",
        "at",
        "this",
        "that",
        "image",
        "picture",
        "photo",
        "photograph",
        "there",
        "of",
        "and",
        "or",
        "to",
        "with",
        "show",
        "showing",
        "can",
        "you",
        "please",
        "describe",
        "tell",
        "me",
        "it",
        "they",
        "them",
        # pronouns that parse as names but aren't objects:
        "something",
        "someone",
        "anything",
        "nothing",
        "everything",
    }
)

_QUOTED: re.Pattern[str] = re.compile(r'"([^"]+)"|' + r"'([^']+)'")
# A capitalized run of one or more Capitalized words ("Hand", "Red Mug").
_CAP_RUN: re.Pattern[str] = re.compile(r"\b(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+)*\b")
# "does/is the <X> <relation> the <Y>" / "are the <X> and <Y> <relation>"
_RELATION = "|".join(_SPATIAL_KEYWORDS)
_SPATIAL_PATTERN: re.Pattern[str] = re.compile(
    r"(?:does|is|are|do)\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+(?:"
    + _RELATION
    + r")\s+(?:the\s+|a\s+|an\s+)?(.+?)[\?\.]?\s*$",
    re.IGNORECASE,
)
# "how many <X>" / "number of <X>" -- take the trailing noun phrase, trimmed.
_COUNT_PATTERN: re.Pattern[str] = re.compile(
    r"(?:how many|number of|count the|total number of)\s+(.+?)[\?\.]?\s*$",
    re.IGNORECASE,
)


def classify(question: str) -> str | None:
    """Return the measurable category for a question, or None if none applies.

    Order matters: ``count`` is tested before ``spatial`` so "how many" is not
    swallowed by a "between" keyword, and ``spatial`` before ``ocr``.
    """
    q = question.lower()
    if any(k in q for k in _COUNT_KEYWORDS):
        return COUNT
    if any(k in q for k in _SPATIAL_KEYWORDS):
        return SPATIAL
    if any(k in q for k in _OCR_KEYWORDS):
        return OCR
    return None


def _raw_names(question: str) -> list[str]:
    """Quoted strings first, then capitalized runs; stopwords filtered out."""
    names: list[str] = []
    for match in _QUOTED.finditer(question):
        token = (match.group(1) or match.group(2) or "").strip()
        if token and token.lower() not in _STOPWORDS:
            names.append(token)
    for match in _CAP_RUN.finditer(question):
        token = match.group(0).strip()
        if token and token.lower() not in _STOPWORDS and token not in names:
            names.append(token)
    return names


def _strip_quotes(token: str) -> str:
    return token.strip().strip("\"'")


# Filler words that end a "how many <noun>" phrase; the noun is the leading chunk
# before the first of these.
_COUNT_FILLER = (
    " are",
    " is",
    " there",
    " do",
    " does",
    " can",
    " will",
    " would",
    " should",
    " in",
    " on",
    " at",
    " with",
    " of",
    " and",
    " that",
    " you",
    " we",
    " they",
    " visible",
    " shown",
    " present",
)


def _trim_count_noun(phrase: str) -> str:
    """Reduce 'dogs are in this image' to 'dogs'; '' if only filler remains."""
    noun = phrase.lower()
    for article in ("the ", "a ", "an "):
        noun = noun.removeprefix(article)
    cut = len(noun)
    for filler in _COUNT_FILLER:
        idx = noun.find(filler)
        if idx != -1 and idx < cut:
            cut = idx
    return noun[:cut].strip()


def names_for(category: str, question: str) -> list[str] | None:
    """Object names the fallback tool needs, or None if they can't be parsed.

    Returns None (meaning "omit the cross-check") rather than guessing when the
    required arity is not met: ``spatial`` needs two names, ``count`` needs one,
    ``ocr`` needs none. Pattern-based extraction is tried before quoted/capitalized
    runs so a plain "does the hand touch the shield" parses even without quotes.
    """
    if category == OCR:
        return []

    if category == SPATIAL:
        match = _SPATIAL_PATTERN.search(question)
        if match:
            a = _strip_quotes(match.group(1)).lower()
            b = _strip_quotes(match.group(2)).lower()
            for article in ("the ", "a ", "an "):
                a = a.removeprefix(article)
                b = b.removeprefix(article)
            if a and b and a not in _STOPWORDS and b not in _STOPWORDS:
                return [a, b]
        names = _raw_names(question)
        return names[:2] if len(names) >= 2 else None

    if category == COUNT:
        match = _COUNT_PATTERN.search(question)
        if match:
            noun = _strip_quotes(_trim_count_noun(match.group(1)))
            if noun and noun not in _STOPWORDS and noun.split()[0] not in _STOPWORDS:
                return [noun]
        names = _raw_names(question)
        return names[:1] if names else None

    return None


__all__ = ["COUNT", "OCR", "SPATIAL", "classify", "names_for"]
