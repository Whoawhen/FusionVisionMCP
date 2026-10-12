"""Detect a generated text that has started looping. Pure Python, no torch: safe on the eager import path."""

import re

from .constants import LOOP_NGRAM, LOOP_RUN

_WORD = re.compile(r"\w+")


def loop_start(text: str, ngram: int = LOOP_NGRAM, run: int = LOOP_RUN) -> int | None:
    """Character offset where `text` starts looping, or None.

    A loop is `run` consecutive `ngram`-word spans that each already appeared earlier (words compared
    case-insensitively, punctuation ignored). The offset is the start of the first repeated span in that
    run, so one copy of the repeated material stays before it. A phrase reused once or twice in a normal
    description never builds a run that long.
    """
    matches = list(_WORD.finditer(text))
    words = [m.group().lower() for m in matches]
    seen: set[tuple[str, ...]] = set()
    streak = 0
    first = 0
    for k in range(len(words) - ngram + 1):
        span = tuple(words[k : k + ngram])
        if span in seen:
            if streak == 0:
                first = k
            streak += 1
            if streak >= run:
                return matches[first].start()
        else:
            streak = 0
            seen.add(span)
    return None


def cut_loop(text: str) -> tuple[str, bool]:
    """`text` with any loop removed, and whether one was found."""
    start = loop_start(text)
    if start is None:
        return text.strip(), False
    return text[:start].rstrip(" \t\n,;:-").strip(), True
