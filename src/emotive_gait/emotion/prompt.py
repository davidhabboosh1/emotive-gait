"""Free-text emotion description to the six emotion intensities.

The package never calls an LLM. This module maps a short English description ("she walks in,
furious") to intensities through a shipped, hand-written lexicon (`data/affect_lexicon.yaml`).
The lexicon is tuned, not cited. The parser never invents an intensity: words it does not know
are returned in `unmatched`, and an empty match is a valid, all-zero result.

Parsing order: lower-case, split on non-letters, then scan left to right. At each position a
two-word term (`grossed out`, `no longer`, `a bit`) is tried before a one-word term. Modifiers
apply to the next lexicon term: `slightly`, `a bit`, `somewhat` multiply the weight by 0.5;
`very`, `extremely`, `really` multiply by 1.0 (they never raise a weight above the lexicon
value); `not`, `no longer`, `never` zero it. Negation survives intensifiers and stop words but
is cleared by an unknown content word. Several terms for one emotion take the maximum.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files

import yaml

from emotive_gait.emotion.evidence import UNSOURCED_EMOTIONS, Emotion

_LEXICON_PATH = files("emotive_gait.emotion") / "data" / "affect_lexicon.yaml"


def _load_lexicon() -> Mapping[str, tuple[Emotion, float]]:
    raw = yaml.safe_load(_LEXICON_PATH.read_text(encoding="utf-8"))["terms"]
    return {term: (Emotion(emotion), float(w)) for term, (emotion, w) in raw.items()}


LEXICON: Mapping[str, tuple[Emotion, float]] = _load_lexicon()

_INTENSIFIERS: Mapping[str, float] = {
    "very": 1.0,
    "extremely": 1.0,
    "really": 1.0,
    "slightly": 0.5,
    "somewhat": 0.5,
    "a bit": 0.5,
}
_NEGATORS = frozenset({"not", "never", "no longer"})
_STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "all",
        "also",
        "am",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "bit",
        "but",
        "by",
        "came",
        "can",
        "come",
        "comes",
        "coming",
        "could",
        "did",
        "do",
        "does",
        "extremely",
        "for",
        "from",
        "go",
        "goes",
        "going",
        "had",
        "has",
        "have",
        "he",
        "her",
        "here",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "like",
        "longer",
        "may",
        "me",
        "might",
        "must",
        "my",
        "never",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "onto",
        "or",
        "our",
        "out",
        "over",
        "quite",
        "rather",
        "really",
        "shall",
        "she",
        "should",
        "slightly",
        "so",
        "somewhat",
        "still",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "too",
        "under",
        "up",
        "us",
        "very",
        "walk",
        "walked",
        "walking",
        "walks",
        "was",
        "we",
        "were",
        "what",
        "when",
        "which",
        "while",
        "who",
        "will",
        "with",
        "would",
        "yet",
        "you",
        "your",
    }
)
_TOKEN = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class PromptResult:
    """Outcome of parsing a prompt.

    `matched` holds (term, emotion, contribution) in text order; a negated term appears with
    contribution 0.0. `grounded` is False when a matched term with a non-zero contribution
    belongs to an unsourced emotion (disgust, surprise).
    """

    intensities: Mapping[Emotion, float]
    matched: tuple[tuple[str, Emotion, float], ...]
    unmatched: tuple[str, ...]
    grounded: bool


def parse_emotion_prompt(text: str) -> PromptResult:
    """Map free text to six emotion intensities (all six keys always present)."""
    tokens = _TOKEN.findall(text.lower())
    intensities = {e: 0.0 for e in Emotion}
    matched: list[tuple[str, Emotion, float]] = []
    unmatched: list[str] = []
    scale = 1.0
    negate = False
    i = 0
    while i < len(tokens):
        word = tokens[i]
        pair = f"{word} {tokens[i + 1]}" if i + 1 < len(tokens) else None
        if pair is not None and (pair in LEXICON or pair in _INTENSIFIERS or pair in _NEGATORS):
            term, step = pair, 2
        else:
            term, step = word, 1
        i += step
        if term in LEXICON:
            emotion, weight = LEXICON[term]
            contribution = 0.0 if negate else min(1.0, weight * scale)
            matched.append((term, emotion, contribution))
            intensities[emotion] = max(intensities[emotion], contribution)
            scale, negate = 1.0, False
        elif term in _INTENSIFIERS:
            scale *= _INTENSIFIERS[term]
        elif term in _NEGATORS:
            negate = True
        elif term not in _STOP_WORDS:
            if term not in unmatched:
                unmatched.append(term)
            scale, negate = 1.0, False
    grounded = not any(c > 0.0 and e in UNSOURCED_EMOTIONS for _, e, c in matched)
    return PromptResult(intensities, tuple(matched), tuple(unmatched), grounded)
