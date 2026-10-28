"""Lexical analyzers for the BM25 leg.

The same analyzer must run at index time and at query time, so each collection
records which analyzer it was built with.
"""

import re
from typing import Protocol

import Stemmer

_STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how
    i if in into is it its may me might my no not of on or our should so such than that
    the their them then there these they this to was we were what when where which who
    whom why will with would you your""".split()
)

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\.]*[A-Za-z0-9]|[A-Za-z0-9]")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


class Analyzer(Protocol):
    name: str

    def tokenize(self, text: str) -> list[str]: ...


class StemmingAnalyzer:
    """lowercase -> tokenize -> stopwords -> Snowball stem, with an abbreviation passthrough.

    Clinical text is full of short mixed-case tokens (HbA1c, eGFR, T2DM, ACE) where
    stemming or splitting destroys meaning, so tokens that look like abbreviations are
    kept verbatim (lowercased) instead of stemmed.
    """

    name = "stemming"

    def __init__(self, language: str = "english") -> None:
        self._stemmer = Stemmer.Stemmer(language)

    @staticmethod
    def _is_abbreviation(token: str) -> bool:
        has_digit = any(c.isdigit() for c in token)
        upper = sum(c.isupper() for c in token)
        return (upper >= 2 and len(token) <= 6) or (has_digit and any(c.isalpha() for c in token))

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for raw in _WORD.findall(text):
            raw = raw.strip(".-")
            if not raw:
                continue
            lower = raw.lower()
            if lower in _STOPWORDS:
                continue
            if self._is_abbreviation(raw):
                tokens.append(lower)
            else:
                # Hyphenated compounds ("beta-blocker") index both the parts and the stem.
                for part in lower.split("-"):
                    if part and part not in _STOPWORDS:
                        tokens.append(self._stemmer.stemWord(part))
        return tokens


class CodeAnalyzer:
    """Splits identifiers (camelCase, snake_case, dotted paths) and keeps the whole too."""

    name = "code"

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for ident in re.findall(r"[A-Za-z_][A-Za-z0-9_\.]*", text):
            lower = ident.lower()
            tokens.append(lower)
            parts = [p for chunk in ident.split(".") for p in chunk.split("_")]
            for part in parts:
                for piece in _CAMEL.split(part):
                    piece = piece.lower()
                    if piece and piece != lower and piece not in _STOPWORDS and len(piece) > 1:
                        tokens.append(piece)
        return tokens


ANALYZERS: dict[str, type[StemmingAnalyzer] | type[CodeAnalyzer]] = {
    StemmingAnalyzer.name: StemmingAnalyzer,
    CodeAnalyzer.name: CodeAnalyzer,
}
