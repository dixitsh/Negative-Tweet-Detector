"""Fuzzy normalization for noisy Hinglish/code-mixed social text."""
from __future__ import annotations
import re
from difflib import SequenceMatcher
from typing import Iterable, Dict, Tuple

DEFAULT_LEXICON = {
    "acha": ["acha", "achha", "achhi", "achi", "achaa", "achaaa", "achhiii"],
    "bahut": ["bahut", "bahoot", "bahot", "bohot", "bahutt", "bahuuut", "bahuttt"],
    "hai": ["hai", "h", "haai", "hainn"],
    "nahi": ["nahi", "nahin", "nai", "nahee", "nhi", "nahiii"],
    "kharab": ["kharab", "khraab", "khrab", "kharabb"],
    "bekar": ["bekar", "bekaar", "bkar", "bekarr"],
    "bakwas": ["bakwas", "bakwaas", "bakwass", "bakvass"],
    "mast": ["mast", "maast", "masst", "mssst"],
    "pyar": ["pyar", "pyaar", "pyaarrr", "piar"],
    "pasand": ["pasand", "psnd", "pasandd"],
}


def _elongation_squash(token: str) -> str:
    # Reduce repeated characters while preserving common Romanized spellings.
    return re.sub(r"(.)\1{2,}", r"\1\1", token.lower())


def normalize_token(token: str, lexicon: Dict[str, Iterable[str]] = DEFAULT_LEXICON,
                     threshold: float = 0.86) -> Tuple[str, float]:
    if not token or not token.isalpha() or len(token) < 3:
        return token, 1.0
    candidate = _elongation_squash(token)
    best_word, best_score = candidate, 0.0
    for canonical, variants in lexicon.items():
        for variant in variants:
            score = SequenceMatcher(None, candidate, variant).ratio()
            if score > best_score:
                best_word, best_score = canonical, score
    if best_score >= threshold:
        return best_word, best_score
    return candidate, best_score


def fuzzy_normalize(text: str, threshold: float = 0.86) -> str:
    """Normalize likely spelling/elongation variants without changing unknown words."""
    tokens = re.findall(r"[A-Za-z]+|[^A-Za-z]+", str(text))
    return "".join(normalize_token(t, threshold=threshold)[0] if t.isalpha() else t for t in tokens)


if __name__ == "__main__":
    examples = ["movie bahuttt achhiii hai", "product bilkul bakwaass hai"]
    for x in examples:
        print(x, "->", fuzzy_normalize(x))
