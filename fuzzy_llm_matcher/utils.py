"""Shared utilities, including a pure-Python fallback scorer.

The package prefers ``rapidfuzz`` for speed, but does not hard-fail if it
is missing (e.g. offline environments, minimal installs). When rapidfuzz
is unavailable we fall back to a slower but dependency-free implementation
built on the standard-library ``difflib`` module, exposing the same
function names (``WRatio``, ``token_sort_ratio``, ``token_set_ratio``,
``ratio``, ``partial_ratio``) so the rest of the codebase does not need to
branch on which backend is active.

Transliteration + phonetic scoring
-----------------------------------
Optional extras ``unidecode`` and ``jellyfish`` unlock:

- ``transliterate_text(text)`` — converts any Unicode script to ASCII using
  the unidecode algorithm.  "Köln" → "Koln", "Москва" → "Moskva", "北京" → "Bei Jing".
- ``phonetic_code(text, algorithm)`` — Soundex / Metaphone / NYSIIS code.
- ``phonetic_similarity_score(a, b, algorithm)`` — 0–100 score based on
  Jaro-Winkler distance of the two phonetic codes.  Matches "Isfahan" /
  "Esfahan" even when WRatio gives only ~70.
- New scorers: ``"transliterated_WRatio"``, ``"soundex"``, ``"metaphone"``,
  ``"nysiis"`` added to ``SCORERS`` when the optional packages are present.
"""

from __future__ import annotations

import difflib
import re
from typing import Iterable

try:
    from rapidfuzz import fuzz as _rf_fuzz  # type: ignore
    HAVE_RAPIDFUZZ = True
except ImportError:
    HAVE_RAPIDFUZZ = False

try:
    from unidecode import unidecode as _unidecode  # type: ignore
    HAVE_UNIDECODE = True
except ImportError:
    HAVE_UNIDECODE = False

try:
    import jellyfish as _jellyfish  # type: ignore
    HAVE_JELLYFISH = True
except ImportError:
    HAVE_JELLYFISH = False


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalize_text(value, transliterate: bool = False) -> str:
    """Lowercase, strip, and collapse whitespace. Non-strings become ''.

    Parameters
    ----------
    value:
        Any value — strings, numbers, NaN, None are all handled safely.
    transliterate:
        When ``True`` (and ``unidecode`` is installed), convert non-ASCII
        characters to their closest ASCII equivalent before normalisation.
        "Köln" → "koln", "Москва" → "moskva", "北京" → "bei jing".
        Falls back silently when unidecode is not installed.
    """
    if value is None:
        return ""
    try:
        if value != value:  # NaN check without importing pandas/numpy here
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if transliterate and HAVE_UNIDECODE:
        text = _unidecode(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text) if t]


def _difflib_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 100.0
    return difflib.SequenceMatcher(None, a, b).ratio() * 100.0


def _difflib_partial_ratio(a: str, b: str) -> float:
    """Approximate rapidfuzz's partial_ratio: best ratio over the shorter
    string aligned against windows of the longer string."""
    if not a or not b:
        return _difflib_ratio(a, b)
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) == 0:
        return 0.0
    best = 0.0
    step = max(1, len(shorter) // 4)
    for start in range(0, max(1, len(longer) - len(shorter) + 1), step):
        window = longer[start:start + len(shorter)]
        best = max(best, _difflib_ratio(shorter, window))
    best = max(best, _difflib_ratio(shorter, longer[-len(shorter):]))
    return best


def _token_sort_ratio(a: str, b: str) -> float:
    ta, tb = sorted(_tokens(a)), sorted(_tokens(b))
    return _difflib_ratio(" ".join(ta), " ".join(tb))


def _token_set_ratio(a: str, b: str) -> float:
    sa, sb = set(_tokens(a)), set(_tokens(b))
    inter = sorted(sa & sb)
    diff_ab = sorted(sa - sb)
    diff_ba = sorted(sb - sa)
    t0 = " ".join(inter)
    t1 = (" ".join(inter) + " " + " ".join(diff_ab)).strip()
    t2 = (" ".join(inter) + " " + " ".join(diff_ba)).strip()
    return max(
        _difflib_ratio(t0, t1),
        _difflib_ratio(t0, t2),
        _difflib_ratio(t1, t2),
    )


def _wratio(a: str, b: str) -> float:
    """Approximate rapidfuzz's WRatio: takes the best of several
    strategies, discounting partial-match strategies slightly so that
    exact/near-exact full matches are still preferred."""
    scores = [
        _difflib_ratio(a, b),
        _token_sort_ratio(a, b),
        _token_set_ratio(a, b) * 0.98,
        _difflib_partial_ratio(a, b) * 0.9,
    ]
    return max(scores)


class _FallbackFuzz:
    """Drop-in replacement exposing the subset of ``rapidfuzz.fuzz`` used
    by this package."""

    @staticmethod
    def ratio(a: str, b: str) -> float:
        return _difflib_ratio(a, b)

    @staticmethod
    def partial_ratio(a: str, b: str) -> float:
        return _difflib_partial_ratio(a, b)

    @staticmethod
    def token_sort_ratio(a: str, b: str) -> float:
        return _token_sort_ratio(a, b)

    @staticmethod
    def token_set_ratio(a: str, b: str) -> float:
        return _token_set_ratio(a, b)

    @staticmethod
    def WRatio(a: str, b: str) -> float:
        return _wratio(a, b)


fuzz = _rf_fuzz if HAVE_RAPIDFUZZ else _FallbackFuzz()

SCORERS = {
    "WRatio": fuzz.WRatio,
    "ratio": fuzz.ratio,
    "partial_ratio": fuzz.partial_ratio,
    "token_sort_ratio": fuzz.token_sort_ratio,
    "token_set_ratio": fuzz.token_set_ratio,
}


def get_scorer(name: str):
    if name not in SCORERS:
        raise ValueError(
            f"Unknown scorer '{name}'. Available scorers: {list(SCORERS)}"
        )
    return SCORERS[name]


def chunked(iterable: Iterable, size: int):
    """Yield successive chunks of `size` from `iterable`."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# ---------------------------------------------------------------------------
# Transliteration helpers
# ---------------------------------------------------------------------------

def transliterate_text(value) -> str:
    """Convert any Unicode text to its closest ASCII representation.

    Uses the ``unidecode`` library (optional dep). Falls back to stripping
    non-ASCII characters if unidecode is not installed.

    Examples
    --------
    >>> transliterate_text("Köln")
    'Koln'
    >>> transliterate_text("Москва")
    'Moskva'
    >>> transliterate_text("北京")
    'Bei Jing'
    >>> transliterate_text("Al Qahirah")
    'Al Qahirah'
    """
    if value is None:
        return ""
    text = str(value)
    if HAVE_UNIDECODE:
        return _unidecode(text)
    # Fallback: remove non-ASCII (loses information but never crashes)
    return text.encode("ascii", errors="ignore").decode("ascii")


# ---------------------------------------------------------------------------
# Phonetic scoring
# ---------------------------------------------------------------------------

def phonetic_code(text: str, algorithm: str = "metaphone") -> str:
    """Return the phonetic code of a string.

    Parameters
    ----------
    text:
        Input string (should already be transliterated to ASCII for best results).
    algorithm:
        One of ``"soundex"``, ``"metaphone"``, ``"nysiis"``.
        Falls back to the raw string if ``jellyfish`` is not installed.

    Returns
    -------
    str  — the phonetic code (empty string for empty input)

    Examples
    --------
    >>> phonetic_code("Isfahan",  "metaphone")
    'ISFHN'
    >>> phonetic_code("Esfahan",  "metaphone")
    'ISFHN'
    >>> phonetic_code("Hannover", "soundex")
    'H516'
    >>> phonetic_code("Hanover",  "soundex")
    'H516'
    """
    if not text:
        return ""
    if not HAVE_JELLYFISH:
        return text.lower()
    alg = algorithm.lower()
    try:
        if alg == "soundex":
            return _jellyfish.soundex(text)
        if alg == "nysiis":
            return _jellyfish.nysiis(text)
        return _jellyfish.metaphone(text)  # default
    except Exception:
        return text.lower()


def phonetic_similarity_score(
    a: str, b: str, algorithm: str = "metaphone"
) -> float:
    """Return a 0–100 phonetic similarity score for two strings.

    Computes the phonetic code of each string and then measures their
    Jaro-Winkler similarity. This catches pairs like "Isfahan"/"Esfahan"
    and "Hannover"/"Hanover" that WRatio scores poorly due to spelling
    differences but whose pronunciation is nearly identical.

    The pipeline for transliterated geo place names:
    1. transliterate_text (unidecode) → remove diacritics
    2. phonetic_code (metaphone/soundex) → pronunciation skeleton
    3. jaro_winkler on the codes → pronunciation similarity

    Parameters
    ----------
    a, b:
        Input strings (pre-transliterated for best results).
    algorithm:
        Phonetic algorithm: ``"metaphone"`` (default, most accurate for
        place names), ``"soundex"``, or ``"nysiis"``.

    Returns
    -------
    float — 0–100 similarity score

    Examples
    --------
    >>> phonetic_similarity_score("Isfahan", "Esfahan")
    100.0
    >>> phonetic_similarity_score("Hannover", "Hanover")
    100.0
    >>> phonetic_similarity_score("München", "Munich")  # after transliteration
    ...
    """
    if not HAVE_JELLYFISH:
        return _difflib_ratio(a.lower(), b.lower())
    ca = phonetic_code(a, algorithm)
    cb = phonetic_code(b, algorithm)
    if not ca and not cb:
        return 100.0
    if not ca or not cb:
        return 0.0
    try:
        return _jellyfish.jaro_winkler_similarity(ca, cb) * 100.0
    except Exception:
        return _difflib_ratio(ca, cb)


# ---------------------------------------------------------------------------
# Transliterated string scorers
# ---------------------------------------------------------------------------

def _transliterated_wratio(a: str, b: str) -> float:
    """WRatio applied after unidecode transliteration on both sides."""
    ta = transliterate_text(a)
    tb = transliterate_text(b)
    return fuzz.WRatio(ta.lower(), tb.lower())


def _metaphone_score(a: str, b: str) -> float:
    ta, tb = transliterate_text(a), transliterate_text(b)
    return phonetic_similarity_score(ta, tb, "metaphone")


def _soundex_score(a: str, b: str) -> float:
    ta, tb = transliterate_text(a), transliterate_text(b)
    return phonetic_similarity_score(ta, tb, "soundex")


def _nysiis_score(a: str, b: str) -> float:
    ta, tb = transliterate_text(a), transliterate_text(b)
    return phonetic_similarity_score(ta, tb, "nysiis")


# Register new scorers (always available — degrade gracefully without jellyfish)
SCORERS["transliterated_WRatio"] = _transliterated_wratio
SCORERS["metaphone"]             = _metaphone_score
SCORERS["soundex"]               = _soundex_score
SCORERS["nysiis"]                = _nysiis_score
