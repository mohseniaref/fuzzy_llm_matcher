"""Shared utilities, including a pure-Python fallback scorer.

The package prefers ``rapidfuzz`` for speed, but does not hard-fail if it
is missing (e.g. offline environments, minimal installs). When rapidfuzz
is unavailable we fall back to a slower but dependency-free implementation
built on the standard-library ``difflib`` module, exposing the same
function names (``WRatio``, ``token_sort_ratio``, ``token_set_ratio``,
``ratio``, ``partial_ratio``) so the rest of the codebase does not need to
branch on which backend is active.
"""

from __future__ import annotations

import difflib
import re
from typing import Iterable

try:
    from rapidfuzz import fuzz as _rf_fuzz  # type: ignore

    HAVE_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - exercised in offline environments
    HAVE_RAPIDFUZZ = False


def normalize_text(value) -> str:
    """Lowercase, strip, and collapse whitespace. Non-strings become ''."""
    if value is None:
        return ""
    try:
        if value != value:  # NaN check without importing pandas/numpy here
            return ""
    except Exception:
        pass
    text = str(value).strip().lower()
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
