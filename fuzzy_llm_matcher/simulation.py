"""Generate noisy variants of clean entity names for benchmarking.

Used to build synthetic ground-truth test sets: each clean name maps to
N dirty variants, and the true mapping is known by construction.
"""

from __future__ import annotations

import random
import re
from typing import Optional

import pandas as pd

LEGAL_SUFFIX_MAP = {
    "gmbh": ["GmbH & Co. KG", "GmbH"],
    "inc": ["Incorporated", "Inc."],
    "inc.": ["Incorporated", "Inc"],
    "ltd": ["Limited", "Ltd."],
    "corp": ["Corporation", "Corp."],
    "llc": ["L.L.C.", "LLC"],
    "co": ["Company", "Co."],
    "university": ["Univ.", "U."],
    "technology": ["Tech"],
    "netherlands": ["NL", "The Netherlands"],
}

ABBREVIATIONS = {
    "university": "Univ.",
    "technology": "Tech",
    "institute": "Inst.",
    "international": "Intl.",
    "corporation": "Corp.",
    "company": "Co.",
    "and": "&",
    "street": "St.",
    "avenue": "Ave.",
}


def _abbreviate(name: str, rng: random.Random) -> str:
    words = name.split()
    out = []
    for w in words:
        key = w.strip(".,").lower()
        if key in ABBREVIATIONS and rng.random() < 0.7:
            out.append(ABBREVIATIONS[key])
        else:
            out.append(w)
    return " ".join(out)


def _drop_word(name: str, rng: random.Random) -> str:
    words = name.split()
    if len(words) <= 1:
        return name
    idx = rng.randrange(len(words))
    del words[idx]
    return " ".join(words)


def _swap_word_order(name: str, rng: random.Random) -> str:
    words = name.split()
    if len(words) < 2:
        return name
    i, j = rng.sample(range(len(words)), 2)
    words[i], words[j] = words[j], words[i]
    return " ".join(words)


def _punctuation_noise(name: str, rng: random.Random) -> str:
    choices = [
        lambda s: s.replace(" ", "-"),
        lambda s: s.replace(",", ""),
        lambda s: s + ".",
        lambda s: s.replace("&", "and"),
        lambda s: re.sub(r"[.,]", "", s),
    ]
    fn = rng.choice(choices)
    return fn(name)

def _legal_suffix_change(name: str, rng: random.Random) -> str:
    words = name.split()
    last = words[-1].strip(".,").lower() if words else ""
    if last in LEGAL_SUFFIX_MAP:
        replacement = rng.choice(LEGAL_SUFFIX_MAP[last])
        words[-1] = replacement
        return " ".join(words)
    # otherwise append a plausible suffix
    suffix = rng.choice(["GmbH", "Inc.", "Ltd.", "LLC", "Corp."])
    return f"{name} {suffix}"


def _spelling_noise(name: str, rng: random.Random) -> str:
    if len(name) < 4:
        return name
    chars = list(name)
    idx = rng.randrange(1, len(chars) - 1)
    op = rng.choice(["swap", "drop", "double"])
    if op == "swap" and idx < len(chars) - 1:
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    elif op == "drop":
        del chars[idx]
    elif op == "double":
        chars.insert(idx, chars[idx])
    return "".join(chars)


def _capitalization_noise(name: str, rng: random.Random) -> str:
    choice = rng.choice(["upper", "lower", "title"])
    if choice == "upper":
        return name.upper()
    if choice == "lower":
        return name.lower()
    return name.title()


def _extra_location(name: str, rng: random.Random) -> str:
    locations = ["Germany", "Netherlands", "USA", "UK", "Berlin", "Amsterdam"]
    return f"{name} {rng.choice(locations)}"


def _partial_name(name: str, rng: random.Random) -> str:
    words = name.split()
    if len(words) <= 1:
        return name
    keep = max(1, len(words) - rng.randrange(1, len(words)))
    return " ".join(words[:keep])


def _initialize_first_word(name: str, rng: random.Random) -> str:
    words = name.split()
    if len(words) <= 1 or not words[0][:1].isalpha():
        return name
    words[0] = words[0][0] + "."
    return " ".join(words)


TRANSFORMS = [
    _abbreviate,
    _drop_word,
    _swap_word_order,
    _punctuation_noise,
    _legal_suffix_change,
    _spelling_noise,
    _capitalization_noise,
    _extra_location,
    _partial_name,
    _initialize_first_word,
]


def simulate_dirty_entities(
    clean_entities: list[str],
    n_variants: int = 3,
    n_transforms_per_variant: int = 2,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """Generate noisy variants of clean entity names for benchmarking.

    Parameters
    ----------
    clean_entities:
        List of clean, canonical entity name strings.
    n_variants:
        Number of dirty variants to generate per clean entity.
    n_transforms_per_variant:
        Number of distinct noise transformations to chain per variant.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    DataFrame with columns: entity_id, clean_name, dirty_name, transforms
    (a comma-separated list of the transform names applied).
    """
    rng = random.Random(random_state)
    rows = []
    for entity_id, clean_name in enumerate(clean_entities):
        for _ in range(n_variants):
            name = clean_name
            chosen = rng.sample(
                TRANSFORMS, k=min(n_transforms_per_variant, len(TRANSFORMS))
            )
            applied = []
            for transform in chosen:
                new_name = transform(name, rng)
                if new_name != name:
                    applied.append(transform.__name__.lstrip("_"))
                name = new_name
            rows.append(
                {
                    "entity_id": entity_id,
                    "clean_name": clean_name,
                    "dirty_name": name,
                    "transforms": ",".join(applied) if applied else "none",
                }
            )
    return pd.DataFrame(rows, columns=["entity_id", "clean_name", "dirty_name", "transforms"])
