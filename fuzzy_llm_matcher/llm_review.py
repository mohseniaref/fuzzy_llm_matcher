"""Optional LLM-assisted review for medium-confidence candidate pairs.

The LLM is never used as the primary matcher -- it only adjudicates pairs
that the deterministic reliability layer already flagged as uncertain
(default: `reliability_label == "medium_review"`). Results are stored as
a review *signal* (`llm_same_entity`, `llm_confidence`, `llm_reason`),
not as ground truth, and are combined with the fuzzy score in
`final_decision`.

This module has zero hard dependency on any specific LLM SDK. Pass any
`client` object exposing a `.review(left_value, right_value) -> dict`
method, or leave `client=None` to use a lightweight built-in heuristic
reviewer (useful for demos/tests without API access). This keeps
`review_uncertain_pairs_with_llm` fully mockable in unit tests.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import pandas as pd

REVIEW_LABEL_DEFAULT = "medium_review"

LLM_REVIEW_PROMPT_TEMPLATE = """You are comparing two short text records to decide if they refer to the same real-world entity (e.g. the same company, product, publication, or organization), allowing for abbreviations, legal-suffix differences, punctuation, and word-order changes.

Record A: {left_value}
Record B: {right_value}

Respond with STRICT JSON only, no extra text, matching exactly this schema:
{{"same_entity": true or false, "confidence": "low" | "medium" | "high", "reason": "short explanation"}}
"""


class LLMClient(Protocol):
    """Minimal interface expected of a user-supplied LLM client."""

    def review(self, left_value: str, right_value: str) -> dict:
        ...


@dataclass
class MockLLMClient:
    """Deterministic, offline stand-in for a real LLM client.

    Uses simple heuristics (token overlap + legal-suffix stripping) so
    tests are fast, free, and reproducible. Swap in a real client (e.g.
    wrapping the Anthropic API) for production use -- the rest of the
    pipeline does not need to change.
    """

    legal_suffixes: tuple[str, ...] = (
        "inc", "inc.", "llc", "ltd", "ltd.", "gmbh", "co", "co.", "corp",
        "corp.", "b.v.", "bv", "s.a.", "sa", "plc", "pte", "kg", "gmbh & co. kg",
    )

    def _strip_suffix(self, text: str) -> str:
        tokens = text.lower().split()
        while tokens and tokens[-1].strip(".,") in [s.strip(".,") for s in self.legal_suffixes]:
            tokens.pop()
        return " ".join(tokens)

    def review(self, left_value: str, right_value: str) -> dict:
        a = self._strip_suffix(str(left_value))
        b = self._strip_suffix(str(right_value))
        ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

        if ratio >= 0.85:
            return {
                "same_entity": True,
                "confidence": "high",
                "reason": "Names match closely after removing legal suffixes.",
            }
        if ratio >= 0.6:
            return {
                "same_entity": True,
                "confidence": "medium",
                "reason": "Names are similar with minor variation; likely the same entity.",
            }
        return {
            "same_entity": False,
            "confidence": "medium",
            "reason": "Names differ substantially even after normalization.",
        }


def _parse_llm_json(raw: Any) -> dict:
    """Best-effort parse of an LLM response into the expected schema."""
    if isinstance(raw, dict):
        data = raw
    else:
        text = str(raw).strip()
        # Strip common code-fence wrapping if a real LLM added it.
        text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"same_entity": None, "confidence": "low", "reason": "unparseable LLM response"}

    return {
        "llm_same_entity": data.get("same_entity"),
        "llm_confidence": data.get("confidence"),
        "llm_reason": data.get("reason"),
    }


def review_uncertain_pairs_with_llm(
    candidates_df: pd.DataFrame,
    client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    review_labels: tuple[str, ...] = (REVIEW_LABEL_DEFAULT,),
    label_col: str = "reliability_label",
) -> pd.DataFrame:
    """Send only uncertain pairs to an LLM and return structured review results.

    Parameters
    ----------
    candidates_df:
        Output of `assign_reliability` (must contain `label_col`,
        `left_value`, `right_value`).
    client:
        Object with a `.review(left_value, right_value) -> dict` method.
        Defaults to `MockLLMClient()` if not provided, so this function
        runs fully offline out of the box.
    model:
        Optional model name, forwarded to the client if it accepts one
        (kept for API compatibility with real LLM clients; unused by the
        mock client).
    review_labels:
        Which reliability labels should be sent for LLM review.
    label_col:
        Column holding the reliability label.

    Returns
    -------
    A copy of `candidates_df` with three new columns: `llm_same_entity`,
    `llm_confidence`, `llm_reason`. Rows not selected for review get
    ``None`` in all three columns.
    """
    df = candidates_df.copy()
    for col in ("llm_same_entity", "llm_confidence", "llm_reason"):
        if col not in df.columns:
            df[col] = None

    if df.empty:
        return df

    active_client = client if client is not None else MockLLMClient()
    mask = df[label_col].isin(review_labels) if label_col in df.columns else pd.Series(False, index=df.index)

    for idx in df.index[mask]:
        left_value = df.at[idx, "left_value"]
        right_value = df.at[idx, "right_value"]
        try:
            if hasattr(active_client, "review"):
                raw = active_client.review(left_value, right_value)
            else:  # allow a bare callable as `client`
                raw = active_client(left_value, right_value)  # type: ignore
            parsed = _parse_llm_json(raw)
        except Exception as exc:  # never let one bad LLM call kill the batch
            parsed = {
                "llm_same_entity": None,
                "llm_confidence": "low",
                "llm_reason": f"LLM review failed: {exc}",
            }
        for k, v in parsed.items():
            df.at[idx, k] = v

    return df


def build_prompt(left_value: str, right_value: str) -> str:
    """Expose the prompt template for callers wiring up a real LLM client."""
    return LLM_REVIEW_PROMPT_TEMPLATE.format(left_value=left_value, right_value=right_value)
