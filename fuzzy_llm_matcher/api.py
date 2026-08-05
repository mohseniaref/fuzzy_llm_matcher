"""High-level, user-facing API."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .candidate_generation import generate_candidates
from .fuzzy_scores import compute_similarity_features
from .llm_review import LLMClient, review_uncertain_pairs_with_llm
from .reliability import assign_reliability


def match_tables(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_on: str,
    right_on: str,
    left_id: Optional[str] = None,
    right_id: Optional[str] = None,
    block_on: Optional[str] = None,
    top_k: int = 5,
    scorer: str = "WRatio",
    high_threshold: float = 92,
    medium_threshold: float = 80,
    min_margin_high: float = 8,
    reject_threshold: float = 60,
    use_llm: bool = False,
    llm_client: Optional[LLMClient] = None,
    llm_model: Optional[str] = None,
    keep_all_candidates: bool = False,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """End-to-end fuzzy + reliability (+ optional LLM) matching pipeline.

    Returns a DataFrame with one row per candidate pair (or, if
    `keep_all_candidates=False`, only the best candidate per left row)
    with columns:

        left_id, right_id, left_value, right_value, fuzzy_score,
        score_margin_to_second_best, reliability_label,
        llm_same_entity, llm_confidence, final_decision

    `final_decision` is True when reliability_label == "high", or when
    the LLM confirmed `same_entity=True` for a medium_review pair
    (only computed if `use_llm=True`).

    Parameters
    ----------
    n_jobs:
        Number of parallel threads for candidate scoring.
        ``1`` (default) = single-threaded. ``-1`` = all CPU cores.
    """
    candidates = generate_candidates(
        left_df=left_df,
        right_df=right_df,
        left_on=left_on,
        right_on=right_on,
        left_id=left_id,
        right_id=right_id,
        block_on=block_on,
        top_k=top_k,
        scorer=scorer,
        n_jobs=n_jobs,
    )

    scored = compute_similarity_features(candidates)

    labeled = assign_reliability(
        scored,
        high_threshold=high_threshold,
        medium_threshold=medium_threshold,
        min_margin_high=min_margin_high,
        reject_threshold=reject_threshold,
    )

    if use_llm:
        labeled = review_uncertain_pairs_with_llm(
            labeled, client=llm_client, model=llm_model
        )
    else:
        for col in ("llm_same_entity", "llm_confidence", "llm_reason"):
            if col not in labeled.columns:
                labeled[col] = None

    def _final_decision(row) -> bool:
        if row["reliability_label"] == "high":
            return True
        if row["reliability_label"] == "medium_review" and row.get("llm_same_entity") is True:
            return True
        return False

    labeled["final_decision"] = labeled.apply(_final_decision, axis=1)

    if not keep_all_candidates and not labeled.empty:
        labeled = labeled.sort_values(["left_id", "score"], ascending=[True, False])
        labeled = labeled.groupby("left_id", as_index=False).first()

    result_cols = [
        "left_id",
        "right_id",
        "left_value",
        "right_value",
        "score_wratio",
        "score_margin_to_second_best",
        "reliability_label",
        "llm_same_entity",
        "llm_confidence",
        "final_decision",
    ]
    result = labeled[[c for c in result_cols if c in labeled.columns]].copy()
    result = result.rename(columns={"score_wratio": "fuzzy_score"})
    return result.reset_index(drop=True)
