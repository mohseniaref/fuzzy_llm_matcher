"""Assign reliability labels to candidate matches.

Labels
------
high         : strong score and a clear margin over the next-best candidate.
medium_review: plausible but not clearly separated from a runner-up, or a
               mid-strength score -- worth a second look (human or LLM).
low          : weak score, unlikely to be correct but not rejected outright.
reject       : below the minimum score threshold.
"""

from __future__ import annotations

import pandas as pd

REJECT_THRESHOLD_DEFAULT = 60


def assign_reliability(
    candidates_df: pd.DataFrame,
    high_threshold: float = 92,
    medium_threshold: float = 80,
    min_margin_high: float = 8,
    reject_threshold: float = REJECT_THRESHOLD_DEFAULT,
    score_col: str = "score_wratio",
) -> pd.DataFrame:
    """Add a `reliability_label` column to a scored candidates DataFrame.

    Parameters
    ----------
    candidates_df:
        Output of `compute_similarity_features` (must contain `score_col`
        and, ideally, `score_margin_to_second_best`).
    high_threshold, medium_threshold, reject_threshold:
        Score cutoffs (0-100 scale).
    min_margin_high:
        Minimum margin over the second-best candidate required for a
        `high` label, even if the score itself clears `high_threshold`.
        This catches cases where two candidates are both excellent
        matches and the pick is therefore ambiguous.
    score_col:
        Which similarity column to use as the primary score. Defaults to
        the combined WRatio score.
    """
    if candidates_df.empty:
        out = candidates_df.copy()
        out["reliability_label"] = pd.Series(dtype="object")
        return out

    df = candidates_df.copy()

    if score_col not in df.columns:
        raise KeyError(
            f"'{score_col}' not found in candidates_df. "
            "Did you run compute_similarity_features() first?"
        )

    margin = df["score_margin_to_second_best"] if "score_margin_to_second_best" in df.columns else pd.Series(
        [float("inf")] * len(df), index=df.index
    )

    def _label(score: float, margin_val: float) -> str:
        if score < reject_threshold:
            return "reject"
        if score >= high_threshold and margin_val >= min_margin_high:
            return "high"
        if score >= medium_threshold:
            return "medium_review"
        if score >= reject_threshold:
            return "low"
        return "reject"

    df["reliability_label"] = [
        _label(s, m) for s, m in zip(df[score_col], margin)
    ]

    return df


def false_confident_matches(
    df: pd.DataFrame,
    correct_col: str,
    label_col: str = "reliability_label",
    confident_labels: tuple[str, ...] = ("high",),
) -> pd.DataFrame:
    """Return rows labeled confident (default: 'high') that are actually wrong.

    `correct_col` should be a boolean column indicating whether the
    predicted pair is a true match (typically produced during evaluation
    against ground truth).
    """
    mask = df[label_col].isin(confident_labels) & (~df[correct_col].astype(bool))
    return df[mask]
