"""Compute multiple string-similarity features for candidate pairs."""

from __future__ import annotations

import pandas as pd

from .utils import fuzz, normalize_text


def compute_similarity_features(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-metric similarity columns to a candidates DataFrame.

    Expects columns: left_value, right_value (and typically left_id,
    right_id, score, rank from `generate_candidates`).

    Adds columns:
        score_wratio, score_token_sort, score_token_set,
        score_partial, score_simple, length_diff,
        normalized_length_diff, best_rank, score_margin_to_second_best
    """
    if candidates_df.empty:
        out = candidates_df.copy()
        for col in [
            "score_wratio",
            "score_token_sort",
            "score_token_set",
            "score_partial",
            "score_simple",
            "length_diff",
            "normalized_length_diff",
            "best_rank",
            "score_margin_to_second_best",
        ]:
            out[col] = pd.Series(dtype="float64")
        return out

    df = candidates_df.copy()

    left_norm = df["left_value"].map(normalize_text)
    right_norm = df["right_value"].map(normalize_text)

    df["score_wratio"] = [
        fuzz.WRatio(a, b) for a, b in zip(left_norm, right_norm)
    ]
    df["score_token_sort"] = [
        fuzz.token_sort_ratio(a, b) for a, b in zip(left_norm, right_norm)
    ]
    df["score_token_set"] = [
        fuzz.token_set_ratio(a, b) for a, b in zip(left_norm, right_norm)
    ]
    df["score_partial"] = [
        fuzz.partial_ratio(a, b) for a, b in zip(left_norm, right_norm)
    ]
    df["score_simple"] = [
        fuzz.ratio(a, b) for a, b in zip(left_norm, right_norm)
    ]

    len_left = left_norm.map(len)
    len_right = right_norm.map(len)
    df["length_diff"] = (len_left - len_right).abs()
    max_len = pd.concat([len_left, len_right], axis=1).max(axis=1).replace(0, 1)
    df["normalized_length_diff"] = df["length_diff"] / max_len

    # best_rank / score_margin_to_second_best are computed per left_id group
    if "left_id" in df.columns:
        if "score" not in df.columns:
            df["score"] = df["score_wratio"]

        df["best_rank"] = df.groupby("left_id")["score"].rank(
            ascending=False, method="first"
        )

        top_score = df.groupby("left_id")["score"].transform("max")
        second_score = df.groupby("left_id")["score"].transform(
            lambda s: float(s.nlargest(2).iloc[1]) if len(s) > 1 else 0.0
        )
        df["score_margin_to_second_best"] = top_score - second_score
    else:
        df["best_rank"] = 1
        df["score_margin_to_second_best"] = 0.0

    return df
