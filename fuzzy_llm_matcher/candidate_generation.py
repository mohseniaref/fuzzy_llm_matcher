"""Generate candidate matches between two pandas DataFrames."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .utils import get_scorer, normalize_text


def generate_candidates(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_on: str,
    right_on: str,
    left_id: Optional[str] = None,
    right_id: Optional[str] = None,
    block_on: Optional[str] = None,
    top_k: int = 5,
    scorer: str = "WRatio",
) -> pd.DataFrame:
    """Return top-k candidate matches per left row, scored with a fuzzy scorer.

    Parameters
    ----------
    left_df, right_df:
        Input DataFrames to match between.
    left_on, right_on:
        Column names holding the text to compare.
    left_id, right_id:
        Optional column names holding stable record IDs. If omitted, the
        DataFrame index is used.
    block_on:
        Optional column name (present in both DataFrames with the same
        name, or a tuple of two names) used to restrict comparisons to
        rows that share the same normalized block value. Pass ``None`` to
        compare every left row against every right row (fine for small
        datasets, expensive for large ones).
    top_k:
        Number of best right-side candidates to keep per left row.
    scorer:
        One of ``"WRatio"``, ``"ratio"``, ``"partial_ratio"``,
        ``"token_sort_ratio"``, ``"token_set_ratio"``.

    Returns
    -------
    pd.DataFrame with columns:
        left_id, right_id, left_value, right_value, score, rank
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    score_fn = get_scorer(scorer)

    left = left_df.copy()
    right = right_df.copy()

    left["_left_id"] = left[left_id] if left_id else left.index
    right["_right_id"] = right[right_id] if right_id else right.index

    left["_left_norm"] = left[left_on].map(normalize_text)
    right["_right_norm"] = right[right_on].map(normalize_text)

    if block_on is not None:
        left_block_col, right_block_col = (
            block_on if isinstance(block_on, (tuple, list)) else (block_on, block_on)
        )
        left["_block"] = left[left_block_col].map(normalize_text)
        right["_block"] = right[right_block_col].map(normalize_text)
        right_groups = {
            key: grp for key, grp in right.groupby("_block")
        }
    else:
        right_groups = None

    rows = []
    for _, lrow in left.iterrows():
        left_val = lrow[left_on]
        left_norm = lrow["_left_norm"]

        if right_groups is not None:
            candidates_right = right_groups.get(lrow["_block"])
            if candidates_right is None or len(candidates_right) == 0:
                continue
        else:
            candidates_right = right

        scored = []
        for _, rrow in candidates_right.iterrows():
            s = score_fn(left_norm, rrow["_right_norm"])
            scored.append((s, rrow["_right_id"], rrow[right_on]))

        if not scored:
            continue

        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:top_k]

        for rank, (s, right_id_val, right_val) in enumerate(top, start=1):
            rows.append(
                {
                    "left_id": lrow["_left_id"],
                    "right_id": right_id_val,
                    "left_value": left_val,
                    "right_value": right_val,
                    "score": s,
                    "rank": rank,
                }
            )

    columns = ["left_id", "right_id", "left_value", "right_value", "score", "rank"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)
