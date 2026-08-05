"""Generate candidate matches between two pandas DataFrames.

Parallelization
---------------
Pass ``n_jobs > 1`` (or ``-1`` for all CPU cores) to split the left table
into chunks scored in parallel via ``concurrent.futures.ThreadPoolExecutor``.
rapidfuzz releases the GIL during scoring so threads give a real speedup.

When rapidfuzz is installed each left row is scored with the vectorized
``rapidfuzz.process.extract`` (a single C-level call), which is 10-100x
faster than a Python loop even before any parallelism.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd

from .utils import HAVE_RAPIDFUZZ, get_scorer, normalize_text

if HAVE_RAPIDFUZZ:
    from rapidfuzz import process as _rf_process  # type: ignore


def _score_one_row(
    left_id_val,
    left_val: str,
    left_norm: str,
    r_ids: list,
    r_vals: list,
    r_norms: list,
    score_fn,
    top_k: int,
) -> list[dict]:
    """Score a single left row against candidate right rows, return top-k."""
    if HAVE_RAPIDFUZZ:
        hits = _rf_process.extract(left_norm, r_norms, scorer=score_fn, limit=top_k)
        return [
            {
                "left_id": left_id_val,
                "right_id": r_ids[idx],
                "left_value": left_val,
                "right_value": r_vals[idx],
                "score": score,
                "rank": rank,
            }
            for rank, (_, score, idx) in enumerate(hits, start=1)
        ]

    # pure-Python fallback
    scored = sorted(
        [(score_fn(left_norm, rn), rid, rv) for rn, rid, rv in zip(r_norms, r_ids, r_vals)],
        key=lambda t: t[0],
        reverse=True,
    )
    return [
        {"left_id": left_id_val, "right_id": rid, "left_value": left_val,
         "right_value": rv, "score": s, "rank": rank}
        for rank, (s, rid, rv) in enumerate(scored[:top_k], start=1)
    ]


def _process_chunk(
    chunk: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    right_groups: Optional[dict],
    score_fn,
    top_k: int,
) -> list[dict]:
    r_ids_all = right["_right_id"].tolist()
    r_vals_all = right[right_on].tolist()
    r_norms_all = right["_right_norm"].tolist()

    rows: list[dict] = []
    for _, lrow in chunk.iterrows():
        if right_groups is not None:
            cand = right_groups.get(lrow["_block"])
            if cand is None or len(cand) == 0:
                continue
            r_ids, r_vals, r_norms = (
                cand["_right_id"].tolist(), cand[right_on].tolist(), cand["_right_norm"].tolist()
            )
        else:
            r_ids, r_vals, r_norms = r_ids_all, r_vals_all, r_norms_all

        rows.extend(_score_one_row(
            lrow["_left_id"], lrow[left_on], lrow["_left_norm"],
            r_ids, r_vals, r_norms, score_fn, top_k,
        ))
    return rows


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
    n_jobs: int = 1,
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
    n_jobs:
        Number of parallel threads.  ``1`` (default) = single-threaded.
        ``-1`` = use all CPU cores. rapidfuzz releases the GIL so threads
        give a real speedup even on CPython.

    Returns
    -------
    pd.DataFrame with columns:
        left_id, right_id, left_value, right_value, score, rank
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

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
        right_groups = {key: grp for key, grp in right.groupby("_block")}
    else:
        right_groups = None

    if n_jobs == 1 or len(left) < 2:
        rows = _process_chunk(left, right, left_on, right_on, right_groups, score_fn, top_k)
    else:
        chunk_size = max(1, math.ceil(len(left) / n_jobs))
        chunks = [left.iloc[i:i + chunk_size] for i in range(0, len(left), chunk_size)]
        rows = []
        with ThreadPoolExecutor(max_workers=n_jobs) as exc:
            futures = [
                exc.submit(_process_chunk, chunk, right, left_on, right_on,
                           right_groups, score_fn, top_k)
                for chunk in chunks
            ]
            for fut in as_completed(futures):
                rows.extend(fut.result())

    columns = ["left_id", "right_id", "left_value", "right_value", "score", "rank"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)
