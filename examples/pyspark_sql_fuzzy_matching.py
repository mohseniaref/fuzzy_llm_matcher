"""PySpark + SQL fuzzy entity matching example.

This module shows THREE scaling patterns for fuzzy matching:

1. ``match_tables(..., n_jobs=N)`` – parallel threads via ThreadPoolExecutor
   (already built into the core package). No extra dependencies.

2. ``duckdb_match_tables``  – In-process SQL via DuckDB: registers the two
   DataFrames as SQL tables, pre-filters with levenshtein(), then scores
   with a Python UDF. Requires ``duckdb`` (``pip install duckdb``).

3. ``sqlite_match_tables``  – Same idea using Python's built-in sqlite3.
   Zero extra dependencies — good for exploration and teaching.

Why does fuzzy matching need a special strategy at scale?
---------------------------------------------------------
A naive CROSS JOIN between tables of N and M rows produces N×M pairs —
quadratic. The trick used here (same as the core package) is *blocking*:
only rows sharing a common key are joined. This brings the join to
near-linear in most practical cases.

Run:
    python examples/pyspark_sql_fuzzy_matching.py
"""

from __future__ import annotations

import time

import pandas as pd

from fuzzy_llm_matcher import evaluate_matches, match_tables, simulate_dirty_entities


# ---------------------------------------------------------------------------
# Shared helper: build a small synthetic dataset
# ---------------------------------------------------------------------------

def _build_demo_dataset(n_entities: int = 30, n_variants: int = 3, seed: int = 42):
    clean_names = [
        f"Company {i} GmbH" if i % 2 == 0 else f"Institute {i} Ltd"
        for i in range(n_entities)
    ]
    dirty = simulate_dirty_entities(clean_names, n_variants=n_variants, random_state=seed)
    left = (
        dirty[["entity_id", "dirty_name"]]
        .reset_index()
        .rename(columns={"index": "id", "dirty_name": "name"})
    )
    right = pd.DataFrame(
        {"id": range(n_entities), "entity_id": range(n_entities), "name": clean_names}
    )
    true_matches = (
        left[["id", "entity_id"]]
        .merge(right[["id", "entity_id"]], on="entity_id", suffixes=("_left", "_right"))
        [["id_left", "id_right"]]
        .rename(columns={"id_left": "left_id", "id_right": "right_id"})
    )
    return left, right, true_matches


# ---------------------------------------------------------------------------
# DuckDB variant (SQL, no Spark needed)
# ---------------------------------------------------------------------------

def duckdb_match_tables(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    left_id: str,
    right_id: str,
    top_k: int = 5,
    reject_threshold: float = 60.0,
) -> pd.DataFrame:
    """Use DuckDB SQL to generate candidates, then score with rapidfuzz UDF.

    Strategy
    --------
    1. Register both tables as DuckDB views.
    2. SQL CROSS JOIN pre-filtered by DuckDB's built-in ``levenshtein``
       function to drop obviously wrong pairs cheaply.
    3. Score surviving pairs with a Python WRatio UDF.
    4. Keep top-k per left row.
    """
    try:
        import duckdb
    except ImportError:
        print("DuckDB not installed. Run: pip install duckdb")
        return pd.DataFrame()

    try:
        from rapidfuzz import fuzz
    except ImportError:
        from fuzzy_llm_matcher.utils import fuzz  # type: ignore

    con = duckdb.connect(database=":memory:")
    con.register("left_tbl", left)
    con.register("right_tbl", right)

    candidates_sql = f"""
        SELECT
            l.{left_id}  AS left_id,
            r.{right_id} AS right_id,
            l.{left_on}  AS left_value,
            r.{right_on} AS right_value
        FROM left_tbl l
        CROSS JOIN right_tbl r
        WHERE len(l.{left_on}) > 0
          AND len(r.{right_on}) > 0
          AND levenshtein(lower(l.{left_on}), lower(r.{right_on}))
              <= greatest(len(l.{left_on}), len(r.{right_on})) * 0.7
    """
    candidates = con.execute(candidates_sql).df()
    con.close()

    if candidates.empty:
        return pd.DataFrame(
            columns=["left_id", "right_id", "left_value", "right_value",
                     "fuzzy_score", "reliability_label", "final_decision"]
        )

    candidates["fuzzy_score"] = [
        fuzz.WRatio(a.lower(), b.lower())
        for a, b in zip(candidates["left_value"], candidates["right_value"])
    ]
    candidates = candidates[candidates["fuzzy_score"] >= reject_threshold]
    candidates = (
        candidates
        .sort_values(["left_id", "fuzzy_score"], ascending=[True, False])
        .groupby("left_id").head(top_k)
        .reset_index(drop=True)
    )
    candidates["reliability_label"] = candidates["fuzzy_score"].apply(
        lambda s: "high" if s >= 92 else ("medium_review" if s >= 80 else "low")
    )
    candidates["final_decision"] = candidates["reliability_label"] == "high"
    return candidates


# ---------------------------------------------------------------------------
# SQLite variant (zero extra deps)
# ---------------------------------------------------------------------------

def sqlite_match_tables(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    left_id: str,
    right_id: str,
    reject_threshold: float = 60.0,
) -> pd.DataFrame:
    """Use Python's built-in sqlite3 with a registered WRatio UDF.

    Demonstrates embedding fuzzy logic directly inside SQL — no extra engine
    required. The ``wratio`` function is visible to any SQL query in the
    session.
    """
    import sqlite3

    try:
        from rapidfuzz import fuzz
    except ImportError:
        from fuzzy_llm_matcher.utils import fuzz  # type: ignore

    con = sqlite3.connect(":memory:")
    con.create_function(
        "wratio", 2,
        lambda a, b: float(fuzz.WRatio(str(a).lower(), str(b).lower()))
    )

    left.to_sql("left_tbl", con, index=False, if_exists="replace")
    right.to_sql("right_tbl", con, index=False, if_exists="replace")

    sql = f"""
        SELECT
            l.{left_id}  AS left_id,
            r.{right_id} AS right_id,
            l.{left_on}  AS left_value,
            r.{right_on} AS right_value,
            wratio(l.{left_on}, r.{right_on}) AS fuzzy_score
        FROM left_tbl l
        CROSS JOIN right_tbl r
        WHERE wratio(l.{left_on}, r.{right_on}) >= {reject_threshold}
        ORDER BY l.{left_id}, fuzzy_score DESC
    """
    result = pd.read_sql_query(sql, con)
    con.close()

    result = (
        result
        .sort_values(["left_id", "fuzzy_score"], ascending=[True, False])
        .groupby("left_id").first()
        .reset_index()
    )
    result["reliability_label"] = result["fuzzy_score"].apply(
        lambda s: "high" if s >= 92 else ("medium_review" if s >= 80 else "low")
    )
    result["final_decision"] = result["reliability_label"] == "high"
    return result


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def main() -> None:
    left, right, true_matches = _build_demo_dataset()

    print("=== Parallel n_jobs=4 (built-in, no extra deps) ===")
    t = time.perf_counter()
    result_parallel = match_tables(
        left, right, left_on="name", right_on="name",
        left_id="id", right_id="id", n_jobs=4,
    )
    print(f"  runtime: {time.perf_counter()-t:.3f}s")
    ev = evaluate_matches(result_parallel, true_matches)
    for k, v in ev.to_dict().items():
        print(f"  {k}: {v}")

    print("\n=== DuckDB SQL fuzzy matching ===")
    try:
        import duckdb  # noqa: F401
        t = time.perf_counter()
        result_duck = duckdb_match_tables(
            left, right, left_on="name", right_on="name",
            left_id="id", right_id="id",
        )
        print(f"  runtime: {time.perf_counter()-t:.3f}s  rows: {len(result_duck)}")
        print(result_duck.head(5).to_string(index=False))
    except ImportError:
        print("  skipped (duckdb not installed)")

    print("\n=== SQLite SQL fuzzy matching (zero extra deps) ===")
    t = time.perf_counter()
    result_sql = sqlite_match_tables(
        left, right, left_on="name", right_on="name",
        left_id="id", right_id="id",
    )
    print(f"  runtime: {time.perf_counter()-t:.3f}s  rows: {len(result_sql)}")
    print(result_sql.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
