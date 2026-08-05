"""Benchmark fuzzy_llm_matcher against the FEBRL4 synthetic dataset.

Requires: pip install recordlinkage

FEBRL provides generated records with known true links, so this gives a
clean measurement of precision/recall/F1 on synthetic duplicate data.

Run:
    python benchmarks/febrl_test.py
"""

from __future__ import annotations

import time

import pandas as pd

from fuzzy_llm_matcher import evaluate_matches, match_tables


def main() -> None:
    try:
        import recordlinkage
    except ImportError:
        print(
            "This benchmark requires the 'recordlinkage' package.\n"
            "Install it with: pip install recordlinkage"
        )
        return

    df_a, df_b, true_links = recordlinkage.datasets.load_febrl4(return_links=True)

    df_a = df_a.reset_index().rename(columns={"rec_id": "id"})
    df_b = df_b.reset_index().rename(columns={"rec_id": "id"})

    # Build a single comparison field from given_name + surname for a
    # simple, package-agnostic text-matching benchmark.
    df_a["full_name"] = (
        df_a["given_name"].fillna("") + " " + df_a["surname"].fillna("")
    )
    df_b["full_name"] = (
        df_b["given_name"].fillna("") + " " + df_b["surname"].fillna("")
    )

    true_pairs = pd.DataFrame(list(true_links), columns=["left_id", "right_id"])

    start = time.perf_counter()
    result = match_tables(
        df_a, df_b,
        left_on="full_name", right_on="full_name",
        left_id="id", right_id="id",
        block_on=None,
        use_llm=False,
    )
    elapsed = time.perf_counter() - start

    ev = evaluate_matches(result, true_pairs, runtime_seconds=elapsed)
    print("FEBRL4 benchmark (RapidFuzz-only baseline):")
    for k, v in ev.to_dict().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
