"""Benchmark on the published Abt-Buy dataset used in DeepMatcher (SIGMOD'18).

Dataset source:
https://raw.githubusercontent.com/anhaidgroup/deepmatcher/master/Datasets.md

Run:
    python benchmarks/abt_buy_paper_dataset_test.py
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from fuzzy_llm_matcher import evaluate_matches, match_tables

DATA_URL = (
    "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/"
    "Textual/Abt-Buy/abt_buy_exp_data.zip"
)


def _load_abt_buy() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load tableA, tableB, and test labels from the published zip."""
    with urlopen(DATA_URL) as resp:
        payload = resp.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        table_a = pd.read_csv(zf.open("exp_data/tableA.csv"))
        table_b = pd.read_csv(zf.open("exp_data/tableB.csv"))
        test_pairs = pd.read_csv(zf.open("exp_data/test.csv"))

    return table_a, table_b, test_pairs


def main() -> None:
    table_a, table_b, test_pairs = _load_abt_buy()

    left_ids = set(test_pairs["ltable_id"].unique())
    left = table_a[table_a["id"].isin(left_ids)][["id", "name"]].copy()
    right = table_b[["id", "name"]].copy()

    true_pairs = test_pairs[test_pairs["label"] == 1][
        ["ltable_id", "rtable_id"]
    ].rename(columns={"ltable_id": "left_id", "rtable_id": "right_id"})

    start = time.perf_counter()
    result = match_tables(
        left_df=left,
        right_df=right,
        left_on="name",
        right_on="name",
        left_id="id",
        right_id="id",
        top_k=5,
        use_llm=False,
    )
    elapsed = time.perf_counter() - start

    ev = evaluate_matches(result, true_pairs, runtime_seconds=elapsed)

    print("Abt-Buy benchmark (published DeepMatcher dataset, test split):")
    print(f"  rows_left_test_subset: {len(left)}")
    print(f"  rows_right_full: {len(right)}")
    print(f"  positive_pairs_in_test: {len(true_pairs)}")
    for k, v in ev.to_dict().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
