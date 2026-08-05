"""Benchmark on noisy company/organization names using the bundled sample
data and the synthetic dirty-data simulator.

Run:
    python benchmarks/company_name_test.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from fuzzy_llm_matcher import evaluate_matches, match_tables, simulate_dirty_entities

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def run_on_bundled_sample() -> None:
    left = pd.read_csv(DATA_DIR / "sample_dirty_left.csv")
    right = pd.read_csv(DATA_DIR / "sample_dirty_right.csv")
    true_matches = pd.read_csv(DATA_DIR / "sample_ground_truth.csv")

    start = time.perf_counter()
    result = match_tables(
        left, right,
        left_on="name", right_on="name",
        left_id="id", right_id="id",
        use_llm=True,
    )
    elapsed = time.perf_counter() - start

    ev = evaluate_matches(result, true_matches, runtime_seconds=elapsed)
    print("Bundled sample company-name benchmark:")
    print(result.to_string(index=False))
    print()
    for k, v in ev.to_dict().items():
        print(f"  {k}: {v}")


def run_on_simulated_data(n_entities: int = 30, n_variants: int = 3) -> None:
    clean_names = [
        f"Company {i} GmbH" if i % 2 == 0 else f"Institute {i} Ltd" for i in range(n_entities)
    ]
    dirty = simulate_dirty_entities(clean_names, n_variants=n_variants, random_state=42)

    left = dirty[["entity_id", "dirty_name"]].reset_index().rename(
        columns={"index": "id", "dirty_name": "name"}
    )
    right = pd.DataFrame(
        {"id": range(len(clean_names)), "entity_id": range(len(clean_names)), "name": clean_names}
    )
    true_matches = left[["id", "entity_id"]].merge(
        right[["id", "entity_id"]], on="entity_id", suffixes=("_left", "_right")
    )[["id_left", "id_right"]].rename(columns={"id_left": "left_id", "id_right": "right_id"})

    start = time.perf_counter()
    result = match_tables(
        left, right, left_on="name", right_on="name", left_id="id", right_id="id", use_llm=True,
    )
    elapsed = time.perf_counter() - start

    ev = evaluate_matches(result, true_matches, runtime_seconds=elapsed)
    print(f"\nSimulated data benchmark ({n_entities} entities x {n_variants} variants):")
    for k, v in ev.to_dict().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    run_on_bundled_sample()
    run_on_simulated_data()
