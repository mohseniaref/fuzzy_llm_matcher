"""Optional baseline comparison against Splink.

Splink is NOT a dependency of fuzzy_llm_matcher -- this script is only
useful if you already have it installed and want a second opinion on
benchmark accuracy.

Requires: pip install splink

Run:
    python benchmarks/splink_baseline.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fuzzy_llm_matcher import evaluate_matches, match_tables

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def run_fuzzy_llm_matcher():
    left = pd.read_csv(DATA_DIR / "sample_dirty_left.csv")
    right = pd.read_csv(DATA_DIR / "sample_dirty_right.csv")
    true_matches = pd.read_csv(DATA_DIR / "sample_ground_truth.csv")

    result = match_tables(
        left, right, left_on="name", right_on="name",
        left_id="id", right_id="id", use_llm=True,
    )
    ev = evaluate_matches(result, true_matches)
    print("fuzzy_llm_matcher:")
    for k, v in ev.to_dict().items():
        print(f"  {k}: {v}")


def run_splink():
    try:
        import splink  # noqa: F401
    except ImportError:
        print(
            "\nSplink is not installed -- skipping baseline comparison.\n"
            "Install it with: pip install splink"
        )
        return

    print(
        "\nSplink baseline: see https://moj-analytical-services.github.io/splink/ "
        "for setting up a linking model on data/sample_dirty_left.csv and "
        "data/sample_dirty_right.csv. This script intentionally does not "
        "hard-code a Splink model, since good Splink configs are dataset-"
        "specific (blocking rules, comparison levels, etc.)."
    )


if __name__ == "__main__":
    run_fuzzy_llm_matcher()
    run_splink()
