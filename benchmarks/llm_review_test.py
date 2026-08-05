"""Compare RapidFuzz-only, +reliability, and +LLM-review strategies on the
same simulated dataset, reporting precision/recall/F1/false-confident
matches for each.

Run:
    python benchmarks/llm_review_test.py
"""

from __future__ import annotations

import pandas as pd

from fuzzy_llm_matcher import evaluate_matches, match_tables, simulate_dirty_entities


def build_dataset(n_entities: int = 25, n_variants: int = 3, seed: int = 7):
    clean_names = [f"Research Group {i} University" for i in range(n_entities)]
    dirty = simulate_dirty_entities(clean_names, n_variants=n_variants, random_state=seed)

    left = dirty[["entity_id", "dirty_name"]].reset_index().rename(
        columns={"index": "id", "dirty_name": "name"}
    )
    right = pd.DataFrame(
        {"id": range(n_entities), "entity_id": range(n_entities), "name": clean_names}
    )
    true_matches = left[["id", "entity_id"]].merge(
        right[["id", "entity_id"]], on="entity_id", suffixes=("_left", "_right")
    )[["id_left", "id_right"]].rename(columns={"id_left": "left_id", "id_right": "right_id"})
    return left, right, true_matches


def main() -> None:
    left, right, true_matches = build_dataset()

    strategies = {
        "RapidFuzz only (accept everything above reject threshold)": dict(use_llm=False),
        "RapidFuzz + reliability (accept only 'high')": dict(use_llm=False),
        "RapidFuzz + reliability + LLM review": dict(use_llm=True),
    }

    for name, kwargs in strategies.items():
        result = match_tables(
            left, right, left_on="name", right_on="name",
            left_id="id", right_id="id", **kwargs,
        )
        if name.startswith("RapidFuzz only"):
            # treat every non-reject candidate as accepted, to show the
            # cost of skipping the reliability layer entirely
            result = result.copy()
            result["final_decision"] = result["reliability_label"] != "reject"

        ev = evaluate_matches(result, true_matches)
        print(f"\n{name}")
        for k, v in ev.to_dict().items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
