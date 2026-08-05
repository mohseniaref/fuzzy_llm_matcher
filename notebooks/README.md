# Notebooks and reports

This folder contains interactive notebooks and generated validation reports.

## Contents

- `01_febrl_benchmark.ipynb`: FEBRL-style benchmarking walkthrough
- `02_company_name_matching.ipynb`: bundled company-name matching demo
- `03_llm_review_demo.ipynb`: reliability + LLM-review demonstration
- `results_synthetic_validation.md`: generated markdown report with benchmark figures
- `figures/`: generated SVG plots used by the markdown report

## Reproduce the synthetic validation report

From the repository root:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import pandas as pd
from fuzzy_llm_matcher import evaluate_matches, match_tables, simulate_dirty_entities

out_dir = Path('notebooks/figures')
out_dir.mkdir(parents=True, exist_ok=True)

n_entities, n_variants, seed = 25, 3, 7
clean_names = [f"Research Group {i} University" for i in range(n_entities)]
dirty = simulate_dirty_entities(clean_names, n_variants=n_variants, random_state=seed)
left = dirty[["entity_id", "dirty_name"]].reset_index().rename(columns={"index": "id", "dirty_name": "name"})
right = pd.DataFrame({"id": range(n_entities), "entity_id": range(n_entities), "name": clean_names})
true_matches = left[["id", "entity_id"]].merge(
    right[["id", "entity_id"]], on="entity_id", suffixes=("_left", "_right")
)[["id_left", "id_right"]].rename(columns={"id_left": "left_id", "id_right": "right_id"})

strategies = [
    ("RapidFuzz only", False, True),
    ("RapidFuzz + reliability", False, False),
    ("RapidFuzz + reliability + LLM", True, False),
]

rows = []
for name, use_llm, rapidfuzz_only in strategies:
    result = match_tables(left, right, left_on="name", right_on="name", left_id="id", right_id="id", use_llm=use_llm)
    if rapidfuzz_only:
        result = result.copy()
        result["final_decision"] = result["reliability_label"] != "reject"
    ev = evaluate_matches(result, true_matches)
    rows.append({"strategy": name, **ev.to_dict()})

metrics = pd.DataFrame(rows)
metrics.to_csv("notebooks/synthetic_benchmark_metrics.csv", index=False)
print(metrics[["strategy", "precision", "recall", "f1"]])
PY
```

Then open `results_synthetic_validation.md` to view the report and figures.
