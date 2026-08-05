# Step-by-step guide: understand `fuzzy_llm_matcher`

This guide explains the package in the order most users need.

## 1) What this package solves

Most fuzzy match tools return a "best match" even when confidence is weak.
This package adds a **reliability layer** so you can separate:

- safe matches (`high`)
- uncertain matches that need review (`medium_review`)
- weak/rejected matches (`low`, `reject`)

The optional LLM step only reviews uncertain pairs. It does not replace
deterministic matching.

## 2) Minimal data format

You need two tables:

- **left table**: noisy records to match
- **right table**: reference records

Required columns:

- text columns to compare (`left_on`, `right_on`)
- id columns (`left_id`, `right_id`)

Example files:

- `data/sample_dirty_left.csv`
- `data/sample_dirty_right.csv`
- `data/sample_ground_truth.csv` (for evaluation)

## 3) First run (no LLM)

```python
import pandas as pd
from fuzzy_llm_matcher import match_tables

left = pd.read_csv("data/sample_dirty_left.csv")
right = pd.read_csv("data/sample_dirty_right.csv")

matches = match_tables(
    left_df=left,
    right_df=right,
    left_on="name",
    right_on="name",
    left_id="id",
    right_id="id",
    use_llm=False,
)
```

## 4) Read the output correctly

Important columns:

- `fuzzy_score`: string similarity score
- `score_margin_to_second_best`: how far best candidate is from runner-up
- `reliability_label`: `high`, `medium_review`, `low`, `reject`
- `final_decision`: accepted/rejected match decision

Interpretation:

- High score + high margin => usually safe (`high`)
- High score + low margin => ambiguous (`medium_review`)

## 5) Add LLM review only for ambiguous cases

```python
matches = match_tables(
    left_df=left,
    right_df=right,
    left_on="name",
    right_on="name",
    left_id="id",
    right_id="id",
    use_llm=True,   # reviews medium_review rows only
)
```

LLM-related columns:

- `llm_same_entity`
- `llm_confidence`
- `llm_reason`

## 6) Evaluate quality against ground truth

```python
from fuzzy_llm_matcher import evaluate_matches

truth = pd.read_csv("data/sample_ground_truth.csv")
ev = evaluate_matches(matches, truth)
print(ev.to_dict())
```

Key metrics:

- precision, recall, f1
- false positives / false negatives
- false confident matches (critical trust metric)

## 7) Tune behavior

Main knobs in `match_tables(...)`:

- `high_threshold`
- `medium_threshold`
- `min_margin_high`
- `reject_threshold`
- `top_k`

Typical direction:

- More precision: raise `high_threshold` / `min_margin_high`
- More recall: lower `high_threshold` or review more medium cases with LLM

## 8) Use blocking for scale

If both tables share a trusted field (for example `country`), use:

```python
match_tables(..., block_on="country")
```

This reduces comparisons and runtime on large datasets.

## 9) Why the new sample dataset is better

The bundled sample now includes realistic difficult cases:

- legal suffix changes (LLC, GmbH, plc)
- abbreviations (MIT, UCL, NYU, TUM)
- acronym collisions (ESA vs ESADE, WHO vs WHO Foundation)
- same-family distractors (Google vs Alphabet, Siemens vs Siemens Healthineers)

This makes demos and benchmarks much closer to real entity-resolution work.
