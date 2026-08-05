# Abt-Buy (published external dataset) benchmark result

This extra example uses the open **Abt-Buy** dataset distributed for the
DeepMatcher SIGMOD 2018 paper benchmark collection.

- Dataset index: https://raw.githubusercontent.com/anhaidgroup/deepmatcher/master/Datasets.md
- Source zip used by the script:
  `http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Abt-Buy/abt_buy_exp_data.zip`
- Script: `benchmarks/abt_buy_paper_dataset_test.py`

## Run command

```bash
python benchmarks/abt_buy_paper_dataset_test.py
```

## Observed output (current run)

- rows_left_test_subset: `737`
- rows_right_full: `1092`
- positive_pairs_in_test: `206`
- precision: `0.1964`
- recall: `0.0534`
- f1: `0.0840`
- true_positives: `11`
- false_positives: `45`
- false_negatives: `195`
- false_confident_matches: `45`
- runtime_seconds: `51.92`

This dataset is significantly harder than the bundled sample and is intended
as a realistic external stress-test example.
