"""Evaluate predicted matches against ground truth."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class EvaluationResult:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    false_confident_matches: int
    n_pairs_sent_to_llm: int
    estimated_llm_cost_per_1000_rows: float
    runtime_seconds: float
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "false_confident_matches": self.false_confident_matches,
            "n_pairs_sent_to_llm": self.n_pairs_sent_to_llm,
            "estimated_llm_cost_per_1000_rows": self.estimated_llm_cost_per_1000_rows,
            "runtime_seconds": self.runtime_seconds,
            **self.details,
        }


def evaluate_matches(
    predicted_matches: pd.DataFrame,
    true_matches: pd.DataFrame,
    left_id_col: str = "left_id",
    right_id_col: str = "right_id",
    accept_col: Optional[str] = "final_decision",
    accept_values: tuple = (True, "accept", "match"),
    confident_label_col: str = "reliability_label",
    confident_labels: tuple = ("high",),
    llm_col: str = "llm_same_entity",
    cost_per_llm_call: float = 0.001,
    runtime_seconds: Optional[float] = None,
) -> EvaluationResult:
    """Compute precision/recall/F1 and reliability-layer diagnostics.

    Parameters
    ----------
    predicted_matches:
        DataFrame of predicted pairs (e.g. output of `match_tables`).
        Rows are only counted as "predicted matches" if `accept_col` is
        missing (all rows counted) or its value is in `accept_values`.
    true_matches:
        Ground-truth DataFrame with at least `left_id_col`/`right_id_col`
        columns identifying true matching pairs.
    confident_label_col, confident_labels:
        Used to compute `false_confident_matches`: predicted pairs
        labeled confident (default "high") that are not in the true set.
    llm_col:
        If present in `predicted_matches`, non-null values are counted
        towards `n_pairs_sent_to_llm`.
    cost_per_llm_call:
        Used only to estimate a rough LLM cost per 1,000 rows processed.
    runtime_seconds:
        If not supplied, defaults to 0.0 (pass this in explicitly when
        timing an actual run -- see `benchmarks/` for examples).
    """
    start = time.perf_counter()

    pred = predicted_matches.copy()
    if accept_col and accept_col in pred.columns:
        pred = pred[pred[accept_col].isin(accept_values)]

    pred_pairs = set(zip(pred[left_id_col], pred[right_id_col]))
    true_pairs = set(zip(true_matches[left_id_col], true_matches[right_id_col]))

    true_positives = len(pred_pairs & true_pairs)
    false_positives = len(pred_pairs - true_pairs)
    false_negatives = len(true_pairs - pred_pairs)

    precision = true_positives / len(pred_pairs) if pred_pairs else 0.0
    recall = true_positives / len(true_pairs) if true_pairs else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    false_confident_matches = 0
    if confident_label_col in predicted_matches.columns:
        confident_rows = predicted_matches[
            predicted_matches[confident_label_col].isin(confident_labels)
        ]
        confident_pairs = set(zip(confident_rows[left_id_col], confident_rows[right_id_col]))
        false_confident_matches = len(confident_pairs - true_pairs)

    n_pairs_sent_to_llm = 0
    if llm_col in predicted_matches.columns:
        n_pairs_sent_to_llm = int(predicted_matches[llm_col].notna().sum())

    total_rows = max(1, len(predicted_matches))
    estimated_llm_cost_per_1000_rows = (
        (n_pairs_sent_to_llm / total_rows) * 1000 * cost_per_llm_call
    )

    elapsed = runtime_seconds if runtime_seconds is not None else (time.perf_counter() - start)

    return EvaluationResult(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        false_confident_matches=false_confident_matches,
        n_pairs_sent_to_llm=n_pairs_sent_to_llm,
        estimated_llm_cost_per_1000_rows=estimated_llm_cost_per_1000_rows,
        runtime_seconds=elapsed,
    )
