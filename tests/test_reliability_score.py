import pandas as pd

from fuzzy_llm_matcher.reliability import assign_reliability, false_confident_matches


def _make_df(rows):
    return pd.DataFrame(rows)


def test_high_label_requires_score_and_margin():
    df = _make_df([
        {"left_id": 1, "right_id": 10, "score_wratio": 95, "score_margin_to_second_best": 10},
        {"left_id": 2, "right_id": 20, "score_wratio": 95, "score_margin_to_second_best": 1},
    ])
    out = assign_reliability(df, high_threshold=92, min_margin_high=8)
    assert out.iloc[0]["reliability_label"] == "high"
    # score clears high_threshold but margin is too small -> not "high"
    assert out.iloc[1]["reliability_label"] != "high"


def test_medium_and_low_and_reject_bins():
    df = _make_df([
        {"left_id": 1, "right_id": 10, "score_wratio": 85, "score_margin_to_second_best": 20},
        {"left_id": 2, "right_id": 20, "score_wratio": 65, "score_margin_to_second_best": 20},
        {"left_id": 3, "right_id": 30, "score_wratio": 40, "score_margin_to_second_best": 20},
    ])
    out = assign_reliability(df, high_threshold=92, medium_threshold=80, reject_threshold=60)
    assert out.iloc[0]["reliability_label"] == "medium_review"
    assert out.iloc[1]["reliability_label"] == "low"
    assert out.iloc[2]["reliability_label"] == "reject"


def test_assign_reliability_empty_df():
    df = pd.DataFrame(columns=["score_wratio", "score_margin_to_second_best"])
    out = assign_reliability(df)
    assert out.empty
    assert "reliability_label" in out.columns


def test_false_confident_matches():
    df = _make_df([
        {"left_id": 1, "right_id": 10, "reliability_label": "high", "is_correct": True},
        {"left_id": 2, "right_id": 20, "reliability_label": "high", "is_correct": False},
        {"left_id": 3, "right_id": 30, "reliability_label": "medium_review", "is_correct": False},
    ])
    fcm = false_confident_matches(df, correct_col="is_correct")
    assert len(fcm) == 1
    assert fcm.iloc[0]["left_id"] == 2


def test_assign_reliability_missing_score_col_raises():
    df = _make_df([{"left_id": 1, "right_id": 10}])
    try:
        assign_reliability(df, score_col="does_not_exist")
        assert False, "expected KeyError"
    except KeyError:
        pass
