import pandas as pd

from fuzzy_llm_matcher.llm_review import (
    MockLLMClient,
    review_uncertain_pairs_with_llm,
)


def test_only_uncertain_pairs_are_sent_to_llm():
    df = pd.DataFrame([
        {"left_id": 1, "right_id": 10, "left_value": "Statista GmbH", "right_value": "Statista GmbH & Co. KG", "reliability_label": "high"},
        {"left_id": 2, "right_id": 20, "left_value": "Acme Corp", "right_value": "Acme Widgets", "reliability_label": "medium_review"},
        {"left_id": 3, "right_id": 30, "left_value": "Zzz", "right_value": "Qqq", "reliability_label": "reject"},
    ])
    out = review_uncertain_pairs_with_llm(df)
    # only the medium_review row should have been reviewed
    assert out.loc[out["left_id"] == 1, "llm_same_entity"].isna().all()
    assert out.loc[out["left_id"] == 3, "llm_same_entity"].isna().all()
    assert out.loc[out["left_id"] == 2, "llm_same_entity"].notna().all()


def test_mock_llm_client_direct():
    client = MockLLMClient()
    result = client.review("Statista Strategy GmbH", "Statista Strategy GmbH & Co. KG")
    assert result["same_entity"] is True
    assert result["confidence"] in {"high", "medium"}
    assert "reason" in result


def test_custom_client_is_used():
    class AlwaysTrueClient:
        def review(self, left_value, right_value):
            return {"same_entity": True, "confidence": "high", "reason": "test stub"}

    df = pd.DataFrame([
        {"left_id": 1, "right_id": 10, "left_value": "A", "right_value": "B", "reliability_label": "medium_review"},
    ])
    out = review_uncertain_pairs_with_llm(df, client=AlwaysTrueClient())
    assert out.iloc[0]["llm_same_entity"] is True
    assert out.iloc[0]["llm_reason"] == "test stub"


def test_llm_review_handles_broken_client_gracefully():
    class BrokenClient:
        def review(self, left_value, right_value):
            raise RuntimeError("boom")

    df = pd.DataFrame([
        {"left_id": 1, "right_id": 10, "left_value": "A", "right_value": "B", "reliability_label": "medium_review"},
    ])
    out = review_uncertain_pairs_with_llm(df, client=BrokenClient())
    assert out.iloc[0]["llm_same_entity"] is None
    assert "failed" in out.iloc[0]["llm_reason"]


def test_review_empty_df():
    df = pd.DataFrame(columns=["left_id", "right_id", "left_value", "right_value", "reliability_label"])
    out = review_uncertain_pairs_with_llm(df)
    assert out.empty
