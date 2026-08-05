import pandas as pd

from fuzzy_llm_matcher.candidate_generation import generate_candidates
from fuzzy_llm_matcher.fuzzy_scores import compute_similarity_features


def test_generate_candidates_returns_top_k():
    left = pd.DataFrame({"id": [1, 2], "name": ["Apple Inc", "Google LLC"]})
    right = pd.DataFrame(
        {"id": [10, 11, 12], "name": ["Apple Incorporated", "Alphabet Inc", "Google Inc"]}
    )
    result = generate_candidates(
        left, right, left_on="name", right_on="name",
        left_id="id", right_id="id", top_k=2,
    )
    assert set(result.columns) == {
        "left_id", "right_id", "left_value", "right_value", "score", "rank",
    }
    # 2 left rows * up to top_k=2 candidates each
    assert len(result) <= 4
    assert (result.groupby("left_id").size() <= 2).all()


def test_generate_candidates_exact_match_scores_high():
    left = pd.DataFrame({"id": [1], "name": ["Statista Strategy GmbH"]})
    right = pd.DataFrame({"id": [1], "name": ["Statista Strategy GmbH"]})
    result = generate_candidates(left, right, left_on="name", right_on="name", left_id="id", right_id="id")
    assert result.iloc[0]["score"] >= 99.0


def test_generate_candidates_with_blocking():
    left = pd.DataFrame({"id": [1, 2], "name": ["Alpha", "Beta"], "country": ["DE", "NL"]})
    right = pd.DataFrame(
        {"id": [10, 11], "name": ["Alpha Corp", "Beta Corp"], "country": ["DE", "NL"]}
    )
    result = generate_candidates(
        left, right, left_on="name", right_on="name",
        left_id="id", right_id="id", block_on="country",
    )
    # each left row should only match right rows in the same block
    assert set(result[result["left_id"] == 1]["right_id"]) == {10}
    assert set(result[result["left_id"] == 2]["right_id"]) == {11}


def test_compute_similarity_features_adds_expected_columns():
    left = pd.DataFrame({"id": [1], "name": ["Alpha Corp"]})
    right = pd.DataFrame({"id": [1], "name": ["Alpha Corporation"]})
    candidates = generate_candidates(left, right, left_on="name", right_on="name", left_id="id", right_id="id")
    scored = compute_similarity_features(candidates)
    expected = {
        "score_wratio", "score_token_sort", "score_token_set",
        "score_partial", "score_simple", "length_diff",
        "normalized_length_diff", "best_rank", "score_margin_to_second_best",
    }
    assert expected.issubset(scored.columns)
    assert scored["score_wratio"].iloc[0] > 50


def test_generate_candidates_empty_inputs():
    left = pd.DataFrame({"id": [], "name": []})
    right = pd.DataFrame({"id": [], "name": []})
    result = generate_candidates(left, right, left_on="name", right_on="name", left_id="id", right_id="id")
    assert result.empty
