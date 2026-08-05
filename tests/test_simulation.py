from fuzzy_llm_matcher.simulation import simulate_dirty_entities


def test_simulate_dirty_entities_shape():
    clean = ["Alpha Corp", "Beta LLC", "Gamma University"]
    df = simulate_dirty_entities(clean, n_variants=4, random_state=1)
    assert len(df) == len(clean) * 4
    assert set(df.columns) == {"entity_id", "clean_name", "dirty_name", "transforms"}
    assert set(df["entity_id"].unique()) == {0, 1, 2}


def test_simulate_dirty_entities_reproducible_with_seed():
    clean = ["Alpha Corp", "Beta LLC"]
    df1 = simulate_dirty_entities(clean, n_variants=3, random_state=99)
    df2 = simulate_dirty_entities(clean, n_variants=3, random_state=99)
    assert df1["dirty_name"].tolist() == df2["dirty_name"].tolist()


def test_simulate_dirty_entities_produces_variation():
    clean = ["Statista Strategy GmbH"]
    df = simulate_dirty_entities(clean, n_variants=10, random_state=5)
    # with 10 variants and randomized transforms, expect some diversity
    assert df["dirty_name"].nunique() > 1


def test_simulate_dirty_entities_empty_input():
    df = simulate_dirty_entities([], n_variants=3)
    assert df.empty
