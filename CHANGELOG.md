# Changelog

All notable changes to this project are documented in this file.

## [Unreleased] — 2026-08-06

### Added — Geospatial matching extensions

**Core library**
- `fuzzy_scores.py`:
  - `haversine_km(lat1, lon1, lat2, lon2)` — great-circle distance
  - `geo_distance_score(...)` — converts distance to 0–100 similarity score
  - `add_geo_distance_score(df, max_km)` — adds `score_geo_distance` column
    to any candidates DataFrame that carries lat/lon columns
- `llm_review.py`:
  - `LLM_REVIEW_GEO_PROMPT_TEMPLATE` — geo-aware prompt template including
    exact coordinates and haversine distance
  - `build_prompt(left, right, geo_context=None)` — selects geo template
    when `geo_context` is supplied
  - `MockLLMClient.review(geo_context=None)` — proximity boost from context
  - `review_uncertain_pairs_with_llm` now auto-detects `left_lat/lon/right_lat/lon`
    columns and injects `geo_context` into every uncertain-pair LLM call
- `api.py`:
  - `match_geodataframes()` — end-to-end GeoDataFrame matching pipeline:
    centroid extraction, attribute or spatial blocking, geo-distance score,
    geo-aware LLM review, geometry join-back. Accepts plain DataFrame on
    right side (CSV + shapefile pattern).
  - `fuzzy_join()` — fuzzy `pd.merge()` for plain DataFrames;
    `how=inner/left/all`
  - `fuzzy_join_geodataframes()` — fuzzy `gpd.sjoin()`: spatial/attribute
    blocking + geo score + attribute join + geometry attachment (`left/right/both`)
  - `fuzzy_dissolve()` — match + merge geometries of confirmed pairs via
    `union/intersection/envelope/centroid/left/right`; supports `aggfunc`
  - `block_on` parameter added to `match_geodataframes()`,
    `fuzzy_join_geodataframes()`, `fuzzy_dissolve()` for attribute-column
    blocking (takes priority over spatial grid blocking)
  - All geo functions accept a plain `pd.DataFrame` on the right side

**New examples**
- `examples/geo_spatial_confidence.py` — `score_geo_distance` + geo-aware LLM
  on 40-city hard-case dataset; 4 figures
- `examples/geo_match_geodataframes.py` — full `match_geodataframes()` workflow;
  GeoJSON export; 4 figures including spatial-blocking grid map
- `examples/geo_fuzzy_join_dissolve.py` — `fuzzy_join()`, `fuzzy_join_geodataframes()`,
  `fuzzy_dissolve()` with all four dissolve_op modes; 4 figures
- `examples/geo_srilanka_admin_join.py` — Sri Lanka GN Division admin join,
  inspired by https://www.geopythontutorials.com/notebooks/geopandas_fuzzy_table_join.html.
  Shows exact-join vs fuzzy-join match rate improvement; attribute blocking by District;
  4 figures

**New documentation**
- `docs/GEO_GUIDE.md` — complete geo-matching reference (blocking strategies, API
  parameters, real-world use cases, figures gallery, API summary table)
- `docs/STEP_BY_STEP_GUIDE.md` — extended with GeoDataFrame, fuzzy join, and
  fuzzy dissolve sections
- `README.md` — fully rewritten with geo quickstart, pipeline diagram, confidence
  label table, per-function code examples, figures gallery, updated future extensions

**New figures** (`notebooks/figures/`):
  `geo_spatial_score_scatter`, `geo_combined_heatmap`, `geo_llm_boost_comparison`,
  `geo_world_map_geo_score`, `geo_gdf_scatter`, `geo_gdf_spatial_blocks`,
  `geo_gdf_match_map`, `geo_gdf_score_bars`, `geo_fuzzy_join_table`,
  `geo_fuzzy_join_map`, `geo_fuzzy_dissolve_union`, `geo_fuzzy_dissolve_ops`,
  `geo_srilanka_match_rate`, `geo_srilanka_reliability`, `geo_srilanka_score_hist`,
  `geo_srilanka_map`

**Infra**
- `.gitignore` — added `_private/` pattern for personal notes

### Fixed
- Removed orphaned dead-code block in `api.py` left after previous refactor

---

## [0.1.0-alpha] - 2026-08-05

> ⚠️ **Experimental / alpha software.** APIs may change without notice.
> Not yet independently validated on production data. Use at your own risk.
> Bug reports and feedback via [GitHub Issues](https://github.com/mohseniaref/fuzzy_llm_matcher/issues) welcome.

First public preview release.

### Added
- Core fuzzy matching pipeline: RapidFuzz string similarity, score-margin
  based confidence estimation, optional LLM review for ambiguous matches.
- `match_tables()` public API with blocking, top-k candidate generation,
  and configurable reliability thresholds.
- Parallelized candidate generation (`n_jobs` parameter) using
  `ThreadPoolExecutor` + vectorized `rapidfuzz.process.extract`.
- `simulate_dirty_entities()` synthetic noisy-data generator.
- Step-by-step documentation (`docs/STEP_BY_STEP_GUIDE.md`).
- Realistic bundled sample dataset (tech companies, universities, orgs
  with abbreviations, acronym collisions, legal-suffix variants).
- Benchmarks: bundled sample, FEBRL, LLM review comparison, Splink
  baseline, and the published Abt-Buy dataset (SIGMOD'18 DeepMatcher
  benchmark set).
- Examples: SQL/DuckDB/SQLite matching patterns, documented PySpark
  scaling pattern, offline OpenStreetMap/GeoNames place-name benchmark
  (40 hard transliteration/abbreviation cases), and a world-map
  visualization of the geo matching results.
- Markdown result reports with figures in `notebooks/`.
- Full pytest suite (19 tests).

[0.1.0]: https://github.com/mohseniaref/fuzzy_llm_matcher/releases/tag/v0.1.0

