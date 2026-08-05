# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - 2026-08-05

First public release.

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
