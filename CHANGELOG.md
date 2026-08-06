# Changelog

All notable changes to this project are documented in this file.
Versions follow the pattern `MAJOR.MINOR.PATCH-stage`.

---

## [0.2.0-alpha] — 2026-08-06

> ⚠️ **Experimental / alpha software.** APIs may change without notice.
> Bug reports and feedback via [GitHub Issues](https://github.com/mohseniaref/fuzzy_llm_matcher/issues) welcome.

Major geo-matching release. All 0.1.0-alpha features are preserved and extended.

### Added — Geospatial matching

**Spatial scoring & proximity**
- `haversine_km()` — great-circle distance between two lat/lon points
- `geo_distance_score()` — converts distance to 0–100 similarity score
- `add_geo_distance_score()` — adds `score_geo_distance` to any candidates DataFrame
- `geo_uncertainty_score()` — P(same location) from Gaussian positional uncertainty
  model (seismology / GPS / archaeology standard)
- `add_geo_uncertainty_score()` — batch version with per-row sigma columns

**Geometry similarity (Feature 4)**
- `geometry_similarity_score(geom1, geom2, method)` — 0–100 score via:
  - `"hausdorff"` — admin boundaries, polygons (any shapely version)
  - `"frechet"` — rivers, roads, fault traces (shapely ≥ 2.0)
  - `"iou"` — Intersection-over-Union / Jaccard for polygons
  - `"distance"` — minimum distance, any geometry type
- `add_geometry_similarity_score()` — batch version with projected CRS reprojection

**Transliteration + phonetic scoring (Feature 1)**
- `transliterate_text()` — Unicode → ASCII via unidecode ("Köln" → "Koln", "Москва" → "Moskva")
- `phonetic_code(text, algorithm)` — Soundex / Metaphone / NYSIIS via jellyfish
- `phonetic_similarity_score(a, b, algorithm)` — 0–100 Jaro-Winkler on phonetic codes
- Four new scorers in `SCORERS` dict: `"transliterated_WRatio"`, `"metaphone"`, `"soundex"`, `"nysiis"`
- New optional extras: `[nlp]` (unidecode, jellyfish) and `[all]` meta-extra

**GeoDataFrame API**
- `match_geodataframes()` — end-to-end GeoDataFrame pipeline: centroid extraction,
  attribute or spatial blocking, geo-distance score, geo-aware LLM review,
  geometry join-back. Accepts plain `pd.DataFrame` on right side.
- `fuzzy_join()` — fuzzy `pd.merge()` for plain DataFrames (`how=inner/left/all`)
- `fuzzy_join_geodataframes()` — fuzzy `gpd.sjoin()` with spatial blocking + geo score
- `fuzzy_dissolve()` — match + merge geometries (union/intersection/envelope/centroid/left/right)
- `block_on` parameter on all geo functions for attribute-column blocking

**Hierarchical blocking (Feature 2)**
- `hierarchical_block_match()` — tries district first, falls back to province, then country.
  Eliminates boundary artefacts of flat blocking. Returns `_block_level` column.

**Hexagonal grid blocking**
- `create_hexagon()` — single flat-top hexagon (pure Shapely, no gemgis/h3 needed)
- `create_hexagon_grid(gdf, radius_m)` — hex tessellation over any GeoDataFrame extent
- `assign_hex_ids(gdf, hex_grid)` — spatial join of features onto hex cells
- `hex_block_match()` — full pipeline with hex-cell blocking + choropleth support

**Spatial proximity candidates**
- `sjoin_nearest_candidates()` — spatial-first approach via `gpd.sjoin_nearest()` + rapidfuzz
- `combined_score()` — weighted name + distance score (`w_name × S_name + w_dist × S_dist`)

**Tile basemaps**
- `add_basemap(ax, style)` — adds OSM / CartoDB / ESRI satellite / Google-equivalent tiles
  to any matplotlib axes via contextily (no API key required for all free providers)
- `TileBasemap` — preset class: `.DARK`, `.LIGHT`, `.OSM`, `.SATELLITE`, `.GOOGLE_LIKE`, `.TOPO`

**Geo-aware LLM prompt**
- `build_prompt(left, right, geo_context=None)` — selects geo-enriched template automatically
- `MockLLMClient.review(geo_context=None)` — proximity boost from coordinate context
- `review_uncertain_pairs_with_llm` — auto-injects coordinates + haversine distance

**Documentation**
- `docs/GEO_GUIDE.md` — complete geo-matching reference (13 sections)
- `docs/STEP_BY_STEP_GUIDE.md` — extended with geo sections
- `README.md` — fully rewritten with geo quickstart, pipeline diagram, figures gallery

**New examples** (all produce publication-quality figures)
- `geo_spatial_confidence.py`, `geo_match_geodataframes.py`, `geo_fuzzy_join_dissolve.py`
- `geo_srilanka_admin_join.py` — Sri Lanka GN Division join (inspired by GeoPython tutorial)
- `geo_sjoin_basemap.py` — sjoin_nearest + combined score + tile basemaps
- `geo_hexagon_matching.py` — hexagonal grid at 3 resolutions + choropleths
- `geo_advanced_matching.py` — all 4 advanced features in one demo

**32+ new figures** in `notebooks/figures/` covering all above features.

### Fixed
- Removed orphaned dead-code block in `api.py` left after previous refactor
- Corrected coordinate-uncertainty formula (erfc, not 0.5×erfc)

---

## [0.1.0-alpha] — 2026-08-05

First public preview release.

### Added
- Core fuzzy matching pipeline: RapidFuzz string similarity, score-margin
  based confidence estimation, optional LLM review for ambiguous matches.
- `match_tables()` public API with blocking, top-k candidate generation,
  and configurable reliability thresholds.
- Parallelized candidate generation (`n_jobs`) via `ThreadPoolExecutor` +
  vectorized `rapidfuzz.process.extract`.
- `simulate_dirty_entities()` synthetic noisy-data generator.
- `docs/STEP_BY_STEP_GUIDE.md` step-by-step walkthrough.
- Realistic bundled sample dataset (tech companies, universities, orgs with
  abbreviations, acronym collisions, legal-suffix variants).
- Benchmarks: bundled sample, FEBRL, LLM review comparison, Splink baseline,
  Abt-Buy dataset (SIGMOD'18 DeepMatcher benchmark set).
- Examples: SQL/DuckDB/SQLite patterns, PySpark scaling, OSM/GeoNames
  place-name benchmark (40 hard transliteration cases), world-map visualisation.
- Full pytest suite (19 tests).

