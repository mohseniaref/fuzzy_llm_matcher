# Geo Matching Guide — fuzzy_llm_matcher

Complete reference for all geospatial matching capabilities.
For a quick overview see `README.md`. For a step-by-step walkthrough see `STEP_BY_STEP_GUIDE.md`.

---

## Table of Contents

1. [Why geo fuzzy matching?](#1-why-geo-fuzzy-matching)
2. [Install](#2-install)
3. [The geo pipeline at a glance](#3-the-geo-pipeline-at-a-glance)
4. [score\_geo\_distance — spatial proximity as a score](#4-score_geo_distance--spatial-proximity-as-a-score)
5. [match\_geodataframes() — the core geo API](#5-match_geodataframes--the-core-geo-api)
6. [fuzzy\_join() — fuzzy pd.merge()](#6-fuzzy_join--fuzzy-pdmerge)
7. [fuzzy\_join\_geodataframes() — fuzzy gpd.sjoin()](#7-fuzzy_join_geodataframes--fuzzy-gpdsjoin)
8. [fuzzy\_dissolve() — match + merge geometries](#8-fuzzy_dissolve--match--merge-geometries)
9. [Blocking strategies](#9-blocking-strategies)
10. [Geo-aware LLM prompt](#10-geo-aware-llm-prompt)
11. [Real-world use cases](#11-real-world-use-cases)
12. [Figures gallery](#12-figures-gallery)
13. [API reference summary](#13-api-reference-summary)

---

## 1. Why geo fuzzy matching?

Geographic datasets from different agencies rarely share a common unique key. The same
feature appears under different spellings, transliterations, abbreviations, or entirely
different names depending on the language and source of the data. Exact joins on name
columns fail silently — they return fewer matches than exist without any warning.

**Classic examples:**
- "Köln" (German) vs "Cologne" (English)
- "Sammanthranapura" (shapefile) vs "Sammanthranpura" (census CSV)
- "Kandyan North" vs "Kandy N" vs "Kandy North"
- "Al Qahirah" (transliteration) vs "Cairo" (English)

Pure string matching without spatial context makes things worse: two cities with similar
names on opposite sides of the world will match. Geography provides a powerful second
signal: **if the names are similar AND the features are close, the match is almost
certainly correct**.

---

## 2. Install

```bash
pip install -e ".[geo]"
# installs: geopandas, shapely, matplotlib, geodatasets (alongside core deps)
```

---

## 3. The geo pipeline at a glance

```
left_gdf (GeoDataFrame)  ──┐
                            ├─ extract centroids
right_gdf / right_df ──────┘      │
                                   ▼
                         spatial or attribute blocking
                         (restrict comparisons to nearby/same-admin pairs)
                                   │
                                   ▼
                         generate_candidates() — top-k fuzzy string matches
                                   │
                                   ▼
                         compute_similarity_features()
                         (WRatio, token_sort, token_set, partial, simple, margin)
                                   │
                                   ▼
                         add_geo_distance_score()
                         haversine distance → 0-100 score
                                   │
                                   ▼
                         assign_reliability()
                         high / medium_review / low / reject
                                   │
                              use_llm=True?
                             /            \
                            yes            no
                             ▼
                  review_uncertain_pairs()
                  (geo context auto-included in prompt)
                             │
                             └──────────────┐
                                            ▼
                                     final_decision
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    ▼                       ▼                       ▼
          match_geodataframes()   fuzzy_join_geodataframes()  fuzzy_dissolve()
          match result +          all columns from both       dissolved geometry
          geometry attached       tables + geometry           per matched pair
```

---

## 4. score\_geo\_distance — spatial proximity as a score

### Concept

`score_geo_distance` converts the haversine distance between two features into a
0–100 similarity score:

```
score_geo_distance = max(0, 100 × (1 - dist_km / max_km))
```

- Distance = 0 km → score = 100
- Distance = max_km → score = 0
- Distance > max_km → score = 0

### Usage

```python
from fuzzy_llm_matcher import add_geo_distance_score, haversine_km

# Direct distance
dist = haversine_km(lat1=48.14, lon1=11.58, lat2=50.94, lon2=6.96)
# → 454.4 km (Munich to Cologne)

# Add to any candidates DataFrame that has lat/lon columns
candidates = add_geo_distance_score(
    candidates,
    left_lat_col="left_lat",
    left_lon_col="left_lon",
    right_lat_col="right_lat",
    right_lon_col="right_lon",
    max_km=500.0,
    score_col="score_geo_distance",  # name of the new column
)
```

### Tuning `max_km`

| Dataset type | Recommended `max_km` |
|---|---|
| Sub-district / neighbourhood | 5–20 km |
| City / municipality | 50–100 km |
| Province / state | 200–500 km |
| Country | 1,000–2,000 km |
| Continent / global | 5,000 km |

### Using geo score as a reliability override

```python
from fuzzy_llm_matcher import assign_reliability

# Use geo distance as the primary score instead of string score
labeled = assign_reliability(scored, score_col="score_geo_distance",
                             high_threshold=80, reject_threshold=30)
```

---

## 5. match\_geodataframes() — the core geo API

The end-to-end geo matching pipeline in one call.

```python
from fuzzy_llm_matcher import match_geodataframes

result = match_geodataframes(
    left_gdf,  right_gdf,
    left_on="name",   right_on="name",
    left_id="id",     right_id="id",

    # ── Blocking ──────────────────────────────────────────
    block_on=None,             # attribute column for blocking (e.g. "district")
                               # takes priority over spatial_block_degrees
    spatial_block_degrees=5.0, # degree-grid spatial blocking (ignored if block_on set)
                               # None = no blocking (compare all pairs)

    # ── Geo-distance score ────────────────────────────────
    max_distance_km=500.0,     # distance at which geo score → 0

    # ── String matching ───────────────────────────────────
    top_k=5,
    scorer="WRatio",

    # ── Reliability thresholds ────────────────────────────
    high_threshold=92,
    medium_threshold=80,
    min_margin_high=8,
    reject_threshold=60,

    # ── LLM review ───────────────────────────────────────
    use_llm=True,              # MockLLMClient runs offline; pass llm_client for real LLM
    llm_client=None,

    # ── Output ───────────────────────────────────────────
    return_geometry=True,      # False = plain DataFrame
    n_jobs=1,                  # -1 = all CPU cores
)
```

### Output columns

| Column | Description |
|--------|-------------|
| `left_id`, `right_id` | Record IDs |
| `left_value`, `right_value` | Matched text values |
| `fuzzy_score` | WRatio string similarity (0–100) |
| `score_geo_distance` | Haversine proximity score (0–100) |
| `left_lat`, `left_lon` | Left centroid |
| `right_lat`, `right_lon` | Right centroid |
| `score_margin_to_second_best` | Gap to runner-up candidate |
| `reliability_label` | `high` / `medium_review` / `low` / `reject` |
| `llm_same_entity` | LLM verdict (True/False/None) |
| `llm_confidence` | `"low"` / `"medium"` / `"high"` |
| `final_decision` | True = confirmed match |
| `geometry` | Left-side geometry (when `return_geometry=True`) |

### Accepts a plain DataFrame on the right

`right_gdf` can be a plain `pd.DataFrame` (e.g. a CSV population table with no
geometry). In that case `score_geo_distance` will be `NaN` for all pairs and
geometry is attached from the left side only.

```python
# GeoDataFrame on left (shapefile), plain DataFrame on right (CSV)
result = match_geodataframes(
    shapefile_gdf, census_csv_df,
    left_on="gn_name", right_on="gn_name",
    block_on="district",           # attribute blocking by admin column
    spatial_block_degrees=None,    # no spatial grid needed
)
```

---

## 6. fuzzy\_join() — fuzzy pd.merge()

Drop-in fuzzy replacement for `pd.merge()`. Matches on a text column instead of an
exact key, then brings all columns from both tables into a single result.

```python
from fuzzy_llm_matcher import fuzzy_join

# Exact merge (pandas)
result = pd.merge(left, right, left_on="city", right_on="name")

# Fuzzy merge (this package — same shape, same how= options)
result = fuzzy_join(
    left, right,
    left_on="city",   right_on="name",
    left_id="id",     right_id="id",
    how="inner",      # "inner" | "left" | "all"
    suffixes=("", "_ref"),
    use_llm=True,
)
# New columns: _fuzzy_score, _reliability
```

`how` options:

| `how` | Rows returned |
|-------|--------------|
| `"inner"` | Only `final_decision=True` pairs |
| `"left"` | All left rows; unmatched get `NaN` in right columns |
| `"all"` | All candidate pairs including low/reject |

---

## 7. fuzzy\_join\_geodataframes() — fuzzy gpd.sjoin()

Fuzzy name join for GeoDataFrames: spatial blocking + geo score + attribute join +
geometry attachment in one call.

```python
from fuzzy_llm_matcher import fuzzy_join_geodataframes

# Exact spatial join (geopandas)
result = gpd.sjoin(left, right, how="inner", predicate="intersects")

# Fuzzy name join (this package)
result = fuzzy_join_geodataframes(
    left_gdf, right_gdf,
    left_on="NAME_1",  right_on="name",
    left_id="id",      right_id="id",
    how="left",

    # Blocking: use attribute column (takes priority) or spatial grid
    block_on="district",          # e.g. "district", "province", "country_code"
    spatial_block_degrees=None,   # set to e.g. 3.0 if no attribute block available

    max_distance_km=200.0,
    geometry="left",              # "left" | "right" | "both"
    use_llm=True,
)
# Result: GeoDataFrame with all attribute columns from both sides
# + _fuzzy_score + _geo_score + _reliability
```

### Attribute blocking vs spatial blocking

| Situation | Use |
|-----------|-----|
| Tables share an admin column (district, province, country) | `block_on="district"` |
| No shared column but features are geographically organised | `spatial_block_degrees=3.0` |
| Small dataset (< ~5,000 rows) | `block_on=None, spatial_block_degrees=None` |
| Right side is a plain CSV (no geometry) | `block_on="district"` (attribute only) |

---

## 8. fuzzy\_dissolve() — match + merge geometries

Match two GeoDataFrames and merge the geometries of confirmed pairs into a single
feature per match. The fuzzy equivalent of `geopandas.GeoDataFrame.dissolve()`.

```python
from fuzzy_llm_matcher import fuzzy_dissolve

dissolved = fuzzy_dissolve(
    left_gdf, right_gdf,
    left_on="NAME_1",   right_on="name",
    left_id="id",       right_id="id",
    dissolve_op="union",
    aggfunc={"population": "sum", "area_km2": "mean"},
    block_on="district",
)

dissolved.plot(column="_reliability", legend=True)
dissolved.to_file("dissolved.geojson", driver="GeoJSON")
```

### dissolve\_op options

| Mode | Result geometry | Typical use case |
|------|-----------------|-----------------|
| `"union"` | Spatial union of left + right | Merge admin boundaries from two agencies |
| `"intersection"` | Overlapping area only | Quantify spatial agreement between datasets |
| `"envelope"` | Bounding box of both | Coarse extent of the matched pair |
| `"centroid"` | Midpoint of the two centroids | Reconcile GPS positions from two surveys |
| `"left"` | Left geometry unchanged | Enrich left features with right attributes |
| `"right"` | Right geometry unchanged | Replace with authoritative right geometry |

### Output columns

| Column | Description |
|--------|-------------|
| `_fuzzy_score` | String similarity of the matched names |
| `_geo_score` | Geo-distance score of the pair |
| `_reliability` | Reliability label of the match |
| `_left_name` | Left text value |
| `_right_name` | Right text value |
| `_dissolve_op` | Which dissolve operation was used |
| `geometry` | Dissolved geometry |
| Any `aggfunc` columns | Aggregated numeric attributes |

---

## 9. Blocking strategies

Blocking restricts which pairs are compared, reducing the O(n²) problem to O(n × k)
where k is the average block size.

### Attribute blocking (recommended for admin data)

```python
# Block by a shared administrative column
match_geodataframes(..., block_on="district")
fuzzy_join_geodataframes(..., block_on="province")
fuzzy_dissolve(..., block_on="country_code")
```

Works for any column present in both datasets with matching values (after normalisation
to lowercase). Typical examples: `district`, `province`, `country_code`, `admin1_name`.

### Spatial grid blocking (for purely spatial data)

```python
# Block by 5-degree grid cell (~500 km at equator)
match_geodataframes(..., spatial_block_degrees=5.0)

# Smaller cell = tighter blocking, fewer false comparisons, but may miss cross-border pairs
match_geodataframes(..., spatial_block_degrees=2.0)
```

### Combining both (hierarchical)

Set `block_on` for attribute blocking. Spatial blocking is skipped when `block_on` is set.

```python
# For intra-district matching with attribute block:
result = match_geodataframes(
    left, right,
    left_on="gn_name", right_on="gn_name",
    block_on="district",           # exact district must match
    spatial_block_degrees=None,    # grid not needed
    max_distance_km=50.0,          # geo score only for nearby pairs
)
```

### No blocking (small datasets)

```python
match_geodataframes(..., block_on=None, spatial_block_degrees=None)
```

---

## 10. Geo-aware LLM prompt

When coordinate columns (`left_lat`, `left_lon`, `right_lat`, `right_lon`) are present
in the candidates DataFrame, `review_uncertain_pairs_with_llm` automatically injects
spatial context into each uncertain-pair prompt:

```
You are comparing two geographic place-name records ...

Record A: München
Record B: Munich

Spatial context:
Record A coordinates: (48.1400°N, 11.5800°E)
Record B coordinates: (48.1400°N, 11.5800°E)
Distance: 0.0 km
```

You can also build the prompt manually:

```python
from fuzzy_llm_matcher import build_prompt

# Plain prompt
prompt = build_prompt("München", "Munich")

# Geo-enriched prompt
prompt = build_prompt(
    "München", "Munich",
    geo_context="Record A: (48.14°N, 11.58°E)  Record B: (48.14°N, 11.58°E)  Distance: 0.0 km"
)
```

### Implementing a real geo-aware LLM client

```python
import json
from anthropic import Anthropic
from fuzzy_llm_matcher import build_prompt

class GeoAnthropicClient:
    def __init__(self, model="claude-sonnet-5"):
        self.client = Anthropic()
        self.model  = model

    def review(self, left_value, right_value, geo_context=None):
        prompt = build_prompt(left_value, right_value, geo_context=geo_context)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return json.loads(resp.content[0].text)

result = match_geodataframes(
    left_gdf, right_gdf,
    left_on="name", right_on="name",
    use_llm=True,
    llm_client=GeoAnthropicClient(),
)
```

---

## 11. Real-world use cases

### Sri Lanka GN Division admin join
*(Inspired by https://www.geopythontutorials.com/notebooks/geopandas_fuzzy_table_join.html)*

**Problem:** ~14,000 Grama Niladhari Division polygons in a shapefile from one agency;
~14,000 rows in a census population CSV from a different agency. Exact join on
concatenated name key → 10,747 matches (23% missed). Manual rapidfuzz loop → 7m 50s.

**Solution with fuzzy_llm_matcher:**

```python
from fuzzy_llm_matcher import fuzzy_join_geodataframes

# Block by District — only GN Divisions in the same district are compared
result = fuzzy_join_geodataframes(
    shapefile_gdf, census_df,       # right side can be a plain CSV DataFrame
    left_on="gn_name",
    right_on="gn_name",
    block_on="district",            # admin blocking: huge speed-up, no grid needed
    how="left",                     # keep all shapefile records
    use_llm=True,
    high_threshold=85,
)
# → ~13,400+ matches with reliability labels per match
# → unmatched records get NaN (visible for human review)
# → runs in seconds
```

See `examples/geo_srilanka_admin_join.py` for a full runnable demo.

---

### City name transliteration matching (OSM vs GeoNames)

```python
from fuzzy_llm_matcher import match_geodataframes

result = match_geodataframes(
    osm_cities_gdf, geonames_gdf,
    left_on="name",     right_on="asciiname",
    left_id="osm_id",   right_id="geonameid",
    block_on="country_code",      # compare only within same country
    max_distance_km=100.0,
    use_llm=True,
)
```

See `examples/osm_geonames_place_matching.py` and `examples/geo_spatial_confidence.py`.

---

### GADM / OpenStreetMap admin boundary merge

```python
from fuzzy_llm_matcher import fuzzy_dissolve

dissolved = fuzzy_dissolve(
    gadm_gdf, osm_gdf,
    left_on="NAME_1", right_on="name:en",
    block_on="admin_level",
    dissolve_op="union",
    aggfunc={"population": "mean"},
    use_llm=True,
)
dissolved.to_file("merged_admin.geojson", driver="GeoJSON")
```

---

### GNSS / GPS station cross-referencing

```python
from fuzzy_llm_matcher import match_geodataframes

result = match_geodataframes(
    unr_stations_gdf, unavco_gdf,
    left_on="site_name", right_on="station_name",
    block_on="country",
    max_distance_km=5.0,   # GNSS stations: tight spatial tolerance
    use_llm=True,
)
```

---

## 12. Figures gallery

All figures in `notebooks/figures/`:

| File | Description |
|------|-------------|
| `geo_spatial_score_scatter.png` | Fuzzy score vs geo-distance score, coloured by label |
| `geo_combined_heatmap.png` | 2-D density heat-map of both scores |
| `geo_llm_boost_comparison.png` | Label distribution before/after geo-LLM review |
| `geo_world_map_geo_score.png` | World arcs coloured by geo-distance score |
| `geo_gdf_spatial_blocks.png` | 5° spatial blocking grid on world map |
| `geo_gdf_match_map.png` | Confirmed matches, arc colour = geo score |
| `geo_gdf_score_bars.png` | Per-pair bar chart: string + geo score |
| `geo_fuzzy_join_table.png` | `fuzzy_join()` result — colour-coded attribute table |
| `geo_fuzzy_join_map.png` | World map after `fuzzy_join_geodataframes()` |
| `geo_fuzzy_dissolve_union.png` | Three-panel: left / right / dissolved union |
| `geo_fuzzy_dissolve_ops.png` | Four-panel: union / intersection / envelope / centroid |
| `geo_srilanka_match_rate.png` | Exact vs fuzzy match count comparison |
| `geo_srilanka_reliability.png` | Label distribution, Sri Lanka admin join |
| `geo_srilanka_score_hist.png` | Score histogram coloured by reliability label |
| `geo_srilanka_map.png` | Sri Lanka GN Division map coloured by reliability |

---

## 13. API reference summary

| Function | Input | Output | Analogue |
|----------|-------|--------|----------|
| `match_tables()` | two DataFrames | match result DataFrame | — |
| `match_geodataframes()` | GeoDF + GeoDF/DF | match GeoDataFrame | — |
| `fuzzy_join()` | two DataFrames | enriched DataFrame | `pd.merge()` |
| `fuzzy_join_geodataframes()` | GeoDF + GeoDF/DF | enriched GeoDataFrame | `gpd.sjoin()` |
| `fuzzy_dissolve()` | two GeoDFs | dissolved GeoDataFrame | `gpd.dissolve()` |
| `add_geo_distance_score()` | candidates DF | candidates + geo score | — |
| `haversine_km()` | two lat/lon pairs | distance in km | — |
| `build_prompt()` | two values + geo_context | LLM prompt string | — |

All functions are importable from the top-level package:

```python
from fuzzy_llm_matcher import (
    match_tables,
    match_geodataframes,
    fuzzy_join,
    fuzzy_join_geodataframes,
    fuzzy_dissolve,
    add_geo_distance_score,
    haversine_km,
    build_prompt,
)
```
