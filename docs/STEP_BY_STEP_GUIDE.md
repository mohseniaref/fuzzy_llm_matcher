# Step-by-step guide: understand `fuzzy_llm_matcher`

This guide explains the package in the order most users need.
For geo-specific features, see also `docs/GEO_GUIDE.md`.

---

## 1) What this package solves

Most fuzzy match tools return a "best match" even when confidence is weak.
This package adds a **reliability layer** so you can separate:

- safe matches (`high`)
- uncertain matches that need review (`medium_review`)
- weak/rejected matches (`low`, `reject`)

For **geospatial data** the reliability layer gains a second signal:
geographic proximity. Two features with similar names but 5,000 km apart
are almost certainly not the same entity; two features with dissimilar
names at the exact same location almost certainly are.

The optional LLM step only reviews uncertain pairs. It does not replace
deterministic matching.

---

## 2) Minimal data format

### Plain tables

You need two tables:

- **left table**: noisy records to match
- **right table**: reference records

Required columns:

- text columns to compare (`left_on`, `right_on`)
- id columns (`left_id`, `right_id`)

Example files:

- `data/sample_dirty_left.csv`
- `data/sample_dirty_right.csv`
- `data/sample_ground_truth.csv` (for evaluation)

### GeoDataFrames

Any `geopandas.GeoDataFrame` in a geographic CRS (EPSG:4326 recommended).
Geometry type can be Point, Polygon, or MultiPolygon — centroids are
extracted automatically.

---

## 3) First run (no LLM)

```python
import pandas as pd
from fuzzy_llm_matcher import match_tables

left  = pd.read_csv("data/sample_dirty_left.csv")
right = pd.read_csv("data/sample_dirty_right.csv")

matches = match_tables(
    left_df=left,
    right_df=right,
    left_on="name",
    right_on="name",
    left_id="id",
    right_id="id",
    use_llm=False,
)
```

---

## 4) Read the output correctly

Important columns:

| Column | Description |
|--------|-------------|
| `fuzzy_score` | String similarity score (0–100) |
| `score_margin_to_second_best` | Gap between best and second-best candidate |
| `reliability_label` | `high` / `medium_review` / `low` / `reject` |
| `final_decision` | True = accepted match |

Interpretation:

- High score + high margin → usually safe (`high`)
- High score + low margin → ambiguous (`medium_review`) — two candidates are
  nearly equally good, the pick is genuinely uncertain
- Low score → `low` or `reject`

---

## 5) Add LLM review only for ambiguous cases

```python
matches = match_tables(
    left_df=left,
    right_df=right,
    left_on="name",
    right_on="name",
    left_id="id",
    right_id="id",
    use_llm=True,   # reviews medium_review rows only
)
```

LLM-related columns added:

| Column | Values |
|--------|--------|
| `llm_same_entity` | `True` / `False` / `None` |
| `llm_confidence` | `"low"` / `"medium"` / `"high"` |
| `llm_reason` | Short text explanation |

No API key is needed for testing — `MockLLMClient` runs entirely offline.

---

## 6) First run with GeoDataFrames

```python
import geopandas as gpd
from fuzzy_llm_matcher import match_geodataframes

result = match_geodataframes(
    left_gdf, right_gdf,
    left_on="name", right_on="name",
    left_id="id",  right_id="id",
    spatial_block_degrees=5.0,   # ~500 km grid cells for blocking
    max_distance_km=500.0,
    use_llm=True,
    return_geometry=True,
)

# result is a GeoDataFrame — plot directly
result.plot(column="reliability_label", legend=True)
result[result["final_decision"]].to_file("matches.geojson", driver="GeoJSON")
```

Extra output columns for geo results:

| Column | Description |
|--------|-------------|
| `score_geo_distance` | 0–100 proximity score (100 = same location) |
| `left_lat`, `left_lon` | Centroid of the left feature |
| `right_lat`, `right_lon` | Centroid of the right feature |
| `geometry` | Left-side geometry |

---

## 7) Fuzzy join — enrich one table with the other's columns

Use `fuzzy_join()` exactly like `pd.merge()` — it brings all columns
from both tables into a single result, matched by fuzzy name:

```python
from fuzzy_llm_matcher import fuzzy_join, fuzzy_join_geodataframes

# Plain DataFrames
enriched = fuzzy_join(
    dirty_df, canonical_df,
    left_on="city", right_on="name",
    how="inner",       # "inner" | "left" | "all"
)
# → all columns from both tables + _fuzzy_score + _reliability

# GeoDataFrames
enriched_gdf = fuzzy_join_geodataframes(
    admin_gdf, reference_gdf,
    left_on="NAME_1", right_on="name",
    how="left",
    geometry="left",   # "left" | "right" | "both"
)
# → GeoDataFrame + all attributes from both sides
```

`how="left"` is particularly useful: every left row appears in the result;
rows without a confirmed match have `NaN` in the right-side columns
(same behaviour as `pd.merge(..., how="left")`).

---

## 8) Fuzzy dissolve — match + merge geometries

Use `fuzzy_dissolve()` when you want to **merge the geometries** of
confirmed matched pairs, not just their attributes:

```python
from fuzzy_llm_matcher import fuzzy_dissolve

dissolved = fuzzy_dissolve(
    gadm_polygons, osm_polygons,
    left_on="NAME_1", right_on="name",
    dissolve_op="union",            # merge the two polygons
    aggfunc={"population": "mean"}, # average population across the pair
)

dissolved.plot(column="_reliability", legend=True)
```

`dissolve_op` choices:

| Mode | Result |
|------|--------|
| `"union"` | Merge the two geometries into one |
| `"intersection"` | Keep only the overlapping area |
| `"envelope"` | Bounding box of both |
| `"centroid"` | Midpoint between centroids |
| `"left"` / `"right"` | Keep one geometry, enrich with the other's attributes |

---

## 9) Evaluate quality against ground truth

```python
from fuzzy_llm_matcher import evaluate_matches

truth = pd.read_csv("data/sample_ground_truth.csv")
ev = evaluate_matches(matches, truth)
print(ev.to_dict())
```

Key metrics:

- `precision`, `recall`, `f1`
- `false_positives`, `false_negatives`
- `false_confident` — pairs labeled `high` that are actually wrong
  (the most important trust metric: these are the silent errors)

---

## 10) Tune behaviour

Main parameters in `match_tables()` and `match_geodataframes()`:

| Parameter | Effect |
|-----------|--------|
| `high_threshold` | Minimum score for `high` label (default 92) |
| `medium_threshold` | Minimum score for `medium_review` (default 80) |
| `min_margin_high` | Minimum margin over runner-up for `high` (default 8) |
| `reject_threshold` | Below this → `reject` (default 60) |
| `top_k` | Number of candidates per left row |
| `max_distance_km` | Distance at which geo score reaches 0 |
| `spatial_block_degrees` | Grid-cell size for spatial blocking |

Typical direction:

- **More precision** → raise `high_threshold` / `min_margin_high`
- **More recall** → lower `high_threshold`, or use `use_llm=True` to
  confirm more `medium_review` pairs
- **Geo-heavy dataset** → lower `high_threshold` and rely more on
  `score_geo_distance` as a tie-breaker

---

## 11) Use blocking for scale

**Text blocking** — if both tables share a trusted field (e.g. `country`):

```python
match_tables(..., block_on="country")
```

**Spatial blocking** — done automatically in `match_geodataframes()`:

```python
match_geodataframes(..., spatial_block_degrees=5.0)
# Only pairs within the same ~500 km grid cell are compared
# Set to None to disable (compare every left row against every right row)
```

Spatial blocking can reduce the comparison space by 99% on large datasets
without missing any matches whose true locations are in the same region.

---

## 12) Why the new sample dataset is better

The bundled sample includes realistic difficult cases:

- legal suffix changes (LLC, GmbH, plc)
- abbreviations (MIT, UCL, NYU, TUM)
- acronym collisions (ESA vs ESADE, WHO vs WHO Foundation)
- same-family distractors (Google vs Alphabet, Siemens vs Siemens Healthineers)

This makes demos and benchmarks much closer to real entity-resolution work.

