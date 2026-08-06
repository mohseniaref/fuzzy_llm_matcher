"""High-level, user-facing API."""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from .candidate_generation import generate_candidates
from .fuzzy_scores import add_geo_distance_score, compute_similarity_features
from .llm_review import LLMClient, review_uncertain_pairs_with_llm
from .reliability import assign_reliability


def match_tables(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_on: str,
    right_on: str,
    left_id: Optional[str] = None,
    right_id: Optional[str] = None,
    block_on: Optional[str] = None,
    top_k: int = 5,
    scorer: str = "WRatio",
    high_threshold: float = 92,
    medium_threshold: float = 80,
    min_margin_high: float = 8,
    reject_threshold: float = 60,
    use_llm: bool = False,
    llm_client: Optional[LLMClient] = None,
    llm_model: Optional[str] = None,
    keep_all_candidates: bool = False,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """End-to-end fuzzy + reliability (+ optional LLM) matching pipeline.

    Returns a DataFrame with one row per candidate pair (or, if
    `keep_all_candidates=False`, only the best candidate per left row)
    with columns:

        left_id, right_id, left_value, right_value, fuzzy_score,
        score_margin_to_second_best, reliability_label,
        llm_same_entity, llm_confidence, final_decision

    `final_decision` is True when reliability_label == "high", or when
    the LLM confirmed `same_entity=True` for a medium_review pair
    (only computed if `use_llm=True`).

    Parameters
    ----------
    n_jobs:
        Number of parallel threads for candidate scoring.
        ``1`` (default) = single-threaded. ``-1`` = all CPU cores.
    """
    candidates = generate_candidates(
        left_df=left_df,
        right_df=right_df,
        left_on=left_on,
        right_on=right_on,
        left_id=left_id,
        right_id=right_id,
        block_on=block_on,
        top_k=top_k,
        scorer=scorer,
        n_jobs=n_jobs,
    )

    scored = compute_similarity_features(candidates)

    labeled = assign_reliability(
        scored,
        high_threshold=high_threshold,
        medium_threshold=medium_threshold,
        min_margin_high=min_margin_high,
        reject_threshold=reject_threshold,
    )

    if use_llm:
        labeled = review_uncertain_pairs_with_llm(
            labeled, client=llm_client, model=llm_model
        )
    else:
        for col in ("llm_same_entity", "llm_confidence", "llm_reason"):
            if col not in labeled.columns:
                labeled[col] = None

    def _final_decision(row) -> bool:
        if row["reliability_label"] == "high":
            return True
        if row["reliability_label"] == "medium_review" and row.get("llm_same_entity") is True:
            return True
        return False

    labeled["final_decision"] = labeled.apply(_final_decision, axis=1)

    if not keep_all_candidates and not labeled.empty:
        labeled = labeled.sort_values(["left_id", "score"], ascending=[True, False])
        labeled = labeled.groupby("left_id", as_index=False).first()

    result_cols = [
        "left_id",
        "right_id",
        "left_value",
        "right_value",
        "score_wratio",
        "score_margin_to_second_best",
        "reliability_label",
        "llm_same_entity",
        "llm_confidence",
        "final_decision",
    ]
    result = labeled[[c for c in result_cols if c in labeled.columns]].copy()
    result = result.rename(columns={"score_wratio": "fuzzy_score"})
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# GeoDataFrame wrapper
# ---------------------------------------------------------------------------

def match_geodataframes(
    left_gdf,
    right_gdf,
    left_on: str,
    right_on: str,
    left_id: Optional[str] = None,
    right_id: Optional[str] = None,
    block_on: Optional[str] = None,
    spatial_block_degrees: float = 5.0,
    max_distance_km: float = 500.0,
    top_k: int = 5,
    scorer: str = "WRatio",
    high_threshold: float = 92,
    medium_threshold: float = 80,
    min_margin_high: float = 8,
    reject_threshold: float = 60,
    use_llm: bool = False,
    llm_client: Optional[LLMClient] = None,
    llm_model: Optional[str] = None,
    keep_all_candidates: bool = False,
    n_jobs: int = 1,
    return_geometry: bool = True,
):
    """Fuzzy match two GeoDataFrames, using spatial proximity as a confidence signal.

    This wrapper around :func:`match_tables` adds three geo-specific steps:

    1. **Automatic spatial blocking** — each geometry's centroid is snapped
       to a ``spatial_block_degrees``-degree grid cell (e.g. 5° ≈ 500 km).
       Only candidates that share a grid cell are compared, reducing the
       O(n²) comparison space for large datasets without network round-trips.

    2. **Geo-distance score** — after string matching,
       :func:`~fuzzy_llm_matcher.fuzzy_scores.add_geo_distance_score`
       computes ``score_geo_distance`` (0–100) for every candidate pair
       from the exact haversine distance between centroids.

    3. **Geometry join-back** — when ``return_geometry=True`` the result
       DataFrame is promoted to a GeoDataFrame carrying the **left**
       geometry of each matched pair, so you can immediately visualise or
       export the results with ``result.to_file(…)`` or ``result.plot()``.

    Parameters
    ----------
    left_gdf, right_gdf:
        GeoPandas GeoDataFrames.  Their active geometry column must contain
        Point, Polygon, or MultiPolygon geometries in a geographic CRS
        (EPSG:4326 or equivalent — decimal degrees).  Polygon/MultiPolygon
        centroids are used automatically.
    left_on, right_on:
        Column names holding the text to fuzzy-match.
    left_id, right_id:
        Optional stable ID column names.  Defaults to the DataFrame index.
    spatial_block_degrees:
        Grid-cell size in decimal degrees used for blocking.  Smaller values
        produce tighter blocks (fewer false comparisons) but may miss pairs
        that straddle a cell boundary.  Default ``5.0`` works well for
        city-level data.  Set to ``None`` to disable spatial blocking and
        compare every left row against every right row.
    max_distance_km:
        Distance at which ``score_geo_distance`` reaches 0.  Tune to the
        expected spread of your dataset (e.g. 50 for neighbourhoods, 500 for
        country-level admin regions, 5000 for continent-scale data).
    top_k:
        Number of best right-side candidates to retain per left row.
    scorer:
        Fuzzy string scorer name (``"WRatio"``, ``"token_sort_ratio"``, …).
    high_threshold, medium_threshold, reject_threshold, min_margin_high:
        Reliability thresholds forwarded to :func:`assign_reliability`.
    use_llm:
        When ``True``, uncertain pairs are sent to ``llm_client`` for
        review.  Coordinate context is included automatically if the
        client accepts a ``geo_context`` kwarg.
    return_geometry:
        When ``True`` (default), the result is a GeoDataFrame with the
        left-side geometry attached.  When ``False`` a plain DataFrame is
        returned (no geopandas dependency needed at call time).
    n_jobs:
        Number of parallel threads for candidate scoring.

    Returns
    -------
    GeoDataFrame (or DataFrame when ``return_geometry=False``) with columns:

        left_id, right_id, left_value, right_value,
        fuzzy_score, score_geo_distance,
        score_margin_to_second_best, reliability_label,
        llm_same_entity, llm_confidence, final_decision,
        [geometry]  ← left centroid, present when return_geometry=True

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Point
    >>> from fuzzy_llm_matcher import match_geodataframes
    >>> left = gpd.GeoDataFrame(
    ...     {"name": ["München", "Köln"]},
    ...     geometry=[Point(11.58, 48.14), Point(6.96, 50.94)],
    ...     crs="EPSG:4326",
    ... )
    >>> right = gpd.GeoDataFrame(
    ...     {"name": ["Munich", "Cologne"]},
    ...     geometry=[Point(11.58, 48.14), Point(6.96, 50.94)],
    ...     crs="EPSG:4326",
    ... )
    >>> result = match_geodataframes(left, right, left_on="name", right_on="name")
    >>> result[["left_value", "right_value", "fuzzy_score", "score_geo_distance"]]
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "match_geodataframes() requires geopandas and shapely. "
            "Install with: pip install 'fuzzy_llm_matcher[geo]'"
        ) from exc

    # ── 1. Work on plain copies so we don't mutate the caller's GeoDataFrames ──
    left  = left_gdf.copy()
    # right may be a plain DataFrame (e.g. a CSV table) — handle both cases
    try:
        import geopandas as _gpd
        right_is_geo = isinstance(right_gdf, _gpd.GeoDataFrame)
    except ImportError:
        right_is_geo = False
    right = right_gdf.copy()

    # ── 2. Extract centroids (works for Point, Polygon, MultiPolygon) ─────────
    import warnings

    def _centroid_lat_lon(gdf):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            centroids = gdf.geometry.centroid
        return centroids.y, centroids.x  # lat, lon

    left_lat, left_lon = _centroid_lat_lon(left)
    left["_lat"] = left_lat.values
    left["_lon"] = left_lon.values

    if right_is_geo:
        right_lat, right_lon = _centroid_lat_lon(right)
        right["_lat"] = right_lat.values
        right["_lon"] = right_lon.values
    else:
        # No geometry on the right — geo distance score will be NaN
        right["_lat"] = float("nan")
        right["_lon"] = float("nan")

    # ── 3. Blocking: attribute column takes priority over spatial grid ────────
    block_col: Optional[str] = None
    if block_on is not None:
        # Attribute blocking: normalise values so the comparison is case-insensitive
        from .utils import normalize_text
        left["_attr_block"]  = left[block_on].map(normalize_text)
        right["_attr_block"] = right[block_on].map(normalize_text)
        block_col = "_attr_block"
    elif spatial_block_degrees is not None:
        d = float(spatial_block_degrees)

        def _grid_cell(lat, lon):
            row = math.floor(lat / d)
            col = math.floor(lon / d)
            return f"{row}_{col}"

        left["_geo_block"]  = [_grid_cell(la, lo) for la, lo in zip(left["_lat"],  left["_lon"])]
        right["_geo_block"] = [_grid_cell(la, lo) for la, lo in zip(right["_lat"], right["_lon"])]
        block_col = "_geo_block"

    # ── 4. Candidate generation (string matching + optional blocking) ─────────
    candidates = generate_candidates(
        left_df=left,
        right_df=right,
        left_on=left_on,
        right_on=right_on,
        left_id=left_id,
        right_id=right_id,
        block_on=block_col,
        top_k=top_k,
        scorer=scorer,
        n_jobs=n_jobs,
    )

    if candidates.empty:
        cols = [
            "left_id", "right_id", "left_value", "right_value",
            "fuzzy_score", "score_geo_distance",
            "score_margin_to_second_best", "reliability_label",
            "llm_same_entity", "llm_confidence", "final_decision",
        ]
        result = pd.DataFrame(columns=cols)
        if return_geometry:
            return gpd.GeoDataFrame(result, geometry=[], crs=left_gdf.crs)
        return result

    # ── 5. String-similarity features ────────────────────────────────────────
    scored = compute_similarity_features(candidates)

    # ── 6. Join centroid coordinates onto candidate pairs ─────────────────────
    id_col_left  = left_id  if left_id  else left.index.name  or "index"
    id_col_right = right_id if right_id else right.index.name or "index"

    # Build lookup: id → (lat, lon)
    if left_id:
        left_coords  = left.set_index(left_id)[["_lat", "_lon"]]
    else:
        left_coords  = left[["_lat", "_lon"]].copy()
        left_coords.index.name = "index"

    if right_id:
        right_coords = right.set_index(right_id)[["_lat", "_lon"]]
    else:
        right_coords = right[["_lat", "_lon"]].copy()
        right_coords.index.name = "index"

    scored = scored.merge(
        left_coords.rename(columns={"_lat": "left_lat", "_lon": "left_lon"}),
        left_on="left_id", right_index=True, how="left",
    )
    scored = scored.merge(
        right_coords.rename(columns={"_lat": "right_lat", "_lon": "right_lon"}),
        left_on="right_id", right_index=True, how="left",
    )

    # ── 7. Geo-distance score ─────────────────────────────────────────────────
    scored = add_geo_distance_score(scored, max_km=max_distance_km)

    # ── 8. Reliability labelling ──────────────────────────────────────────────
    labeled = assign_reliability(
        scored,
        high_threshold=high_threshold,
        medium_threshold=medium_threshold,
        min_margin_high=min_margin_high,
        reject_threshold=reject_threshold,
    )

    # ── 9. Optional geo-aware LLM review ─────────────────────────────────────
    if use_llm:
        labeled = review_uncertain_pairs_with_llm(
            labeled, client=llm_client, model=llm_model
        )
    else:
        for col in ("llm_same_entity", "llm_confidence", "llm_reason"):
            if col not in labeled.columns:
                labeled[col] = None

    # ── 10. Final decision ────────────────────────────────────────────────────
    def _final_decision(row) -> bool:
        if row["reliability_label"] == "high":
            return True
        if row["reliability_label"] == "medium_review" and row.get("llm_same_entity") is True:
            return True
        return False

    labeled["final_decision"] = labeled.apply(_final_decision, axis=1)

    if not keep_all_candidates and not labeled.empty:
        labeled = labeled.sort_values(["left_id", "score"], ascending=[True, False])
        labeled = labeled.groupby("left_id", as_index=False).first()

    # ── 11. Build result columns ──────────────────────────────────────────────
    result_cols = [
        "left_id", "right_id", "left_value", "right_value",
        "score_wratio", "score_geo_distance",
        "left_lat", "left_lon", "right_lat", "right_lon",
        "score_margin_to_second_best", "reliability_label",
        "llm_same_entity", "llm_confidence", "final_decision",
    ]
    result = labeled[[c for c in result_cols if c in labeled.columns]].copy()
    result = result.rename(columns={"score_wratio": "fuzzy_score"})
    result = result.reset_index(drop=True)

    # ── 12. Attach left geometry ──────────────────────────────────────────────
    if return_geometry:
        if left_id:
            geom_lookup = left_gdf.set_index(left_id).geometry
        else:
            geom_lookup = left_gdf.geometry
            geom_lookup.index.name = "index"

        result["geometry"] = result["left_id"].map(geom_lookup)
        result = gpd.GeoDataFrame(result, geometry="geometry", crs=left_gdf.crs)

    return result


# ---------------------------------------------------------------------------
# Fuzzy table join
# ---------------------------------------------------------------------------

def fuzzy_join(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_on: str,
    right_on: str,
    left_id: Optional[str] = None,
    right_id: Optional[str] = None,
    how: str = "inner",
    suffixes: tuple = ("_left", "_right"),
    block_on: Optional[str] = None,
    top_k: int = 5,
    scorer: str = "WRatio",
    high_threshold: float = 92,
    medium_threshold: float = 80,
    min_margin_high: float = 8,
    reject_threshold: float = 60,
    use_llm: bool = False,
    llm_client: Optional[LLMClient] = None,
    n_jobs: int = 1,
    match_score_col: str = "_fuzzy_score",
    reliability_col: str = "_reliability",
) -> pd.DataFrame:
    """Fuzzy merge of two DataFrames — the fuzzy equivalent of ``pd.merge()``.

    Instead of matching on an exact key, it uses fuzzy string similarity to
    align the ``left_on`` column in ``left_df`` with the ``right_on`` column
    in ``right_df``, then performs a standard join to bring all columns from
    both tables into a single result.

    This is intentionally designed to feel like ``pd.merge()``:

    .. code-block:: python

        # Exact join (pandas)
        result = pd.merge(left, right, left_on="city", right_on="name")

        # Fuzzy join (this function)
        result = fuzzy_join(left, right, left_on="city", right_on="name")

    Parameters
    ----------
    left_df, right_df:
        Input DataFrames.
    left_on, right_on:
        Column names to fuzzy-match on.
    left_id, right_id:
        Optional stable ID column names. Defaults to DataFrame index.
    how:
        Join type:
        - ``"inner"``  (default) — only confirmed matches (``final_decision=True``)
        - ``"left"``   — all left rows; unmatched right columns are NaN
        - ``"all"``    — all candidate pairs including low/reject
    suffixes:
        Column-name suffixes when both tables share non-key column names.
    block_on:
        Optional blocking column name (same in both tables, or a 2-tuple).
    match_score_col:
        Name of the fuzzy score column added to the result. Set to ``None``
        to suppress it.
    reliability_col:
        Name of the reliability label column added to the result. Set to
        ``None`` to suppress it.

    Returns
    -------
    pd.DataFrame with all columns from both tables aligned by fuzzy match,
    plus ``match_score_col`` and ``reliability_col`` appended.

    Examples
    --------
    >>> import pandas as pd
    >>> from fuzzy_llm_matcher import fuzzy_join
    >>> left  = pd.DataFrame({"city": ["München", "Köln"],  "pop": [1.5e6, 1.1e6]})
    >>> right = pd.DataFrame({"name": ["Munich",  "Cologne"], "country": ["DE", "DE"]})
    >>> fuzzy_join(left, right, left_on="city", right_on="name")
       city     pop    name country  _fuzzy_score _reliability
    0  München  1500000.0  Munich   DE           97.3       high
    1  Köln     1100000.0  Cologne  DE           88.2  medium_review
    """
    # ── 1. Run the core matching pipeline ────────────────────────────────────
    matches = match_tables(
        left_df=left_df,
        right_df=right_df,
        left_on=left_on,
        right_on=right_on,
        left_id=left_id,
        right_id=right_id,
        block_on=block_on,
        top_k=top_k,
        scorer=scorer,
        high_threshold=high_threshold,
        medium_threshold=medium_threshold,
        min_margin_high=min_margin_high,
        reject_threshold=reject_threshold,
        use_llm=use_llm,
        llm_client=llm_client,
        keep_all_candidates=(how == "all"),
        n_jobs=n_jobs,
    )

    # ── 2. Filter by `how` ────────────────────────────────────────────────────
    if how == "inner":
        matches = matches[matches["final_decision"] == True].copy()
    # "left" and "all" keep everything — unmatched left rows handled below

    # ── 3. Build left/right index lookups ─────────────────────────────────────
    left_copy  = left_df.copy()
    right_copy = right_df.copy()

    # Use the id column as join key; fall back to a synthetic _left_row_idx
    if left_id:
        left_key = left_id
    else:
        left_copy = left_copy.reset_index(drop=False)
        left_key  = left_copy.columns[0]  # "index" or the original index name

    if right_id:
        right_key = right_id
    else:
        right_copy = right_copy.reset_index(drop=False)
        right_key  = right_copy.columns[0]

    # ── 4. Merge matches → left columns ───────────────────────────────────────
    # Rename columns that clash with right table
    left_data_cols  = [c for c in left_copy.columns  if c != left_key]
    right_data_cols = [c for c in right_copy.columns if c != right_key]
    clashing = set(left_data_cols) & set(right_data_cols)

    left_rename  = {c: c + suffixes[0]  for c in clashing}
    right_rename = {c: c + suffixes[1] for c in clashing}
    left_copy  = left_copy.rename(columns=left_rename)
    right_copy = right_copy.rename(columns=right_rename)

    result = matches.merge(
        left_copy,
        left_on="left_id", right_on=left_key, how="left",
    ).drop(columns=[left_key], errors="ignore")

    result = result.merge(
        right_copy,
        left_on="right_id", right_on=right_key, how="left",
    ).drop(columns=[right_key], errors="ignore")

    # ── 5. Handle "left" — add unmatched left rows with NaN right columns ─────
    if how == "left":
        matched_left_ids = set(result["left_id"].dropna())
        all_left_ids     = set(left_df[left_id].tolist() if left_id
                               else left_df.index.tolist())
        unmatched = all_left_ids - matched_left_ids
        if unmatched:
            unmatched_df = left_df[
                (left_df[left_id].isin(unmatched) if left_id
                 else left_df.index.isin(unmatched))
            ].copy()
            # Pad with NaN for all match + right columns
            for col in result.columns:
                if col not in unmatched_df.columns:
                    unmatched_df[col] = pd.NA
            result = pd.concat([result, unmatched_df[result.columns]], ignore_index=True)

    # ── 6. Tidy up: rename match-score columns, drop internal plumbing ────────
    drop_cols = ["left_id", "right_id", "left_value", "right_value",
                 "final_decision", "llm_same_entity", "llm_confidence",
                 "score_margin_to_second_best"]
    result = result.drop(columns=[c for c in drop_cols if c in result.columns],
                         errors="ignore")

    if match_score_col:
        result = result.rename(columns={"fuzzy_score": match_score_col})
    else:
        result = result.drop(columns=["fuzzy_score"], errors="ignore")

    if reliability_col:
        result = result.rename(columns={"reliability_label": reliability_col})
    else:
        result = result.drop(columns=["reliability_label"], errors="ignore")

    return result.reset_index(drop=True)


def fuzzy_join_geodataframes(
    left_gdf,
    right_gdf,
    left_on: str,
    right_on: str,
    left_id: Optional[str] = None,
    right_id: Optional[str] = None,
    how: str = "inner",
    suffixes: tuple = ("_left", "_right"),
    block_on: Optional[str] = None,
    spatial_block_degrees: float = 5.0,
    max_distance_km: float = 500.0,
    top_k: int = 5,
    scorer: str = "WRatio",
    high_threshold: float = 92,
    medium_threshold: float = 80,
    min_margin_high: float = 8,
    reject_threshold: float = 60,
    use_llm: bool = False,
    llm_client: Optional[LLMClient] = None,
    n_jobs: int = 1,
    geometry: str = "left",
    match_score_col: str = "_fuzzy_score",
    geo_score_col: str = "_geo_score",
    reliability_col: str = "_reliability",
):
    """Fuzzy merge of two GeoDataFrames — the spatial equivalent of ``gpd.sjoin()``.

    Performs a fuzzy name-based join enriched with spatial proximity, then
    brings all attribute columns from both GeoDataFrames into a single
    result — exactly like ``geopandas.GeoDataFrame.merge()`` or ``sjoin()``,
    but matching on fuzzy names instead of exact keys or spatial predicates.

    .. code-block:: python

        # Exact spatial join (geopandas)
        result = gpd.sjoin(left, right, how="inner", predicate="intersects")

        # Fuzzy name join with spatial confidence (this function)
        result = fuzzy_join_geodataframes(
            left, right, left_on="name", right_on="name"
        )

    Parameters
    ----------
    left_gdf, right_gdf:
        Input GeoDataFrames in geographic CRS (EPSG:4326 recommended).
    left_on, right_on:
        Column names to fuzzy-match on.
    how:
        Join type — ``"inner"`` (default), ``"left"``, or ``"all"``.
    suffixes:
        Suffixes for clashing non-key column names.
    spatial_block_degrees:
        Grid-cell size for spatial blocking (degrees). ``None`` disables blocking.
    max_distance_km:
        Distance at which geo score reaches 0. Tune to dataset spread.
    geometry:
        Which geometry to carry in the result: ``"left"`` (default), ``"right"``,
        or ``"both"`` (adds ``geometry_left`` and ``geometry_right`` columns).
    match_score_col, geo_score_col, reliability_col:
        Names of the diagnostic columns appended to the result.
        Set to ``None`` to suppress any of them.

    Returns
    -------
    GeoDataFrame with all columns from both tables plus diagnostic columns,
    with geometry from the chosen side (or both).

    Examples
    --------
    >>> result = fuzzy_join_geodataframes(
    ...     admin_left, admin_right,
    ...     left_on="NAME_1", right_on="name",
    ...     how="left",
    ...     spatial_block_degrees=3.0,
    ...     max_distance_km=200.0,
    ... )
    >>> result.plot(column="_reliability", legend=True)
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "fuzzy_join_geodataframes() requires geopandas. "
            "Install with: pip install 'fuzzy_llm_matcher[geo]'"
        ) from exc

    # ── 1. Run the geo matching pipeline ─────────────────────────────────────
    matches = match_geodataframes(
        left_gdf=left_gdf,
        right_gdf=right_gdf,
        left_on=left_on,
        right_on=right_on,
        left_id=left_id,
        right_id=right_id,
        block_on=block_on,
        spatial_block_degrees=spatial_block_degrees,
        max_distance_km=max_distance_km,
        top_k=top_k,
        scorer=scorer,
        high_threshold=high_threshold,
        medium_threshold=medium_threshold,
        min_margin_high=min_margin_high,
        reject_threshold=reject_threshold,
        use_llm=use_llm,
        llm_client=llm_client,
        n_jobs=n_jobs,
        keep_all_candidates=(how == "all"),
        return_geometry=False,
    )

    # ── 2. Filter by join type ────────────────────────────────────────────────
    if how == "inner":
        matches = matches[matches["final_decision"] == True].copy()

    # ── 3. Prepare attribute tables (drop geometry column from GeoDataFrames) ──
    right_is_geo_join = hasattr(right_gdf, "geometry") and hasattr(right_gdf, "crs")
    left_attr  = pd.DataFrame(left_gdf.drop(columns=left_gdf.geometry.name))
    right_attr = pd.DataFrame(
        right_gdf.drop(columns=right_gdf.geometry.name) if right_is_geo_join else right_gdf
    )

    left_key  = left_id  if left_id  else "_left_idx"
    right_key = right_id if right_id else "_right_idx"
    if not left_id:
        left_attr[left_key]  = left_gdf.index
    if not right_id:
        right_attr[right_key] = right_gdf.index
    # Rename clashing attribute columns
    left_data  = [c for c in left_attr.columns  if c != left_key]
    right_data = [c for c in right_attr.columns if c != right_key]
    clashing   = set(left_data) & set(right_data)
    left_attr  = left_attr.rename(columns={c: c + suffixes[0]  for c in clashing})
    right_attr = right_attr.rename(columns={c: c + suffixes[1] for c in clashing})

    # ── 4. Merge attributes into result ──────────────────────────────────────
    result = matches.merge(left_attr,  left_on="left_id",  right_on=left_key,  how="left"
                           ).drop(columns=[left_key], errors="ignore")
    result = result.merge(right_attr, left_on="right_id", right_on=right_key, how="left"
                          ).drop(columns=[right_key], errors="ignore")

    # ── 5. Unmatched left rows for "left" join ────────────────────────────────
    if how == "left":
        matched_ids = set(result["left_id"].dropna())
        all_ids     = set(left_gdf[left_id].tolist() if left_id
                          else left_gdf.index.tolist())
        unmatched   = all_ids - matched_ids
        if unmatched:
            mask = (left_gdf[left_id].isin(unmatched) if left_id
                    else left_gdf.index.isin(unmatched))
            unmatched_rows = pd.DataFrame(left_gdf[mask].drop(
                columns=left_gdf.geometry.name))
            for col in result.columns:
                if col not in unmatched_rows.columns:
                    unmatched_rows[col] = pd.NA
            result = pd.concat([result, unmatched_rows[result.columns]],
                                ignore_index=True)

    # ── 6. Attach geometry / geometries ──────────────────────────────────────
    if left_id:
        left_geom_lookup = left_gdf.set_index(left_id).geometry
    else:
        left_geom_lookup = left_gdf.geometry.copy()
        left_geom_lookup.index.name = "_left_idx"

    # Right geometry only available when right_gdf is a GeoDataFrame
    right_geom_lookup = None
    if right_is_geo_join:
        if right_id:
            right_geom_lookup = right_gdf.set_index(right_id).geometry
        else:
            right_geom_lookup = right_gdf.geometry.copy()
            right_geom_lookup.index.name = "_right_idx"

    if geometry in ("left", "both"):
        result["geometry"] = result["left_id"].map(left_geom_lookup)
    if geometry == "right":
        # Fall back to left geometry if right has no geometry
        if right_geom_lookup is not None:
            result["geometry"] = result["right_id"].map(right_geom_lookup)
        else:
            result["geometry"] = result["left_id"].map(left_geom_lookup)
    if geometry == "both":
        if right_geom_lookup is not None:
            result["geometry_right"] = result["right_id"].map(right_geom_lookup)
        result = result.rename(columns={"geometry": "geometry_left"})
        result = gpd.GeoDataFrame(result, geometry="geometry_left", crs=left_gdf.crs)
    else:
        result = gpd.GeoDataFrame(result, geometry="geometry", crs=left_gdf.crs)

    # ── 7. Rename / suppress diagnostic columns ───────────────────────────────
    drop_cols = ["left_id", "right_id", "left_value", "right_value",
                 "final_decision", "llm_same_entity", "llm_confidence",
                 "score_margin_to_second_best",
                 "left_lat", "left_lon", "right_lat", "right_lon"]
    result = result.drop(columns=[c for c in drop_cols if c in result.columns],
                         errors="ignore")

    if match_score_col:
        result = result.rename(columns={"fuzzy_score": match_score_col})
    else:
        result = result.drop(columns=["fuzzy_score"], errors="ignore")

    if geo_score_col:
        result = result.rename(columns={"score_geo_distance": geo_score_col})
    else:
        result = result.drop(columns=["score_geo_distance"], errors="ignore")

    if reliability_col:
        result = result.rename(columns={"reliability_label": reliability_col})
    else:
        result = result.drop(columns=["reliability_label"], errors="ignore")

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fuzzy dissolve
# ---------------------------------------------------------------------------

def fuzzy_dissolve(
    left_gdf,
    right_gdf,
    left_on: str,
    right_on: str,
    left_id: Optional[str] = None,
    right_id: Optional[str] = None,
    dissolve_op: str = "union",
    aggfunc: Optional[dict] = None,
    block_on: Optional[str] = None,
    spatial_block_degrees: float = 5.0,
    max_distance_km: float = 500.0,
    scorer: str = "WRatio",
    high_threshold: float = 92,
    medium_threshold: float = 80,
    min_margin_high: float = 8,
    reject_threshold: float = 60,
    use_llm: bool = False,
    llm_client: Optional[LLMClient] = None,
    n_jobs: int = 1,
):
    """Fuzzy-match and dissolve/union geometries of matched pairs.

    First finds confirmed fuzzy matches between ``left_gdf`` and
    ``right_gdf``, then **dissolves** (merges) the geometry of each
    matched pair using the chosen spatial operation. This is analogous to
    ``geopandas.GeoDataFrame.dissolve()``, but the grouping is defined by
    fuzzy name matching rather than an exact shared attribute.

    Use cases
    ---------
    - **Merging admin boundaries** that represent the same region in two
      different datasets (e.g. GADM and OpenStreetMap admin polygons for
      the same province) → ``dissolve_op="union"``
    - **Finding the intersection** of matched polygon pairs to quantify
      spatial agreement → ``dissolve_op="intersection"``
    - **Centroid-based collapse** of matched point clouds from two
      surveys → ``dissolve_op="centroid"``
    - **Deduplication** of overlapping polygon datasets — merge the
      geometries of fuzzy duplicates into a single feature.

    Parameters
    ----------
    left_gdf, right_gdf:
        Input GeoDataFrames in geographic CRS (EPSG:4326 recommended).
    left_on, right_on:
        Columns to fuzzy-match on.
    dissolve_op:
        How to combine geometries of matched pairs:
        - ``"union"``        — spatial union of left + right geometry
        - ``"intersection"`` — spatial intersection (overlap only)
        - ``"envelope"``     — bounding box of both geometries
        - ``"centroid"``     — midpoint between the two centroids
        - ``"left"``         — keep left geometry unchanged (default join behaviour)
        - ``"right"``        — keep right geometry unchanged
    aggfunc:
        Dictionary of ``{column: aggregation_function}`` applied to
        non-geometry columns of the dissolved pairs, e.g.
        ``{"population": "sum", "area_km2": "mean"}``.
        Columns not in ``aggfunc`` are taken from the left row.
    spatial_block_degrees, max_distance_km, scorer,
    high_threshold, medium_threshold, min_margin_high, reject_threshold,
    use_llm, llm_client, n_jobs:
        Forwarded to :func:`match_geodataframes`.

    Returns
    -------
    GeoDataFrame with one row per confirmed fuzzy match. Geometry is the
    dissolved result of the pair. Diagnostic columns
    ``_fuzzy_score``, ``_geo_score``, ``_reliability`` are included.

    Examples
    --------
    >>> dissolved = fuzzy_dissolve(
    ...     gadm_polygons, osm_polygons,
    ...     left_on="NAME_1", right_on="name",
    ...     dissolve_op="union",
    ...     aggfunc={"population": "mean"},
    ... )
    >>> dissolved.plot(column="_reliability", legend=True, figsize=(12, 8))
    """
    try:
        import geopandas as gpd
        from shapely.ops import unary_union
        import warnings
    except ImportError as exc:
        raise ImportError(
            "fuzzy_dissolve() requires geopandas and shapely. "
            "Install with: pip install 'fuzzy_llm_matcher[geo]'"
        ) from exc

    valid_ops = ("union", "intersection", "envelope", "centroid", "left", "right")
    if dissolve_op not in valid_ops:
        raise ValueError(f"dissolve_op must be one of {valid_ops}, got {dissolve_op!r}")

    # ── 1. Match ──────────────────────────────────────────────────────────────
    matches = match_geodataframes(
        left_gdf=left_gdf,
        right_gdf=right_gdf,
        left_on=left_on,
        right_on=right_on,
        left_id=left_id,
        right_id=right_id,
        block_on=block_on,
        spatial_block_degrees=spatial_block_degrees,
        max_distance_km=max_distance_km,
        top_k=5,
        scorer=scorer,
        high_threshold=high_threshold,
        medium_threshold=medium_threshold,
        min_margin_high=min_margin_high,
        reject_threshold=reject_threshold,
        use_llm=use_llm,
        llm_client=llm_client,
        n_jobs=n_jobs,
        keep_all_candidates=False,
        return_geometry=False,
    )
    matches = matches[matches["final_decision"] == True].copy()

    if matches.empty:
        return gpd.GeoDataFrame(columns=["geometry"], crs=left_gdf.crs)

    # ── 2. Build geometry lookups ─────────────────────────────────────────────
    right_is_geo_dissolve = hasattr(right_gdf, "geometry") and hasattr(right_gdf, "crs")

    if left_id:
        left_geom  = left_gdf.set_index(left_id).geometry
        left_attrs = left_gdf.set_index(left_id).drop(columns=left_gdf.geometry.name)
    else:
        left_geom  = left_gdf.geometry
        left_attrs = left_gdf.drop(columns=left_gdf.geometry.name)

    if right_is_geo_dissolve:
        if right_id:
            right_geom = right_gdf.set_index(right_id).geometry
        else:
            right_geom = right_gdf.geometry
    else:
        # Plain DataFrame on right — no geometry available; treat as left-only
        right_geom = None
        if dissolve_op not in ("left", "centroid"):
            dissolve_op = "left"  # silently fall back

    # ── 3. Dissolve geometry per matched pair ─────────────────────────────────
    dissolved_geoms = []
    agg_rows        = []

    for _, row in matches.iterrows():
        lid = row["left_id"]
        rid = row["right_id"]

        lg = left_geom.get(lid)
        rg = right_geom.get(rid) if right_geom is not None else None

        if lg is None and rg is None:
            continue

        if dissolve_op == "union":
            geom = unary_union([g for g in (lg, rg) if g is not None])
        elif dissolve_op == "intersection":
            geom = lg.intersection(rg) if (lg is not None and rg is not None) else (lg or rg)
        elif dissolve_op == "envelope":
            geom = unary_union([g for g in (lg, rg) if g is not None]).envelope
        elif dissolve_op == "centroid":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pts = [g.centroid for g in (lg, rg) if g is not None]
            from shapely.geometry import MultiPoint
            geom = MultiPoint(pts).centroid
        elif dissolve_op == "left":
            geom = lg
        else:  # "right"
            geom = rg

        dissolved_geoms.append(geom)

        # Base attribute row from left
        base = {}
        if lid in left_attrs.index:
            base = left_attrs.loc[lid].to_dict()

        # Apply aggfunc for numeric columns
        if aggfunc:
            for col, func in aggfunc.items():
                collected = []
                # Only look up the column if it exists in that table
                if col in left_gdf.columns:
                    mask_l = (left_gdf[left_id] == lid) if left_id else (left_gdf.index == lid)
                    collected.extend(left_gdf.loc[mask_l, col].dropna().tolist())
                if col in right_gdf.columns:
                    mask_r = (right_gdf[right_id] == rid) if right_id else (right_gdf.index == rid)
                    collected.extend(right_gdf.loc[mask_r, col].dropna().tolist())
                combined = [v for v in collected if v is not None and v == v]
                if combined:
                    if func in ("sum", sum):
                        base[col] = sum(combined)
                    elif func in ("mean", "average"):
                        base[col] = sum(combined) / len(combined)
                    elif func in ("min", min):
                        base[col] = min(combined)
                    elif func in ("max", max):
                        base[col] = max(combined)
                    elif callable(func):
                        base[col] = func(combined)

        base["_fuzzy_score"]  = row.get("fuzzy_score")
        base["_geo_score"]    = row.get("score_geo_distance")
        base["_reliability"]  = row.get("reliability_label")
        base["_left_name"]    = row.get("left_value")
        base["_right_name"]   = row.get("right_value")
        base["_dissolve_op"]  = dissolve_op

        agg_rows.append(base)

    result = gpd.GeoDataFrame(agg_rows, geometry=dissolved_geoms, crs=left_gdf.crs)
    return result.reset_index(drop=True)
