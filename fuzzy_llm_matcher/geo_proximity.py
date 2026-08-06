"""Spatial proximity candidate generation and combined scoring.

This module provides two capabilities that complement the string-based
fuzzy matching pipeline:

1. **sjoin_nearest_candidates()** — uses ``geopandas.sjoin_nearest()`` to
   find spatially nearby candidates first, then scores each pair with a
   fuzzy string scorer.  This is the spatial-first approach:

       spatial proximity → candidate pairs → name similarity scoring

   It is the cleanest approach for datasets where both sources carry
   reliable geometry (points, polygons, lines) and you want a hard distance
   cut-off before any name comparison.

2. **combined_score()** — computes a weighted combination of name similarity
   and distance scores:

       combined = w_name × name_score + w_dist × distance_score

   Mirrors the standard formula used in the spatial data-integration
   literature.

3. **TileBasemap** — a thin wrapper around ``contextily`` that adds
   OpenStreetMap, CartoDB, ESRI Satellite, or any XYZ tile provider as a
   background to any ``matplotlib`` axes, including axes created by
   geopandas ``.plot()``.

Quick example
-------------
>>> import geopandas as gpd
>>> from fuzzy_llm_matcher.geo_proximity import sjoin_nearest_candidates, combined_score
>>>
>>> candidates = sjoin_nearest_candidates(
...     left_gdf, right_gdf,
...     left_name_col="name", right_name_col="name",
...     max_distance_m=500,
...     projected_crs="EPSG:32633",
... )
>>> matches = combined_score(candidates, w_name=0.7, w_dist=0.3, threshold=75)
"""

from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd


# ── Tile providers ──────────────────────────────────────────────────────────

class TileBasemap:
    """Add a tile-based background map to a matplotlib axes object.

    Uses ``contextily`` to fetch and reproject web map tiles from any
    XYZ tile provider (OpenStreetMap, CartoDB, ESRI, Stamen, etc.).

    Parameters
    ----------
    provider:
        A ``contextily`` / ``xyzservices`` tile provider object, or a
        URL template string ``"https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"``.
        Defaults to ``CartoDB.DarkMatter`` which matches the dark theme
        used by the existing fuzzy_llm_matcher figures.

    Preset class attributes
    -----------------------
    ``TileBasemap.DARK``         CartoDB DarkMatter  (default, matches package theme)
    ``TileBasemap.LIGHT``        CartoDB Positron
    ``TileBasemap.OSM``          OpenStreetMap Mapnik
    ``TileBasemap.SATELLITE``    ESRI WorldImagery  (closest to Google Satellite)
    ``TileBasemap.GOOGLE_LIKE``  ESRI WorldStreetMap (closest to Google Maps Streets)
    ``TileBasemap.TOPO``         ESRI WorldTopoMap

    Examples
    --------
    >>> import geopandas as gpd
    >>> ax = gdf.to_crs(3857).plot(alpha=0.6, figsize=(10, 8))
    >>> TileBasemap.DARK.add(ax, zoom=10)

    >>> # Or use the convenience function:
    >>> from fuzzy_llm_matcher.geo_proximity import add_basemap
    >>> add_basemap(ax, style="satellite", zoom=10)
    """

    def __init__(self, provider):
        self._provider = provider

    def add(self, ax, zoom: int = "auto", alpha: float = 1.0, **kwargs):
        """Add this basemap to ``ax``.

        The axes must already be in Web Mercator (EPSG:3857), or the CRS
        must be set. Use ``gdf.to_crs(3857)`` before plotting.

        Parameters
        ----------
        ax:
            A matplotlib axes object (e.g. from ``gdf.plot()``).
        zoom:
            Tile zoom level. ``"auto"`` lets contextily choose. Higher values
            load more detail but fetch more tiles (slow for large extents).
        alpha:
            Tile opacity (0–1).
        """
        try:
            import contextily as ctx
        except ImportError:
            warnings.warn(
                "contextily is required for basemap support. "
                "Install with: pip install 'fuzzy_llm_matcher[geo]'",
                stacklevel=2,
            )
            return ax

        zoom_kwarg = {} if zoom == "auto" else {"zoom": zoom}
        ctx.add_basemap(ax, source=self._provider, alpha=alpha, **zoom_kwarg, **kwargs)
        return ax


def _make_presets():
    try:
        import contextily as ctx
        return {
            "DARK":        TileBasemap(ctx.providers.CartoDB.DarkMatter),
            "LIGHT":       TileBasemap(ctx.providers.CartoDB.Positron),
            "OSM":         TileBasemap(ctx.providers.OpenStreetMap.Mapnik),
            "SATELLITE":   TileBasemap(ctx.providers.Esri.WorldImagery),
            "GOOGLE_LIKE": TileBasemap(ctx.providers.Esri.WorldStreetMap),
            "TOPO":        TileBasemap(ctx.providers.Esri.WorldTopoMap),
        }
    except Exception:
        return {}


_PRESETS = _make_presets()

# Expose as class attributes for convenient access
for _name, _obj in _PRESETS.items():
    setattr(TileBasemap, _name, _obj)


def add_basemap(
    ax,
    style: str = "dark",
    zoom: int = "auto",
    alpha: float = 1.0,
    provider=None,
    **kwargs,
):
    """Add a tile-based background map to a matplotlib axes.

    The axes must be in Web Mercator (EPSG:3857). Convert before plotting:
    ``gdf.to_crs(3857).plot(...)``.

    Parameters
    ----------
    ax:
        Matplotlib axes from ``gdf.plot()``.
    style:
        One of ``"dark"`` (CartoDB DarkMatter), ``"light"`` (CartoDB Positron),
        ``"osm"`` (OpenStreetMap), ``"satellite"`` (ESRI WorldImagery —
        closest to Google Satellite), ``"google"`` (ESRI WorldStreetMap —
        closest to Google Maps Streets), ``"topo"`` (ESRI WorldTopoMap).
    zoom:
        Tile zoom level. ``"auto"`` = let contextily decide.
        Typical values: 5 (country), 8 (region), 10 (city), 12 (district).
    alpha:
        Tile opacity (0–1). 0.6–0.8 works well over data layers.
    provider:
        Override: pass any ``contextily``/``xyzservices`` provider object
        directly. If set, ``style`` is ignored.

    Examples
    --------
    >>> ax = gdf.to_crs(3857).plot(alpha=0.7, figsize=(10, 8))
    >>> add_basemap(ax, style="satellite", zoom=10)

    >>> # Google Maps-like street map
    >>> add_basemap(ax, style="google", zoom=9)

    >>> # Dark theme matching the package's own figures
    >>> add_basemap(ax, style="dark")
    """
    try:
        import contextily as ctx
    except ImportError:
        warnings.warn(
            "contextily is required for basemap support. "
            "pip install 'fuzzy_llm_matcher[geo]'",
            stacklevel=2,
        )
        return ax

    if provider is None:
        style_map = {
            "dark":      ctx.providers.CartoDB.DarkMatter,
            "light":     ctx.providers.CartoDB.Positron,
            "osm":       ctx.providers.OpenStreetMap.Mapnik,
            "satellite": ctx.providers.Esri.WorldImagery,
            "google":    ctx.providers.Esri.WorldStreetMap,
            "topo":      ctx.providers.Esri.WorldTopoMap,
        }
        provider = style_map.get(style.lower(), ctx.providers.CartoDB.DarkMatter)

    zoom_kwarg = {} if zoom == "auto" else {"zoom": zoom}
    ctx.add_basemap(ax, source=provider, alpha=alpha, **zoom_kwarg, **kwargs)
    return ax


# ── Spatial candidate generation ────────────────────────────────────────────

def sjoin_nearest_candidates(
    left_gdf,
    right_gdf,
    left_name_col: str,
    right_name_col: str,
    left_id_col: Optional[str] = None,
    right_id_col: Optional[str] = None,
    max_distance_m: float = 500.0,
    projected_crs: str = "EPSG:3857",
    scorer: str = "token_set_ratio",
    keep_geometry: bool = False,
) -> pd.DataFrame:
    """Find spatially nearby candidate pairs and score their name similarity.

    This is the **spatial-first** approach: start with geometry proximity to
    generate candidates, then score the name pairs. The complementary
    **name-first** approach is :func:`~fuzzy_llm_matcher.match_geodataframes`
    which starts with string similarity and adds a geo score afterwards.

    Use this function when:
    - You have a hard distance requirement (e.g. only match features within 50 m)
    - Geometry quality is high (accurate GPS points, cadastre polygons)
    - The name match is a secondary filter, not the primary signal

    Use :func:`~fuzzy_llm_matcher.match_geodataframes` when:
    - Names are the primary signal (transliterations, abbreviations)
    - Geometry is approximate or missing on one side
    - You need reliability labels and LLM review

    Parameters
    ----------
    left_gdf, right_gdf:
        Input GeoDataFrames in any CRS — they will be reprojected to
        ``projected_crs`` for distance computation.
    left_name_col, right_name_col:
        Column names holding the names to compare.
    left_id_col, right_id_col:
        Optional ID column names. Defaults to index.
    max_distance_m:
        Maximum distance (in metres after reprojection) for a pair to be
        included as a candidate. Equivalent to a hard spatial cut-off.
    projected_crs:
        A projected CRS with metre units for accurate distance computation.
        ``"EPSG:3857"`` (Web Mercator) works globally but distorts at high
        latitudes. Use a UTM zone (e.g. ``"EPSG:32633"`` for central Europe,
        ``"EPSG:32644"`` for South Asia) for more accurate local distances.
    scorer:
        RapidFuzz scorer name: ``"token_set_ratio"`` (default — best for
        admin names with word-order differences), ``"WRatio"``,
        ``"token_sort_ratio"``, ``"ratio"``.
    keep_geometry:
        If True, include left-side geometry in the result.

    Returns
    -------
    pd.DataFrame with columns:
        ``left_id``, ``right_id``, ``left_name``, ``right_name``,
        ``distance_m``, ``name_score``, [``geometry``]

    Examples
    --------
    >>> candidates = sjoin_nearest_candidates(
    ...     osm_points, geonames_points,
    ...     left_name_col="name", right_name_col="asciiname",
    ...     max_distance_m=1000,
    ...     projected_crs="EPSG:32648",  # UTM zone 48N for SE Asia
    ... )
    >>> matches = combined_score(candidates, threshold=70)
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "sjoin_nearest_candidates() requires geopandas. "
            "pip install 'fuzzy_llm_matcher[geo]'"
        ) from exc

    from .utils import get_scorer, normalize_text

    score_fn = get_scorer(scorer)

    # ── 1. Reproject to projected CRS for accurate metre distances ────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        left_proj  = left_gdf.to_crs(projected_crs).copy()
        right_proj = right_gdf.to_crs(projected_crs).copy()

    # ── 2. Set up id columns ──────────────────────────────────────────────────
    if left_id_col:
        left_proj["_left_id"]  = left_proj[left_id_col]
    else:
        left_proj["_left_id"]  = left_proj.index

    if right_id_col:
        right_proj["_right_id"] = right_proj[right_id_col]
    else:
        right_proj["_right_id"] = right_proj.index

    # ── 3. gpd.sjoin_nearest: spatial proximity candidates ───────────────────
    try:
        joined = gpd.sjoin_nearest(
            left_proj[["_left_id", left_name_col, left_proj.geometry.name]],
            right_proj[["_right_id", right_name_col, right_proj.geometry.name]],
            how="left",
            max_distance=max_distance_m,
            distance_col="distance_m",
            lsuffix="left",
            rsuffix="right",
        )
    except TypeError:
        # Older geopandas versions without max_distance parameter
        joined = gpd.sjoin_nearest(
            left_proj[["_left_id", left_name_col, left_proj.geometry.name]],
            right_proj[["_right_id", right_name_col, right_proj.geometry.name]],
            how="left",
            distance_col="distance_m",
            lsuffix="left",
            rsuffix="right",
        )
        joined = joined[joined["distance_m"] <= max_distance_m]

    # ── 4. Drop rows with no match (distance NaN = beyond max_distance) ───────
    joined = joined.dropna(subset=["distance_m"]).copy()

    if joined.empty:
        cols = ["left_id", "right_id", "left_name", "right_name",
                "distance_m", "name_score"]
        if keep_geometry:
            cols.append("geometry")
        return pd.DataFrame(columns=cols)

    # ── 5. Compute fuzzy name score ───────────────────────────────────────────
    # After sjoin_nearest columns are suffixed: name → name_left / name_right
    left_name_joined  = f"{left_name_col}_left"
    right_name_joined = f"{right_name_col}_right"

    # If columns weren't suffixed (identical names), fall back gracefully
    if left_name_joined not in joined.columns:
        left_name_joined = left_name_col
    if right_name_joined not in joined.columns:
        right_name_joined = right_name_col

    left_norms  = joined[left_name_joined].map(normalize_text).tolist()
    right_norms = joined[right_name_joined].map(normalize_text).tolist()

    try:
        from rapidfuzz import process as _rf_process  # type: ignore
        scores = [score_fn(a, b) for a, b in zip(left_norms, right_norms)]
    except ImportError:
        scores = [score_fn(a, b) for a, b in zip(left_norms, right_norms)]

    joined["name_score"] = scores

    # ── 6. Build clean output ─────────────────────────────────────────────────
    result = joined.rename(columns={
        "_left_id":        "left_id",
        "_right_id":       "right_id",
        left_name_joined:  "left_name",
        right_name_joined: "right_name",
    })[["left_id", "right_id", "left_name", "right_name", "distance_m", "name_score"]]

    if keep_geometry:
        result = result.copy()
        result["geometry"] = joined[left_proj.geometry.name].values

    return result.reset_index(drop=True)


# ── Combined score ───────────────────────────────────────────────────────────

def combined_score(
    candidates: pd.DataFrame,
    w_name: float = 0.7,
    w_dist: float = 0.3,
    max_distance_m: float = 500.0,
    threshold: float = 75.0,
    name_col: str = "name_score",
    dist_col: str = "distance_m",
    output_col: str = "combined_score",
    dist_score_col: str = "distance_score",
) -> pd.DataFrame:
    """Compute a weighted combined score from name similarity and distance.

    Implements the standard spatial data-integration formula:

    .. math::

        S_{combined} = w_{name} \\cdot S_{name} + w_{dist} \\cdot S_{dist}

    where:

    - :math:`S_{name}` is the fuzzy string similarity score (0–100)
    - :math:`S_{dist}` is the normalised distance similarity score (100 at
      distance 0, 0 at ``max_distance_m``) — identical to
      :func:`~fuzzy_llm_matcher.fuzzy_scores.geo_distance_score`

    Parameters
    ----------
    candidates:
        Output of :func:`sjoin_nearest_candidates` or any DataFrame with
        ``name_score`` and ``distance_m`` columns.
    w_name, w_dist:
        Weights for name and distance scores (should sum to 1.0 but any
        positive values are accepted).
    max_distance_m:
        Distance at which ``distance_score`` reaches 0. Should match the
        value used in :func:`sjoin_nearest_candidates`.
    threshold:
        Minimum ``combined_score`` to retain in the output.
    name_col, dist_col:
        Column names for name score and distance in ``candidates``.
    output_col:
        Name of the new combined-score column.
    dist_score_col:
        Name of the intermediate normalised distance-score column.

    Returns
    -------
    pd.DataFrame — filtered copy with ``distance_score`` and ``combined_score``
    columns added. Only rows with ``combined_score >= threshold`` are returned.

    Examples
    --------
    >>> candidates = sjoin_nearest_candidates(
    ...     left, right, "name", "name", max_distance_m=500
    ... )
    >>> matches = combined_score(candidates, w_name=0.7, w_dist=0.3, threshold=75)
    >>> matches[["left_name", "right_name", "distance_m", "combined_score"]]
    """
    import numpy as np

    df = candidates.copy()

    # Normalised distance score: 100 at 0 m, 0 at max_distance_m
    df[dist_score_col] = np.clip(
        100.0 * (1.0 - df[dist_col] / max_distance_m), 0.0, 100.0
    )

    df[output_col] = w_name * df[name_col] + w_dist * df[dist_score_col]

    return df[df[output_col] >= threshold].reset_index(drop=True)
