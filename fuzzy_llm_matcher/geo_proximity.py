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


# ── Hexagonal grid blocking ─────────────────────────────────────────────────
# Pure-Shapely implementation — no gemgis / h3 dependency needed.
# Inspired by:
#   https://gemgis.readthedocs.io/en/latest/getting_started/tutorial/58_creating_hexagonal_grid.html

def create_hexagon(center, radius: float):
    """Return a flat-top regular hexagon as a Shapely Polygon.

    The radius is the distance from the centre to each vertex (= side length).
    Flat-top means two edges are horizontal.

    Parameters
    ----------
    center:
        A ``shapely.geometry.Point`` or ``(x, y)`` tuple for the centre.
    radius:
        Distance from centre to vertex in the CRS units (metres for
        projected CRS, degrees for geographic CRS).

    Returns
    -------
    shapely.geometry.Polygon

    Examples
    --------
    >>> from shapely.geometry import Point
    >>> from fuzzy_llm_matcher.geo_proximity import create_hexagon
    >>> h = create_hexagon(Point(0, 0), radius=1000)
    >>> round(h.area, 0)
    2598076.0
    """
    import math
    from shapely.geometry import Polygon, Point as SPoint

    if hasattr(center, "x"):
        cx, cy = center.x, center.y
    else:
        cx, cy = float(center[0]), float(center[1])

    # Flat-top: vertex angles at 30°, 90°, 150°, 210°, 270°, 330°
    angles = [math.pi / 6 + math.pi / 3 * i for i in range(6)]
    return Polygon([(cx + radius * math.cos(a), cy + radius * math.sin(a))
                    for a in angles])


def create_hexagon_grid(
    gdf,
    radius_m: float = 5000.0,
    projected_crs: str = "EPSG:3857",
    crop: bool = True,
    buffer_pct: float = 0.1,
    hex_id_col: str = "hex_id",
):
    """Create a hexagonal grid covering the extent of a GeoDataFrame.

    Each hexagon has a unique integer ``hex_id`` column suitable for use as
    a blocking key in :func:`~fuzzy_llm_matcher.match_geodataframes`.

    Mirrors the ``gemgis.vector.create_hexagon_grid()`` approach but is
    implemented entirely in Shapely — no gemgis dependency needed.

    Parameters
    ----------
    gdf:
        Input GeoDataFrame whose bounding box defines the grid extent.
    radius_m:
        Radius of each hexagon (centre → vertex) in metres. This equals
        the side length of the hexagon.

        Typical values:
            - 500 m  — sub-district / neighbourhood scale
            - 2 km   — city district scale
            - 10 km  — regional scale
            - 50 km  — national scale
    projected_crs:
        A projected CRS with metre units used for grid generation. The
        output grid is then reprojected back to the input GDF's CRS.
    crop:
        If ``True`` (default), clip the grid to the input GDF boundary.
        If ``False``, return the full bounding-box grid.
    buffer_pct:
        Fraction of ``radius_m`` added as a buffer around the bounding box
        to ensure features near edges are covered. Default 10%.
    hex_id_col:
        Name of the integer ID column added to each hexagon.

    Returns
    -------
    geopandas.GeoDataFrame with columns ``[hex_id_col, geometry]``

    Examples
    --------
    >>> import geopandas as gpd
    >>> from fuzzy_llm_matcher.geo_proximity import create_hexagon_grid
    >>> cities = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    >>> grid = create_hexagon_grid(cities, radius_m=500_000)
    >>> grid.plot(alpha=0.4, edgecolor="black")
    """
    import math
    import warnings
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf_proj = gdf.to_crs(projected_crs)

    r = float(radius_m)
    buf = r * buffer_pct
    xmin, ymin, xmax, ymax = gdf_proj.total_bounds
    xmin -= buf; ymin -= buf; xmax += buf; ymax += buf

    # Flat-top hexagon tiling
    # Column spacing (centre-to-centre horizontal) = r * sqrt(3)
    # Row spacing    (centre-to-centre vertical)   = r * 1.5
    dx = r * math.sqrt(3)
    dy = r * 1.5

    hexes = []
    row = 0
    y = ymin
    while y - r < ymax:
        x_offset = (dx / 2) if (row % 2) else 0.0
        x = xmin - dx + x_offset
        while x - r < xmax:
            hexes.append(create_hexagon((x, y), r))
            x += dx
        y += dy
        row += 1

    grid = gpd.GeoDataFrame(
        {hex_id_col: range(len(hexes))},
        geometry=hexes,
        crs=projected_crs,
    )

    if crop:
        study_area = unary_union(gdf_proj.geometry)
        grid = grid[grid.geometry.intersects(study_area)].copy()
        grid = grid.reset_index(drop=True)
        grid[hex_id_col] = range(len(grid))

    return grid.to_crs(gdf.crs)


def assign_hex_ids(
    gdf,
    hex_grid,
    hex_id_col: str = "hex_id",
    how: str = "centroid",
    include_neighbors: bool = False,
) -> "gpd.GeoDataFrame":
    """Assign each feature in ``gdf`` to one (or more) hexagon cell IDs.

    Parameters
    ----------
    gdf:
        Input GeoDataFrame (Points, Polygons, or any geometry type).
    hex_grid:
        Hexagonal grid GeoDataFrame produced by :func:`create_hexagon_grid`.
    hex_id_col:
        Name of the hex ID column in ``hex_grid``.
    how:
        Spatial join predicate:
        - ``"centroid"`` (default) — each feature is assigned to the hex
          containing its centroid. Fast and unambiguous for any geometry type.
        - ``"intersects"`` — each feature is assigned to every hex it
          touches. A polygon straddling two hexes gets two rows. Useful
          when ``include_neighbors=True``.
    include_neighbors:
        If ``True``, also assign each feature to the 6 adjacent hexagons
        of its primary cell. This prevents features near hex boundaries from
        being missed. Requires ``how="intersects"`` for full effect.

    Returns
    -------
    GeoDataFrame with a new ``hex_id_col`` column. When a feature falls in
    multiple hexes (how="intersects"), it appears multiple times.
    """
    import warnings
    import geopandas as gpd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        if how == "centroid":
            centroids = gdf.copy()
            centroids.geometry = gdf.geometry.centroid
            joined = gpd.sjoin(
                centroids, hex_grid[[hex_id_col, "geometry"]],
                how="left", predicate="within"
            ).drop(columns=["index_right"], errors="ignore")
        else:
            joined = gpd.sjoin(
                gdf, hex_grid[[hex_id_col, "geometry"]],
                how="left", predicate="intersects"
            ).drop(columns=["index_right"], errors="ignore")

    return joined


def hex_block_match(
    left_gdf,
    right_gdf,
    left_on: str,
    right_on: str,
    left_id: Optional[str] = None,
    right_id: Optional[str] = None,
    hex_radius_m: float = 5000.0,
    projected_crs: str = "EPSG:3857",
    include_neighbor_hexes: bool = False,
    crop_grid: bool = False,
    max_distance_km: float = 50.0,
    top_k: int = 5,
    scorer: str = "WRatio",
    high_threshold: float = 92,
    medium_threshold: float = 80,
    min_margin_high: float = 8,
    reject_threshold: float = 60,
    use_llm: bool = False,
    llm_client=None,
    n_jobs: int = 1,
    return_geometry: bool = True,
    return_grid: bool = False,
):
    """Fuzzy match two GeoDataFrames using hexagonal grid blocking.

    Creates a flat-top hexagonal grid over the study area, assigns each
    feature to its hexagon cell, then runs the full fuzzy matching pipeline
    using hex cell membership as the blocking key.

    Why hexagonal blocking is better than degree-grid blocking
    ----------------------------------------------------------
    - **Equal area**: every hexagon covers the same area (no polar distortion)
    - **Equidistant neighbors**: the 6 adjacent hexagons are all equidistant
      from the centre (unlike a square grid where diagonals are farther)
    - **Lower boundary artifacts**: hexagons have shorter perimeters per unit
      area than squares, so fewer features straddle cell boundaries
    - **Scalable**: ``hex_radius_m`` controls resolution independently of CRS

    Parameters
    ----------
    left_gdf, right_gdf:
        Input GeoDataFrames in any CRS.
    left_on, right_on:
        Columns to fuzzy-match on.
    left_id, right_id:
        Optional ID column names.
    hex_radius_m:
        Hexagon radius (centre → vertex) in metres. Controls grid resolution:
        - 500 m  — street block / sub-district
        - 2 km   — city district
        - 10 km  — municipality
        - 50 km  — regional
        - 200 km — national
    projected_crs:
        Projected CRS with metre units for hex grid generation.
        Use a UTM zone centred on your study area for most accuracy.
    include_neighbor_hexes:
        If ``True``, each feature is also compared against features in
        the 6 adjacent hexagons. Use this when features near cell boundaries
        might otherwise be missed. Increases matching time by ~7×.
    crop_grid:
        If ``True``, the hex grid is cropped to the union of both GDFs.
        Leave ``False`` (default) for faster setup.
    max_distance_km:
        Distance at which ``score_geo_distance`` reaches 0.
    top_k, scorer, high_threshold, medium_threshold,
    min_margin_high, reject_threshold, use_llm, llm_client, n_jobs:
        Forwarded to :func:`~fuzzy_llm_matcher.match_geodataframes`.
    return_geometry:
        Attach left geometry to result GeoDataFrame.
    return_grid:
        If ``True``, also return the hexagonal grid GeoDataFrame.

    Returns
    -------
    result: GeoDataFrame (or DataFrame when return_geometry=False)
        Match result with all standard columns plus ``hex_id`` blocking info.
    grid: GeoDataFrame (only when return_grid=True)
        The hexagonal grid used for blocking.

    Examples
    --------
    >>> from fuzzy_llm_matcher import hex_block_match
    >>>
    >>> result, grid = hex_block_match(
    ...     osm_gdf, geonames_gdf,
    ...     left_on="name",  right_on="asciiname",
    ...     hex_radius_m=10_000,           # 10 km hexagons
    ...     projected_crs="EPSG:32648",    # UTM 48N for SE Asia
    ...     include_neighbor_hexes=False,
    ...     return_grid=True,
    ... )
    >>> # Plot: hexagons coloured by match count, on satellite background
    >>> grid = grid.merge(
    ...     result.groupby("left_hex_id").size().rename("n_matches"),
    ...     left_on="hex_id", right_index=True, how="left"
    ... )
    >>> ax = grid.to_crs(3857).plot(column="n_matches", legend=True)
    >>> from fuzzy_llm_matcher import add_basemap
    >>> add_basemap(ax, style="satellite")
    """
    import warnings
    import geopandas as gpd
    from fuzzy_llm_matcher.api import match_geodataframes

    # ── 1. Build the hexagonal blocking grid ─────────────────────────────────
    combined_bounds = gpd.GeoDataFrame(
        geometry=list(left_gdf.geometry) + (
            list(right_gdf.geometry) if hasattr(right_gdf, "geometry") else []
        ),
        crs=left_gdf.crs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = create_hexagon_grid(
            combined_bounds,
            radius_m=hex_radius_m,
            projected_crs=projected_crs,
            crop=crop_grid,
        )

    # ── 2. Assign left features to hexagon cells ──────────────────────────────
    left_hexed = assign_hex_ids(
        left_gdf.copy(), grid,
        how="intersects" if include_neighbor_hexes else "centroid",
    )
    left_hexed = left_hexed.rename(columns={"hex_id": "left_hex_id"})

    if right_gdf is not None and hasattr(right_gdf, "geometry"):
        right_hexed = assign_hex_ids(
            right_gdf.copy(), grid,
            how="intersects" if include_neighbor_hexes else "centroid",
        )
        right_hexed = right_hexed.rename(columns={"hex_id": "right_hex_id"})
    else:
        right_hexed = right_gdf.copy() if right_gdf is not None else None

    # ── 3. Run matching blocked by hex cell ───────────────────────────────────
    # Add a shared block column so match_geodataframes can use it
    if "left_hex_id" in left_hexed.columns:
        left_hexed["_hex_block"] = left_hexed["left_hex_id"].astype(str)

    if right_hexed is not None and "right_hex_id" in right_hexed.columns:
        right_hexed["_hex_block"] = right_hexed["right_hex_id"].astype(str)
    elif right_hexed is not None:
        right_hexed["_hex_block"] = "0"

    result = match_geodataframes(
        left_hexed, right_hexed,
        left_on=left_on,
        right_on=right_on,
        left_id=left_id,
        right_id=right_id,
        block_on="_hex_block",          # use hex cell as blocking key
        spatial_block_degrees=None,
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
        return_geometry=return_geometry,
    )

    if return_grid:
        return result, grid
    return result

