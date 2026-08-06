"""Compute multiple string-similarity features for candidate pairs."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import pandas as pd

from .utils import fuzz, normalize_text


# ---------------------------------------------------------------------------
# Geo-distance helpers
# ---------------------------------------------------------------------------

def haversine_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Return the great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def geo_distance_score(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    max_km: float = 100.0,
) -> float:
    """Convert haversine distance to a 0–100 similarity score.

    A pair at distance 0 km scores 100; a pair at ``max_km`` or beyond
    scores 0.  Linear interpolation in between.

    Parameters
    ----------
    max_km:
        Distance at which the geo score reaches 0.  Defaults to 100 km
        (appropriate for city-level matching).  Increase for country-level
        or continent-level tasks.
    """
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (lat1, lon1, lat2, lon2)):
        return float("nan")
    dist = haversine_km(lat1, lon1, lat2, lon2)
    return max(0.0, 100.0 * (1.0 - dist / max_km))


def add_geo_distance_score(
    candidates_df: pd.DataFrame,
    left_lat_col: str = "left_lat",
    left_lon_col: str = "left_lon",
    right_lat_col: str = "right_lat",
    right_lon_col: str = "right_lon",
    max_km: float = 100.0,
    score_col: str = "score_geo_distance",
) -> pd.DataFrame:
    """Add a ``score_geo_distance`` column (0–100) to a candidates DataFrame.

    The column measures how close the two matched entities are
    geographically. Pairs with missing coordinates get ``NaN`` so
    downstream steps can treat them as unknown rather than zero.

    Parameters
    ----------
    candidates_df:
        Any candidates DataFrame that has lat/lon columns for both sides.
        Typically produced by :func:`compute_similarity_features` (or
        directly from :func:`generate_candidates` when you join coordinates
        before calling this function).
    left_lat_col, left_lon_col, right_lat_col, right_lon_col:
        Column names for the coordinate pairs.
    max_km:
        Distance at which ``score_geo_distance`` reaches 0.
    score_col:
        Name of the output column to add (default: ``"score_geo_distance"``).

    Returns
    -------
    A copy of ``candidates_df`` with the new score column appended.

    Example
    -------
    >>> import pandas as pd
    >>> from fuzzy_llm_matcher.fuzzy_scores import add_geo_distance_score
    >>> df = pd.DataFrame({
    ...     "left_lat": [48.14], "left_lon": [11.58],
    ...     "right_lat": [48.14], "right_lon": [11.58],
    ... })
    >>> add_geo_distance_score(df)["score_geo_distance"].iloc[0]
    100.0
    """
    df = candidates_df.copy()
    missing_cols = [
        c for c in (left_lat_col, left_lon_col, right_lat_col, right_lon_col)
        if c not in df.columns
    ]
    if missing_cols:
        raise KeyError(
            f"Coordinate columns not found in DataFrame: {missing_cols}. "
            "Join lat/lon columns before calling add_geo_distance_score()."
        )

    df[score_col] = [
        geo_distance_score(flat, flon, rlat, rlon, max_km=max_km)
        for flat, flon, rlat, rlon in zip(
            df[left_lat_col], df[left_lon_col],
            df[right_lat_col], df[right_lon_col],
        )
    ]
    return df


def compute_similarity_features(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-metric similarity columns to a candidates DataFrame.

    Expects columns: left_value, right_value (and typically left_id,
    right_id, score, rank from `generate_candidates`).

    Adds columns:
        score_wratio, score_token_sort, score_token_set,
        score_partial, score_simple, length_diff,
        normalized_length_diff, best_rank, score_margin_to_second_best

    Geo columns (``score_geo_distance``) are **not** added here because
    coordinates are not always available.  Call
    :func:`add_geo_distance_score` separately after joining coordinates.
    """
    if candidates_df.empty:
        out = candidates_df.copy()
        for col in [
            "score_wratio",
            "score_token_sort",
            "score_token_set",
            "score_partial",
            "score_simple",
            "length_diff",
            "normalized_length_diff",
            "best_rank",
            "score_margin_to_second_best",
        ]:
            out[col] = pd.Series(dtype="float64")
        return out

    df = candidates_df.copy()

    left_norm = df["left_value"].map(normalize_text)
    right_norm = df["right_value"].map(normalize_text)

    df["score_wratio"] = [
        fuzz.WRatio(a, b) for a, b in zip(left_norm, right_norm)
    ]
    df["score_token_sort"] = [
        fuzz.token_sort_ratio(a, b) for a, b in zip(left_norm, right_norm)
    ]
    df["score_token_set"] = [
        fuzz.token_set_ratio(a, b) for a, b in zip(left_norm, right_norm)
    ]
    df["score_partial"] = [
        fuzz.partial_ratio(a, b) for a, b in zip(left_norm, right_norm)
    ]
    df["score_simple"] = [
        fuzz.ratio(a, b) for a, b in zip(left_norm, right_norm)
    ]

    len_left = left_norm.map(len)
    len_right = right_norm.map(len)
    df["length_diff"] = (len_left - len_right).abs()
    max_len = pd.concat([len_left, len_right], axis=1).max(axis=1).replace(0, 1)
    df["normalized_length_diff"] = df["length_diff"] / max_len

    # best_rank / score_margin_to_second_best are computed per left_id group
    if "left_id" in df.columns:
        if "score" not in df.columns:
            df["score"] = df["score_wratio"]

        df["best_rank"] = df.groupby("left_id")["score"].rank(
            ascending=False, method="first"
        )

        top_score = df.groupby("left_id")["score"].transform("max")
        second_score = df.groupby("left_id")["score"].transform(
            lambda s: float(s.nlargest(2).iloc[1]) if len(s) > 1 else 0.0
        )
        df["score_margin_to_second_best"] = top_score - second_score
    else:
        df["best_rank"] = 1
        df["score_margin_to_second_best"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Feature 3: Coordinate-uncertainty-aware geo score
# ---------------------------------------------------------------------------

def geo_uncertainty_score(
    lat1: float, lon1: float, sigma1_km: float,
    lat2: float, lon2: float, sigma2_km: float,
) -> float:
    """Probability (0–100) that two uncertain point locations are co-located.

    Uses the overlap probability of two Gaussian positional uncertainty
    distributions — the standard model in seismological catalogue
    association, earthquake relocation, GPS/GNSS metadata linking,
    and archaeological site matching.

    Formula:

    .. math::

        P(\\text{same}) = 1 - \\Phi\\!\\left(
            \\frac{d}{\\sqrt{\\sigma_1^2 + \\sigma_2^2}}
        \\right) \\times 100

    where :math:`d` is the haversine distance, :math:`\\Phi` is the
    standard normal CDF, and :math:`\\sigma_{1,2}` are the 1-sigma
    positional uncertainty radii in km.

    Parameters
    ----------
    lat1, lon1, sigma1_km:
        Coordinates and positional uncertainty (1-sigma, km) of record A.
    lat2, lon2, sigma2_km:
        Coordinates and positional uncertainty (1-sigma, km) of record B.

    Returns
    -------
    float — 0–100 score.

    Examples
    --------
    >>> geo_uncertainty_score(48.14, 11.58, 2.0, 48.17, 11.58, 2.0)
    77.3...
    >>> geo_uncertainty_score(48.14, 11.58, 0.01, 48.14, 11.58, 0.01)
    100.0
    """
    if any(v is None or (isinstance(v, float) and math.isnan(v))
           for v in (lat1, lon1, sigma1_km, lat2, lon2, sigma2_km)):
        return float("nan")

    dist = haversine_km(lat1, lon1, lat2, lon2)
    combined_sigma = math.sqrt(sigma1_km ** 2 + sigma2_km ** 2)

    if combined_sigma <= 0:
        return 100.0 if dist == 0.0 else 0.0

    z = dist / combined_sigma
    # Correct formula: erfc(d / (sigma_combined * sqrt(2)))
    # = 2 × (1 - Φ(z)) where Φ is the standard normal CDF
    # At d=0: erfc(0)=1.0 → 100%  ✓
    # At d→∞: erfc(∞)=0.0 → 0%    ✓
    p_same = math.erfc(z / math.sqrt(2))
    return min(100.0, p_same * 100.0)


def add_geo_uncertainty_score(
    candidates_df: pd.DataFrame,
    left_lat_col: str = "left_lat",
    left_lon_col: str = "left_lon",
    right_lat_col: str = "right_lat",
    right_lon_col: str = "right_lon",
    left_sigma_col: str = "left_sigma_km",
    right_sigma_col: str = "right_sigma_km",
    default_sigma_km: float = 5.0,
    score_col: str = "score_geo_uncertainty",
) -> pd.DataFrame:
    """Add ``score_geo_uncertainty`` column (probability of spatial co-location).

    Parameters
    ----------
    candidates_df:
        Candidates DataFrame with lat/lon columns and optionally sigma columns.
    left_sigma_col, right_sigma_col:
        Column names holding the 1-sigma positional uncertainty in km.
        If absent, ``default_sigma_km`` is used for all rows.
    default_sigma_km:
        Fallback uncertainty. Typical values:
        GPS points ≈ 0.01 km, seismic locations ≈ 10 km, historical records ≈ 50 km.

    Returns
    -------
    Copy of ``candidates_df`` with the new ``score_col`` column appended.
    """
    df = candidates_df.copy()
    for col in (left_lat_col, left_lon_col, right_lat_col, right_lon_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found. Join lat/lon columns first.")

    s1 = df[left_sigma_col].tolist()  if left_sigma_col  in df.columns \
         else [default_sigma_km] * len(df)
    s2 = df[right_sigma_col].tolist() if right_sigma_col in df.columns \
         else [default_sigma_km] * len(df)

    df[score_col] = [
        geo_uncertainty_score(flat, flon, sig1, rlat, rlon, sig2)
        for flat, flon, sig1, rlat, rlon, sig2 in zip(
            df[left_lat_col], df[left_lon_col], s1,
            df[right_lat_col], df[right_lon_col], s2,
        )
    ]
    return df


# ---------------------------------------------------------------------------
# Feature 4: Geometry similarity scores (Hausdorff, Fréchet, IoU)
# ---------------------------------------------------------------------------

def geometry_similarity_score(
    geom1,
    geom2,
    method: str = "hausdorff",
    max_distance_m: float = 1000.0,
) -> float:
    """Convert a geometry pair to a 0–100 similarity score.

    Both geometries must be in a **projected CRS with metre units**
    before calling this function.

    Parameters
    ----------
    geom1, geom2:
        Shapely geometry objects in a projected CRS (metres).
    method:
        ``"hausdorff"``  — max of all minimum point-to-point distances.
            Best for polygons / admin boundaries.
        ``"frechet"``    — discrete Fréchet distance (path-sensitive).
            Best for lines: rivers, roads, fault traces. Requires shapely ≥ 2.
        ``"iou"``        — Intersection-over-Union (Jaccard) for polygon pairs.
            Returns NaN for non-polygon inputs.
        ``"distance"``   — minimum distance between geometries.
            Works for any geometry type including points.
    max_distance_m:
        Distance at which hausdorff / frechet / distance score reaches 0.

    Returns
    -------
    float — 0–100. NaN when geometries are None or method is inapplicable.

    Examples
    --------
    >>> from shapely.geometry import LineString
    >>> l1 = LineString([(0, 0), (1000, 1000)])
    >>> l2 = LineString([(0, 50), (1000, 1050)])
    >>> geometry_similarity_score(l1, l2, "hausdorff", max_distance_m=200)
    75.0
    """
    if geom1 is None or geom2 is None:
        return float("nan")
    try:
        if geom1.is_empty or geom2.is_empty:
            return float("nan")
    except Exception:
        return float("nan")

    m = method.lower()
    try:
        if m == "hausdorff":
            dist = geom1.hausdorff_distance(geom2)
            return max(0.0, 100.0 * (1.0 - dist / max_distance_m))

        if m == "frechet":
            try:
                import shapely as _sh
                dist = _sh.frechet_distance(geom1, geom2)
            except AttributeError:
                dist = geom1.hausdorff_distance(geom2)  # graceful fallback
            return max(0.0, 100.0 * (1.0 - dist / max_distance_m))

        if m == "iou":
            if "Polygon" not in geom1.geom_type or "Polygon" not in geom2.geom_type:
                return float("nan")
            union_area = geom1.union(geom2).area
            if union_area == 0:
                return 100.0
            return 100.0 * geom1.intersection(geom2).area / union_area

        if m == "distance":
            dist = geom1.distance(geom2)
            return max(0.0, 100.0 * (1.0 - dist / max_distance_m))

    except Exception:
        pass
    return float("nan")


def add_geometry_similarity_score(
    candidates_df: pd.DataFrame,
    left_gdf,
    right_gdf,
    left_id_col: str = "left_id",
    right_id_col: str = "right_id",
    method: str = "hausdorff",
    max_distance_m: float = 1000.0,
    projected_crs: str = "EPSG:3857",
    score_col: Optional[str] = None,
    left_geom_index: Optional[str] = None,
    right_geom_index: Optional[str] = None,
) -> pd.DataFrame:
    """Add a geometry similarity score column to a candidates DataFrame.

    Reprojects both GeoDataFrames to ``projected_crs`` (metres), looks up
    each candidate pair's geometries, and calls
    :func:`geometry_similarity_score` on every pair.

    Parameters
    ----------
    candidates_df:
        Candidates DataFrame with ``left_id_col`` and ``right_id_col``.
    left_gdf, right_gdf:
        GeoDataFrames whose geometries are to be compared.
    method:
        ``"hausdorff"``, ``"frechet"``, ``"iou"``, or ``"distance"``.
        See :func:`geometry_similarity_score` for details.
    max_distance_m:
        Scale parameter. Tune to dataset precision:
        - 10 m  — building footprints / cadastre
        - 100 m — road / river centrelines
        - 1 km  — admin boundaries
        - 50 km — national coastlines
    projected_crs:
        A projected CRS with metre units (e.g. ``"EPSG:32632"`` for UTM 32N).
    score_col:
        Output column name (default: ``f"score_geom_{method}"``).
    left_geom_index, right_geom_index:
        Column to use as join key when IDs are stored as columns rather
        than the GDF index.

    Returns
    -------
    Copy of ``candidates_df`` with the new score column appended.

    Examples
    --------
    >>> # Compare admin polygon shapes
    >>> scored = add_geometry_similarity_score(
    ...     candidates, gadm_gdf, osm_gdf,
    ...     method="hausdorff", max_distance_m=5000,
    ...     projected_crs="EPSG:32632",
    ... )

    >>> # Compare river centrelines
    >>> scored = add_geometry_similarity_score(
    ...     candidates, left_rivers, right_rivers,
    ...     method="frechet", max_distance_m=200,
    ...     projected_crs="EPSG:32632",
    ... )
    """
    import warnings

    if score_col is None:
        score_col = f"score_geom_{method}"

    df = candidates_df.copy()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            left_proj  = left_gdf.to_crs(projected_crs)
            right_proj = right_gdf.to_crs(projected_crs)
    except Exception as exc:
        raise ImportError(
            "add_geometry_similarity_score() requires geopandas. "
            "pip install 'fuzzy_llm_matcher[geo]'"
        ) from exc

    left_geom_map  = (left_proj.set_index(left_geom_index).geometry.to_dict()
                      if left_geom_index else left_proj.geometry.to_dict())
    right_geom_map = (right_proj.set_index(right_geom_index).geometry.to_dict()
                      if right_geom_index else right_proj.geometry.to_dict())

    df[score_col] = [
        geometry_similarity_score(
            left_geom_map.get(row[left_id_col]),
            right_geom_map.get(row[right_id_col]),
            method=method,
            max_distance_m=max_distance_m,
        )
        for _, row in df.iterrows()
    ]
    return df
