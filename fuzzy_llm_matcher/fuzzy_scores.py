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
