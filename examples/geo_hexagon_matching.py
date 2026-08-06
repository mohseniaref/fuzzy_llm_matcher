"""Hexagonal grid blocking for fuzzy entity matching on geodata.

Inspired by:
  https://gemgis.readthedocs.io/en/latest/getting_started/tutorial/58_creating_hexagonal_grid.html

Why hexagonal grids for blocking?
----------------------------------
A hexagonal grid is the optimal tessellation of the plane for proximity
blocking:
  • Equal area per cell — no polar distortion (unlike degree-grid)
  • 6 equidistant neighbours — fewer boundary artefacts than square grids
  • Compact shape — shortest perimeter per unit area, minimising edge cases
  • Scale-independent — radius_m controls resolution in metres, not degrees

This example demonstrates:
  1. Creating hexagonal grids with create_hexagon_grid()
  2. Assigning features to hex cells with assign_hex_ids()
  3. Running fuzzy matching blocked by hex cell with hex_block_match()
  4. Comparing degree-grid vs hex-grid blocking coverage
  5. All figures on tile basemaps (OSM / satellite / CartoDB)

Figures produced → notebooks/figures/
  geo_hex_grid_overview.png        – hex grid over study area (3 resolutions)
  geo_hex_matching_results.png     – matched pairs per hex cell on OSM tiles
  geo_hex_vs_grid_comparison.png   – hex vs degree-grid blocking comparison
  geo_hex_satellite_choropleth.png – match confidence choropleth on satellite

Run:
    python examples/geo_hexagon_matching.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ── Synthetic European city dataset ───────────────────────────────────────
# Left = OSM-sourced names (local language), Right = GeoNames English names
# Deliberately introduces the same kinds of discrepancies found in real data

CITIES = [
    # (lid, left_name,            lat,    lon,
    #  rid, right_name)
    (1,  "München",            48.137, 11.576,   1,  "Munich"),
    (2,  "Köln",               50.938,  6.960,   2,  "Cologne"),
    (3,  "Nürnberg",           49.453, 11.077,   3,  "Nuremberg"),
    (4,  "Düsseldorf",         51.228,  6.773,   4,  "Dusseldorf"),
    (5,  "Stuttgart",          48.775,  9.183,   5,  "Stuttgart"),
    (6,  "Frankfurt am Main",  50.110,  8.682,   6,  "Frankfurt"),
    (7,  "Hamburg",            53.551,  9.994,   7,  "Hamburg"),
    (8,  "Berlin",             52.520, 13.405,   8,  "Berlin"),
    (9,  "Leipzig",            51.340, 12.374,   9,  "Leipzig"),
    (10, "Dresden",            51.050, 13.738,  10,  "Dresden"),
    (11, "Hannover",           52.374,  9.738,  11,  "Hanover"),
    (12, "Bremen",             53.079,  8.801,  12,  "Bremen"),
    (13, "Dortmund",           51.514,  7.466,  13,  "Dortmund"),
    (14, "Essen",              51.455,  7.011,  14,  "Essen"),
    (15, "Bochum",             51.482,  7.216,  15,  "Bochum"),
    (16, "Mannheim",           49.487,  8.466,  16,  "Mannheim"),
    (17, "Karlsruhe",          49.007,  8.404,  17,  "Karlsruhe"),
    (18, "Freiburg im Breisgau",47.995, 7.849,  18,  "Freiburg"),
    (19, "Augsburg",           48.370, 10.897,  19,  "Augsburg"),
    (20, "Wiesbaden",          50.082,  8.240,  20,  "Wiesbaden"),
    (21, "Bonn",               50.735,  7.100,  21,  "Bonn"),
    (22, "Münster",            51.962,  7.626,  22,  "Muenster"),
    (23, "Bielefeld",          52.021,  8.532,  23,  "Bielefeld"),
    (24, "Wuppertal",          51.267,  7.186,  24,  "Wuppertal"),
    (25, "Bochum",             51.482,  7.216,  25,  "Bochum"),   # duplicate test
    (26, "Gelsenkirchen",      51.517,  7.085,  26,  "Gelsenkirchen"),
    (27, "Mönchengladbach",    51.196,  6.441,  27,  "Moenchengladbach"),
    (28, "Braunschweig",       52.269, 10.521,  28,  "Brunswick"),
    (29, "Aachen",             50.776,  6.084,  29,  "Aachen"),
    (30, "Kiel",               54.323, 10.122,  30,  "Kiel"),
]

GROUND_TRUTH = {r[0]: r[4] for r in CITIES}


def _build_gdfs():
    import geopandas as gpd
    from shapely.geometry import Point

    left = gpd.GeoDataFrame(
        pd.DataFrame([(r[0], r[1], r[2], r[3]) for r in CITIES],
                     columns=["id", "name", "lat", "lon"]),
        geometry=[Point(r[3], r[2]) for r in CITIES],
        crs="EPSG:4326",
    )
    right = gpd.GeoDataFrame(
        pd.DataFrame([(r[4], r[5], r[2], r[3]) for r in CITIES],
                     columns=["id", "name", "lat", "lon"]),
        geometry=[Point(r[3], r[2]) for r in CITIES],
        crs="EPSG:4326",
    ).drop_duplicates("id").reset_index(drop=True)

    return left, right


# ── Figure 1 – Hex grid overview at 3 resolutions ─────────────────────────

def fig_hex_grid_overview(left, right, out_dir: Path):
    """Show the hexagon grid at three different radii on CartoDB Dark tiles."""
    from fuzzy_llm_matcher import add_basemap, create_hexagon_grid

    radii   = [500_000, 200_000, 100_000]  # metres
    titles  = ["radius = 500 km\n(continental)", "radius = 200 km\n(regional)",
               "radius = 100 km\n(local)"]
    colors  = ["#00e5ff", "#ffb300", "#ff5252"]

    import geopandas as gpd
    combined = gpd.GeoDataFrame(
        geometry=list(left.geometry) + list(right.geometry), crs="EPSG:4326"
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor("#0d1b2a")

    for ax, r, title, color in zip(axes, radii, titles, colors):
        grid = create_hexagon_grid(combined, radius_m=r, crop=False)
        grid_merc = grid.to_crs(3857)
        left_merc  = left.to_crs(3857)

        grid_merc.plot(ax=ax, facecolor=color, alpha=0.15,
                       edgecolor=color, linewidth=0.8, zorder=2)
        left_merc.plot(ax=ax, color="white", markersize=18,
                       zorder=4, alpha=0.9)

        # Mark hex centres
        grid_merc.geometry.centroid.plot(ax=ax, color=color,
                                         markersize=4, zorder=3, alpha=0.6)

        try:
            add_basemap(ax, style="dark", alpha=0.8)
        except Exception:
            ax.set_facecolor("#0d1b2a")

        ax.set_axis_off()
        ax.set_title(
            f"{title}\n{len(grid)} hexagons",
            color="white", fontsize=11, fontweight="bold",
        )

    fig.suptitle(
        "Hexagonal Grid for Fuzzy Matching Blocking\n"
        "Three resolutions — German cities dataset\n"
        "Inspired by: gemgis.readthedocs.io/tutorial/58_creating_hexagonal_grid.html",
        color="white", fontsize=12, fontweight="bold",
    )
    fig.text(0.5, 0.01, "Tiles © CartoDB OpenMapTiles | github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=7, color="#a0bdd8")

    out = out_dir / "geo_hex_grid_overview.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="#0d1b2a")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2 – Matching results with hex cells on OSM tiles ───────────────

def fig_hex_matching_results(result, grid, out_dir: Path):
    """Matched pairs overlaid on hex grid on OpenStreetMap tiles."""
    from fuzzy_llm_matcher import add_basemap
    import geopandas as gpd

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor("white")

    LABEL_COLORS = {
        "high":          "#00cc44",
        "medium_review": "#ffaa00",
        "low":           "#ff4444",
        "reject":        "#999999",
    }

    for ax_idx, (ax, title, style) in enumerate(zip(
        axes,
        ["Hex grid blocking + match results\n(OpenStreetMap)", "Hex grid blocking + match results\n(ESRI Satellite)"],
        ["osm", "satellite"],
    )):
        grid_merc = grid.to_crs(3857)

        # Colour hexes by number of matched pairs
        if "reliability_label" in result.columns and hasattr(result, "geometry"):
            result_merc = result.to_crs(3857)
        else:
            result_merc = result

        # Draw hexagon cells (light)
        grid_merc.plot(ax=ax, facecolor="#aaaaaa", alpha=0.08,
                       edgecolor="#666666", linewidth=0.5, zorder=1)

        # Draw matched points coloured by reliability
        rel_col = "reliability_label"
        if rel_col in result_merc.columns and hasattr(result_merc, "geometry"):
            for label, color in LABEL_COLORS.items():
                subset = result_merc[result_merc[rel_col] == label]
                if len(subset):
                    subset.plot(ax=ax, color=color, markersize=50,
                                edgecolor="white", linewidth=0.4,
                                alpha=0.9, zorder=4)

        try:
            add_basemap(ax, style=style, alpha=0.75)
        except Exception:
            ax.set_facecolor("#e8e8e8")

        ax.set_axis_off()
        ax.set_title(title, fontsize=11, fontweight="bold")

    # Legend
    patches = [mpatches.Patch(color=c, label=l)
               for l, c in LABEL_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=10, framealpha=0.8)

    n_matched = int(result["final_decision"].sum()) \
        if "final_decision" in result.columns else 0
    fig.suptitle(
        f"hex_block_match() — {n_matched} confirmed matches "
        f"from {len(result)} candidates\n"
        f"Hex radius: 200 km | German cities (OSM names → GeoNames canonical)",
        fontsize=12, fontweight="bold",
    )
    fig.text(0.5, 0.01,
             "Tiles: © OpenStreetMap contributors · © Esri, Maxar, Earthstar Geographics | "
             "github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=7, color="#555555")

    out = out_dir / "geo_hex_matching_results.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3 – Hex vs degree-grid blocking comparison ─────────────────────

def fig_hex_vs_grid_comparison(left, right, out_dir: Path):
    """Side-by-side: degree-grid (match_geodataframes) vs hex grid blocking."""
    from fuzzy_llm_matcher import add_basemap, hex_block_match, match_geodataframes
    import geopandas as gpd

    # Run both approaches
    result_grid = match_geodataframes(
        left, right, left_on="name", right_on="name",
        left_id="id", right_id="id",
        spatial_block_degrees=5.0, max_distance_km=500,
        high_threshold=85, use_llm=True, return_geometry=True,
    )
    result_hex, hex_grid = hex_block_match(
        left, right, left_on="name", right_on="name",
        left_id="id", right_id="id",
        hex_radius_m=200_000, projected_crs="EPSG:3857",
        max_distance_km=500, high_threshold=85, use_llm=True,
        return_geometry=True, return_grid=True,
    )

    grid_n_matched = int(result_grid["final_decision"].sum()) \
        if "final_decision" in result_grid.columns else 0
    hex_n_matched  = int(result_hex["final_decision"].sum()) \
        if "final_decision" in result_hex.columns else 0

    print(f"  Degree-grid matched: {grid_n_matched}")
    print(f"  Hex-grid matched:    {hex_n_matched}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor("white")

    configs = [
        (axes[0], result_grid, None,
         f"Degree-grid blocking (5°)\n{grid_n_matched} matched"),
        (axes[1], result_hex, hex_grid,
         f"Hexagonal blocking (r=200 km)\n{hex_n_matched} matched"),
    ]

    for ax, res, hgrid, title in configs:
        if hgrid is not None:
            hgrid.to_crs(3857).plot(
                ax=ax, facecolor="#4466ff", alpha=0.06,
                edgecolor="#4466ff", linewidth=0.6, zorder=1,
            )

        if hasattr(res, "geometry"):
            for label, color in [("high","#00cc44"),("medium_review","#ffaa00"),
                                  ("low","#ff4444"),("reject","#999999")]:
                subset = res[res.get("reliability_label", pd.Series()) == label] \
                    if "reliability_label" in res.columns else pd.DataFrame()
                if len(subset):
                    try:
                        gpd.GeoDataFrame(subset, geometry="geometry",
                                         crs=res.crs).to_crs(3857).plot(
                            ax=ax, color=color, markersize=55,
                            edgecolor="white", linewidth=0.4, alpha=0.9, zorder=3,
                        )
                    except Exception:
                        pass

        try:
            add_basemap(ax, style="google", alpha=0.75)
        except Exception:
            ax.set_facecolor("#e8e8e8")

        ax.set_axis_off()
        ax.set_title(title, fontsize=11, fontweight="bold")

    patches = [mpatches.Patch(color=c, label=l) for l, c in [
        ("high","#00cc44"),("medium_review","#ffaa00"),
        ("low","#ff4444"),("reject","#999999"),
    ]]
    patches += [mpatches.Patch(color="#4466ff", alpha=0.3, label="hex cell")]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9, framealpha=0.8)

    fig.suptitle(
        "Degree-grid blocking vs Hexagonal grid blocking\n"
        "Tiles: ESRI WorldStreetMap (Google Maps-equivalent)",
        fontsize=12, fontweight="bold",
    )
    fig.text(0.5, 0.01,
             "Tiles © Esri, DeLorme, NAVTEQ | github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=7, color="#555555")

    out = out_dir / "geo_hex_vs_grid_comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return result_hex, hex_grid


# ── Figure 4 – Hex choropleth on satellite ─────────────────────────────────

def fig_hex_satellite_choropleth(result, hex_grid, out_dir: Path):
    """Choropleth of match confidence per hexagon on satellite imagery."""
    from fuzzy_llm_matcher import add_basemap
    import geopandas as gpd

    # Compute per-hex stats
    if "reliability_label" not in result.columns:
        print("  No reliability_label column — skipping choropleth")
        return

    label_score = {"high": 3, "medium_review": 2, "low": 1, "reject": 0}
    result_copy = result.copy()
    result_copy["_score"] = result_copy["reliability_label"].map(label_score).fillna(0)

    # We need to find which hex each left_id belongs to
    # Since we used hex blocking, we can join through hex_grid via spatial join
    if "geometry" not in result_copy.columns:
        print("  No geometry in result — skipping choropleth")
        return

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_gdf = gpd.GeoDataFrame(
            result_copy, geometry="geometry", crs=hex_grid.crs
        )
        joined = gpd.sjoin(
            result_gdf[["_score", "final_decision", "geometry"]],
            hex_grid[["hex_id", "geometry"]],
            how="left", predicate="within",
        ).drop(columns=["index_right"], errors="ignore")

    hex_stats = joined.groupby("hex_id").agg(
        mean_score=("_score", "mean"),
        n_matches=("final_decision", "sum"),
        n_total=("_score", "count"),
    ).reset_index()

    grid_enriched = hex_grid.merge(hex_stats, on="hex_id", how="left")
    grid_enriched["mean_score"] = grid_enriched["mean_score"].fillna(0)
    grid_enriched["n_matches"]  = grid_enriched["n_matches"].fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor("white")

    for ax, col, title, cmap in [
        (axes[0], "mean_score", "Mean reliability score\n(3=high, 2=medium, 1=low, 0=reject)", "RdYlGn"),
        (axes[1], "n_matches",  "Number of confirmed matches\nper hexagon", "YlOrRd"),
    ]:
        gm = grid_enriched.to_crs(3857)
        gm.plot(ax=ax, column=col, cmap=cmap, alpha=0.65,
                edgecolor="white", linewidth=0.4, zorder=2, legend=True,
                legend_kwds={"shrink": 0.6, "label": col})

        try:
            add_basemap(ax, style="satellite", alpha=0.8)
        except Exception:
            ax.set_facecolor("#2a2a2a")

        ax.set_axis_off()
        ax.set_title(title, fontsize=11, fontweight="bold")

    fig.suptitle(
        "Hexagonal Choropleth of Fuzzy Match Confidence\n"
        "ESRI WorldImagery (satellite) background — German cities",
        fontsize=12, fontweight="bold",
    )
    fig.text(0.5, 0.01,
             "Imagery © Esri, Maxar, Earthstar Geographics | github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=7, color="#555555")

    out = out_dir / "geo_hex_satellite_choropleth.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ────────────────────────────────────────────────────────────────────

def run():
    from fuzzy_llm_matcher import (
        create_hexagon_grid, assign_hex_ids, hex_block_match,
    )

    out_dir = Path("notebooks/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building German city GeoDataFrames …")
    left, right = _build_gdfs()
    print(f"  Left features:  {len(left)}")
    print(f"  Right features: {len(right)}")

    # ── Show single hexagon ────────────────────────────────────────────────
    from fuzzy_llm_matcher import create_hexagon
    from shapely.geometry import Point
    h = create_hexagon(Point(0, 0), radius=100_000)
    print(f"\n  Single hexagon (r=100 km): area = {h.area/1e6:.0f} km²")

    # ── Create hex grid ────────────────────────────────────────────────────
    print("\n── Creating hexagonal grids ────────────────────────────────")
    for r_km, label in [(500, "500 km"), (200, "200 km"), (100, "100 km")]:
        import geopandas as gpd
        combined = gpd.GeoDataFrame(
            geometry=list(left.geometry)+list(right.geometry), crs="EPSG:4326"
        )
        grid = create_hexagon_grid(combined, radius_m=r_km*1000, crop=False)
        print(f"  r={label:6s}: {len(grid):3d} hexagons covering study area")

    # ── Assign hex IDs ─────────────────────────────────────────────────────
    print("\n── Assigning hex IDs to features ───────────────────────────")
    grid_200 = create_hexagon_grid(
        gpd.GeoDataFrame(geometry=list(left.geometry)+list(right.geometry),
                         crs="EPSG:4326"),
        radius_m=200_000, crop=False,
    )
    left_hexed = assign_hex_ids(left, grid_200)
    print(f"  Left features assigned: {left_hexed['hex_id'].notna().sum()} / {len(left_hexed)}")
    print(f"  Hex cells occupied:     {left_hexed['hex_id'].nunique()}")

    # ── Run hex_block_match ────────────────────────────────────────────────
    print("\n── hex_block_match() (r=200 km) ────────────────────────────")
    result, grid = hex_block_match(
        left, right, left_on="name", right_on="name",
        left_id="id", right_id="id",
        hex_radius_m=200_000,
        projected_crs="EPSG:3857",
        max_distance_km=500,
        high_threshold=85,
        use_llm=True,
        return_geometry=True,
        return_grid=True,
    )

    n_matched = int(result["final_decision"].sum()) \
        if "final_decision" in result.columns else 0
    print(f"  Total candidates:  {len(result)}")
    print(f"  Final decisions:   {n_matched}")
    if "reliability_label" in result.columns:
        print(f"  Label distribution:")
        print(result["reliability_label"].value_counts().to_string())

    # ── Figures ────────────────────────────────────────────────────────────
    print("\nGenerating figures (tiles fetched from internet) …")
    fig_hex_grid_overview(left, right, out_dir)
    fig_hex_matching_results(result, grid, out_dir)
    result_hex, hex_grid = fig_hex_vs_grid_comparison(left, right, out_dir)
    fig_hex_satellite_choropleth(result_hex, hex_grid, out_dir)

    print("\nDone.")
    return result, grid


if __name__ == "__main__":
    run()
