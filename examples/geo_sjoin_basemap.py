"""Spatial proximity matching + tile basemaps (OSM / Satellite / Google-like).

Demonstrates two new capabilities:

1. **sjoin_nearest_candidates()** — spatial-first candidate generation using
   ``geopandas.sjoin_nearest()``, then rapidfuzz name scoring:

       spatial proximity → candidate pairs → name similarity scoring

   Compare with the name-first approach (``match_geodataframes()``):

       name similarity → candidates → geo-distance score

2. **combined_score()** — weighted combination of name and distance scores:

       combined = w_name × name_score + w_dist × distance_score

3. **add_basemap() / TileBasemap** — tile-based backgrounds via contextily:

       - ``"dark"``      CartoDB DarkMatter     (package default theme)
       - ``"light"``     CartoDB Positron
       - ``"osm"``       OpenStreetMap Mapnik
       - ``"satellite"`` ESRI WorldImagery      (Google Satellite-equivalent)
       - ``"google"``    ESRI WorldStreetMap    (Google Maps-equivalent)
       - ``"topo"``      ESRI WorldTopoMap

Figures produced → notebooks/figures/
  geo_sjoin_nearest_results.png    – match map on OSM basemap
  geo_combined_score_scatter.png   – name score vs distance score scatter
                                     on CartoDB Dark background
  geo_basemap_comparison.png       – 4-panel tile style comparison
  geo_satellite_map.png            – matched pairs on ESRI satellite imagery

Tile notes
----------
  Tiles are fetched live from the internet. A network connection is required.
  All providers used here are FREE — no API key needed:
    • CartoDB (OpenMapTiles)
    • ESRI (ArcGIS Online public tiles)
    • OpenStreetMap

  For true Google Maps tiles a Google Maps Platform API key is required:
    url = "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}&key=YOUR_KEY"
    add_basemap(ax, provider=url)

Run:
    python examples/geo_sjoin_basemap.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Synthetic city dataset ─────────────────────────────────────────────────
# Left: OSM-like point layer with local-language names
# Right: GeoNames-like reference layer with English names
# Both carry slightly different coordinates (GPS noise / source offset)

CITIES = [
    # (left_id, left_name,          left_lat, left_lon,
    #  right_id, right_name,         right_lat, right_lon)
    (1,  "München",          48.137, 11.576,   1,  "Munich",            48.137,  11.576),
    (2,  "Köln",             50.938,  6.960,   2,  "Cologne",           50.937,   6.960),
    (3,  "Nürnberg",         49.453, 11.077,   3,  "Nuremberg",         49.453,  11.077),
    (4,  "Düsseldorf",       51.228,  6.773,   4,  "Dusseldorf",        51.228,   6.773),
    (5,  "Stuttgart",        48.775,  9.183,   5,  "Stuttgart",         48.775,   9.182),
    (6,  "Frankfurt am Main",50.110,  8.682,   6,  "Frankfurt",         50.110,   8.682),
    (7,  "Hamburg",          53.551,  9.994,   7,  "Hamburg",           53.551,   9.994),
    (8,  "Berlin",           52.520, 13.405,   8,  "Berlin",            52.520,  13.405),
    (9,  "Leipzig",          51.340, 12.374,   9,  "Leipzig",           51.340,  12.374),
    (10, "Dresden",          51.050, 13.738,  10,  "Dresden",           51.050,  13.738),
    # Deliberate near-misses: close names, slightly shifted coordinates
    (11, "Hannover",         52.374,  9.738,  11,  "Hanover",           52.374,   9.738),
    (12, "Bremen",           53.079,  8.801,  12,  "Bremen",            53.079,   8.801),
    (13, "Dortmund",         51.514,  7.466,  13,  "Dortmund",          51.513,   7.466),
    (14, "Essen",            51.455,  7.011,  14,  "Essen",             51.456,   7.011),
    (15, "Bochum",           51.482,  7.216,  15,  "Bochum",            51.481,   7.216),
    # Hard: names differ but within 200 m of each other
    (16, "Muenchen Mitte",   48.139, 11.578,   1,  "Munich",            48.137,  11.576),
    # Far: similar name but >500 m away (should be filtered by max_distance)
    (17, "Frankfurt Oder",   52.346, 14.550,   6,  "Frankfurt",         50.110,   8.682),
]

GROUND_TRUTH = {r[0]: r[4] for r in CITIES if r[0] != 17}  # 17 is intentionally filtered


def _build_gdfs():
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as e:
        raise ImportError("pip install 'fuzzy_llm_matcher[geo]'") from e

    left = gpd.GeoDataFrame(
        pd.DataFrame(
            [(r[0], r[1], r[2], r[3]) for r in CITIES],
            columns=["id", "name", "lat", "lon"],
        ),
        geometry=[Point(r[3], r[2]) for r in CITIES],
        crs="EPSG:4326",
    )
    right = gpd.GeoDataFrame(
        pd.DataFrame(
            [(r[4], r[5], r[6], r[7]) for r in CITIES],
            columns=["id", "name", "lat", "lon"],
        ),
        geometry=[Point(r[7], r[6]) for r in CITIES],
        crs="EPSG:4326",
    )
    # Remove duplicate right entries (cities that appear in multiple left rows)
    right = right.drop_duplicates("id").reset_index(drop=True)
    return left, right


# ── Run both approaches ────────────────────────────────────────────────────

def run_sjoin_approach(left, right):
    """Spatial-first: sjoin_nearest + combined score."""
    from fuzzy_llm_matcher import combined_score, sjoin_nearest_candidates

    # UTM zone 32N — good for Germany/Central Europe, metre-accurate
    candidates = sjoin_nearest_candidates(
        left, right,
        left_name_col="name",
        right_name_col="name",
        left_id_col="id",
        right_id_col="id",
        max_distance_m=50_000,          # 50 km — generous first cut
        projected_crs="EPSG:32632",     # UTM zone 32N
        scorer="token_set_ratio",
    )
    print(f"  Candidates after spatial filter: {len(candidates)}")

    # Combined score: 70% name, 30% distance
    matches = combined_score(
        candidates,
        w_name=0.7,
        w_dist=0.3,
        max_distance_m=50_000,
        threshold=70.0,
    )
    print(f"  Matches after combined threshold: {len(matches)}")
    return candidates, matches


def run_name_first_approach(left, right):
    """Name-first: match_geodataframes with geo-distance score."""
    from fuzzy_llm_matcher import match_geodataframes

    result = match_geodataframes(
        left, right,
        left_on="name", right_on="name",
        left_id="id", right_id="id",
        spatial_block_degrees=5.0,
        max_distance_km=500.0,
        use_llm=True,
        high_threshold=85,
        return_geometry=True,
    )
    return result


# ── Colour scheme ──────────────────────────────────────────────────────────

LABEL_COLORS = {
    "high":          "#00e5ff",
    "medium_review": "#ffb300",
    "low":           "#ff5252",
    "reject":        "#888888",
}


# ── Figure 1 – Match map on OSM basemap ───────────────────────────────────

def fig_sjoin_osm(matches: pd.DataFrame, left, right, out_dir: Path):
    """Plot matched pairs on OpenStreetMap tiles — Germany focus."""
    from fuzzy_llm_matcher import add_basemap
    import geopandas as gpd
    from shapely.geometry import Point

    fig, ax = plt.subplots(figsize=(10, 12))

    # Build GeoDataFrame for matched left points in Web Mercator
    geoms = []
    for _, row in matches.iterrows():
        lid = row["left_id"]
        city_row = next((r for r in CITIES if r[0] == lid), None)
        if city_row:
            geoms.append(Point(city_row[3], city_row[2]))
        else:
            geoms.append(None)

    matched_gdf = gpd.GeoDataFrame(matches.copy(), geometry=geoms, crs="EPSG:4326")
    matched_gdf = matched_gdf[matched_gdf.geometry.notna()].to_crs(3857)

    # Colour by combined score
    score_norm = (matched_gdf["combined_score"] - 70) / 30
    colors = plt.cm.RdYlGn(score_norm.clip(0, 1))

    matched_gdf.plot(
        ax=ax,
        color=colors,
        markersize=80,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
        alpha=0.9,
    )

    # Label matched pairs
    for _, row in matched_gdf.iterrows():
        ax.annotate(
            f"{row['left_name'][:10]}\n→{row['right_name'][:10]}\n{row['combined_score']:.0f}",
            xy=(row.geometry.x, row.geometry.y),
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=6, color="white", ha="center",
            zorder=5,
        )

    # OSM basemap
    add_basemap(ax, style="osm", alpha=0.65)

    ax.set_axis_off()
    ax.set_title(
        "sjoin_nearest_candidates() + combined_score() on OpenStreetMap tiles\n"
        "German cities: left = local names, right = English canonical names\n"
        "Marker colour: green = high combined score, red = lower",
        fontsize=11, fontweight="bold", pad=10,
    )
    fig.text(0.5, 0.01, "Tiles © OpenStreetMap contributors | github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=7, color="#555555")

    out = out_dir / "geo_sjoin_nearest_results.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2 – Combined score scatter on dark background ──────────────────

def fig_combined_scatter(candidates: pd.DataFrame, matches: pd.DataFrame,
                         out_dir: Path):
    """Name score vs distance score scatter — CartoDB Dark background as canvas."""
    BG = "#0d1b2a"
    MID = "#12263a"

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
    ax.set_facecolor(MID)
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_edgecolor("#2e5f8a")

    # All candidates
    all_ds = np.clip(100 * (1 - candidates["distance_m"] / 50_000), 0, 100)
    ax.scatter(
        candidates["name_score"], all_ds,
        c="#444466", s=35, alpha=0.5, label="candidates (below threshold)",
        edgecolors="none", zorder=2,
    )

    # Matched pairs coloured by combined score
    match_ds = np.clip(100 * (1 - matches["distance_m"] / 50_000), 0, 100)
    sc = ax.scatter(
        matches["name_score"], match_ds,
        c=matches["combined_score"], cmap="RdYlGn",
        vmin=70, vmax=100,
        s=100, alpha=0.95, edgecolors="white", linewidths=0.4,
        zorder=4, label="accepted matches",
    )

    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("combined_score", color="white", fontsize=10)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    # Decision boundary
    w_n, w_d, thr = 0.7, 0.3, 70
    x_line = np.linspace(0, 100, 200)
    y_line = (thr - w_n * x_line) / w_d
    ax.plot(x_line, y_line, color="#ffb300", lw=1.5, ls="--",
            alpha=0.8, label=f"threshold={thr} (w_name=0.7, w_dist=0.3)")

    ax.set_xlabel("name_score (token_set_ratio, 0–100)", color="white", fontsize=12)
    ax.set_ylabel("distance_score (0–100, max=50 km)", color="white", fontsize=12)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_title(
        "combined_score = 0.7 × name_score + 0.3 × distance_score\n"
        "Points above the dashed line pass the threshold=70 filter",
        color="white", fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, framealpha=0.35, facecolor=BG,
              edgecolor="#2e5f8a", labelcolor="white")
    fig.text(0.5, 0.01, "github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=8, color="#a0bdd8")

    out = out_dir / "geo_combined_score_scatter.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3 – 4-panel tile style comparison ──────────────────────────────

def fig_tile_comparison(result, out_dir: Path):
    """4-panel showing same data on 4 different tile backgrounds."""
    from fuzzy_llm_matcher import add_basemap
    import geopandas as gpd
    from shapely.geometry import Point

    styles = [
        ("dark",      "CartoDB DarkMatter\n(package default)"),
        ("osm",       "OpenStreetMap Mapnik\n(free, no key)"),
        ("satellite", "ESRI WorldImagery\n(Google Satellite-equivalent)"),
        ("google",    "ESRI WorldStreetMap\n(Google Maps-equivalent)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes_flat = axes.flatten()

    # Build geo-enriched points in Web Mercator
    rel_col = "reliability_label"
    if result is None or len(result) == 0:
        for ax in axes_flat:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
        out = out_dir / "geo_basemap_comparison.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")
        return

    for ax, (style, title) in zip(axes_flat, styles):
        try:
            for label, color in LABEL_COLORS.items():
                subset = result[result.get(rel_col, pd.Series()) == label] \
                    if rel_col in result.columns else pd.DataFrame()
                if len(subset) and "geometry" in result.columns:
                    try:
                        import geopandas as gpd
                        g = gpd.GeoDataFrame(subset, geometry="geometry",
                                             crs=result.crs if hasattr(result, "crs") else "EPSG:4326")
                        g.to_crs(3857).plot(
                            ax=ax, color=color, markersize=60,
                            edgecolor="white", linewidth=0.4, alpha=0.9, zorder=3,
                        )
                    except Exception:
                        pass

            add_basemap(ax, style=style, alpha=0.7, zoom=5)
            ax.set_axis_off()
            ax.set_title(title, fontsize=10, fontweight="bold")

        except Exception as e:
            ax.text(0.5, 0.5, f"Could not load tiles:\n{e}",
                    transform=ax.transAxes, ha="center", va="center", fontsize=8)
            ax.set_axis_off()

    # Shared legend
    patches = [mpatches.Patch(color=c, label=l) for l, c in LABEL_COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9,
               framealpha=0.4, labelcolor="black")

    fig.suptitle(
        "Same match results on four different tile backgrounds\n"
        "fuzzy_llm_matcher · add_basemap(ax, style=...)",
        fontsize=13, fontweight="bold",
    )
    fig.text(0.5, 0.02,
             "Tiles: CartoDB (© OpenMapTiles, © OpenStreetMap) · ESRI (Esri, HERE, Garmin, FAO, NOAA) · "
             "OSM (© OpenStreetMap contributors)",
             ha="center", fontsize=7, color="#555555")

    out = out_dir / "geo_basemap_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 4 – Satellite map with combined-score results ──────────────────

def fig_satellite_map(matches: pd.DataFrame, out_dir: Path):
    """Plot accepted matches on ESRI satellite imagery — Germany."""
    from fuzzy_llm_matcher import add_basemap
    import geopandas as gpd
    from shapely.geometry import Point, LineString

    fig, ax = plt.subplots(figsize=(11, 12))

    # Build left and right geometry in Web Mercator
    left_pts  = []
    right_pts = []
    arcs      = []

    for _, row in matches.iterrows():
        city_l = next((r for r in CITIES if r[0] == row["left_id"]),  None)
        city_r = next((r for r in CITIES if r[4] == row["right_id"]), None)
        if city_l and city_r:
            left_pts.append( Point(city_l[3], city_l[2]))
            right_pts.append(Point(city_r[7], city_r[6]))
            arcs.append(LineString([(city_l[3], city_l[2]), (city_r[7], city_r[6])]))

    if not left_pts:
        print("  No geometry for satellite map — skipping")
        return

    gdf_lines = gpd.GeoDataFrame(
        matches.head(len(arcs)).copy(),
        geometry=arcs, crs="EPSG:4326",
    ).to_crs(3857)

    gdf_left  = gpd.GeoDataFrame(
        matches.head(len(left_pts)).copy(),
        geometry=left_pts, crs="EPSG:4326",
    ).to_crs(3857)

    gdf_right = gpd.GeoDataFrame(
        matches.head(len(right_pts)).copy(),
        geometry=right_pts, crs="EPSG:4326",
    ).to_crs(3857)

    # Score-coloured arcs
    for _, row in gdf_lines.iterrows():
        norm = max(0, min(1, (row["combined_score"] - 70) / 30))
        color = plt.cm.YlOrRd(1 - norm)  # red=low, yellow=high
        try:
            xs, ys = row.geometry.xy
            ax.plot(xs, ys, color="#00ff88", lw=1.2, alpha=0.7, zorder=2)
        except Exception:
            pass

    gdf_left.plot( ax=ax, color="#00ff88", markersize=70,
                   edgecolor="white", linewidth=0.5, zorder=4, alpha=0.9,
                   label="left (OSM names)")
    gdf_right.plot(ax=ax, color="#ffdd00", markersize=55, marker="*",
                   zorder=5, alpha=0.95, label="right (canonical names)")

    # ESRI satellite background
    add_basemap(ax, style="satellite", alpha=0.85, zoom=6)

    ax.set_axis_off()
    ax.set_title(
        "Fuzzy name matches on ESRI WorldImagery (satellite)\n"
        "green● = OSM local name   ★ = canonical English name\n"
        "green arc = matched pair",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower left", framealpha=0.7)
    fig.text(0.5, 0.01,
             "Imagery © Esri, Maxar, Earthstar Geographics | "
             "github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=7, color="#888888")

    out = out_dir / "geo_satellite_map.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ────────────────────────────────────────────────────────────────────

def run():
    out_dir = Path("notebooks/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    left, right = _build_gdfs()

    print("── sjoin_nearest_candidates() + combined_score() ────────────")
    candidates, matches = run_sjoin_approach(left, right)

    print(f"\n  Sample matches:")
    print(matches[["left_name", "right_name", "distance_m",
                   "name_score", "distance_score", "combined_score"]
                  ].head(8).to_string(index=False))

    print("\n── match_geodataframes() (name-first, for comparison) ───────")
    result = run_name_first_approach(left, right)
    print(f"  Total matched: {result['final_decision'].sum()}")
    print(f"  Reliability distribution:")
    print(result["reliability_label"].value_counts().to_string())

    print("\nGenerating figures (tiles fetched from internet) …")
    fig_sjoin_osm(matches, left, right, out_dir)
    fig_combined_scatter(candidates, matches, out_dir)
    fig_tile_comparison(result, out_dir)
    fig_satellite_map(matches, out_dir)

    print("\nDone.")
    return candidates, matches, result


if __name__ == "__main__":
    run()
