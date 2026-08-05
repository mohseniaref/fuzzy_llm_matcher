"""match_geodataframes() demo – spatial blocking, geo score and geometry join-back.

Demonstrates the full GeoDataFrame workflow in three sections:

A. Basic usage  – match two small GeoDataFrames of world cities (no I/O).
B. Figures      – four publication-quality charts saved to notebooks/figures/.
C. Export demo  – write the matched GeoDataFrame to GeoJSON (if output dir is writable).

Figures produced
----------------
    geo_gdf_scatter.png          – fuzzy score vs geo-distance score, coloured
                                   by reliability_label
    geo_gdf_spatial_blocks.png   – world map showing the spatial blocking grid
                                   and which left/right points share a block
    geo_gdf_match_map.png        – world map of matched pairs, arc thickness =
                                   final_decision, arc colour = geo score
    geo_gdf_score_bars.png       – per-pair bar chart of fuzzy + geo score side
                                   by side, sorted by reliability

Run:
    python examples/geo_match_geodataframes.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Synthetic city GeoDataFrame
# ---------------------------------------------------------------------------

CITIES = [
    # (left_id, left_name,         left_lat, left_lon,  right_id, right_name,       right_lat, right_lon)
    (1,  "München",            48.14,  11.58,   1,  "Munich",          48.14,  11.58),
    (2,  "Köln",               50.94,   6.96,   2,  "Cologne",         50.94,   6.96),
    (3,  "Nürnberg",           49.45,  11.08,   3,  "Nuremberg",       49.45,  11.08),
    (4,  "NYC",                40.71, -74.00,   4,  "New York City",   40.71, -74.00),
    (5,  "San Fran",           37.77,-122.42,   5,  "San Francisco",   37.77,-122.42),
    (6,  "Tokio",              35.69, 139.69,   6,  "Tokyo",           35.69, 139.69),
    (7,  "Moskva",             55.75,  37.62,   7,  "Moscow",          55.75,  37.62),
    (8,  "Al Qahirah",         30.06,  31.25,   8,  "Cairo",           30.06,  31.25),
    (9,  "Peking",             39.93, 116.39,   9,  "Beijing",         39.93, 116.39),
    (10, "Bombay",             19.08,  72.88,  10,  "Mumbai",          19.08,  72.88),
    (11, "Rio de Jan.",       -22.91, -43.18,  11,  "Rio de Janeiro", -22.91, -43.18),
    (12, "Buenos Ayres",      -34.61, -58.37,  12,  "Buenos Aires",   -34.61, -58.37),
    (13, "Instanbul",          41.01,  28.97,  13,  "Istanbul",        41.01,  28.97),
    (14, "Djakarta",           -6.21, 106.85,  14,  "Jakarta",         -6.21, 106.85),
    (15, "Sankt-Peterburg",    59.94,  30.32,  15,  "Saint Petersburg",59.94,  30.32),
    # hard negatives — names sound alike but are distant
    (16, "Cairo (Illinois)",   37.00, -89.18,  16,  "Cairo",           30.06,  31.25),
    (17, "Frankfurt am Main",  50.11,   8.68,  17,  "Frankfurt",       50.11,   8.68),
    # cross-block pair — must share a block to be compared at 5° grid
    (18, "Lyon France",        45.75,   4.83,  18,  "Lyon",            45.75,   4.83),
    (19, "Marseilles",         43.30,   5.37,  19,  "Marseille",       43.30,   5.37),
    (20, "Guadalahara",        20.67,-103.35,  20,  "Guadalajara",     20.67,-103.35),
]

GROUND_TRUTH = {r[0]: r[4] for r in CITIES}  # left_id → right_id (true match)


def _build_gdfs():
    """Return (left_gdf, right_gdf) as GeoDataFrames with EPSG:4326."""
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as e:
        raise ImportError(
            "This example requires geopandas and shapely.\n"
            "Install with: pip install 'fuzzy_llm_matcher[geo]'"
        ) from e

    left_rows  = [(r[0], r[1], r[2], r[3]) for r in CITIES]
    right_rows = [(r[4], r[5], r[6], r[7]) for r in CITIES]

    left = gpd.GeoDataFrame(
        pd.DataFrame(left_rows, columns=["id", "name", "lat", "lon"]),
        geometry=[Point(lon, lat) for _, _, lat, lon in left_rows],
        crs="EPSG:4326",
    )
    right = gpd.GeoDataFrame(
        pd.DataFrame(right_rows, columns=["id", "name", "lat", "lon"]),
        geometry=[Point(lon, lat) for _, _, lat, lon in right_rows],
        crs="EPSG:4326",
    )
    return left, right


# ---------------------------------------------------------------------------
# Section A – Run match_geodataframes()
# ---------------------------------------------------------------------------

def run_matching():
    from fuzzy_llm_matcher import match_geodataframes

    left, right = _build_gdfs()

    result = match_geodataframes(
        left, right,
        left_on="name", right_on="name",
        left_id="id", right_id="id",
        spatial_block_degrees=5.0,
        max_distance_km=500.0,
        top_k=5,
        use_llm=True,
        return_geometry=True,
    )

    result["is_correct"] = result.apply(
        lambda r: GROUND_TRUTH.get(r["left_id"]) == r["right_id"], axis=1
    )

    n_total   = len(result)
    n_decided = int(result["final_decision"].sum())
    n_correct = int((result["final_decision"] & result["is_correct"]).sum())
    precision = n_correct / n_decided if n_decided else 0.0
    recall    = n_correct / n_total
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("── match_geodataframes() results ────────────────")
    print(f"  Total pairs       : {n_total}")
    print(f"  Final decisions   : {n_decided}")
    print(f"  Correct           : {n_correct}")
    print(f"  Precision         : {precision:.3f}")
    print(f"  Recall            : {recall:.3f}")
    print(f"  F1                : {f1:.3f}")
    print(f"  Has geometry col  : {'geometry' in result.columns}")
    print(f"  CRS               : {result.crs}")
    print()
    print("  Reliability distribution:")
    print(result["reliability_label"].value_counts().to_string())
    print()
    print("  Geo-distance score summary:")
    print(result["score_geo_distance"].describe().round(1).to_string())
    print()
    return result


# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------

LABEL_COLORS = {
    "high":          "#00e5ff",
    "medium_review": "#ffb300",
    "low":           "#ff5252",
    "reject":        "#888888",
}

BG  = "#0d1b2a"
MID = "#12263a"
GRID_COLOR = "#2e5f8a"


def _fig_style(ax):
    ax.set_facecolor(MID)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)


# ---------------------------------------------------------------------------
# Figure 1 – Scatter: fuzzy score vs score_geo_distance
# ---------------------------------------------------------------------------

def fig_scatter(result: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
    _fig_style(ax)

    for label, grp in result.groupby("reliability_label"):
        color = LABEL_COLORS.get(label, "#cccccc")
        ok  = grp[grp["is_correct"]]
        bad = grp[~grp["is_correct"]]
        ax.scatter(ok["fuzzy_score"],  ok["score_geo_distance"],
                   c=color, s=90,  marker="o", alpha=0.9,
                   edgecolors="white", linewidths=0.4, zorder=3,
                   label=f"{label} ✓ (n={len(ok)})")
        ax.scatter(bad["fuzzy_score"], bad["score_geo_distance"],
                   c=color, s=100, marker="X", alpha=0.9,
                   edgecolors="white", linewidths=0.4, zorder=3,
                   label=f"{label} ✗ (n={len(bad)})")

    ax.axhline(50, color=GRID_COLOR, lw=0.8, ls="--", alpha=0.6)
    ax.axvline(92, color=GRID_COLOR, lw=0.8, ls="--", alpha=0.6)
    ax.text(93, 2,  "high_threshold=92",   color=GRID_COLOR, fontsize=8)
    ax.text(2,  51, "geo=50 (250 km mark)", color=GRID_COLOR, fontsize=8)

    ax.set_xlabel("Fuzzy string score (WRatio, 0–100)", color="white", fontsize=12)
    ax.set_ylabel("Geo-distance score (0–100)",          color="white", fontsize=12)
    ax.set_title("match_geodataframes() – Spatial Confidence vs String Similarity\n"
                 "○ = correct match   ✗ = wrong match",
                 color="white", fontsize=12, fontweight="bold")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8, framealpha=0.35, facecolor=BG,
              edgecolor=GRID_COLOR, labelcolor="white", loc="lower right")
    fig.text(0.5, 0.01, "github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=8, color="#a0bdd8")

    out = out_dir / "geo_gdf_scatter.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2 – Spatial blocking grid map
# ---------------------------------------------------------------------------

def _arc(ax, x0, y0, x1, y1, color, lw=1.2, alpha=0.75, n=60):
    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2 + abs(x1 - x0) * 0.12
    t  = np.linspace(0, 1, n)
    xs = (1 - t)**2 * x0 + 2*(1-t)*t * mid_x + t**2 * x1
    ys = (1 - t)**2 * y0 + 2*(1-t)*t * mid_y + t**2 * y1
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=3, solid_capstyle="round")


def _load_world():
    """Return a geopandas GeoDataFrame for the world land, with fallback."""
    try:
        import geopandas as gpd
        try:
            import geodatasets
            return gpd.read_file(geodatasets.get_path("naturalearth.land"))
        except Exception:
            return gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception:
        return None


def fig_spatial_blocks(result: pd.DataFrame, out_dir: Path,
                       block_degrees: float = 5.0) -> None:
    world = _load_world()

    fig, ax = plt.subplots(figsize=(20, 10), facecolor=BG)
    ax.set_facecolor(BG)

    if world is not None:
        world.plot(ax=ax, color="#1c3557", edgecolor=GRID_COLOR,
                   linewidth=0.4, zorder=1)

    # Draw the spatial blocking grid
    d = block_degrees
    for lon in np.arange(-180, 181, d):
        ax.axvline(lon, color=GRID_COLOR, lw=0.35, alpha=0.35, zorder=2)
    for lat in np.arange(-90, 91, d):
        ax.axhline(lat, color=GRID_COLOR, lw=0.35, alpha=0.35, zorder=2)

    # Shade shared block cells
    plotted_cells: set = set()
    for row in CITIES:
        la_l, lo_l = row[2], row[3]
        la_r, lo_r = row[6], row[7]
        cell_l = (math.floor(la_l / d), math.floor(lo_l / d))
        cell_r = (math.floor(la_r / d), math.floor(lo_r / d))
        if cell_l == cell_r and cell_l not in plotted_cells:
            plotted_cells.add(cell_l)
            rx = cell_l[1] * d
            ry = cell_l[0] * d
            rect = plt.Rectangle((rx, ry), d, d, linewidth=0,
                                  facecolor="#00e5ff", alpha=0.08, zorder=2)
            ax.add_patch(rect)

    # Plot city points and arcs
    for _, r in result.iterrows():
        lid = r["left_id"]
        rid = r["right_id"]
        city = next((c for c in CITIES if c[0] == lid), None)
        rcity = next((c for c in CITIES if c[4] == rid), None)
        if city is None or rcity is None:
            continue
        lo_l, la_l = city[3],  city[2]
        lo_r, la_r = rcity[7], rcity[6]

        label = r.get("reliability_label", "reject")
        color = LABEL_COLORS.get(label, "#888888")
        geo_s = float(r.get("score_geo_distance", 50) or 50)
        alpha = 0.35 + 0.55 * geo_s / 100.0

        _arc(ax, lo_l, la_l, lo_r, la_r, color, lw=1.5, alpha=alpha)
        ax.scatter(lo_l, la_l, s=30, c=color, edgecolors="white",
                   linewidths=0.3, zorder=5, alpha=0.9)
        ax.scatter(lo_r, la_r, s=45, c=color, marker="*",
                   zorder=6, alpha=0.95)

    legend_items = [
        mpatches.Patch(facecolor="#00e5ff", alpha=0.2, label=f"Shared {d}° block cell"),
        *[mpatches.Patch(color=c, label=l) for l, c in LABEL_COLORS.items()],
        mpatches.Patch(color="white", label="● left entity   ★ right entity"),
    ]
    leg = ax.legend(handles=legend_items, loc="lower left", fontsize=9,
                    framealpha=0.3, facecolor=BG, edgecolor=GRID_COLOR,
                    labelcolor="white", title=f"Spatial blocking ({d}°)",
                    title_fontsize=10)
    leg.get_title().set_color("white")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.axis("off")
    ax.set_title(
        f"Spatial Blocking Grid ({d}° cells) — only pairs in the same cell are compared\n"
        "fuzzy_llm_matcher · match_geodataframes()",
        color="white", fontsize=14, fontweight="bold", pad=14,
    )
    fig.text(0.5, 0.02, "github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=9, color="#a0bdd8")

    out = out_dir / "geo_gdf_spatial_blocks.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3 – Match map coloured by geo-distance score
# ---------------------------------------------------------------------------

def fig_match_map(result: pd.DataFrame, out_dir: Path) -> None:
    world = _load_world()
    cmap  = plt.get_cmap("RdYlGn")

    fig, ax = plt.subplots(figsize=(20, 10), facecolor=BG)
    ax.set_facecolor(BG)
    if world is not None:
        world.plot(ax=ax, color="#1c3557", edgecolor=GRID_COLOR,
                   linewidth=0.4, zorder=1)

    for _, r in result.iterrows():
        lid = r["left_id"]
        rid = r["right_id"]
        city  = next((c for c in CITIES if c[0] == lid),  None)
        rcity = next((c for c in CITIES if c[4] == rid),  None)
        if city is None or rcity is None:
            continue

        lo_l, la_l = city[3],  city[2]
        lo_r, la_r = rcity[7], rcity[6]

        geo_s  = float(r.get("score_geo_distance", 50) or 50)
        color  = cmap(geo_s / 100.0)
        decided = bool(r.get("final_decision", False))
        lw    = 2.4 if decided else 0.8
        alpha = 0.90 if decided else 0.40

        _arc(ax, lo_l, la_l, lo_r, la_r, color, lw=lw, alpha=alpha)
        ax.scatter(lo_l, la_l, s=28, color=color, zorder=5,
                   edgecolors="white", linewidths=0.3, alpha=0.9)
        ax.scatter(lo_r, la_r, s=44, color=color, marker="*",
                   zorder=6, alpha=0.95)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 100))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                      fraction=0.025, pad=0.03, aspect=45)
    cb.set_label("score_geo_distance (0=distant, 100=same location)",
                 color="white", fontsize=10)
    cb.ax.xaxis.set_tick_params(color="white")
    plt.setp(cb.ax.xaxis.get_ticklabels(), color="white")

    legend_items = [
        mpatches.Patch(color="white",   label="● left entity   ★ right entity"),
        mpatches.Patch(color="#aaaaaa", label="thick arc = final_decision=True"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=9,
              framealpha=0.25, facecolor=BG, edgecolor=GRID_COLOR,
              labelcolor="white")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.axis("off")
    ax.set_title(
        "match_geodataframes() — Matched Pairs with Geo-Distance Score\n"
        "Arc colour: green = nearby, red = far  |  thick arc = confirmed match",
        color="white", fontsize=14, fontweight="bold", pad=14,
    )
    fig.text(0.5, 0.02, "github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=9, color="#a0bdd8")

    out = out_dir / "geo_gdf_match_map.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 4 – Per-pair bar chart
# ---------------------------------------------------------------------------

def fig_score_bars(result: pd.DataFrame, out_dir: Path) -> None:
    df = result.sort_values("reliability_label").reset_index(drop=True)
    n  = len(df)
    x  = np.arange(n)
    w  = 0.38

    fig, ax = plt.subplots(figsize=(max(10, n * 0.65), 5), facecolor=BG)
    _fig_style(ax)

    colors = [LABEL_COLORS.get(l, "#888888") for l in df["reliability_label"]]

    bars1 = ax.bar(x - w/2, df["fuzzy_score"],        width=w, color=colors,
                   alpha=0.75, label="fuzzy_score (string)",
                   edgecolor="white", linewidth=0.3)
    bars2 = ax.bar(x + w/2, df["score_geo_distance"], width=w, color=colors,
                   alpha=1.00, label="score_geo_distance",
                   edgecolor="white", linewidth=0.3, hatch="//")

    # Mark final decisions with a crown marker on top of the string bar
    for i, row in df.iterrows():
        if row.get("final_decision"):
            ax.text(i - w/2, row["fuzzy_score"] + 1.5, "✓",
                    ha="center", va="bottom", color="white", fontsize=9)

    # x-tick labels: left_value truncated
    labels = [str(v)[:12] for v in df["left_value"]]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", color="white", fontsize=8)

    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (0–100)", color="white", fontsize=11)
    ax.set_title(
        "Per-Pair Scores from match_geodataframes()\n"
        "Solid = fuzzy string score   Hatched = geo-distance score   ✓ = final_decision",
        color="white", fontsize=12, fontweight="bold",
    )

    legend_patches = [
        mpatches.Patch(color="#00e5ff",  label="high"),
        mpatches.Patch(color="#ffb300",  label="medium_review"),
        mpatches.Patch(color="#ff5252",  label="low"),
        mpatches.Patch(color="#888888",  label="reject"),
        mpatches.Patch(facecolor="white", hatch="//", label="score_geo_distance"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, framealpha=0.35,
              facecolor=BG, edgecolor=GRID_COLOR, labelcolor="white",
              loc="upper right")

    fig.text(0.5, 0.0, "github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=8, color="#a0bdd8")

    out = out_dir / "geo_gdf_score_bars.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Section C – GeoJSON export demo
# ---------------------------------------------------------------------------

def export_demo(result, out_dir: Path) -> None:
    out_json = out_dir / "geo_gdf_matches.geojson"
    try:
        import geopandas as gpd
        if isinstance(result, gpd.GeoDataFrame) and result.crs is not None:
            # Export only decided matches
            decided = result[result["final_decision"] == True].copy()
            # GeoJSON doesn't support some dtypes; cast boolean/object columns
            for col in decided.select_dtypes(include=["object"]).columns:
                decided[col] = decided[col].astype(str)
            decided.to_file(out_json, driver="GeoJSON")
            print(f"  GeoJSON export: {out_json}  ({len(decided)} features)")
    except Exception as exc:
        print(f"  GeoJSON export skipped: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    out_dir = Path("notebooks/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running match_geodataframes() …\n")
    result = run_matching()

    print("Generating figures …")
    fig_scatter(result, out_dir)
    fig_spatial_blocks(result, out_dir)
    fig_match_map(result, out_dir)
    fig_score_bars(result, out_dir)

    print("\nExport demo …")
    export_demo(result, out_dir)

    print("\nDone. Figures in", out_dir)
    return result


if __name__ == "__main__":
    run()
