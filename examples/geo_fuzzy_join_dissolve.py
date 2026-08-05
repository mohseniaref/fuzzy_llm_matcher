"""Fuzzy table join and dissolve for GeoDataFrames.

Demonstrates three new operations:

1. ``fuzzy_join()``               — fuzzy pd.merge() for plain DataFrames
2. ``fuzzy_join_geodataframes()`` — fuzzy gpd.sjoin() for GeoDataFrames
3. ``fuzzy_dissolve()``           — match + merge geometries of matched pairs

Figures produced (→ notebooks/figures/)
-----------------------------------------
    geo_fuzzy_join_table.png      – side-by-side left/right attributes after
                                    fuzzy_join(), coloured by reliability
    geo_fuzzy_join_map.png        – world map: matched features with right-side
                                    attributes joined, sized by joined population
    geo_fuzzy_dissolve_union.png  – dissolved union polygons on a map grid
    geo_fuzzy_dissolve_ops.png    – four subplots comparing dissolve_op modes
                                    (union / intersection / envelope / centroid)

Run:
    python examples/geo_fuzzy_join_dissolve.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Synthetic datasets ─────────────────────────────────────────────────────
#
# left_gdf  = "dirty" city records (as they appear in field data)
# right_gdf = "canonical" city records (authoritative reference table)
#
# Each record has: id, name, population (left), area_km2 (right), coordinates

DIRTY_CITIES = [
    # (id, name,              lat,    lon,    population)
    (1,  "München",          48.14,  11.58,  1_484_226),
    (2,  "Köln",             50.94,   6.96,  1_083_498),
    (3,  "Nürnberg",         49.45,  11.08,    522_443),
    (4,  "NYC",              40.71,  -74.00, 8_336_817),
    (5,  "San Fran",         37.77, -122.42,   873_965),
    (6,  "Tokio",            35.69,  139.69, 13_960_000),
    (7,  "Moskva",           55.75,   37.62, 12_500_000),
    (8,  "Al Qahirah",       30.06,   31.25, 10_100_000),
    (9,  "Peking",           39.93,  116.39, 21_540_000),
    (10, "Bombay",           19.08,   72.88, 12_478_447),
    (11, "Rio de Jan.",     -22.91,  -43.18,  6_748_000),
    (12, "Buenos Ayres",    -34.61,  -58.37, 15_150_000),
    (13, "Instanbul",        41.01,   28.97, 15_462_000),
    (14, "Djakarta",         -6.21,  106.85, 10_560_000),
    (15, "Lyon France",      45.75,    4.83,    516_092),
]

CANONICAL_CITIES = [
    # (id, name,               lat,    lon,    area_km2)
    (1,  "Munich",           48.14,  11.58,    310.43),
    (2,  "Cologne",          50.94,   6.96,    405.15),
    (3,  "Nuremberg",        49.45,  11.08,    186.37),
    (4,  "New York City",    40.71,  -74.00,   783.84),
    (5,  "San Francisco",    37.77, -122.42,   121.40),
    (6,  "Tokyo",            35.69,  139.69,  2_194.0),
    (7,  "Moscow",           55.75,   37.62,  2_511.0),
    (8,  "Cairo",            30.06,   31.25,  3_085.1),
    (9,  "Beijing",          39.93,  116.39, 16_411.0),
    (10, "Mumbai",           19.08,   72.88,    603.4),
    (11, "Rio de Janeiro",  -22.91,  -43.18,  1_221.3),
    (12, "Buenos Aires",    -34.61,  -58.37,    203.0),
    (13, "Istanbul",         41.01,   28.97,  5_343.0),
    (14, "Jakarta",          -6.21,  106.85,    664.0),
    (15, "Lyon",             45.75,    4.83,     47.87),
    # Extra canonical entries that have no dirty counterpart
    (16, "Berlin",           52.52,   13.41,    891.68),
    (17, "Madrid",           40.41,   -3.70,    604.45),
]

GROUND_TRUTH = {r[0]: r[0] for r in DIRTY_CITIES}  # id matches id for all 15


# ── Build GeoDataFrames ────────────────────────────────────────────────────

def _build_gdfs():
    try:
        import geopandas as gpd
        from shapely.geometry import Point, box
    except ImportError as e:
        raise ImportError(
            "This example requires geopandas + shapely.\n"
            "pip install 'fuzzy_llm_matcher[geo]'"
        ) from e

    left = gpd.GeoDataFrame(
        pd.DataFrame(DIRTY_CITIES,     columns=["id", "name", "lat", "lon", "population"]),
        geometry=[Point(lon, lat) for _, _, lat, lon, _ in DIRTY_CITIES],
        crs="EPSG:4326",
    )
    right = gpd.GeoDataFrame(
        pd.DataFrame(CANONICAL_CITIES, columns=["id", "name", "lat", "lon", "area_km2"]),
        geometry=[Point(lon, lat) for _, _, lat, lon, _ in CANONICAL_CITIES],
        crs="EPSG:4326",
    )

    # Also build small synthetic polygon GDFs (±0.3° boxes around each city)
    # so we can demonstrate dissolve on polygon features
    def _box(lat, lon, d=0.3):
        return box(lon - d, lat - d, lon + d, lat + d)

    left_poly = left.copy()
    left_poly["geometry"] = [_box(r.lat, r.lon) for r in left.itertuples()]

    right_poly = right.copy()
    right_poly["geometry"] = [_box(r.lat, r.lon, d=0.25) for r in right.itertuples()]

    return left, right, left_poly, right_poly


# ── Colour scheme ──────────────────────────────────────────────────────────

BG         = "#0d1b2a"
MID        = "#12263a"
GRID_COLOR = "#2e5f8a"
LABEL_COLORS = {
    "high":          "#00e5ff",
    "medium_review": "#ffb300",
    "low":           "#ff5252",
    "reject":        "#888888",
}


def _fig_style(ax):
    ax.set_facecolor(MID)
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COLOR)


def _load_world():
    try:
        import geopandas as gpd
        try:
            import geodatasets
            return gpd.read_file(geodatasets.get_path("naturalearth.land"))
        except Exception:
            return gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception:
        return None


# ── Figure 1 – fuzzy_join() attribute table ────────────────────────────────

def fig_join_table(joined: pd.DataFrame, out_dir: Path) -> None:
    """Visualise the joined attribute columns as a colour-coded table."""
    cols_to_show = ["name", "name_right", "population", "area_km2",
                    "_fuzzy_score", "_reliability"]
    cols_to_show = [c for c in cols_to_show if c in joined.columns]
    df = joined[cols_to_show].copy()

    # Format numbers
    for col in ("population", "area_km2"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: f"{v:,.0f}" if pd.notna(v) and v == v else "—"
            )
    if "_fuzzy_score" in df.columns:
        df["_fuzzy_score"] = df["_fuzzy_score"].apply(
            lambda v: f"{v:.1f}" if pd.notna(v) and v == v else "—"
        )

    n_rows = len(df)
    fig_h  = max(4, 0.38 * n_rows + 1.2)
    fig, ax = plt.subplots(figsize=(12, fig_h), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    col_labels = {
        "name":         "Left name (dirty)",
        "name_right":   "Right name (canonical)",
        "population":   "Population",
        "area_km2":     "Area km²",
        "_fuzzy_score": "Fuzzy score",
        "_reliability": "Reliability",
    }
    headers = [col_labels.get(c, c) for c in df.columns]

    # Row colours from reliability
    row_colors = []
    for _, row in df.iterrows():
        rel   = row.get("_reliability", "reject")
        color = LABEL_COLORS.get(rel, "#888888")
        # very transparent background
        row_colors.append([color + "22"] * len(df.columns))

    tbl = ax.table(
        cellText=df.values,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    # Style header row
    for j, _ in enumerate(headers):
        cell = tbl[0, j]
        cell.set_facecolor(GRID_COLOR)
        cell.set_text_props(color="white", fontweight="bold")
    # Style data cells
    for i in range(1, n_rows + 1):
        for j in range(len(df.columns)):
            cell = tbl[i, j]
            cell.set_text_props(color="white")
            rel = df.iloc[i - 1].get("_reliability", "reject")
            cell.set_facecolor(LABEL_COLORS.get(rel, "#888888") + "33")

    # Legend
    patches = [mpatches.Patch(color=c, label=l) for l, c in LABEL_COLORS.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=8,
              framealpha=0.4, facecolor=BG, edgecolor=GRID_COLOR,
              labelcolor="white")

    ax.set_title(
        "fuzzy_join() result — left attributes enriched with right table columns\n"
        "Population (left, dirty) · Area km² (right, canonical)",
        color="white", fontsize=12, fontweight="bold", pad=12,
    )
    fig.text(0.5, 0.01, "github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=8, color="#a0bdd8")

    out = out_dir / "geo_fuzzy_join_table.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2 – fuzzy_join_geodataframes() world map ───────────────────────

def _arc(ax, x0, y0, x1, y1, color, lw=1.2, alpha=0.75, n=60):
    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2 + abs(x1 - x0) * 0.12
    t  = np.linspace(0, 1, n)
    xs = (1 - t)**2 * x0 + 2*(1-t)*t*mid_x + t**2 * x1
    ys = (1 - t)**2 * y0 + 2*(1-t)*t*mid_y + t**2 * y1
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=3,
            solid_capstyle="round")


def fig_join_map(joined_gdf, out_dir: Path) -> None:
    world = _load_world()

    fig, ax = plt.subplots(figsize=(20, 10), facecolor=BG)
    ax.set_facecolor(BG)
    if world is not None:
        world.plot(ax=ax, color="#1c3557", edgecolor=GRID_COLOR,
                   linewidth=0.4, zorder=1)

    # Size markers by population; colour by reliability
    pop_col = "population" if "population" in joined_gdf.columns else None
    max_pop = joined_gdf["population"].max() if pop_col else 1

    for _, row in joined_gdf.iterrows():
        geom  = row.geometry
        if geom is None:
            continue
        lx, ly = geom.x, geom.y
        rel   = row.get("_reliability", "reject")
        color = LABEL_COLORS.get(rel, "#888888")
        size  = 30 + 220 * (row.get("population", 0) or 0) / max_pop if pop_col else 60

        ax.scatter(lx, ly, s=size, color=color, zorder=5,
                   edgecolors="white", linewidths=0.4, alpha=0.9)

        # Label with both names
        left_n  = str(row.get("name",       ""))[:10]
        right_n = str(row.get("name_right", ""))[:10]
        if left_n and left_n != right_n:
            ax.annotate(
                f"{left_n} → {right_n}",
                xy=(lx, ly), xytext=(lx + 1.5, ly + 1.5),
                fontsize=6, color="white", alpha=0.75,
                arrowprops=dict(arrowstyle="-", color="white", lw=0.4),
            )

    legend_items = [
        mpatches.Patch(color=c, label=l) for l, c in LABEL_COLORS.items()
    ] + [mpatches.Patch(color="white",
                        label="marker size ∝ population")]
    leg = ax.legend(handles=legend_items, loc="lower left", fontsize=9,
                    framealpha=0.3, facecolor=BG, edgecolor=GRID_COLOR,
                    labelcolor="white", title="fuzzy_join_geodataframes()",
                    title_fontsize=10)
    leg.get_title().set_color("white")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.axis("off")
    ax.set_title(
        "fuzzy_join_geodataframes() — Left GeoDataFrame enriched with Right attributes\n"
        "Marker colour = reliability label   Marker size = population (left table)",
        color="white", fontsize=13, fontweight="bold", pad=14,
    )
    fig.text(0.5, 0.02, "github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=9, color="#a0bdd8")

    out = out_dir / "geo_fuzzy_join_map.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3 – dissolve union polygons ────────────────────────────────────

def fig_dissolve_union(dissolved, left_poly, right_poly, out_dir: Path) -> None:
    """Show three layers: left polygons, right polygons, dissolved unions."""
    # Focus on Europe for clarity
    x0, x1, y0, y1 = -2, 17, 43, 56

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=BG)
    titles = ["Left GeoDataFrame\n(dirty names)", "Right GeoDataFrame\n(canonical)",
              "fuzzy_dissolve(op='union')\nmatched & merged"]

    layers = [left_poly, right_poly, dissolved]
    colors = ["#00e5ff", "#ffb300", "#7cfc00"]

    for ax, title, gdf, color in zip(axes, titles, layers, colors):
        ax.set_facecolor(BG)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_edgecolor(GRID_COLOR)

        # Filter to Europe extent
        try:
            subset = gdf.cx[x0:x1, y0:y1]
        except Exception:
            subset = gdf

        for _, row in subset.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type in ("Polygon", "MultiPolygon"):
                if geom.geom_type == "Polygon":
                    xs, ys = geom.exterior.xy
                    ax.fill(xs, ys, color=color, alpha=0.35, zorder=2)
                    ax.plot(xs, ys, color=color, lw=0.8, zorder=3)
                else:
                    for part in geom.geoms:
                        xs, ys = part.exterior.xy
                        ax.fill(xs, ys, color=color, alpha=0.35, zorder=2)
                        ax.plot(xs, ys, color=color, lw=0.8, zorder=3)
                # Label
                c = geom.centroid
                name_col = "name" if "name" in row.index else "_left_name"
                label = str(row.get(name_col, ""))[:8]
                ax.text(c.x, c.y, label, ha="center", va="center",
                        color="white", fontsize=7, zorder=5)

        ax.set_title(title, color="white", fontsize=10, fontweight="bold")

    fig.suptitle(
        "Fuzzy Dissolve — Matching and Merging Spatial Features\n"
        "Example: European cities (±0.3° synthetic polygons)",
        color="white", fontsize=13, fontweight="bold",
    )
    fig.text(0.5, 0.01, "github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=8, color="#a0bdd8")

    out = out_dir / "geo_fuzzy_dissolve_union.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 4 – four dissolve operations comparison ────────────────────────

def fig_dissolve_ops(left_poly, right_poly, out_dir: Path) -> None:
    """4-panel plot comparing union / intersection / envelope / centroid."""
    from fuzzy_llm_matcher import fuzzy_dissolve

    ops   = ["union", "intersection", "envelope", "centroid"]
    colors = ["#00e5ff", "#ffb300", "#7cfc00", "#ff69b4"]

    # Focus on Munich / Cologne / Nuremberg / Lyon
    x0, x1, y0, y1 = 3, 15, 43, 55

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG)
    axes_flat = axes.flatten()

    for ax, op, color in zip(axes_flat, ops, colors):
        ax.set_facecolor(BG)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_edgecolor(GRID_COLOR)

        # Draw the input polygons faintly
        for gdf, c in [(left_poly, "#ffffff"), (right_poly, "#aaaaaa")]:
            try:
                subset = gdf.cx[x0:x1, y0:y1]
            except Exception:
                subset = gdf
            for _, row in subset.iterrows():
                geom = row.geometry
                if geom is None:
                    continue
                if geom.geom_type == "Polygon":
                    xs, ys = geom.exterior.xy
                    ax.fill(xs, ys, color=c, alpha=0.06, zorder=1)
                    ax.plot(xs, ys, color=c, lw=0.5, alpha=0.4, zorder=2)

        # Compute dissolve
        try:
            dissolved = fuzzy_dissolve(
                left_poly, right_poly,
                left_on="name", right_on="name",
                left_id="id", right_id="id",
                dissolve_op=op,
                spatial_block_degrees=5.0,
                max_distance_km=500.0,
                use_llm=True,
                high_threshold=85,
            )
            try:
                subset = dissolved.cx[x0:x1, y0:y1]
            except Exception:
                subset = dissolved

            for _, row in subset.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                if geom.geom_type == "Point":
                    ax.scatter(geom.x, geom.y, s=80, color=color, zorder=5,
                               edgecolors="white", linewidths=0.6)
                elif geom.geom_type == "MultiPoint":
                    for pt in geom.geoms:
                        ax.scatter(pt.x, pt.y, s=80, color=color, zorder=5,
                                   edgecolors="white", linewidths=0.6)
                else:
                    parts = [geom] if geom.geom_type in ("Polygon", "LineString") \
                            else list(geom.geoms)
                    for part in parts:
                        if hasattr(part, "exterior"):
                            xs, ys = part.exterior.xy
                            ax.fill(xs, ys, color=color, alpha=0.45, zorder=3)
                            ax.plot(xs, ys, color=color, lw=1.2, zorder=4)
                        elif hasattr(part, "xy"):
                            ax.plot(*part.xy, color=color, lw=1.5, zorder=4)

                # Label with both names
                try:
                    c_pt = geom.centroid
                    label = f"{row.get('_left_name','')[:8]}"
                    ax.text(c_pt.x, c_pt.y, label,
                            ha="center", va="center", color="white",
                            fontsize=7, zorder=6)
                except Exception:
                    pass

        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", transform=ax.transAxes,
                    ha="center", va="center", color="red", fontsize=8)

        ax.set_title(f"dissolve_op='{op}'", color=color,
                     fontsize=11, fontweight="bold")
        # Light input-layer legend
        ax.plot([], [], color="white",   lw=1, alpha=0.4, label="left polygon")
        ax.plot([], [], color="#aaaaaa", lw=1, alpha=0.4, label="right polygon")
        ax.fill([], [], color=color, alpha=0.45, label="dissolved result")
        ax.legend(fontsize=7, framealpha=0.3, facecolor=BG,
                  edgecolor=GRID_COLOR, labelcolor="white")

    fig.suptitle(
        "fuzzy_dissolve() — Four Geometry Combination Modes\n"
        "White = left polygons   Grey = right polygons   Colour = dissolved result",
        color="white", fontsize=13, fontweight="bold",
    )
    fig.text(0.5, 0.01, "github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=8, color="#a0bdd8")

    out = out_dir / "geo_fuzzy_dissolve_ops.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ────────────────────────────────────────────────────────────────────

def run():
    from fuzzy_llm_matcher import fuzzy_dissolve, fuzzy_join, fuzzy_join_geodataframes

    out_dir = Path("notebooks/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    left_gdf, right_gdf, left_poly, right_poly = _build_gdfs()

    # ── Demo 1: fuzzy_join() on plain DataFrames ─────────────────────────────
    print("── fuzzy_join() ──────────────────────────────")
    left_df  = pd.DataFrame(DIRTY_CITIES,     columns=["id", "name", "lat", "lon", "population"])
    right_df = pd.DataFrame(CANONICAL_CITIES, columns=["id", "name", "lat", "lon", "area_km2"])

    joined_df = fuzzy_join(
        left_df, right_df,
        left_on="name", right_on="name",
        left_id="id", right_id="id",
        how="inner",
        suffixes=("", "_right"),
        use_llm=True,          # MockLLMClient confirms medium_review pairs
        high_threshold=85,     # slightly relaxed for plain-text matching
    )
    print(f"  Rows matched (inner):  {len(joined_df)}")
    print(f"  Columns: {list(joined_df.columns)}")
    print()

    joined_left = fuzzy_join(
        left_df, right_df,
        left_on="name", right_on="name",
        left_id="id", right_id="id",
        how="left",
        suffixes=("", "_right"),
        use_llm=True,
        high_threshold=85,
    )
    print(f"  Rows (left join):  {len(joined_left)}  "
          f"(unmatched get NaN in right columns)")
    print()

    # ── Demo 2: fuzzy_join_geodataframes() ───────────────────────────────────
    print("── fuzzy_join_geodataframes() ────────────────")
    joined_gdf = fuzzy_join_geodataframes(
        left_gdf, right_gdf,
        left_on="name", right_on="name",
        left_id="id", right_id="id",
        how="inner",
        suffixes=("", "_right"),
        spatial_block_degrees=5.0,
        max_distance_km=500.0,
        geometry="left",
        use_llm=True,
        high_threshold=85,
    )
    print(f"  Rows matched:  {len(joined_gdf)}")
    print(f"  Is GeoDataFrame:  {hasattr(joined_gdf, 'crs')}")
    print(f"  CRS:  {joined_gdf.crs}")
    print(f"  Columns: {list(joined_gdf.columns)}")
    print()

    # ── Demo 3: fuzzy_dissolve() ──────────────────────────────────────────────
    print("── fuzzy_dissolve() ──────────────────────────")
    dissolved_union = fuzzy_dissolve(
        left_poly, right_poly,
        left_on="name", right_on="name",
        left_id="id", right_id="id",
        dissolve_op="union",
        aggfunc={"population": "mean"},
        spatial_block_degrees=5.0,
        max_distance_km=500.0,
        use_llm=True,
        high_threshold=85,
    )
    print(f"  Pairs dissolved (union):  {len(dissolved_union)}")
    print(f"  Columns: {list(dissolved_union.columns)}")
    print()

    dissolved_inter = fuzzy_dissolve(
        left_poly, right_poly,
        left_on="name", right_on="name",
        left_id="id", right_id="id",
        dissolve_op="intersection",
        spatial_block_degrees=5.0,
        max_distance_km=500.0,
    )
    non_empty = dissolved_inter[~dissolved_inter.geometry.is_empty]
    print(f"  Pairs dissolved (intersection): {len(non_empty)} non-empty")
    print()

    # ── Figures ───────────────────────────────────────────────────────────────
    print("Generating figures …")
    fig_join_table(joined_df, out_dir)
    fig_join_map(joined_gdf, out_dir)
    fig_dissolve_union(dissolved_union, left_poly, right_poly, out_dir)
    fig_dissolve_ops(left_poly, right_poly, out_dir)

    print("\nDone.")
    return joined_df, joined_gdf, dissolved_union


if __name__ == "__main__":
    run()
