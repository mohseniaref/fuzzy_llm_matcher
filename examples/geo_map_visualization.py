"""World map visualization of the geo place-name fuzzy matching results.

Produces a publication-quality map showing:
  - Each city pair (dirty left name → canonical right name)
  - Arcs connecting matched pairs, colored by reliability
  - City markers sized by matching confidence
  - Clean legend and title suitable for LinkedIn / presentations

Output: notebooks/figures/geo_matching_world_map.png

Run:
    python examples/geo_map_visualization.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fuzzy_llm_matcher import match_tables

# ---------------------------------------------------------------------------
# City coordinates (lon, lat)
# ---------------------------------------------------------------------------
LEFT_COORDS = {
    "Köln":                  (6.96,  50.94),
    "München":               (11.58, 48.14),
    "Nürnberg":              (11.08, 49.45),
    "Düsseldorf":            (6.79,  51.22),
    "NYC":                   (-74.0, 40.71),
    "LA":                    (-118.24, 34.05),
    "San Fran":              (-122.42, 37.77),
    "Philly":                (-75.16, 39.95),
    "Saint-Denis":           (2.36,  48.93),
    "Lyon France":           (4.83,  45.75),
    "Marseilles":            (5.37,  43.30),
    "Strasbourg-Alsace":     (7.75,  48.58),
    "Al Qahirah":            (31.25, 30.06),
    "Aleksandria":           (29.92, 31.20),
    "Ispahan":               (51.68, 32.66),
    "Teheran":               (51.42, 35.69),
    "Moskva":                (37.62, 55.75),
    "Sankt-Peterburg":       (30.32, 59.94),
    "Peking":                (116.39, 39.93),
    "Canton":                (113.27, 23.13),
    "Bombay":                (72.88, 19.08),
    "Calcutta":              (88.37, 22.57),
    "Madras":                (80.28, 13.09),
    "Bangalore":             (77.59, 12.97),
    "Rio de Jan.":           (-43.18, -22.91),
    "Sao Paolo":             (-46.63, -23.55),
    "Bogotá":                (-74.08, 4.71),
    "Buenos Ayres":          (-58.37, -34.61),
    "Ciudad de Mexico":      (-99.13, 19.43),
    "Guadalahara":           (-103.35, 20.67),
    "Instanbul":             (28.97, 41.01),
    "Izmir Turkey":          (27.14, 38.41),
    "Soul":                  (126.98, 37.57),
    "Busan Korea":           (129.04, 35.10),
    "Tokio":                 (139.69, 35.69),
    "Osaca":                 (135.52, 34.69),
    "Djakarta":              (106.85, -6.21),
    "Djokdjakarta":          (110.37, -7.80),
    "Kairo":                 (8.68,  50.11),   # hard negative — Frankfurt area
    "Frankfurt am Main":     (8.68,  50.11),
}

RIGHT_COORDS = {
    "Cologne":               (6.96,  50.94),
    "Munich":                (11.58, 48.14),
    "Nuremberg":             (11.08, 49.45),
    "Dusseldorf":            (6.79,  51.22),
    "Frankfurt":             (8.68,  50.11),
    "New York City":         (-74.0, 40.71),
    "Los Angeles":           (-118.24, 34.05),
    "San Francisco":         (-122.42, 37.77),
    "Philadelphia":          (-75.16, 39.95),
    "Saint-Denis":           (2.36,  48.93),
    "Lyon":                  (4.83,  45.75),
    "Marseille":             (5.37,  43.30),
    "Strasbourg":            (7.75,  48.58),
    "Cairo":                 (31.25, 30.06),
    "Alexandria":            (29.92, 31.20),
    "Isfahan":               (51.68, 32.66),
    "Tehran":                (51.42, 35.69),
    "Moscow":                (37.62, 55.75),
    "Saint Petersburg":      (30.32, 59.94),
    "Beijing":               (116.39, 39.93),
    "Guangzhou":             (113.27, 23.13),
    "Mumbai":                (72.88, 19.08),
    "Kolkata":               (88.37, 22.57),
    "Chennai":               (80.28, 13.09),
    "Bengaluru":             (77.59, 12.97),
    "Rio de Janeiro":        (-43.18, -22.91),
    "Sao Paulo":             (-46.63, -23.55),
    "Bogota":                (-74.08, 4.71),
    "Buenos Aires":          (-58.37, -34.61),
    "Mexico City":           (-99.13, 19.43),
    "Guadalajara":           (-103.35, 20.67),
    "Istanbul":              (28.97, 41.01),
    "Izmir":                 (27.14, 38.41),
    "Seoul":                 (126.98, 37.57),
    "Busan":                 (129.04, 35.10),
    "Tokyo":                 (139.69, 35.69),
    "Osaka":                 (135.52, 34.69),
    "Jakarta":               (106.85, -6.21),
    "Yogyakarta":            (110.37, -7.80),
}


# ---------------------------------------------------------------------------
# Run matching
# ---------------------------------------------------------------------------
def run_matching():
    from examples.osm_geonames_place_matching import (
        LEFT_PLACES, RIGHT_PLACES, GROUND_TRUTH,
    )
    left = pd.DataFrame(LEFT_PLACES, columns=["id", "name", "country_code"])
    right = pd.DataFrame(RIGHT_PLACES, columns=["id", "name", "country_code"])
    true_matches = pd.DataFrame(GROUND_TRUTH, columns=["left_id", "right_id"])

    result = match_tables(
        left, right,
        left_on="name", right_on="name",
        left_id="id", right_id="id",
        top_k=5, block_on="country_code", use_llm=True,
    )
    # Add coordinates
    left_coord = left.set_index("id")[["name"]].copy()
    right_coord = right.set_index("id")[["name"]].copy()
    result = result.merge(left_coord.rename(columns={"name": "_ln"}), left_on="left_id", right_index=True)
    result = result.merge(right_coord.rename(columns={"name": "_rn"}), left_on="right_id", right_index=True)
    result["left_lon"]  = result["_ln"].map(lambda n: LEFT_COORDS.get(n, (None, None))[0])
    result["left_lat"]  = result["_ln"].map(lambda n: LEFT_COORDS.get(n, (None, None))[1])
    result["right_lon"] = result["_rn"].map(lambda n: RIGHT_COORDS.get(n, (None, None))[0])
    result["right_lat"] = result["_rn"].map(lambda n: RIGHT_COORDS.get(n, (None, None))[1])
    return result, true_matches


# ---------------------------------------------------------------------------
# Draw curved arc between two points
# ---------------------------------------------------------------------------
def _arc(ax, x0, y0, x1, y1, color, lw, alpha, n=60):
    """Draw a smooth quadratic bezier arc between two lon/lat points."""
    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2 + abs(x1 - x0) * 0.18
    t = np.linspace(0, 1, n)
    xs = (1 - t)**2 * x0 + 2 * (1 - t) * t * mid_x + t**2 * x1
    ys = (1 - t)**2 * y0 + 2 * (1 - t) * t * mid_y + t**2 * y1
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=3, solid_capstyle="round")


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------
def make_map():
    out_dir = Path("notebooks/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    result, true_matches = run_matching()

    import geodatasets
    world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

    fig, ax = plt.subplots(figsize=(20, 11), facecolor="#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    world.plot(
        ax=ax,
        color="#1c3557",
        edgecolor="#2e5f8a",
        linewidth=0.4,
        zorder=1,
    )

    # color scheme
    COLORS = {
        "high":           ("#00e5ff", 2.2, 0.90),   # cyan
        "medium_review":  ("#ffb300", 1.6, 0.75),   # amber
        "low":            ("#ff5252", 1.0, 0.45),   # red
        "reject":         ("#888888", 0.7, 0.25),   # grey
    }
    true_set = set(zip(true_matches["left_id"], true_matches["right_id"]))

    plotted = {"high": 0, "medium_review": 0, "low": 0, "reject": 0}

    for _, row in result.iterrows():
        lx, ly = row.get("left_lon"), row.get("left_lat")
        rx, ry = row.get("right_lon"), row.get("right_lat")
        if lx is None or rx is None:
            continue

        label = row["reliability_label"] if row["reliability_label"] in COLORS else "reject"
        col, lw, alpha = COLORS[label]

        # Thicker arc for correct final decisions
        is_correct = (row["left_id"], row["right_id"]) in true_set
        lw_use = lw * (1.8 if is_correct and row["final_decision"] else 1.0)
        alpha_use = min(1.0, alpha * (1.4 if is_correct else 1.0))

        _arc(ax, lx, ly, rx, ry, col, lw_use, alpha_use)

        # Dirty name dot (left)
        ax.scatter(lx, ly, s=22, color=col, zorder=5, alpha=0.9, edgecolors="white", linewidths=0.3)
        # Canonical dot (right) — slightly larger
        ax.scatter(rx, ry, s=38, color=col, marker="*", zorder=6, alpha=0.95)
        plotted[label] += 1

    # Legend
    legend_items = [
        mpatches.Patch(color="#00e5ff", label=f"High confidence  (n={plotted['high']})"),
        mpatches.Patch(color="#ffb300", label=f"Medium / LLM reviewed (n={plotted['medium_review']})"),
        mpatches.Patch(color="#ff5252", label=f"Low confidence  (n={plotted['low']})"),
        mpatches.Patch(color="#888888", label=f"Rejected  (n={plotted['reject']})"),
        mpatches.Patch(color="white",   label="● dirty name    ★ canonical name"),
    ]
    leg = ax.legend(
        handles=legend_items,
        loc="lower left",
        fontsize=10,
        framealpha=0.25,
        facecolor="#0d1b2a",
        edgecolor="#2e5f8a",
        labelcolor="white",
        title="Match reliability",
        title_fontsize=11,
    )
    leg.get_title().set_color("white")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.axis("off")

    ax.set_title(
        "Fuzzy Entity Matching on Real-World Place Names\n"
        "OSM / GeoNames · 40 hard city pairs · fuzzy_llm_matcher",
        color="white", fontsize=16, fontweight="bold", pad=18,
    )

    fig.text(
        0.5, 0.02,
        "Arcs connect noisy spellings (●) to canonical names (★)  |  "
        "github.com/mohseniaref/fuzzy_llm_matcher",
        ha="center", fontsize=9, color="#a0bdd8",
    )

    out = out_dir / "geo_matching_world_map.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    make_map()
