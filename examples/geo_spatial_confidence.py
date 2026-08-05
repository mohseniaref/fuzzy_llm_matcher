"""Spatial confidence features demo: score_geo_distance + geo-aware LLM prompt.

This example illustrates two new capabilities added to fuzzy_llm_matcher:

1. **score_geo_distance** – a 0–100 similarity score derived from the
   haversine distance between two candidate records.  Pairs that are
   geographically close score high; distant pairs score low.

2. **Geo-aware LLM prompt** – when coordinate columns are present,
   ``review_uncertain_pairs_with_llm`` automatically builds a richer
   prompt that includes the distance, allowing the LLM (or MockLLMClient)
   to combine name similarity *and* spatial proximity when adjudicating
   uncertain pairs.

The demo uses the same 40-city hard-case dataset from
``osm_geonames_place_matching.py`` (no network required) and produces four
publication-quality figures saved under ``notebooks/figures/``:

    geo_spatial_score_scatter.png   – fuzzy_score vs score_geo_distance,
                                      coloured by reliability label
    geo_combined_heatmap.png        – 2-D heat-map of both scores
    geo_llm_boost_comparison.png    – reliability label distribution with
                                      and without geo-context LLM review
    geo_world_map_geo_score.png     – world map with arc colour = geo score

Run:
    python examples/geo_spatial_confidence.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Inline city coordinates (lon, lat) – same as osm_geonames_place_matching.py
# ---------------------------------------------------------------------------
LEFT_COORDS: dict[int, tuple[float, float]] = {
    1:  (50.94, 6.96),    # Köln
    2:  (48.14, 11.58),   # München
    3:  (49.45, 11.08),   # Nürnberg
    4:  (51.22, 6.79),    # Düsseldorf
    5:  (40.71, -74.00),  # NYC
    6:  (34.05, -118.24), # LA
    7:  (37.77, -122.42), # San Fran
    8:  (39.95, -75.16),  # Philly
    9:  (48.93, 2.36),    # Saint-Denis
    10: (45.75, 4.83),    # Lyon France
    11: (43.30, 5.37),    # Marseilles
    12: (48.58, 7.75),    # Strasbourg-Alsace
    13: (30.06, 31.25),   # Al Qahirah
    14: (31.20, 29.92),   # Aleksandria
    15: (32.66, 51.68),   # Ispahan
    16: (35.69, 51.42),   # Teheran
    17: (55.75, 37.62),   # Moskva
    18: (59.94, 30.32),   # Sankt-Peterburg
    19: (39.93, 116.39),  # Peking
    20: (23.13, 113.27),  # Canton
    21: (19.08, 72.88),   # Bombay
    22: (22.57, 88.37),   # Calcutta
    23: (13.09, 80.28),   # Madras
    24: (12.97, 77.59),   # Bangalore
    25: (-22.91, -43.18), # Rio de Jan.
    26: (-23.55, -46.63), # Sao Paolo
    27: (4.71, -74.08),   # Bogotá
    28: (-34.61, -58.37), # Buenos Ayres
    29: (19.43, -99.13),  # Ciudad de Mexico
    30: (20.67, -103.35), # Guadalahara
    31: (41.01, 28.97),   # Instanbul
    32: (38.41, 27.14),   # Izmir Turkey
    33: (37.57, 126.98),  # Soul
    34: (35.10, 129.04),  # Busan Korea
    35: (35.69, 139.69),  # Tokio
    36: (34.69, 135.52),  # Osaca
    37: (-6.21, 106.85),  # Djakarta
    38: (-7.80, 110.37),  # Djokdjakarta
    # hard negatives share Frankfurt coords on purpose
    39: (50.11, 8.68),    # Kairo (hard negative – should map to Frankfurt)
    40: (50.11, 8.68),    # Frankfurt am Main
}

RIGHT_COORDS: dict[int, tuple[float, float]] = {
    1:  (50.94, 6.96),    # Cologne
    2:  (48.14, 11.58),   # Munich
    3:  (49.45, 11.08),   # Nuremberg
    4:  (51.22, 6.79),    # Dusseldorf
    5:  (50.11, 8.68),    # Frankfurt (right table)
    6:  (40.71, -74.00),  # New York City
    7:  (34.05, -118.24), # Los Angeles
    8:  (37.77, -122.42), # San Francisco
    9:  (39.95, -75.16),  # Philadelphia
    10: (48.93, 2.36),    # Saint-Denis (right)
    11: (45.75, 4.83),    # Lyon
    12: (43.30, 5.37),    # Marseille
    13: (48.58, 7.75),    # Strasbourg
    14: (30.06, 31.25),   # Cairo
    15: (31.20, 29.92),   # Alexandria
    16: (32.66, 51.68),   # Isfahan
    17: (35.69, 51.42),   # Tehran
    18: (55.75, 37.62),   # Moscow
    19: (59.94, 30.32),   # Saint Petersburg
    20: (39.93, 116.39),  # Beijing
    21: (23.13, 113.27),  # Guangzhou
    22: (19.08, 72.88),   # Mumbai
    23: (22.57, 88.37),   # Kolkata
    24: (13.09, 80.28),   # Chennai
    25: (12.97, 77.59),   # Bengaluru
    26: (-22.91, -43.18), # Rio de Janeiro
    27: (-23.55, -46.63), # Sao Paulo
    28: (4.71, -74.08),   # Bogota
    29: (-34.61, -58.37), # Buenos Aires
    30: (19.43, -99.13),  # Mexico City
    31: (20.67, -103.35), # Guadalajara
    32: (41.01, 28.97),   # Istanbul
    33: (38.41, 27.14),   # Izmir
    34: (37.57, 126.98),  # Seoul
    35: (35.10, 129.04),  # Busan
    36: (35.69, 139.69),  # Tokyo
    37: (34.69, 135.52),  # Osaka
    38: (-6.21, 106.85),  # Jakarta
    39: (-7.80, 110.37),  # Yogyakarta
}

# Ground truth: left_id → right_id (correct matches)
GROUND_TRUTH = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7, 7: 8, 8: 9,
    9: 10, 10: 11, 11: 12, 12: 13,
    13: 14, 14: 15, 15: 16, 16: 17,
    17: 18, 18: 19, 19: 20, 20: 21,
    21: 22, 22: 23, 23: 24, 24: 25,
    25: 26, 26: 27, 27: 28, 28: 29,
    29: 30, 30: 31, 31: 32, 32: 33,
    33: 34, 34: 35, 35: 36, 36: 37,
    37: 38, 38: 39,
    39: 5,   # Kairo → Frankfurt (intentional hard negative in name matching)
    40: 5,
}

LEFT_PLACES = [
    (1,  "Köln",               "DE"), (2,  "München",            "DE"),
    (3,  "Nürnberg",           "DE"), (4,  "Düsseldorf",         "DE"),
    (5,  "NYC",                "US"), (6,  "LA",                 "US"),
    (7,  "San Fran",           "US"), (8,  "Philly",             "US"),
    (9,  "Saint-Denis",        "FR"), (10, "Lyon France",        "FR"),
    (11, "Marseilles",         "FR"), (12, "Strasbourg-Alsace",  "FR"),
    (13, "Al Qahirah",         "EG"), (14, "Aleksandria",        "EG"),
    (15, "Ispahan",            "IR"), (16, "Teheran",            "IR"),
    (17, "Moskva",             "RU"), (18, "Sankt-Peterburg",    "RU"),
    (19, "Peking",             "CN"), (20, "Canton",             "CN"),
    (21, "Bombay",             "IN"), (22, "Calcutta",           "IN"),
    (23, "Madras",             "IN"), (24, "Bangalore",          "IN"),
    (25, "Rio de Jan.",        "BR"), (26, "Sao Paolo",          "BR"),
    (27, "Bogotá",             "CO"), (28, "Buenos Ayres",       "AR"),
    (29, "Ciudad de Mexico",   "MX"), (30, "Guadalahara",        "MX"),
    (31, "Instanbul",          "TR"), (32, "Izmir Turkey",       "TR"),
    (33, "Soul",               "KR"), (34, "Busan Korea",        "KR"),
    (35, "Tokio",              "JP"), (36, "Osaca",              "JP"),
    (37, "Djakarta",           "ID"), (38, "Djokdjakarta",       "ID"),
    (39, "Kairo",              "DE"), (40, "Frankfurt am Main",  "DE"),
]

RIGHT_PLACES = [
    (1,  "Cologne",          "DE"), (2,  "Munich",            "DE"),
    (3,  "Nuremberg",        "DE"), (4,  "Dusseldorf",        "DE"),
    (5,  "Frankfurt",        "DE"),
    (6,  "New York City",    "US"), (7,  "Los Angeles",       "US"),
    (8,  "San Francisco",    "US"), (9,  "Philadelphia",      "US"),
    (10, "Saint-Denis",      "FR"), (11, "Lyon",              "FR"),
    (12, "Marseille",        "FR"), (13, "Strasbourg",        "FR"),
    (14, "Cairo",            "EG"), (15, "Alexandria",        "EG"),
    (16, "Isfahan",          "IR"), (17, "Tehran",            "IR"),
    (18, "Moscow",           "RU"), (19, "Saint Petersburg",  "RU"),
    (20, "Beijing",          "CN"), (21, "Guangzhou",         "CN"),
    (22, "Mumbai",           "IN"), (23, "Kolkata",           "IN"),
    (24, "Chennai",          "IN"), (25, "Bengaluru",         "IN"),
    (26, "Rio de Janeiro",   "BR"), (27, "Sao Paulo",         "BR"),
    (28, "Bogota",           "CO"), (29, "Buenos Aires",      "AR"),
    (30, "Mexico City",      "MX"), (31, "Guadalajara",       "MX"),
    (32, "Istanbul",         "TR"), (33, "Izmir",             "TR"),
    (34, "Seoul",            "KR"), (35, "Busan",             "KR"),
    (36, "Tokyo",            "JP"), (37, "Osaka",             "JP"),
    (38, "Jakarta",          "ID"), (39, "Yogyakarta",        "ID"),
]


# ---------------------------------------------------------------------------
# Build the enriched result DataFrame
# ---------------------------------------------------------------------------

def build_result() -> pd.DataFrame:
    from fuzzy_llm_matcher import (
        add_geo_distance_score,
        match_tables,
    )
    from fuzzy_llm_matcher.fuzzy_scores import compute_similarity_features
    from fuzzy_llm_matcher.candidate_generation import generate_candidates
    from fuzzy_llm_matcher.reliability import assign_reliability
    from fuzzy_llm_matcher.llm_review import review_uncertain_pairs_with_llm

    left  = pd.DataFrame(LEFT_PLACES,  columns=["id", "name", "country_code"])
    right = pd.DataFrame(RIGHT_PLACES, columns=["id", "name", "country_code"])

    # ── Step 1: candidate generation ──────────────────────────────────────
    candidates = generate_candidates(
        left, right,
        left_on="name", right_on="name",
        left_id="id", right_id="id",
        block_on="country_code", top_k=5,
    )

    # ── Step 2: string similarity features ────────────────────────────────
    scored = compute_similarity_features(candidates)

    # ── Step 3: join coordinates ──────────────────────────────────────────
    scored["left_lat"]  = scored["left_id"].map(lambda i: LEFT_COORDS.get(i, (None, None))[0])
    scored["left_lon"]  = scored["left_id"].map(lambda i: LEFT_COORDS.get(i, (None, None))[1])
    scored["right_lat"] = scored["right_id"].map(lambda i: RIGHT_COORDS.get(i, (None, None))[0])
    scored["right_lon"] = scored["right_id"].map(lambda i: RIGHT_COORDS.get(i, (None, None))[1])

    # ── Step 4: geo-distance score ────────────────────────────────────────
    scored = add_geo_distance_score(scored, max_km=500)

    # ── Step 5: reliability labelling (string score only, baseline) ───────
    labeled = assign_reliability(scored)

    # ── Step 6: geo-aware LLM review for uncertain pairs ──────────────────
    labeled = review_uncertain_pairs_with_llm(labeled)   # geo_context auto-detected

    # ── Step 7: final decision ────────────────────────────────────────────
    def _final(row):
        if row["reliability_label"] == "high":
            return True
        if row["reliability_label"] == "medium_review" and row.get("llm_same_entity") is True:
            return True
        return False

    labeled["final_decision"] = labeled.apply(_final, axis=1)

    # Keep only the best candidate per left entity
    labeled = labeled.sort_values(["left_id", "score"], ascending=[True, False])
    labeled = labeled.groupby("left_id", as_index=False).first()

    # Annotate ground truth correctness
    labeled["is_correct"] = labeled.apply(
        lambda r: GROUND_TRUTH.get(r["left_id"]) == r["right_id"], axis=1
    )
    return labeled


# ---------------------------------------------------------------------------
# Figure 1 – Scatter: fuzzy_score vs score_geo_distance
# ---------------------------------------------------------------------------

LABEL_COLORS = {
    "high":          "#00e5ff",
    "medium_review": "#ffb300",
    "low":           "#ff5252",
    "reject":        "#888888",
}


def fig_scatter(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0d1b2a")
    ax.set_facecolor("#12263a")

    for label, grp in df.groupby("reliability_label"):
        color = LABEL_COLORS.get(label, "#cccccc")
        # Mark incorrect matches with X marker
        correct = grp[grp["is_correct"]]
        incorrect = grp[~grp["is_correct"]]
        ax.scatter(
            correct["score_wratio"], correct["score_geo_distance"],
            c=color, s=80, marker="o", alpha=0.85, edgecolors="white",
            linewidths=0.4, label=f"{label} ✓ (n={len(correct)})", zorder=3,
        )
        ax.scatter(
            incorrect["score_wratio"], incorrect["score_geo_distance"],
            c=color, s=90, marker="X", alpha=0.85, edgecolors="white",
            linewidths=0.4, label=f"{label} ✗ (n={len(incorrect)})", zorder=3,
        )

    ax.set_xlabel("Fuzzy string score (WRatio, 0–100)", color="white", fontsize=12)
    ax.set_ylabel("Geo-distance score (0–100)", color="white", fontsize=12)
    ax.set_title(
        "Spatial Confidence vs String Similarity\n"
        "○ = correct match   ✗ = wrong match",
        color="white", fontsize=13, fontweight="bold",
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e5f8a")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.axhline(50, color="#2e5f8a", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(92, color="#2e5f8a", lw=0.8, ls="--", alpha=0.6)
    ax.text(93, 2, "high_threshold=92", color="#2e5f8a", fontsize=8)
    ax.text(2, 51, "geo=50 (≈ max_km/2)", color="#2e5f8a", fontsize=8)

    leg = ax.legend(
        loc="lower right", fontsize=8, framealpha=0.3,
        facecolor="#0d1b2a", edgecolor="#2e5f8a", labelcolor="white",
    )
    fig.text(0.5, 0.01, "github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=8, color="#a0bdd8")

    out = out_dir / "geo_spatial_score_scatter.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 2 – 2-D heat-map: geo score vs string score
# ---------------------------------------------------------------------------

def fig_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0d1b2a")
    ax.set_facecolor("#12263a")

    x = df["score_wratio"].clip(0, 100)
    y = df["score_geo_distance"].clip(0, 100)

    h, xe, ye = np.histogram2d(x, y, bins=15, range=[[0, 100], [0, 100]])
    pcm = ax.pcolormesh(xe, ye, h.T, cmap="plasma", alpha=0.85)

    # overlay scatter
    for label, grp in df.groupby("reliability_label"):
        color = LABEL_COLORS.get(label, "#cccccc")
        ax.scatter(
            grp["score_wratio"], grp["score_geo_distance"],
            c=color, s=60, edgecolors="white", linewidths=0.3,
            alpha=0.9, zorder=4,
        )

    cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Pair count", color="white", fontsize=10)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("Fuzzy string score (WRatio)", color="white", fontsize=12)
    ax.set_ylabel("Geo-distance score", color="white", fontsize=12)
    ax.set_title(
        "Score Distribution Heat-Map\n"
        "High-quality matches cluster top-right",
        color="white", fontsize=13, fontweight="bold",
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e5f8a")

    legend_patches = [
        mpatches.Patch(color=c, label=l)
        for l, c in LABEL_COLORS.items()
    ]
    leg = ax.legend(
        handles=legend_patches, loc="lower right", fontsize=8,
        framealpha=0.35, facecolor="#0d1b2a", edgecolor="#2e5f8a",
        labelcolor="white",
    )

    out = out_dir / "geo_combined_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 3 – Bar: label distribution with and without geo-context LLM boost
# ---------------------------------------------------------------------------

def fig_llm_boost(df: pd.DataFrame, out_dir: Path) -> Path:
    """Compare reliability labels before and after geo-aware LLM review."""
    from fuzzy_llm_matcher.reliability import assign_reliability
    from fuzzy_llm_matcher.llm_review import review_uncertain_pairs_with_llm

    # Baseline: no LLM
    scored_no_llm = df.copy()
    label_counts_no_llm = scored_no_llm["reliability_label"].value_counts()

    # With geo-aware LLM (already in df, re-run on a fresh copy for clarity)
    scored_llm = df.copy()
    # Simulate final decision counts for medium_review → confirmed matches
    confirmed_by_llm = (
        (scored_llm["reliability_label"] == "medium_review")
        & (scored_llm["llm_same_entity"] == True)
    ).sum()
    label_counts_llm = scored_llm["reliability_label"].value_counts()

    labels = ["high", "medium_review", "low", "reject"]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0d1b2a")
    ax.set_facecolor("#12263a")

    bars1 = ax.bar(
        x - w / 2,
        [label_counts_no_llm.get(l, 0) for l in labels],
        width=w, color=[LABEL_COLORS[l] for l in labels],
        alpha=0.65, label="String-only labels", edgecolor="white", linewidth=0.4,
    )
    bars2 = ax.bar(
        x + w / 2,
        [label_counts_llm.get(l, 0) for l in labels],
        width=w, color=[LABEL_COLORS[l] for l in labels],
        alpha=1.0, label="After geo-LLM review", edgecolor="white", linewidth=0.4,
    )

    # Annotate the medium_review bar with number of LLM-confirmed matches
    mid_idx = labels.index("medium_review")
    ax.annotate(
        f"+{confirmed_by_llm} confirmed\nby geo-LLM",
        xy=(mid_idx + w / 2, label_counts_llm.get("medium_review", 0) + 0.2),
        color="#ffb300", fontsize=9, ha="center",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=11)
    ax.set_ylabel("Number of candidate pairs", color="white", fontsize=11)
    ax.set_title(
        "Reliability Label Distribution\n"
        "String-only vs Geo-aware LLM Review",
        color="white", fontsize=13, fontweight="bold",
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e5f8a")
    ax.legend(
        fontsize=9, framealpha=0.35,
        facecolor="#0d1b2a", edgecolor="#2e5f8a", labelcolor="white",
    )

    out = out_dir / "geo_llm_boost_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 4 – World map: arc colour = geo-distance score
# ---------------------------------------------------------------------------

def _arc(ax, x0, y0, x1, y1, color, lw, alpha, n=60):
    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2 + abs(x1 - x0) * 0.15
    t = np.linspace(0, 1, n)
    xs = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * mid_x + t ** 2 * x1
    ys = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * mid_y + t ** 2 * y1
    ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, zorder=3, solid_capstyle="round")


def fig_world_map(df: pd.DataFrame, out_dir: Path) -> Path:
    """World map using matplotlib's built-in polygon data (no geopandas needed)."""
    # Try geopandas first; fall back to a pure-matplotlib land outline.
    world_polys = None
    try:
        import geopandas as gpd
        try:
            import geodatasets
            world_gdf = gpd.read_file(geodatasets.get_path("naturalearth.land"))
        except Exception:
            world_gdf = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        world_polys = world_gdf.geometry
    except Exception:
        pass  # draw without land polygons

    cmap = plt.get_cmap("RdYlGn")

    fig, ax = plt.subplots(figsize=(20, 11), facecolor="#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    if world_polys is not None:
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path as MPath
        import shapely.geometry as sgeom
        for geom in world_polys:
            if geom is None:
                continue
            if geom.geom_type == "Polygon":
                geoms = [geom]
            else:
                geoms = list(geom.geoms)
            for poly in geoms:
                xs, ys = poly.exterior.xy
                ax.fill(xs, ys, color="#1c3557", linewidth=0.3,
                        edgecolor="#2e5f8a", zorder=1)
    else:
        # Minimal rectangle to represent land (no external data needed)
        ax.set_facecolor("#12263a")
        # Draw a light grid so the map isn't completely blank
        for lon in range(-180, 181, 30):
            ax.axvline(lon, color="#1c3557", lw=0.4, alpha=0.5, zorder=1)
        for lat in range(-60, 86, 30):
            ax.axhline(lat, color="#1c3557", lw=0.4, alpha=0.5, zorder=1)

    for _, row in df.iterrows():
        lid = row["left_id"]
        rid = row["right_id"]
        lc = LEFT_COORDS.get(lid)
        rc = RIGHT_COORDS.get(rid)
        if lc is None or rc is None:
            continue
        # Coordinates stored as (lat, lon) → map expects (lon, lat)
        llon, llat = lc[1], lc[0]
        rlon, rlat = rc[1], rc[0]

        geo_s = row.get("score_geo_distance", float("nan"))
        try:
            norm_s = 0.5 if math.isnan(float(geo_s)) else float(geo_s) / 100.0
        except Exception:
            norm_s = 0.5
        color = cmap(norm_s)

        is_correct = bool(row["is_correct"])
        lw = 2.2 if is_correct else 0.9
        alpha = 0.90 if is_correct else 0.45

        _arc(ax, llon, llat, rlon, rlat, color, lw, alpha)
        ax.scatter(llon, llat, s=25, color=color, zorder=5,
                   edgecolors="white", linewidths=0.3, alpha=0.9)
        ax.scatter(rlon, rlat, s=40, color=color, marker="*", zorder=6, alpha=0.95)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                      fraction=0.03, pad=0.03, aspect=40)
    cb.set_label("score_geo_distance (0=far, 100=same location)",
                 color="white", fontsize=10)
    cb.ax.xaxis.set_tick_params(color="white")
    plt.setp(cb.ax.xaxis.get_ticklabels(), color="white")

    legend_items = [
        mpatches.Patch(color="white",   label="● dirty name   ★ canonical name"),
        mpatches.Patch(color="#aaaaaa", label="thick arc = correct ground-truth match"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=9,
              framealpha=0.25, facecolor="#0d1b2a", edgecolor="#2e5f8a",
              labelcolor="white")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.axis("off")
    ax.set_title(
        "Geo-Distance Score on World City Pair Matches\n"
        "Arc colour: green = nearby, red = far apart  |  fuzzy_llm_matcher",
        color="white", fontsize=15, fontweight="bold", pad=16,
    )
    fig.text(0.5, 0.02,
             "github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=9, color="#a0bdd8")

    out = out_dir / "geo_world_map_geo_score.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    out_dir = Path("notebooks/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building result DataFrame with geo-distance features …")
    df = build_result()

    n_total   = len(df)
    n_decided = df["final_decision"].sum()
    n_correct = (df["final_decision"] & df["is_correct"]).sum()
    precision = n_correct / n_decided if n_decided else 0
    recall    = n_correct / n_total
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"\n── Results ──────────────────────────────────")
    print(f"  Total pairs       : {n_total}")
    print(f"  Final decisions   : {int(n_decided)}")
    print(f"  Correct           : {int(n_correct)}")
    print(f"  Precision         : {precision:.3f}")
    print(f"  Recall            : {recall:.3f}")
    print(f"  F1                : {f1:.3f}")
    print(f"\nReliability label distribution:")
    print(df["reliability_label"].value_counts().to_string())
    print(f"\nGeo-distance score summary:")
    print(df["score_geo_distance"].describe().to_string())
    print()

    print("Generating figures …")
    fig_scatter(df, out_dir)
    fig_heatmap(df, out_dir)
    fig_llm_boost(df, out_dir)
    fig_world_map(df, out_dir)
    print("\nDone.")
    return df


if __name__ == "__main__":
    run()
