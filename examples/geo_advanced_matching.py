"""Advanced geo matching: transliteration, hierarchical blocking,
coordinate uncertainty, and geometry similarity scores.

This example demonstrates all four new capabilities in one script:

1. **Transliteration + Phonetic scoring** — match place names across scripts
   ("Москва"/"Moscow", "Köln"/"Cologne", "Isfahan"/"Esfahan") using
   unidecode + jellyfish metaphone/soundex/NYSIIS.

2. **Hierarchical admin blocking** — match at district level first, fall back
   to province, then country. No boundary artefacts.

3. **Coordinate-uncertainty-aware geo score** — probabilistic spatial overlap
   (P(same location)) using the Gaussian uncertainty model from seismology.

4. **Geometry similarity scores** — compare polygon/line shapes using
   Hausdorff distance, Fréchet distance, and Intersection-over-Union.

Figures produced → notebooks/figures/
  geo_adv_phonetic_comparison.png   – WRatio vs transliterated_WRatio vs metaphone
  geo_adv_hierarchical_blocking.png – match rates by blocking level
  geo_adv_uncertainty_score.png     – geo uncertainty score vs haversine distance
  geo_adv_geometry_similarity.png   – Hausdorff / Fréchet / IoU comparison

Run:
    python examples/geo_advanced_matching.py
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

BG   = "#0d1b2a"
MID  = "#12263a"
GRID = "#2e5f8a"

def _ax(ax):
    ax.set_facecolor(MID)
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_edgecolor(GRID)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 1 — Transliteration + Phonetic Scoring
# ══════════════════════════════════════════════════════════════════════════════

# Hard transliteration / cross-language place-name pairs
TRANSLIT_PAIRS = [
    # (left_name,             right_name,          ground_truth)
    ("Köln",                 "Cologne",            True),
    ("München",              "Munich",             True),
    ("Isfahan",              "Esfahan",            True),
    ("Moskva",               "Moscow",             True),
    ("Al Qahirah",           "Cairo",              True),
    ("Peking",               "Beijing",            True),
    ("Bombay",               "Mumbai",             True),
    ("Instanbul",            "Istanbul",           True),
    ("Hannover",             "Hanover",            True),
    ("Braunschweig",         "Brunswick",          True),
    ("Nürnberg",             "Nuremberg",          True),
    # Hard negatives — similar names, different cities
    ("Frankfurt am Main",    "Frankfurt an der Oder", False),
    ("Cairo (Illinois)",     "Cairo",              False),
    ("New York",             "York",               False),
]


def run_phonetic_demo():
    from fuzzy_llm_matcher import (
        transliterate_text, phonetic_code, phonetic_similarity_score,
        HAVE_UNIDECODE, HAVE_JELLYFISH,
    )
    from fuzzy_llm_matcher.utils import fuzz, normalize_text

    print(f"  unidecode available: {HAVE_UNIDECODE}")
    print(f"  jellyfish available: {HAVE_JELLYFISH}")

    rows = []
    for left, right, gt in TRANSLIT_PAIRS:
        # Standard WRatio (no transliteration)
        wrat = fuzz.WRatio(normalize_text(left), normalize_text(right))

        # WRatio after transliteration
        tl = transliterate_text(left).lower()
        tr = transliterate_text(right).lower()
        twrat = fuzz.WRatio(tl, tr)

        # Metaphone phonetic score (after transliteration)
        mph = phonetic_similarity_score(tl, tr, "metaphone")

        # Soundex phonetic score
        sx  = phonetic_similarity_score(tl, tr, "soundex")

        rows.append({
            "left": left, "right": right, "ground_truth": gt,
            "WRatio": wrat, "transliterated_WRatio": twrat,
            "metaphone": mph, "soundex": sx,
        })

    df = pd.DataFrame(rows)
    print("\n  Scorer comparison on transliteration pairs:")
    print(df[["left","right","WRatio","transliterated_WRatio","metaphone","soundex"]].to_string(index=False))
    return df


def fig_phonetic_comparison(df: pd.DataFrame, out_dir: Path):
    true_pairs  = df[df["ground_truth"] == True]
    false_pairs = df[df["ground_truth"] == False]

    scorers = ["WRatio", "transliterated_WRatio", "metaphone", "soundex"]
    colors  = ["#888888", "#00e5ff", "#ffb300", "#ff5252"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)

    for ax, data, title in [
        (axes[0], true_pairs,  "True pairs (same city)"),
        (axes[1], false_pairs, "Hard negatives (different cities)"),
    ]:
        _ax(ax)
        x = np.arange(len(data))
        w = 0.2
        for i, (scorer, color) in enumerate(zip(scorers, colors)):
            offset = (i - 1.5) * w
            ax.bar(x + offset, data[scorer], width=w, color=color,
                   alpha=0.85, edgecolor="white", linewidth=0.3,
                   label=scorer)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{r.left[:8]}\n→{r.right[:8]}" for r in data.itertuples()],
            color="white", fontsize=8, rotation=35, ha="right",
        )
        ax.set_ylim(0, 115)
        ax.set_ylabel("Score (0–100)", color="white")
        ax.set_title(title, color="white", fontweight="bold")
        ax.axhline(80, color="#2e5f8a", lw=1, ls="--", alpha=0.6)

    axes[0].legend(fontsize=8, framealpha=0.35, facecolor=BG,
                   edgecolor=GRID, labelcolor="white", loc="lower left")

    fig.suptitle(
        "Transliteration + Phonetic Scoring\n"
        "WRatio / transliterated_WRatio / metaphone / soundex",
        color="white", fontsize=12, fontweight="bold",
    )
    fig.text(0.5, 0.01, "github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=8, color="#a0bdd8")

    out = out_dir / "geo_adv_phonetic_comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 2 — Hierarchical Admin Blocking
# ══════════════════════════════════════════════════════════════════════════════

# Sri Lanka GN Division synthetic dataset — simulates the boundary problem
HIER_LEFT = [
    # (id, gn_name,             district,    province)
    (1, "Sammanthranapura",  "Colombo",   "Western"),
    (2, "Mattakkuliya",      "Colombo",   "Western"),
    (3, "Peliyagoda",        "Colombo",   "Western"),
    (4, "Kaduwela",          "Colombo",   "Western"),
    (5, "Biyagama",          "Gampaha",   "Western"),
    (6, "Kelaniya",          "Gampaha",   "Western"),
    # Feature near district boundary — wrong district in one source
    (7, "Bordering Village A", "Colombo", "Western"),  # correct district
    (8, "Bordering Village B", "Gampaha", "Western"),  # correct district
]

HIER_RIGHT = [
    # Census CSV — slightly different names, CORRECT districts
    (1, "Sammanthranpura",   "Colombo",   "Western"),   # spelling variant
    (2, "Mattakkulia",       "Colombo",   "Western"),   # variant
    (3, "Peliyagoda",        "Colombo",   "Western"),
    (4, "Kaduwella",         "Colombo",   "Western"),   # variant
    (5, "Biyagama",          "Gampaha",   "Western"),
    (6, "Kelaniya",          "Gampaha",   "Western"),
    # The boundary feature is mis-assigned in the shapefile (Gampaha not Colombo)
    (7, "Bordering Village A", "Gampaha", "Western"),   # WRONG district in CSV
    (8, "Bordering Village B", "Colombo", "Western"),   # WRONG district in CSV
]


def run_hierarchical_demo():
    from fuzzy_llm_matcher import hierarchical_block_match

    left  = pd.DataFrame(HIER_LEFT,  columns=["id", "gn_name", "district", "province"])
    right = pd.DataFrame(HIER_RIGHT, columns=["id", "gn_name", "district", "province"])

    # Flat blocking — will miss boundary features 7 and 8
    from fuzzy_llm_matcher import match_tables
    flat_result = match_tables(
        left, right, left_on="gn_name", right_on="gn_name",
        left_id="id", right_id="id",
        block_on="district", high_threshold=70, use_llm=True,
    )

    # Hierarchical blocking — district first, province fallback
    hier_result = hierarchical_block_match(
        left, right, left_on="gn_name", right_on="gn_name",
        block_levels=["district", "province"],
        left_id="id", right_id="id",
        high_threshold=70, use_llm=True,
    )

    print(f"\n  Flat blocking (district only):")
    print(f"    Matched: {flat_result['final_decision'].sum()} / {len(flat_result)}")
    print(f"\n  Hierarchical blocking (district → province):")
    print(f"    Matched: {hier_result['final_decision'].sum()} / {len(hier_result)}")
    if "_block_level" in hier_result.columns:
        print(f"    By level:")
        print(hier_result["_block_level"].value_counts().to_string())

    return flat_result, hier_result


def fig_hierarchical_blocking(flat_result, hier_result, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)

    def _plot_result(ax, result, title):
        _ax(ax)
        if result.empty:
            ax.text(0.5, 0.5, "No results", transform=ax.transAxes,
                    color="white", ha="center")
            return
        rel_col = "reliability_label"
        counts = result[rel_col].value_counts() if rel_col in result.columns \
                 else pd.Series({"no_label": len(result)})
        label_colors = {"high":"#00e5ff","medium_review":"#ffb300",
                        "low":"#ff5252","reject":"#888888"}
        bars = ax.bar(counts.index,
                      counts.values,
                      color=[label_colors.get(l, "#aaaaaa") for l in counts.index],
                      edgecolor="white", linewidth=0.4)
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x() + b.get_width()/2, v + 0.1, str(v),
                    ha="center", color="white", fontsize=10)
        ax.set_ylabel("Count", color="white")
        ax.set_title(title, color="white", fontweight="bold")
        ax.set_xticklabels(counts.index, color="white")

    _plot_result(axes[0], flat_result,
                 f"Flat blocking (district only)\n"
                 f"Matched: {flat_result['final_decision'].sum() if not flat_result.empty else 0}")
    _plot_result(axes[1], hier_result,
                 f"Hierarchical blocking (district→province)\n"
                 f"Matched: {hier_result['final_decision'].sum() if not hier_result.empty else 0}")

    # Annotate boundary features
    if not hier_result.empty and "_block_level" in hier_result.columns:
        level1 = hier_result[hier_result["_block_level"] == 1]
        if len(level1):
            axes[1].annotate(
                f"✓ {len(level1)} boundary feature(s)\nmatched at province level",
                xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top",
                color="#00ff88", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1b2a", alpha=0.7),
            )

    fig.suptitle(
        "Hierarchical Admin Blocking — Eliminating Boundary Artefacts\n"
        "Features near district boundaries matched at province level",
        color="white", fontsize=12, fontweight="bold",
    )
    fig.text(0.5, 0.01, "github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=8, color="#a0bdd8")

    out = out_dir / "geo_adv_hierarchical_blocking.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 3 — Coordinate Uncertainty Score
# ══════════════════════════════════════════════════════════════════════════════

def run_uncertainty_demo():
    from fuzzy_llm_matcher import geo_uncertainty_score, haversine_km

    print("\n  Uncertainty score examples (seismology / GPS context):")
    cases = [
        ("Same location, GPS precision (±0.01 km)",  0.0,   0.01, 0.01),
        ("0.5 km apart, GPS (±0.01 km each)",        0.5,   0.01, 0.01),
        ("3 km apart, urban seismic (±2 km each)",   3.0,   2.0,  2.0),
        ("10 km apart, regional seismic (±5 km)",   10.0,   5.0,  5.0),
        ("50 km apart, historical record (±30 km)", 50.0,  30.0, 30.0),
        ("200 km apart, well-located events",       200.0,  5.0,  5.0),
    ]
    for desc, dist_km, s1, s2 in cases:
        # Place two points dist_km apart at same latitude
        lat1, lon1 = 48.14, 11.58
        dlat = dist_km / 111.32
        score = geo_uncertainty_score(lat1, lon1, s1, lat1 + dlat, lon1, s2)
        print(f"  {desc:45s}: P(same) = {score:5.1f}%")

    return cases


def fig_uncertainty_score(out_dir: Path):
    from fuzzy_llm_matcher import geo_uncertainty_score

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)

    # Panel 1: Score vs distance for different sigma values
    ax = axes[0]
    _ax(ax)
    distances = np.linspace(0, 100, 200)
    sigmas = [0.5, 2.0, 5.0, 10.0, 20.0]
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(sigmas)))

    for sigma, color in zip(sigmas, colors):
        scores = []
        for d in distances:
            dlat = d / 111.32
            scores.append(geo_uncertainty_score(0, 0, sigma, dlat, 0, sigma))
        ax.plot(distances, scores, color=color, lw=2,
                label=f"σ₁=σ₂={sigma} km")

    ax.set_xlabel("Haversine distance (km)", color="white", fontsize=11)
    ax.set_ylabel("P(same location) × 100", color="white", fontsize=11)
    ax.set_title("Uncertainty score vs distance\nfor equal uncertainties",
                 color="white", fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.35, facecolor=BG,
              edgecolor=GRID, labelcolor="white")
    ax.axhline(50, color=GRID, lw=0.8, ls="--")
    ax.text(102, 51, "50%", color=GRID, fontsize=8)

    # Panel 2: Score as a 2-D heatmap (sigma1 vs sigma2, fixed dist=10 km)
    ax2 = axes[1]
    _ax(ax2)
    s_vals = np.linspace(0.1, 30, 50)
    Z = np.array([[
        geo_uncertainty_score(0, 0, s1, 10/111.32, 0, s2)
        for s2 in s_vals] for s1 in s_vals])

    im = ax2.contourf(s_vals, s_vals, Z, levels=20, cmap="RdYlGn")
    cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label("P(same) × 100", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    ax2.set_xlabel("σ₁ (km) — left record uncertainty", color="white", fontsize=11)
    ax2.set_ylabel("σ₂ (km) — right record uncertainty", color="white", fontsize=11)
    ax2.set_title("P(same) heatmap  |  haversine = 10 km\nseismology / GPS / archaeology",
                  color="white", fontweight="bold")

    fig.suptitle(
        "Coordinate-Uncertainty-Aware Geo Score\n"
        "P(same location) = 1 − Φ(d / √(σ₁² + σ₂²))  ×  100",
        color="white", fontsize=12, fontweight="bold",
    )
    fig.text(0.5, 0.01, "github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=8, color="#a0bdd8")

    out = out_dir / "geo_adv_uncertainty_score.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 4 — Geometry Similarity Scores
# ══════════════════════════════════════════════════════════════════════════════

def run_geometry_similarity_demo():
    from fuzzy_llm_matcher import geometry_similarity_score
    from shapely.geometry import LineString, Polygon, Point, MultiPolygon

    print("\n  Geometry similarity examples:")

    # Lines (Fréchet + Hausdorff)
    base_line  = LineString([(0,0),(500,500),(1000,0)])
    close_line = LineString([(0,50),(500,550),(1000,50)])     # 50 m offset
    far_line   = LineString([(0,500),(500,1000),(1000,500)])  # 500 m offset

    for method in ["hausdorff", "frechet"]:
        s_close = geometry_similarity_score(base_line, close_line, method, 500)
        s_far   = geometry_similarity_score(base_line, far_line,   method, 500)
        print(f"  {method:10s} | close offset (50m): {s_close:.1f}  "
              f"| far offset (500m): {s_far:.1f}")

    # Polygons (IoU + Hausdorff)
    p1 = Point(0, 0).buffer(1000)
    p2 = Point(500, 0).buffer(1000)   # 50% overlap
    p3 = Point(3000, 0).buffer(1000)  # no overlap

    for method in ["iou", "hausdorff"]:
        s_overlap = geometry_similarity_score(p1, p2, method, 2000)
        s_noover  = geometry_similarity_score(p1, p3, method, 2000)
        print(f"  {method:10s} | 50% overlap: {s_overlap:.1f}  "
              f"| no overlap: {s_noover:.1f}")


def fig_geometry_similarity(out_dir: Path):
    from fuzzy_llm_matcher import geometry_similarity_score
    from shapely.geometry import LineString, Point, Polygon
    import matplotlib.patches as mpl_patches
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor=BG)
    axes = axes.flatten()

    # ── Panel 0: Line pair — Hausdorff score vs offset ───────────────────────
    ax = axes[0]; _ax(ax)
    offsets  = np.linspace(0, 500, 100)
    max_dist = 500.0
    base     = LineString([(0,0),(1000,1000),(2000,0)])
    h_scores = [geometry_similarity_score(
        base, LineString([(0,o),(1000,1000+o),(2000,o)]),
        "hausdorff", max_dist) for o in offsets]
    f_scores = [geometry_similarity_score(
        base, LineString([(0,o),(1000,1000+o),(2000,o)]),
        "frechet", max_dist) for o in offsets]

    ax.plot(offsets, h_scores, color="#00e5ff", lw=2, label="Hausdorff")
    ax.plot(offsets, f_scores, color="#ffb300", lw=2, label="Fréchet", ls="--")
    ax.set_xlabel("Perpendicular offset (m)", color="white")
    ax.set_ylabel("Score (0–100)", color="white")
    ax.set_title("Line similarity vs perpendicular offset\n(rivers, roads, fault traces)",
                 color="white", fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.35, facecolor=BG, edgecolor=GRID, labelcolor="white")
    ax.axvline(max_dist, color="#ff5252", lw=1, ls=":", alpha=0.7)
    ax.text(max_dist + 5, 5, f"max_distance={max_dist:.0f}m", color="#ff5252", fontsize=8)

    # ── Panel 1: Polygon pair — IoU vs centroid distance ─────────────────────
    ax2 = axes[1]; _ax(ax2)
    displacements = np.linspace(0, 3000, 100)
    iou_scores = [geometry_similarity_score(
        Point(0,0).buffer(1000), Point(d,0).buffer(1000), "iou") for d in displacements]
    haus_scores_poly = [geometry_similarity_score(
        Point(0,0).buffer(1000), Point(d,0).buffer(1000), "hausdorff", 3000)
        for d in displacements]

    ax2.plot(displacements, iou_scores,      color="#7cfc00", lw=2, label="IoU (Jaccard)")
    ax2.plot(displacements, haus_scores_poly, color="#ff69b4", lw=2, label="Hausdorff", ls="--")
    ax2.set_xlabel("Centroid displacement (m)", color="white")
    ax2.set_ylabel("Score (0–100)", color="white")
    ax2.set_title("Polygon similarity vs displacement\n(admin boundaries, land parcels)",
                  color="white", fontweight="bold")
    ax2.legend(fontsize=9, framealpha=0.35, facecolor=BG, edgecolor=GRID, labelcolor="white")

    # ── Panel 2: Visual line comparison ──────────────────────────────────────
    ax3 = axes[2]; ax3.set_facecolor(MID)
    ax3.tick_params(colors="white")
    for s in ax3.spines.values(): s.set_edgecolor(GRID)

    lines = [
        (LineString([(0,0),(500,500),(1000,0)]),   "#00e5ff", "reference line", 2),
        (LineString([(0,50),(500,550),(1000,50)]), "#ffb300", "offset +50m",   1.5),
        (LineString([(0,200),(500,700),(1000,200)]),"#ff5252", "offset +200m", 1.5),
        (LineString([(0,500),(500,1000),(1000,500)]),"#888888","offset +500m", 1.0),
    ]
    ref = lines[0][0]
    for line, color, label, lw in lines:
        xs, ys = line.xy
        score_h = geometry_similarity_score(ref, line, "hausdorff", 500)
        score_f = geometry_similarity_score(ref, line, "frechet",   500)
        ax3.plot(xs, ys, color=color, lw=lw,
                 label=f"{label}\nH={score_h:.0f}  F={score_f:.0f}")
    ax3.legend(fontsize=8, framealpha=0.35, facecolor=BG, edgecolor=GRID, labelcolor="white")
    ax3.set_title("Line geometries — Hausdorff (H) and Fréchet (F) scores",
                  color="white", fontweight="bold")

    # ── Panel 3: Visual polygon comparison ───────────────────────────────────
    ax4 = axes[3]; ax4.set_facecolor(MID)
    ax4.tick_params(colors="white")
    for s in ax4.spines.values(): s.set_edgecolor(GRID)

    polys = [
        (Point(0,0).buffer(1000),       "#00e5ff", "reference", 1.0),
        (Point(600,0).buffer(1000),     "#ffb300", "60% overlap", 0.5),
        (Point(1500,0).buffer(1000),    "#ff5252", "touching",   0.5),
        (Point(3000,0).buffer(1000),    "#888888", "no overlap", 0.3),
    ]
    ref_poly = polys[0][0]
    for poly, color, label, alpha in polys:
        xs, ys = poly.exterior.xy
        ax4.fill(xs, ys, color=color, alpha=alpha)
        ax4.plot(xs, ys, color=color, lw=1.2)
        iou = geometry_similarity_score(ref_poly, poly, "iou")
        haus = geometry_similarity_score(ref_poly, poly, "hausdorff", 4000)
        c = poly.centroid
        ax4.text(c.x, c.y, f"{label}\nIoU={iou:.0f}\nH={haus:.0f}",
                 ha="center", va="center", color="white", fontsize=8)
    ax4.set_title("Polygon geometries — IoU and Hausdorff scores",
                  color="white", fontweight="bold")
    ax4.set_aspect("equal")

    fig.suptitle(
        "Geometry Similarity Scores: Hausdorff / Fréchet / IoU\n"
        "For polygon, line, and point feature matching",
        color="white", fontsize=12, fontweight="bold",
    )
    fig.text(0.5, 0.01, "github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=8, color="#a0bdd8")

    out = out_dir / "geo_adv_geometry_similarity.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run():
    out_dir = Path("notebooks/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("══ Feature 1: Transliteration + Phonetic Scoring ══════════")
    phonetic_df = run_phonetic_demo()
    fig_phonetic_comparison(phonetic_df, out_dir)

    print("\n══ Feature 2: Hierarchical Admin Blocking ══════════════════")
    flat_result, hier_result = run_hierarchical_demo()
    fig_hierarchical_blocking(flat_result, hier_result, out_dir)

    print("\n══ Feature 3: Coordinate Uncertainty Score ═════════════════")
    run_uncertainty_demo()
    fig_uncertainty_score(out_dir)

    print("\n══ Feature 4: Geometry Similarity Scores ═══════════════════")
    run_geometry_similarity_demo()
    fig_geometry_similarity(out_dir)

    print("\nDone. All figures saved to", out_dir)


if __name__ == "__main__":
    run()
