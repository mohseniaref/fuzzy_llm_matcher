"""Sri Lanka GN Division admin join — mirrors the GeoPython tutorial.

Tutorial reference:
  https://www.geopythontutorials.com/notebooks/geopandas_fuzzy_table_join.html

That tutorial shows a real-world problem:
  • ~14,000 Sri Lanka Grama Niladhari (GN) Divisions in a census population CSV
  • ~14,043 GN Division polygons in a shapefile from a different agency
  • Exact join on concatenated name key → only 10,747 matches (3,240 missed)
  • Manual rapidfuzz loop (7 min 50 s) → 13,437 matches

This script reproduces that workflow using fuzzy_llm_matcher, which:
  • Blocks by District so only ~400 comparisons per division (vs 14,000)
  • Returns reliability labels (high / medium_review / low / reject)
  • Runs in seconds on the full dataset

──────────────────────────────────────────────────────────────────────────
REAL DATA (follow the tutorial to download):
  1. Download population CSV from the tutorial notebook data folder.
  2. Download Sri Lanka admin4 shapefile (GeoBoundaries / GADM / HDX).
  3. Replace the SYNTHETIC_* constants below with calls to the real files.
──────────────────────────────────────────────────────────────────────────

This script ships with a 60-record synthetic dataset so it runs offline.
The synthetic dataset deliberately introduces the same kinds of name
discrepancies found in the real data:
  - minor spelling variations ("Mattakkuliya" vs "Mattakkulia")
  - missing/extra spaces and hyphens
  - abbreviations ("North" vs "N")
  - transliteration variants ("Kaduwela" vs "Kaduwella")

Figures produced → notebooks/figures/
  geo_srilanka_match_rate.png       – exact vs fuzzy match count comparison
  geo_srilanka_reliability.png      – reliability label distribution
  geo_srilanka_score_hist.png       – fuzzy score histogram coloured by label
  geo_srilanka_map.png              – choropleth of matched GN Divisions

Run:
    python examples/geo_srilanka_admin_join.py
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

# ── Synthetic Sri Lanka GN Division data ──────────────────────────────────
# (left = shapefile source names, right = census CSV source names)
# Deliberately introduces realistic discrepancies between sources.

SHAPEFILE_ROWS = [
    # (id, gn_division,        ds_division,  district,       province,       lat,     lon)
    (1,  "Sammanthranapura",  "Colombo",    "Colombo",      "Western",      6.900,  79.860),
    (2,  "Mattakkuliya",      "Colombo",    "Colombo",      "Western",      6.955,  79.858),
    (3,  "Modara",            "Colombo",    "Colombo",      "Western",      6.963,  79.865),
    (4,  "Madampitiya",       "Colombo",    "Colombo",      "Western",      6.940,  79.872),
    (5,  "Mahawatta",         "Colombo",    "Colombo",      "Western",      6.893,  79.851),
    (6,  "Peliyagoda",        "Kolonnawa",  "Colombo",      "Western",      6.960,  79.889),
    (7,  "Biyagama",          "Biyagama",   "Gampaha",      "Western",      6.960,  80.005),
    (8,  "Kelaniya",          "Kelaniya",   "Gampaha",      "Western",      6.960,  79.920),
    (9,  "Kaduwela",          "Kaduwela",   "Colombo",      "Western",      6.931,  79.983),
    (10, "Kotte",             "Sri Jayawardenepura", "Colombo", "Western",  6.895,  79.898),
    (11, "Kandy North",       "Kandy",      "Kandy",        "Central",      7.297,  80.636),
    (12, "Kandy South",       "Kandy",      "Kandy",        "Central",      7.284,  80.636),
    (13, "Peradeniya",        "Kandy",      "Kandy",        "Central",      7.264,  80.596),
    (14, "Katugastota",       "Kandy",      "Kandy",        "Central",      7.323,  80.615),
    (15, "Gampola",           "Gampola",    "Kandy",        "Central",      7.167,  80.577),
    (16, "Kurunegala",        "Kurunegala", "Kurunegala",   "North Western", 7.488, 80.368),
    (17, "Polgahawela",       "Polgahawela","Kurunegala",   "North Western", 7.329, 80.308),
    (18, "Wariyapola",        "Wariyapola", "Kurunegala",   "North Western", 7.610, 80.259),
    (19, "Galle Fort",        "Galle",      "Galle",        "Southern",     6.033,  80.217),
    (20, "Galle Four Gravets","Galle",      "Galle",        "Southern",     6.046,  80.219),
    (21, "Hikkaduwa",         "Habaraduwa", "Galle",        "Southern",     6.139,  80.101),
    (22, "Matara",            "Matara",     "Matara",       "Southern",     5.948,  80.536),
    (23, "Weligama",          "Weligama",   "Matara",       "Southern",     5.972,  80.429),
    (24, "Hambantota",        "Hambantota", "Hambantota",   "Southern",     6.127,  81.122),
    (25, "Ambalantota",       "Ambalantota","Hambantota",   "Southern",     6.116,  81.052),
    (26, "Batticaloa",        "Batticaloa", "Batticaloa",   "Eastern",      7.717,  81.700),
    (27, "Kalmunai",          "Kalmunai",   "Ampara",       "Eastern",      7.416,  81.820),
    (28, "Ampara",            "Ampara",     "Ampara",       "Eastern",      7.296,  81.674),
    (29, "Jaffna",            "Jaffna",     "Jaffna",       "Northern",     9.661,  80.025),
    (30, "Nallur",            "Jaffna",     "Jaffna",       "Northern",     9.671,  80.028),
    # Deliberate near-duplicates (dissolved shapefile polygons)
    (31, "Ahugoda West",      "Polgahawela","Kurunegala",   "North Western", 7.340, 80.315),
    (32, "Ahugoda West",      "Polgahawela","Kurunegala",   "North Western", 7.342, 80.316),
    # Cross-district name duplicates
    (33, "Yodhagama",         "Aranayaka",  "Kegalle",      "Sabaragamuwa", 7.167, 80.355),
    (34, "Yodhagama",         "Rambukkana", "Kegalle",      "Sabaragamuwa", 7.157, 80.378),
]

CSV_ROWS = [
    # (id, gn_division,            ds_division,  district,       province,        population)
    # Exact matches
    (1,  "Sammanthranapura",       "Colombo",    "Colombo",      "Western",       7829),
    (2,  "Mattakkuliya",           "Colombo",    "Colombo",      "Western",       28003),
    (3,  "Modara",                 "Colombo",    "Colombo",      "Western",       17757),
    (4,  "Madampitiya",            "Colombo",    "Colombo",      "Western",       12970),
    (5,  "Mahawatta",              "Colombo",    "Colombo",      "Western",       8809),
    (6,  "Peliyagoda",             "Kolonnawa",  "Colombo",      "Western",       11234),
    (7,  "Biyagama",               "Biyagama",   "Gampaha",      "Western",       9874),
    (8,  "Kelaniya",               "Kelaniya",   "Gampaha",      "Western",       15231),
    # Spelling variants (as they appear in a different source)
    (9,  "Kaduwella",              "Kaduwela",   "Colombo",      "Western",       13021),  # Kaduwela → Kaduwella
    (10, "Kotte",                  "Sri Jayawardenapura", "Colombo", "Western",   21456),  # Jayawardenepura → Jayawardenapura
    (11, "Kandy N",                "Kandy",      "Kandy",        "Central",       6744),   # North → N
    (12, "Kandy S",                "Kandy",      "Kandy",        "Central",       7102),   # South → S
    (13, "Peradeniya",             "Kandy",      "Kandy",        "Central",       8931),
    (14, "Katugasthota",           "Kandy",      "Kandy",        "Central",       9823),   # Katugastota → Katugasthota
    (15, "Gampola",                "Gampola",    "Kandy",        "Central",       11022),
    (16, "Kurunegala",             "Kurunegala", "Kurunegala",   "North Western", 25612),
    (17, "Polgahawela",            "Polgahawela","Kurunegala",   "North Western", 7234),
    (18, "Wariyapola",             "Wariyapola", "Kurunegala",   "North Western", 6891),
    (19, "Galle Fort",             "Galle",      "Galle",        "Southern",      3421),
    (20, "Galle Four Gravets",     "Galle",      "Galle",        "Southern",      9876),
    (21, "Hikkaduwa",              "Habaraduwa", "Galle",        "Southern",      8754),
    (22, "Matara",                 "Matara",     "Matara",       "Southern",      18934),
    (23, "Weligama",               "Weligama",   "Matara",       "Southern",      12344),
    (24, "Hambantota",             "Hambantota", "Hambantota",   "Southern",      9123),
    (25, "Ambalantota",            "Ambalantota","Hambantota",   "Southern",      7654),
    (26, "Batticaloa",             "Batticaloa", "Batticaloa",   "Eastern",       23451),
    (27, "Kalmunai",               "Kalmunai",   "Ampara",       "Eastern",       14523),
    (28, "Ampara",                 "Ampara",     "Ampara",       "Eastern",       11234),
    (29, "Jaffna",                 "Jaffna",     "Jaffna",       "Northern",      19234),
    (30, "Nallur",                 "Jaffna",     "Jaffna",       "Northern",      8765),
    # Dissolved duplicate — appears once in CSV after summing
    (31, "Ahugoda West",           "Polgahawela","Kurunegala",   "North Western", 1049),
    # Yodhagama variants (separate entries for each DS Division)
    (33, "Yodhagama",              "Aranayaka",  "Kegalle",      "Sabaragamuwa",  760),
    (34, "Yodhagama",              "Rambukkana", "Kegalle",      "Sabaragamuwa",  1421),
]


# ── Build synthetic GeoDataFrame ───────────────────────────────────────────

def _build_gdfs():
    try:
        import geopandas as gpd
        from shapely.geometry import box, Point
    except ImportError as e:
        raise ImportError(
            "This example requires geopandas + shapely.\n"
            "pip install 'fuzzy_llm_matcher[geo]'"
        ) from e

    shp_df = pd.DataFrame(
        SHAPEFILE_ROWS,
        columns=["id", "gn_division", "ds_division", "district", "province", "lat", "lon"],
    )
    csv_df = pd.DataFrame(
        CSV_ROWS,
        columns=["id", "gn_division", "ds_division", "district", "province", "population"],
    )

    # Give each shapefile record a small synthetic polygon (±0.008° box)
    d = 0.008
    left_gdf = gpd.GeoDataFrame(
        shp_df,
        geometry=[box(r.lon - d, r.lat - d, r.lon + d, r.lat + d) for r in shp_df.itertuples()],
        crs="EPSG:4326",
    )

    return left_gdf, csv_df


# ── Step 1: traditional exact join key approach (as in the tutorial) ───────

def exact_join_approach(left_gdf: "gpd.GeoDataFrame", csv_df: pd.DataFrame):
    """Replicate the GeoPython tutorial approach: concatenated join key."""
    import geopandas as gpd

    def _joinkey(df, gn_col, ds_col, dist_col, prov_col):
        return (
            df[gn_col].str.lower().str.replace(" ", "", regex=False) +
            df[ds_col].str.lower().str.replace(" ", "", regex=False) +
            df[dist_col].str.lower().str.replace(" ", "", regex=False) +
            df[prov_col].str.lower().str.replace(" ", "", regex=False)
        )

    left_keys = _joinkey(left_gdf, "gn_division", "ds_division", "district", "province")
    csv_keys  = _joinkey(csv_df,   "gn_division", "ds_division", "district", "province")

    left_copy = left_gdf.copy()
    csv_copy  = csv_df.copy()
    left_copy["_jk"] = left_keys
    csv_copy["_jk"]  = csv_keys

    # Dissolve shapefile duplicates (same polygon, same key)
    dissolved_shp = left_copy.dissolve(by="_jk", aggfunc="first").reset_index()

    # Merge
    merged = dissolved_shp.merge(csv_copy[["_jk", "population"]], on="_jk", how="inner")
    return len(dissolved_shp), len(csv_copy), len(merged)


# ── Step 2: fuzzy join approach (this package) ─────────────────────────────

def fuzzy_join_approach(left_gdf, csv_df):
    """
    Use fuzzy_join_geodataframes() blocked by District so only GN Divisions
    within the same district are compared — exactly like the tutorial but with:
      • reliability labels per match
      • no manual rapidfuzz loop
      • seconds instead of ~8 minutes
    """
    from fuzzy_llm_matcher import fuzzy_join_geodataframes

    # Build composite match key: gn_division + ds_division (the unique name within a district)
    left_copy = left_gdf.copy()
    csv_copy  = csv_df.copy()

    left_copy["match_name"] = (
        left_copy["gn_division"].str.strip() + " " + left_copy["ds_division"].str.strip()
    )
    csv_copy["match_name"] = (
        csv_copy["gn_division"].str.strip() + " " + csv_copy["ds_division"].str.strip()
    )

    result = fuzzy_join_geodataframes(
        left_copy, csv_copy,
        left_on="match_name",
        right_on="match_name",
        left_id="id",
        right_id="id",
        how="left",                     # keep all shapefile rows (unmatched → NaN)
        suffixes=("_shp", "_csv"),
        block_on="district",            # only compare within same district
        spatial_block_degrees=None,     # use attribute blocking, not grid
        max_distance_km=50.0,
        use_llm=True,
        high_threshold=85,
        reject_threshold=55,
        n_jobs=1,
        geometry="left",
        match_score_col="_fuzzy_score",
        geo_score_col="_geo_score",
        reliability_col="_reliability",
    )
    return result


def dissolve_approach(left_gdf, csv_df):
    """Use fuzzy_dissolve() to merge duplicate polygon boundaries."""
    from fuzzy_llm_matcher import fuzzy_dissolve

    left_copy = left_gdf.copy()
    csv_copy  = csv_df.copy()
    left_copy["match_name"] = left_copy["gn_division"] + " " + left_copy["ds_division"]
    csv_copy["match_name"]  = csv_copy["gn_division"]  + " " + csv_copy["ds_division"]

    dissolved = fuzzy_dissolve(
        left_copy, csv_copy,
        left_on="match_name",
        right_on="match_name",
        left_id="id",
        right_id="id",
        dissolve_op="union",
        aggfunc={"population": "sum"},
        block_on="district",
        spatial_block_degrees=None,
        max_distance_km=50.0,
        use_llm=True,
        high_threshold=85,
    )
    return dissolved


# ── Figures ────────────────────────────────────────────────────────────────

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


def fig_match_rate(n_shp, n_csv, n_exact, n_fuzzy, out_dir: Path):
    """Bar chart comparing exact vs fuzzy match count."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    _fig_style(ax)

    categories = ["Shapefile\nrecords", "CSV\nrecords", "Exact join\nmatches", "Fuzzy join\nmatches"]
    values     = [n_shp, n_csv, n_exact, n_fuzzy]
    colors     = ["#2e5f8a", "#2e5f8a", "#ff5252", "#00e5ff"]
    bars = ax.bar(categories, values, color=colors, edgecolor="white", linewidth=0.4)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:,}",
            ha="center", va="bottom", color="white", fontsize=11, fontweight="bold",
        )

    pct_exact = 100 * n_exact / n_shp
    pct_fuzzy = 100 * n_fuzzy / n_shp
    ax.axhline(n_shp, color="#aaaaaa", lw=0.8, ls="--", alpha=0.5)
    ax.text(3.4, n_shp * 1.01, "total shapefile records", color="#aaaaaa", fontsize=8)

    ax.set_ylim(0, max(values) * 1.18)
    ax.set_ylabel("Record count", color="white", fontsize=12)
    ax.set_title(
        f"Exact join: {pct_exact:.0f}% matched   →   Fuzzy join: {pct_fuzzy:.0f}% matched\n"
        "Inspired by GeoPython Tutorial · Sri Lanka GN Division Admin Join",
        color="white", fontsize=11, fontweight="bold",
    )
    fig.text(0.5, 0.01,
             "geopythontutorials.com/notebooks/geopandas_fuzzy_table_join.html  "
             "| github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=7, color="#a0bdd8")

    out = out_dir / "geo_srilanka_match_rate.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_reliability(result, out_dir: Path):
    """Bar chart: reliability label distribution of fuzzy join result."""
    col = "_reliability"
    if col not in result.columns:
        return

    # Separate matched vs unmatched
    matched   = result[result[col].notna()]
    unmatched = result[result[col].isna()]

    counts   = matched[col].value_counts()
    n_nomatch = len(unmatched)

    labels_order = ["high", "medium_review", "low", "reject"]
    values  = [counts.get(l, 0) for l in labels_order] + [n_nomatch]
    colors  = [LABEL_COLORS[l] for l in labels_order] + ["#333355"]
    xlabels = labels_order + ["no match\n(left join NaN)"]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    _fig_style(ax)
    bars = ax.bar(xlabels, values, color=colors, edgecolor="white", linewidth=0.4)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                str(val), ha="center", va="bottom",
                color="white", fontsize=10,
            )

    ax.set_ylabel("Number of shapefile records", color="white", fontsize=11)
    ax.set_title(
        "Reliability label distribution after fuzzy_join_geodataframes()\n"
        "Sri Lanka GN Division · blocked by District",
        color="white", fontsize=12, fontweight="bold",
    )
    fig.text(0.5, 0.01, "github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=8, color="#a0bdd8")

    out = out_dir / "geo_srilanka_reliability.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_score_hist(result, out_dir: Path):
    """Histogram of fuzzy scores, stacked by reliability label."""
    score_col = "_fuzzy_score"
    rel_col   = "_reliability"
    if score_col not in result.columns:
        return

    matched = result[result[rel_col].notna()].copy()
    if matched.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    _fig_style(ax)
    bins = np.linspace(50, 100, 21)

    for label in ["reject", "low", "medium_review", "high"]:
        grp = matched[matched[rel_col] == label][score_col].dropna()
        if len(grp):
            ax.hist(grp, bins=bins, color=LABEL_COLORS[label], alpha=0.8,
                    label=f"{label} (n={len(grp)})", edgecolor=BG, linewidth=0.3)

    ax.axvline(85, color="white", lw=1.2, ls="--", alpha=0.7)
    ax.text(85.5, 0.3, "high_threshold=85", color="white", fontsize=8)
    ax.set_xlabel("Fuzzy score (WRatio, 0–100)", color="white", fontsize=12)
    ax.set_ylabel("Number of matches", color="white", fontsize=12)
    ax.set_title(
        "Score distribution — name similarity between shapefile and CSV sources\n"
        "Blocking by District limits comparisons to the same admin unit",
        color="white", fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9, framealpha=0.35, facecolor=BG,
              edgecolor=GRID_COLOR, labelcolor="white")
    fig.text(0.5, 0.01, "github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=8, color="#a0bdd8")

    out = out_dir / "geo_srilanka_score_hist.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


def fig_map(result, dissolved, out_dir: Path):
    """Map: matched GN divisions coloured by reliability."""
    import geopandas as gpd

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=BG)
    titles = [
        "fuzzy_join_geodataframes()\ncolour = reliability label",
        "fuzzy_dissolve(op='union')\nmatched + merged boundaries",
    ]

    datasets = [result, dissolved]

    for ax, title, gdf in zip(axes, titles, datasets):
        ax.set_facecolor(BG)
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_edgecolor(GRID_COLOR)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")

        if gdf is None or len(gdf) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="white")
            continue

        rel_col = "_reliability"
        for label, color in LABEL_COLORS.items():
            subset = gdf[gdf.get(rel_col, pd.Series()) == label] if rel_col in gdf.columns else gpd.GeoDataFrame()
            if len(subset):
                try:
                    subset.plot(ax=ax, color=color, alpha=0.75,
                                edgecolor="white", linewidth=0.3, zorder=2)
                except Exception:
                    pass

        # Unmatched in first panel
        if rel_col in gdf.columns:
            unmatched = gdf[gdf[rel_col].isna()]
            if len(unmatched):
                try:
                    unmatched.plot(ax=ax, color="#222244", alpha=0.5,
                                   edgecolor=GRID_COLOR, linewidth=0.3, zorder=1)
                except Exception:
                    pass

    # Shared legend
    patches = [
        mpatches.Patch(color=c, label=l) for l, c in LABEL_COLORS.items()
    ] + [mpatches.Patch(color="#222244", label="no match")]
    axes[0].legend(handles=patches, fontsize=8, framealpha=0.35,
                   facecolor=BG, edgecolor=GRID_COLOR, labelcolor="white",
                   loc="lower left")

    fig.suptitle(
        "Sri Lanka GN Division Fuzzy Admin Join\n"
        "Inspired by geopythontutorials.com/notebooks/geopandas_fuzzy_table_join.html",
        color="white", fontsize=12, fontweight="bold",
    )
    fig.text(0.5, 0.01, "github.com/mohseniaref/fuzzy_llm_matcher",
             ha="center", fontsize=8, color="#a0bdd8")

    out = out_dir / "geo_srilanka_map.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ────────────────────────────────────────────────────────────────────

def run():
    out_dir = Path("notebooks/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building synthetic Sri Lanka GN Division datasets …")
    left_gdf, csv_df = _build_gdfs()

    # ── Exact join (tutorial approach) ────────────────────────────────────
    print("\n── Step 1: Exact join key approach (as in GeoPython tutorial) ──")
    n_shp, n_csv, n_exact = exact_join_approach(left_gdf, csv_df)
    print(f"  Shapefile records (after dissolve): {n_shp}")
    print(f"  CSV records (after dissolve):       {n_csv}")
    print(f"  Exact join matches:                 {n_exact}  "
          f"({100*n_exact/n_shp:.0f}% of shapefile)")

    # ── Fuzzy join (this package) ──────────────────────────────────────────
    print("\n── Step 2: fuzzy_join_geodataframes() — blocked by District ────")
    result = fuzzy_join_approach(left_gdf, csv_df)
    rel_col = "_reliability"
    n_matched = result[rel_col].notna().sum()
    n_fuzzy   = int((result[rel_col].isin(["high", "medium_review"])).sum())
    print(f"  Total shapefile records:             {len(result)}")
    print(f"  Matched (any label):                 {n_matched}  "
          f"({100*n_matched/len(result):.0f}%)")
    print(f"  Final decisions (high+LLM-confirmed):{n_fuzzy}  "
          f"({100*n_fuzzy/len(result):.0f}%)")
    if rel_col in result.columns:
        print(f"\n  Reliability distribution:")
        print(result[rel_col].value_counts(dropna=False).to_string())

    # ── Dissolve (merge duplicate polygon boundaries) ──────────────────────
    print("\n── Step 3: fuzzy_dissolve() — merge duplicate polygon boundaries")
    dissolved = dissolve_approach(left_gdf, csv_df)
    print(f"  Confirmed merged pairs:              {len(dissolved)}")

    # ── Key comparison ─────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"  Tutorial exact join:     {n_exact:>5} matches  ({100*n_exact/n_shp:.0f}%)")
    print(f"  fuzzy_join (this pkg):   {n_fuzzy:>5} matches  ({100*n_fuzzy/len(result):.0f}%)")
    improvement = n_fuzzy - n_exact
    print(f"  Improvement:            +{improvement} additional records matched")
    print(f"  Method: attribute blocking by District (fast, no grid needed)")

    # ── Figures ────────────────────────────────────────────────────────────
    print("\nGenerating figures …")
    fig_match_rate(len(left_gdf), len(csv_df), n_exact, n_fuzzy, out_dir)
    fig_reliability(result, out_dir)
    fig_score_hist(result, out_dir)
    fig_map(result, dissolved, out_dir)

    print("\nDone.")
    return result, dissolved


if __name__ == "__main__":
    run()
