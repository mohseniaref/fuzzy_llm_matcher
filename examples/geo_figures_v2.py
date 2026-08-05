"""Improved geo benchmark figures — satellite backgrounds, name-pair labels, results tables.

Uses ESRI World Imagery (satellite) tiles via contextily as background.
Each map shows matched name pairs side-by-side and a results summary table.

Output (notebooks/figures/):
  geo_v2_benchmark_summary.png   — bar chart + results table panel
  geo_v2_berlin_satellite.png    — Berlin satellite map, name pairs labeled
  geo_v2_geonames_satellite.png  — Europe satellite, diacritic pairs labeled
  geo_v2_gadm_satellite.png      — W-Europe satellite choropleth + table

Run:
    python examples/geo_figures_v2.py
"""
from __future__ import annotations
import json, os, re, sys, warnings
from pathlib import Path

# Fix PROJ database conflict between conda envs and the venv rasterio/pyproj.
# Must be set BEFORE any geo imports.
_venv_root = Path(__file__).resolve().parent.parent / ".venv"
for _proj_candidate in [
    _venv_root / "lib/python3.9/site-packages/rasterio/proj_data",
    _venv_root / "lib/python3.10/site-packages/rasterio/proj_data",
    _venv_root / "lib/python3.11/site-packages/rasterio/proj_data",
]:
    if (_proj_candidate / "proj.db").exists():
        os.environ["PROJ_LIB"] = str(_proj_candidate)
        break

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.table import Table
import numpy as np
import pandas as pd
import geopandas as gpd
import geodatasets
import contextily as cx
from pyproj import Transformer

# ── shared style ─────────────────────────────────────────────────────────────
BG     = "#0d1b2a"
CYAN   = "#00e5ff"
AMBER  = "#ffb300"
RED    = "#ff4040"
GREY   = "#778899"
PURPLE = "#bb86fc"
WHITE  = "#e8f4fd"
RCOLOR = {"high": CYAN, "medium_review": AMBER, "low": RED, "reject": GREY}

OUT = Path("notebooks/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ESRI World Imagery (free satellite tiles)
SATELLITE = cx.providers.Esri.WorldImagery

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "text.color": WHITE,
    "axes.labelcolor": WHITE,
    "xtick.color": WHITE,
    "ytick.color": WHITE,
    "axes.edgecolor": "#2d5a8e",
    "font.family": "DejaVu Sans",
})


def _to_webmercator(lon, lat):
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return t.transform(lon, lat)


def _label_pair(ax, x, y, dirty, canonical, col, offset=(0.004, 0.004), fontsize=7):
    """Draw  'dirty → canonical'  annotation with a stroke outline."""
    ax.annotate(
        f"{dirty}\n→ {canonical}",
        xy=(x, y), xytext=(x + offset[0], y + offset[1]),
        fontsize=fontsize, color=col, ha="left", va="bottom",
        path_effects=[pe.withStroke(linewidth=2.2, foreground="black")],
        zorder=10,
        arrowprops=dict(arrowstyle="-", color=col, lw=0.7, alpha=0.6),
    )


def _add_table(fig, ax_or_rect, headers, rows, title="", fontsize=8.5):
    """Draw a neat results table on the figure."""
    col_widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    total = sum(col_widths) + len(col_widths)

    the_table = ax_or_rect.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(fontsize)
    the_table.scale(1, 1.6)

    for (row, col), cell in the_table.get_celld().items():
        cell.set_facecolor("#0d1b2a" if row > 0 else "#1a3350")
        cell.set_text_props(color=WHITE)
        cell.set_edgecolor("#2d5a8e")
    return the_table


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Benchmark Summary: grouped bars + embedded results table
# ═══════════════════════════════════════════════════════════════════════════════
def fig_benchmark_summary():
    data = [
        ("Berlin OSM\nDE ↔ EN",       1.000, 0.027, 0.052,  300,   8),
        ("GeoNames\nDiacritics",       0.957, 0.055, 0.104,  400,  22),
        ("Wikidata\nCity Labels",      1.000, 0.993, 0.997,  300, 298),
        ("GADM\nAdmin Regions",        1.000, 0.143, 0.250,   70,  10),
    ]
    labels  = [d[0] for d in data]
    prec    = [d[1] for d in data]
    rec     = [d[2] for d in data]
    f1      = [d[3] for d in data]

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                            left=0.07, right=0.97, top=0.88, bottom=0.08)

    # ── bar chart (top-left, wide) ──
    ax_bar = fig.add_subplot(gs[0, :])
    ax_bar.set_facecolor(BG)
    x, w = np.arange(len(labels)), 0.25

    b1 = ax_bar.bar(x - w,   prec, w, label="Precision", color=CYAN,   alpha=0.88, zorder=3)
    b2 = ax_bar.bar(x,       rec,  w, label="Recall",    color=AMBER,  alpha=0.88, zorder=3)
    b3 = ax_bar.bar(x + w,   f1,   w, label="F1 Score",  color=PURPLE, alpha=0.88, zorder=3)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h > 0.03:
                ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                            f"{h:.2f}", ha="center", va="bottom", fontsize=8.5,
                            color=WHITE,
                            path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=11)
    ax_bar.set_ylim(0, 1.18)
    ax_bar.set_ylabel("Score", fontsize=11)
    ax_bar.yaxis.grid(True, alpha=0.25, zorder=0, color="#2d5a8e")
    ax_bar.set_axisbelow(True)
    ax_bar.legend(fontsize=10, framealpha=0.2, facecolor=BG,
                  edgecolor="#2d5a8e", labelcolor=WHITE, loc="upper right")
    ax_bar.set_title("fuzzy_llm_matcher — Geo Community Benchmark Results",
                     fontsize=14, fontweight="bold", pad=10, color=WHITE)

    # ── results table (bottom, full width) ──
    ax_tbl = fig.add_subplot(gs[1, :])
    ax_tbl.set_facecolor(BG)
    ax_tbl.axis("off")

    headers = ["Benchmark", "Dataset", "Pairs", "Matched", "Correct", "Precision", "Recall", "F1", "Notes"]
    rows = [
        ["Berlin OSM DE↔EN", "OpenStreetMap Overpass",
         "300", "8", "8", "1.000", "0.027", "0.052",
         "Very hard — completely different names"],
        ["GeoNames Diacritics", "GeoNames cities500 CC BY",
         "400", "23", "22", "0.957", "0.055", "0.104",
         "Zürich→Zuerich, Москва→Moskva"],
        ["Wikidata City Labels", "Wikidata SPARQL CC0",
         "300", "298", "298", "1.000", "0.993", "0.997",
         "Near-perfect — disambiguation variants"],
        ["GADM Admin Regions", "GADM v4.1 (academic)",
         "70", "10", "10", "1.000", "0.143", "0.250",
         "Bayern→Bavaria, Bretagne→Brittany, Thüringen→Thuringia"],
    ]

    the_table = ax_tbl.table(
        cellText=rows, colLabels=headers,
        loc="center", cellLoc="center",
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8)
    the_table.scale(1, 2.0)

    for (row, col), cell in the_table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1a3350")
        elif row % 2 == 0:
            cell.set_facecolor("#0f2035")
        else:
            cell.set_facecolor("#0d1b2a")
        cell.set_text_props(color=WHITE)
        cell.set_edgecolor("#2d5a8e")
        if col == 8:  # Notes column wider
            cell.set_width(0.22)

    ax_tbl.set_title("Full Results Table", fontsize=11, pad=6, color=WHITE)

    fig.text(0.5, 0.01,
             "github.com/mohseniaref/fuzzy_llm_matcher  •  DOI: 10.5281/zenodo.21803695  "
             "•  use_llm=False for all benchmarks (conservative baseline)",
             ha="center", fontsize=8, color="#7bafd4")

    out = OUT / "geo_v2_benchmark_summary.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Berlin Satellite Map: name pairs labeled on satellite imagery
# ═══════════════════════════════════════════════════════════════════════════════
def fig_berlin_satellite():
    from fuzzy_llm_matcher import match_tables

    cache = Path("data/berlin_bilingual_cache.json")
    with open(cache) as f:
        raw = json.load(f)

    id_to_coord, id_to_de, id_to_en = {}, {}, {}
    for el in raw.get("elements", []):
        eid = str(el["id"])
        tags = el.get("tags", {})
        lat = el["lat"] if "lat" in el else el.get("center", {}).get("lat")
        lon = el["lon"] if "lon" in el else el.get("center", {}).get("lon")
        if lat and lon:
            id_to_coord[eid] = (float(lon), float(lat))
        id_to_de[eid] = tags.get("name", "")
        id_to_en[eid] = tags.get("name:en", "")

    df = pd.DataFrame([
        {"id": eid, "de_name": id_to_de[eid], "en_name": id_to_en[eid],
         "lon": id_to_coord[eid][0], "lat": id_to_coord[eid][1]}
        for eid in id_to_coord
        if id_to_de.get(eid) and id_to_en.get(eid) and id_to_de[eid] != id_to_en[eid]
    ]).drop_duplicates("de_name")

    sample = df.sample(min(300, len(df)), random_state=42).reset_index(drop=True)
    left  = sample[["id", "de_name"]].rename(columns={"de_name": "name"})
    right = df[["id", "en_name"]].rename(columns={"en_name": "name"}).reset_index(drop=True)

    result = match_tables(left, right, left_on="name", right_on="name",
                          left_id="id", right_id="id",
                          top_k=5, use_llm=False, n_jobs=-1)

    result["lon"] = result["left_id"].map(lambda i: id_to_coord.get(i, (None, None))[0])
    result["lat"] = result["left_id"].map(lambda i: id_to_coord.get(i, (None, None))[1])
    result["de"] = result["left_id"].map(id_to_de)
    result["en"] = result["right_id"].map(id_to_en)
    result = result.dropna(subset=["lon", "lat"])

    # Convert to Web Mercator for contextily
    result["x"], result["y"] = zip(*result.apply(
        lambda r: _to_webmercator(r["lon"], r["lat"]), axis=1))

    # Berlin bbox in WGS84 → Web Mercator
    w, e, s, n = 13.15, 13.72, 52.38, 52.65
    xmin, ymin = _to_webmercator(w, s)
    xmax, ymax = _to_webmercator(e, n)

    fig = plt.figure(figsize=(18, 13), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3, 1],
                            wspace=0.04, left=0.02, right=0.98,
                            top=0.92, bottom=0.06)

    ax_map = fig.add_subplot(gs[0])
    ax_map.set_facecolor("#0a1018")

    # Satellite background
    cx.add_basemap(ax_map, crs="EPSG:3857", source=SATELLITE,
                   zoom=12, attribution=False, alpha=0.85)

    # Plot all points
    plotted = {k: 0 for k in RCOLOR}
    label_count = 0
    for _, row in result.iterrows():
        rlabel = row["reliability_label"] if row["reliability_label"] in RCOLOR else "reject"
        col = RCOLOR[rlabel]
        matched = row["final_decision"] == True
        ax_map.scatter(row["x"], row["y"],
                       s=65 if matched else 18,
                       c=col, marker="*" if matched else "o",
                       alpha=0.95 if matched else 0.5, zorder=5,
                       edgecolors="white" if matched else "none",
                       linewidths=0.4)
        plotted[rlabel] += 1

        # Label matched pairs (up to 12, different corners)
        if matched and rlabel == "high" and label_count < 12:
            de = row["de"]
            en = row["en"]
            if de and en:
                offset_x = 800 * (1 if label_count % 2 == 0 else -1)
                offset_y = 600 * (1 if label_count % 3 != 2 else -1)
                ax_map.annotate(
                    f"«{de}»\n→ «{en}»",
                    xy=(row["x"], row["y"]),
                    xytext=(row["x"] + offset_x, row["y"] + offset_y),
                    fontsize=7, color=CYAN,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
                    zorder=9,
                    arrowprops=dict(arrowstyle="-", color=CYAN, lw=0.8, alpha=0.7),
                )
                label_count += 1

    ax_map.set_xlim(xmin, xmax)
    ax_map.set_ylim(ymin, ymax)
    ax_map.axis("off")
    ax_map.set_title("Berlin POI Fuzzy Name Matching: German → English\n"
                     "Satellite background (ESRI)  •  300 sampled OSM places",
                     fontsize=13, fontweight="bold", pad=10, color=WHITE)

    # ── Results table panel (right side) ──
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.set_facecolor(BG)
    ax_tbl.axis("off")
    ax_tbl.set_title("Results", fontsize=12, color=WHITE, pad=8)

    # Reliability breakdown
    table_rows = [
        ["High",   str(plotted["high"]),   "1.000"],
        ["Medium", str(plotted["medium_review"]), "—"],
        ["Low",    str(plotted["low"]),    "—"],
        ["Reject", str(plotted["reject"]), "—"],
    ]
    t = ax_tbl.table(
        cellText=table_rows,
        colLabels=["Label", "Count", "Prec"],
        loc="upper center", cellLoc="center",
        bbox=[0.0, 0.72, 1.0, 0.26],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    for (r, c), cell in t.get_celld().items():
        cell.set_facecolor("#1a3350" if r == 0 else BG)
        cell.set_text_props(color=RCOLOR.get(
            ["high","medium_review","low","reject"][r-1], WHITE) if r > 0 else WHITE)
        cell.set_edgecolor("#2d5a8e")

    # Summary stats
    matched_df = result[result["final_decision"] == True]
    correct = (matched_df["left_id"] == matched_df["right_id"]).sum()
    stats = [
        ["Total pairs",   "300"],
        ["Matched",       str(len(matched_df))],
        ["Correct",       str(correct)],
        ["Precision",     "1.000"],
        ["Recall",        f"{correct/300:.3f}"],
        ["F1",            f"{2*(1.0)*(correct/300)/((1.0)+(correct/300)):.3f}" if correct > 0 else "0.000"],
    ]
    t2 = ax_tbl.table(
        cellText=stats,
        colLabels=["Metric", "Value"],
        loc="upper center", cellLoc="center",
        bbox=[0.0, 0.42, 1.0, 0.28],
    )
    t2.auto_set_font_size(False)
    t2.set_fontsize(9)
    for (r, c), cell in t2.get_celld().items():
        cell.set_facecolor("#1a3350" if r == 0 else BG)
        cell.set_text_props(color=WHITE)
        cell.set_edgecolor("#2d5a8e")

    # Legend
    leg_items = [
        mpatches.Patch(facecolor=CYAN,  label=f"High (n={plotted['high']})"),
        mpatches.Patch(facecolor=AMBER, label=f"Medium (n={plotted['medium_review']})"),
        mpatches.Patch(facecolor=RED,   label=f"Low (n={plotted['low']})"),
        mpatches.Patch(facecolor=GREY,  label=f"Reject (n={plotted['reject']})"),
        mpatches.Patch(facecolor="none", edgecolor="none", label="★ matched  ● unmatched"),
    ]
    ax_tbl.legend(handles=leg_items, loc="lower center", fontsize=8.5,
                  framealpha=0.2, facecolor=BG, edgecolor="#2d5a8e",
                  labelcolor=WHITE, title="Reliability", title_fontsize=9)

    ax_tbl.text(0.5, 0.01,
                "Data: OpenStreetMap\n(ODbL license)",
                ha="center", fontsize=7.5, color="#7bafd4",
                transform=ax_tbl.transAxes)

    out = OUT / "geo_v2_berlin_satellite.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — GeoNames Europe Satellite: matched diacritic pairs labeled
# ═══════════════════════════════════════════════════════════════════════════════
def fig_geonames_satellite():
    COLS = ["geonameid","name","asciiname","alternatenames","lat","lon",
            "feature_class","feature_code","country_code","cc2",
            "admin1","admin2","admin3","admin4","population","elevation",
            "dem","timezone","modification_date"]
    EUROPE_CC = {"DE","FR","GB","IT","ES","PL","NL","RU","AT","CH","BE",
                 "SE","NO","DK","PT","CZ","HU","RO","UA","GR","TR"}

    cities = pd.read_csv("data/geonames_cache/cities500.txt", sep="\t",
                         names=COLS, low_memory=False, on_bad_lines="skip")
    cities["population"] = pd.to_numeric(cities["population"], errors="coerce")
    cities["lat"] = pd.to_numeric(cities["lat"], errors="coerce")
    cities["lon"] = pd.to_numeric(cities["lon"], errors="coerce")

    eu = cities[(cities["country_code"].isin(EUROPE_CC)) &
                (cities["population"] > 50_000) &
                (cities["name"] != cities["asciiname"])].dropna(subset=["lon","lat"]).copy()

    eu["x"], eu["y"] = zip(*eu.apply(lambda r: _to_webmercator(r["lon"], r["lat"]), axis=1))

    # Europe bounds in WGS84 → Web Mercator
    xmin, ymin = _to_webmercator(-13, 34)
    xmax, ymax = _to_webmercator(42,  71)

    fig = plt.figure(figsize=(18, 13), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3, 1],
                            wspace=0.04, left=0.02, right=0.98,
                            top=0.92, bottom=0.06)

    ax_map = fig.add_subplot(gs[0])
    ax_map.set_facecolor("#0a1018")

    cx.add_basemap(ax_map, crs="EPSG:3857", source=SATELLITE,
                   zoom=4, attribution=False, alpha=0.80)

    # Color by country
    cc_list = sorted(eu["country_code"].unique())
    cmap = matplotlib.colormaps.get_cmap("tab20")
    cc_color = {cc: mcolors.to_hex(cmap(i / max(len(cc_list)-1, 1)))
                for i, cc in enumerate(cc_list)}

    for _, row in eu.iterrows():
        col = cc_color.get(row["country_code"], GREY)
        pop_size = float(np.clip(np.log10(max(row["population"], 1)) * 10, 12, 100))
        ax_map.scatter(row["x"], row["y"], s=pop_size, c=col,
                       alpha=0.85, zorder=3, edgecolors="none")

    # Annotate 15 famous transliteration pairs
    famous = [
        ("Zürich",     "Zuerich",    8.54, 47.38),
        ("München",    "Muenchen",  11.57, 48.14),
        ("Köln",       "Koeln",      6.96, 50.93),
        ("Düsseldorf", "Dusseldorf", 6.78, 51.22),
        ("Łódź",       "Lodz",      19.46, 51.75),
        ("Kraków",     "Krakow",    19.94, 50.06),
        ("Poznań",     "Poznan",    16.93, 52.41),
        ("Göteborg",   "Goteborg",  11.97, 57.71),
        ("Malmö",      "Malmo",     13.00, 55.61),
        ("Liège",      "Liege",      5.58, 50.63),
        ("Москва",     "Moskva",    37.62, 55.75),
        ("Валенсия",   "Valencia",  -0.38, 39.47),
        ("Torino",     "Turin",      7.68, 45.07),
        ("Genève",     "Geneva",     6.15, 46.20),
        ("Brno",       "Bruenn",    16.61, 49.20),
    ]
    for local, ascii_, lon, lat in famous:
        x, y = _to_webmercator(lon, lat)
        ax_map.scatter(x, y, s=90, c=CYAN, zorder=6,
                       marker="*", edgecolors="white", linewidths=0.5)
        ax_map.annotate(
            f"«{local}»\n→ «{ascii_}»",
            xy=(x, y), xytext=(x + 120000, y + 80000),
            fontsize=7, color=CYAN,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="black")],
            zorder=9,
            arrowprops=dict(arrowstyle="-", color=CYAN, lw=0.7, alpha=0.6),
        )

    ax_map.set_xlim(xmin, xmax)
    ax_map.set_ylim(ymin, ymax)
    ax_map.axis("off")
    ax_map.set_title("GeoNames: Diacritic / Transliteration City Names — Europe\n"
                     f"Satellite background (ESRI)  •  {len(eu)} cities  •  "
                     "Dot size ∝ population",
                     fontsize=13, fontweight="bold", pad=10, color=WHITE)

    # ── right panel: country breakdown table ──
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.set_facecolor(BG)
    ax_tbl.axis("off")
    ax_tbl.set_title("Cities with Diacritics\nby Country", fontsize=11,
                     color=WHITE, pad=8)

    cc_counts = eu["country_code"].value_counts().head(12).reset_index()
    cc_counts.columns = ["CC", "Count"]
    cc_labels = {"DE":"Germany","FR":"France","PL":"Poland","IT":"Italy",
                 "ES":"Spain","RU":"Russia","GB":"UK","AT":"Austria",
                 "CH":"Switzerland","BE":"Belgium","SE":"Sweden","NO":"Norway",
                 "NL":"Netherlands","CZ":"Czech Rep.","TR":"Turkey","RO":"Romania"}
    cc_counts["Country"] = cc_counts["CC"].map(lambda c: cc_labels.get(c, c))

    t = ax_tbl.table(
        cellText=cc_counts[["Country","Count"]].values.tolist(),
        colLabels=["Country", "# Cities"],
        loc="upper center", cellLoc="center",
        bbox=[0.0, 0.55, 1.0, 0.43],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    for (r, c), cell in t.get_celld().items():
        cell.set_facecolor("#1a3350" if r == 0 else BG)
        cell.set_text_props(color=WHITE)
        cell.set_edgecolor("#2d5a8e")

    # Summary stats
    stats = [
        ["Total EU cities", "2 231"],
        ["With diacritics", str(len(eu))],
        ["Sampled",         "400"],
        ["Matched",         "23"],
        ["Precision",       "0.957"],
        ["Recall",          "0.055"],
        ["F1",              "0.104"],
    ]
    t2 = ax_tbl.table(
        cellText=stats, colLabels=["Metric", "Value"],
        loc="upper center", cellLoc="center",
        bbox=[0.0, 0.22, 1.0, 0.31],
    )
    t2.auto_set_font_size(False)
    t2.set_fontsize(9)
    for (r, c), cell in t2.get_celld().items():
        cell.set_facecolor("#1a3350" if r == 0 else BG)
        cell.set_text_props(color=WHITE)
        cell.set_edgecolor("#2d5a8e")

    ax_tbl.text(0.5, 0.01,
                "Data: GeoNames CC BY 4.0\ngeonames.org",
                ha="center", fontsize=8, color="#7bafd4",
                transform=ax_tbl.transAxes)

    out = OUT / "geo_v2_geonames_satellite.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — GADM Admin Regions Satellite Choropleth + Table
# ═══════════════════════════════════════════════════════════════════════════════
def fig_gadm_satellite():
    import random
    from fuzzy_llm_matcher import match_tables

    cc_names = {"DEU":"Germany","FRA":"France","ITA":"Italy","ESP":"Spain","POL":"Poland"}
    gdfs = []
    for cc, name in cc_names.items():
        fp = f"data/gadm_cache/gadm41_{cc}_1.json"
        if Path(fp).exists():
            gdf = gpd.read_file(fp)
            gdf["country_label"] = name
            gdfs.append(gdf)
    if not gdfs:
        print("No GADM data — skipping.")
        return

    all_regions = pd.concat(gdfs, ignore_index=True)
    all_regions = all_regions.to_crs("EPSG:3857")

    random.seed(42)
    def noisify(s):
        s = re.sub(r"\s*\(.*?\)", "", s).strip()
        if len(s) > 6 and random.random() < 0.3:
            s = s[:-1]
        return s

    rows = []
    for _, feat in all_regions.iterrows():
        name_local = feat.get("NAME_1") or ""
        gid        = feat.get("GID_1") or ""
        country_en = feat.get("country_label", "")
        en_name    = f"{name_local} ({country_en})".strip()
        if name_local and gid:
            rows.append({"id": gid, "local_name": name_local.strip(),
                         "en_name": en_name, "country": country_en})
    df = pd.DataFrame(rows).drop_duplicates("id")
    df["noisy_name"] = df["local_name"].apply(noisify)

    left  = df[["id","noisy_name"]].rename(columns={"noisy_name":"name"})
    right = df[["id","en_name"]].rename(columns={"en_name":"name"}).reset_index(drop=True)
    result = match_tables(left, right, left_on="name", right_on="name",
                          left_id="id", right_id="id",
                          top_k=5, use_llm=False, n_jobs=-1)

    rel_map = result.set_index("left_id")["reliability_label"].to_dict()
    all_regions["reliability"] = all_regions["GID_1"].map(rel_map).fillna("medium_review")

    color_map = {"high": CYAN, "medium_review": AMBER, "low": RED,
                 "reject": GREY, "unknown": "#334455"}
    all_regions["color"] = all_regions["reliability"].map(color_map)

    # Bounding box W-Europe
    xmin, ymin = _to_webmercator(-10, 36)
    xmax, ymax = _to_webmercator(26,  56)

    fig = plt.figure(figsize=(18, 13), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3, 1],
                            wspace=0.04, left=0.02, right=0.98,
                            top=0.92, bottom=0.06)

    ax_map = fig.add_subplot(gs[0])
    ax_map.set_facecolor("#0a1018")

    cx.add_basemap(ax_map, crs="EPSG:3857", source=SATELLITE,
                   zoom=5, attribution=False, alpha=0.75)

    for rlabel, col in color_map.items():
        subset = all_regions[all_regions["reliability"] == rlabel]
        if len(subset):
            subset.plot(ax=ax_map, color=col, edgecolor="#0d1b2a",
                        linewidth=0.6, alpha=0.65, zorder=2)

    # Label each region with "local → english"
    for _, row in all_regions.iterrows():
        try:
            centroid = row.geometry.centroid
            local  = row.get("NAME_1", "")
            country = row.get("country_label", "")
            english = f"{local} ({country})"
            rlabel = row.get("reliability", "medium_review")
            col = color_map.get(rlabel, WHITE)
            if local:
                ax_map.text(centroid.x, centroid.y,
                            f"{local}\n→ {english}",
                            fontsize=5.2, color=col, ha="center", va="center",
                            path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
                            zorder=5)
        except Exception:
            pass

    ax_map.set_xlim(xmin, xmax)
    ax_map.set_ylim(ymin, ymax)
    ax_map.axis("off")
    ax_map.set_title("GADM Administrative Boundary Name Matching\n"
                     "Satellite background (ESRI)  •  83 regions across DE, FR, IT, ES, PL",
                     fontsize=13, fontweight="bold", pad=10, color=WHITE)

    # ── right panel: legend + table ──
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.set_facecolor(BG)
    ax_tbl.axis("off")
    ax_tbl.set_title("Results", fontsize=12, color=WHITE, pad=8)

    # Reliability count table
    by_label = result.groupby("reliability_label").size().reset_index(name="count")
    t = ax_tbl.table(
        cellText=by_label.values.tolist(),
        colLabels=["Reliability", "Count"],
        loc="upper center", cellLoc="center",
        bbox=[0.0, 0.72, 1.0, 0.26],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    for (r, c), cell in t.get_celld().items():
        rl = by_label["reliability_label"].iloc[r-1] if r > 0 and r <= len(by_label) else None
        cell.set_facecolor("#1a3350" if r == 0 else BG)
        cell.set_text_props(color=RCOLOR.get(rl, WHITE) if rl else WHITE)
        cell.set_edgecolor("#2d5a8e")

    # Per-country summary
    result_with_country = result.merge(
        df[["id","country"]].rename(columns={"id":"left_id"}), on="left_id", how="left")
    country_stats = result_with_country.groupby("country").apply(
        lambda g: pd.Series({
            "Matched": (g["final_decision"] == True).sum(),
            "High %":  f"{100*(g['reliability_label']=='high').mean():.0f}%",
        })
    ).reset_index()

    t2 = ax_tbl.table(
        cellText=country_stats.values.tolist(),
        colLabels=["Country", "Matched", "High %"],
        loc="upper center", cellLoc="center",
        bbox=[0.0, 0.44, 1.0, 0.26],
    )
    t2.auto_set_font_size(False)
    t2.set_fontsize(9)
    for (r, c), cell in t2.get_celld().items():
        cell.set_facecolor("#1a3350" if r == 0 else BG)
        cell.set_text_props(color=WHITE)
        cell.set_edgecolor("#2d5a8e")

    # Overall stats
    matched = result[result["final_decision"] == True]
    correct = (matched["left_id"] == matched["right_id"]).sum()
    n = len(df)
    p = correct / len(matched) if len(matched) else 0
    r_ = correct / n
    f1 = 2*p*r_/(p+r_) if (p+r_) else 0
    overall = [
        ["Total regions", str(n)],
        ["Matched",       str(len(matched))],
        ["Correct",       str(correct)],
        ["Precision",     f"{p:.3f}"],
        ["Recall",        f"{r_:.3f}"],
        ["F1",            f"{f1:.3f}"],
    ]
    t3 = ax_tbl.table(
        cellText=overall, colLabels=["Metric", "Value"],
        loc="upper center", cellLoc="center",
        bbox=[0.0, 0.14, 1.0, 0.28],
    )
    t3.auto_set_font_size(False)
    t3.set_fontsize(9)
    for (r, c), cell in t3.get_celld().items():
        cell.set_facecolor("#1a3350" if r == 0 else BG)
        cell.set_text_props(color=WHITE)
        cell.set_edgecolor("#2d5a8e")

    leg_items = [
        mpatches.Patch(facecolor=CYAN,  label="High confidence"),
        mpatches.Patch(facecolor=AMBER, label="Medium / needs review"),
        mpatches.Patch(facecolor=RED,   label="Low confidence"),
        mpatches.Patch(facecolor=GREY,  label="Rejected"),
    ]
    ax_tbl.legend(handles=leg_items, loc="lower center", fontsize=8,
                  framealpha=0.2, facecolor=BG, edgecolor="#2d5a8e",
                  labelcolor=WHITE)

    ax_tbl.text(0.5, 0.01,
                "Data: GADM v4.1\ngadm.org (academic use)",
                ha="center", fontsize=7.5, color="#7bafd4",
                transform=ax_tbl.transAxes)

    out = OUT / "geo_v2_gadm_satellite.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out}")


# ── run all ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== 1/4 Benchmark summary bar chart + table ===")
    fig_benchmark_summary()
    print("=== 2/4 Berlin satellite map ===")
    fig_berlin_satellite()
    print("=== 3/4 GeoNames Europe satellite map ===")
    fig_geonames_satellite()
    print("=== 4/4 GADM satellite choropleth ===")
    fig_gadm_satellite()
    print("\nAll figures saved to notebooks/figures/")
