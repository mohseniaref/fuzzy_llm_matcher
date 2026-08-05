"""GADM administrative boundary name matching: local name vs English/multilingual.

Uses the GADM VARNAME_1 field (genuine alternative / English names) as the
canonical right side, and the local NAME_1 as the noisy left side.

Real matching challenges:
  Bayern        → Bavaria
  Niedersachsen → LowerSaxony
  Thüringen     → Thuringia
  Toscana       → Tuscany (Toscana → Toscane|Tuscany|Toskana)
  Bretagne      → Brittany
  Katalonia     → Catalonia
  Andalucía     → Andalusia

Run:
    python examples/geo_gadm_admin_names.py
"""
from __future__ import annotations
import json, urllib.request
from pathlib import Path
import pandas as pd
import geopandas as gpd
from fuzzy_llm_matcher import match_tables

GADM_URLS = {
    "DEU": "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_DEU_1.json",
    "FRA": "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_FRA_1.json",
    "ITA": "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ITA_1.json",
    "ESP": "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ESP_1.json",
    "POL": "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_POL_1.json",
}
CACHE_DIR = Path("data/gadm_cache")

# Hand-curated English names for regions where VARNAME_1 is missing (NA)
MANUAL_EN = {
    # France (VARNAME_1 mostly empty in GADM)
    "Bretagne":                 "Brittany",
    "Normandie":                "Normandy",
    "Occitanie":                "Occitania",
    "Provence-Alpes-CôtedAzur": "Provence-Alpes-Cote d'Azur",
    "Île-de-France":            "Ile-de-France",
    "GrandEst":                 "Grand Est",
    "Centre-ValdeLoire":        "Centre-Val de Loire",
    "Bourgogne-Franche-Comté":  "Bourgogne-Franche-Comte",
    "Auvergne-Rhône-Alpes":     "Auvergne-Rhone-Alpes",
    "HauvtssdeFrance":          "Hauts-de-France",
    "Hauts-deFrance":           "Hauts-de-France",
    # Poland (all Polish, VARNAME_1 empty)
    "Dolnośląskie":    "Lower Silesian",
    "Kujawsko-Pomorskie": "Kuyavian-Pomeranian",
    "Łódzkie":         "Lodz",
    "Małopolskie":     "Lesser Poland",
    "Mazowieckie":     "Masovian",
    "Opolskie":        "Opole",
    "Podkarpackie":    "Subcarpathian",
    "Podlaskie":       "Podlaskie",
    "Pomorskie":       "Pomeranian",
    "Śląskie":         "Silesian",
    "Świętokrzyskie":  "Holy Cross",
    "Warmińsko-Mazurskie": "Warmian-Masurian",
    "Wielkopolskie":   "Greater Poland",
    "Zachodniopomorskie": "West Pomeranian",
    "Lubelskie":       "Lublin",
    "Lubuskie":        "Lubusz",
}


def _first_varname(varname_field: str) -> str | None:
    """Extract the first entry from pipe/comma-separated VARNAME_1."""
    if not varname_field or str(varname_field).strip() in ("NA", "nan", ""):
        return None
    # Take first pipe-separated variant, strip trailing pipe/space
    first = str(varname_field).split("|")[0].strip().rstrip("|").strip()
    # Insert spaces before uppercase letters that follow lowercase (CamelCase fix)
    import re
    first = re.sub(r'([a-z])([A-Z])', r'\1 \2', first)
    return first if first else None


def load_gadm_pairs() -> pd.DataFrame:
    rows = []
    for cc, url in GADM_URLS.items():
        fp = CACHE_DIR / f"gadm41_{cc}_1.json"
        if not fp.exists():
            print(f"Downloading {cc}...")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as r:
                fp.write_bytes(r.read())
        gdf = gpd.read_file(fp)
        for _, feat in gdf.iterrows():
            local  = str(feat.get("NAME_1") or "").strip()
            gid    = str(feat.get("GID_1") or "").strip()
            varraw = feat.get("VARNAME_1") or ""
            en     = _first_varname(varraw) or MANUAL_EN.get(local)
            if local and gid and en and local != en:
                rows.append({"id": gid, "local": local, "english": en,
                             "country": cc})
    df = pd.DataFrame(rows).drop_duplicates("id")
    print(f"Loaded {len(df)} regions with genuine local↔English name pairs")
    return df


def run():
    df = load_gadm_pairs()
    if df.empty:
        print("No pairs found.")
        return

    print("\nSample pairs:")
    print(df[["country","local","english"]].head(20).to_string(index=False))

    left  = df[["id","local"]].rename(columns={"local":"name"})
    right = df[["id","english"]].rename(columns={"english":"name"}).reset_index(drop=True)

    result = match_tables(
        left, right,
        left_on="name", right_on="name",
        left_id="id", right_id="id",
        top_k=5, use_llm=False, n_jobs=-1,
    )

    matched   = result[result["final_decision"] == True]
    correct   = (matched["left_id"] == matched["right_id"]).sum()
    precision = correct / len(matched) if len(matched) else 0
    recall    = correct / len(df)
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0

    print(f"\n--- GADM Admin Names (genuine local ↔ English) ---")
    print(f"Pairs: {len(df)}, Matched: {len(matched)}, Correct: {correct}")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    by_label = result.groupby("reliability_label").size()
    print("By label:\n", by_label.to_string())

    print("\nMatched pairs (left=local, right=english):")
    hits = matched.merge(df[["id","local","english","country"]].rename(columns={"id":"left_id"}),
                         on="left_id", how="left")
    print(hits[["country","local","english","fuzzy_score","reliability_label"]].to_string(index=False))

    out = Path("notebooks/results_gadm_admin_names.md")
    out.write_text(f"""# GADM Administrative Boundary Name Matching

**Real challenge**: local-language region names vs genuine English translations.

Examples: `Bayern → Bavaria`, `Niedersachsen → Lower Saxony`,
`Thüringen → Thuringia`, `Toscana → Tuscany`, `Bretagne → Brittany`,
`Andalucía → Andalusia`, `Dolnośląskie → Lower Silesian`

Countries: Germany (16), France (13), Italy (20), Spain (17), Poland (16)
Total pairs: **{len(df)}** regions

| Metric | Value |
|--------|-------|
| Pairs | {len(df)} |
| Matched | {len(matched)} |
| Correct (same GADM GID) | {correct} |
| Precision | {precision:.3f} |
| Recall | {recall:.3f} |
| F1 | {f1:.3f} |

## By reliability label
```
{by_label.to_string()}
```

## Matched pairs
{hits[["country","local","english","fuzzy_score","reliability_label"]].to_markdown(index=False)}

## Data source
GADM version 4.1 (https://gadm.org). Free for academic and non-commercial use.
""")
    print(f"\nReport saved: {out}")
    return result, df


if __name__ == "__main__":
    run()
