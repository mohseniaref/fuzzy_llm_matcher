"""GADM administrative boundary name matching: local name vs English.

Downloads GADM level-1 (states/provinces) for Germany, France, Italy,
Spain, and Poland. Matches local-language names against English variants.

GADM data: https://gadm.org/data.html (free for academic/non-commercial use)
We use the GeoJSON simplified version (small files).

Run:
    python examples/geo_gadm_admin_names.py
"""
from __future__ import annotations
import io, json, urllib.request, zipfile
from pathlib import Path
import pandas as pd
from fuzzy_llm_matcher import match_tables

# GADM level-1 GeoJSON for selected countries (small files ~100-300KB each)
GADM_URLS = {
    "DEU": "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_DEU_1.json",
    "FRA": "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_FRA_1.json",
    "ITA": "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ITA_1.json",
    "ESP": "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ESP_1.json",
    "POL": "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_POL_1.json",
}
CACHE_DIR = Path("data/gadm_cache")


def _fetch_gadm(country: str, url: str) -> list[dict]:
    cache_file = CACHE_DIR / f"gadm41_{country}_1.json"
    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
    else:
        print(f"Downloading GADM {country}...")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.load(r)
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"  Could not download {country}: {e}")
            return []
    rows = []
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        name_local = props.get("NAME_1") or props.get("VARNAME_1") or ""
        name_en    = props.get("ENGTYPE_1") or ""
        gid        = props.get("GID_1") or props.get("HASC_1") or ""
        # Build English variant: "NAME_1, Country"
        country_en = {"DEU":"Germany","FRA":"France","ITA":"Italy","ESP":"Spain","POL":"Poland"}.get(country, country)
        full_local = name_local.strip()
        full_en    = f"{name_local} ({country_en})".strip()
        if full_local and full_local != full_en:
            rows.append({"id": gid, "local_name": full_local, "en_name": full_en, "country": country})
    return rows


def run():
    all_rows = []
    for cc, url in GADM_URLS.items():
        all_rows.extend(_fetch_gadm(cc, url))

    if not all_rows:
        print("No GADM data downloaded — check connectivity or try later.")
        return

    df = pd.DataFrame(all_rows).drop_duplicates("id")
    print(f"Total admin regions: {len(df)}")

    # Introduce realistic noise on left side: drop parenthetical, add typo
    import re, random
    random.seed(42)
    def noisify(s):
        s = re.sub(r"\s*\(.*?\)", "", s).strip()
        # Occasionally drop last letter
        if len(s) > 6 and random.random() < 0.3:
            s = s[:-1]
        return s

    df["noisy_name"] = df["local_name"].apply(noisify)

    left  = df[["id","noisy_name"]].rename(columns={"noisy_name":"name"})
    right = df[["id","en_name"]].rename(columns={"en_name":"name"}).reset_index(drop=True)

    result = match_tables(
        left, right,
        left_on="name", right_on="name",
        left_id="id", right_id="id",
        top_k=5, use_llm=False, n_jobs=-1,
    )

    matched   = result[result["final_decision"] == True]
    correct   = (matched["left_id"] == matched["right_id"]).sum()
    precision = correct / len(df) if len(df) else 0  # denominator = all regions
    recall    = correct / len(df) if len(df) else 0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0

    print(f"\n--- GADM Admin Names Results ---")
    print(f"Regions: {len(df)}, Matched: {len(matched)}, Correct: {correct}")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    by_label = result.groupby("reliability_label").size()
    print("By label:\n", by_label.to_string())

    out = Path("notebooks/results_gadm_admin_names.md")
    out.write_text(f"""# GADM Administrative Boundary Name Matching

Local-language names vs English-annotated variants for level-1 administrative
regions in Germany, France, Italy, Spain, and Poland ({len(df)} regions).

| Metric | Value |
|--------|-------|
| Regions | {len(df)} |
| Matched | {len(matched)} |
| Correct (same GADM GID) | {correct} |
| Precision | {precision:.3f} |
| Recall | {recall:.3f} |
| F1 | {f1:.3f} |

## By reliability label
```
{by_label.to_string()}
```

## Data source
GADM version 4.1 (https://gadm.org). Free for academic and non-commercial use.
""")
    print(f"Report saved: {out}")
    return result

if __name__ == "__main__":
    run()
