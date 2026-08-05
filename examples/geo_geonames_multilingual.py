"""GeoNames multilingual city-name matching.

Uses the cities500 dataset (already downloaded for other benchmarks) to
match local-language city names (with diacritics/umlauts) against their
ASCII transliteration. Examples: Zürich→Zuerich, München→Muenchen,
Köln→Koeln, Łódź→Lodz, Москва→Moskva.

Also uses the comma-separated alternate names column to find cross-language
variant pairs (e.g. German city, English alternate name).

Run:
    python examples/geo_geonames_multilingual.py
"""
from __future__ import annotations
import urllib.request, zipfile, io
from pathlib import Path
import pandas as pd
from fuzzy_llm_matcher import match_tables

CITIES_URL = "https://download.geonames.org/export/dump/cities500.zip"
CACHE_DIR  = Path("data/geonames_cache")

COLS = ["geonameid","name","asciiname","alternatenames","lat","lon",
        "feature_class","feature_code","country_code","cc2",
        "admin1","admin2","admin3","admin4","population","elevation",
        "dem","timezone","modification_date"]

EUROPE_CC = {"DE","FR","GB","IT","ES","PL","NL","RU","AT","CH","BE",
             "SE","NO","DK","PT","CZ","HU","RO","UA","GR","TR"}


def load_cities() -> pd.DataFrame:
    cache_file = CACHE_DIR / "cities500.txt"
    if not cache_file.exists():
        print(f"Downloading {CITIES_URL} ...")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(CITIES_URL, timeout=120) as r:
            data = r.read()
        zf = zipfile.ZipFile(io.BytesIO(data))
        txt_name = [n for n in zf.namelist() if n.endswith(".txt")][0]
        cache_file.write_bytes(zf.read(txt_name))
        print(f"  Saved to {cache_file}")
    else:
        print(f"Using cached {cache_file}")
    return pd.read_csv(cache_file, sep="\t", names=COLS, low_memory=False, on_bad_lines="skip")


def run():
    cities = load_cities()
    cities["population"] = pd.to_numeric(cities["population"], errors="coerce")

    eu = cities[
        (cities["country_code"].isin(EUROPE_CC)) &
        (cities["population"] > 50_000)
    ].copy()
    print(f"European cities > 50k population: {len(eu)}")

    # --- Benchmark 1: local name (with diacritics) → ASCII transliteration ---
    diacritic_pairs = eu[eu["name"] != eu["asciiname"]][
        ["geonameid","name","asciiname","country_code"]
    ].copy()
    diacritic_pairs["geonameid"] = diacritic_pairs["geonameid"].astype(str)
    print(f"Pairs with diacritics: {len(diacritic_pairs)}")

    # --- Benchmark 2: extract English alternate names from the alternatenames column ---
    # The alternatenames field is a comma-separated list of all language variants
    # We pick the first alternate name that uses only ASCII letters (rough English proxy)
    import re
    def first_ascii_alt(row):
        alts = str(row.get("alternatenames","")).split(",")
        native = row["name"]
        for a in alts:
            a = a.strip()
            if a and a != native and re.match(r'^[A-Za-z\s\-\.]+$', a) and len(a) > 2:
                return a
        return None

    eu["en_alt"] = eu.apply(first_ascii_alt, axis=1)
    cross_lang = eu[
        eu["en_alt"].notna() &
        (eu["name"] != eu["en_alt"])
    ][["geonameid","name","en_alt","country_code"]].copy()
    cross_lang["geonameid"] = cross_lang["geonameid"].astype(str)
    print(f"Cross-language pairs (local → ASCII alt): {len(cross_lang)}")

    # Combine both sets
    all_pairs = pd.concat([
        diacritic_pairs[["geonameid","name","asciiname"]].rename(columns={"asciiname":"target_name"}),
        cross_lang[["geonameid","name","en_alt"]].rename(columns={"en_alt":"target_name"}),
    ]).drop_duplicates("geonameid").reset_index(drop=True)
    print(f"Total unique city pairs: {len(all_pairs)}")

    sample = all_pairs.sample(min(400, len(all_pairs)), random_state=42).reset_index(drop=True)

    left  = sample[["geonameid","name"]].rename(columns={"geonameid":"id"})
    right = all_pairs[["geonameid","target_name"]].rename(columns={"geonameid":"id","target_name":"name"}).reset_index(drop=True)

    result = match_tables(
        left, right,
        left_on="name", right_on="name",
        left_id="id", right_id="id",
        top_k=5, use_llm=False, n_jobs=-1,
    )

    matched   = result[result["final_decision"] == True]
    correct   = (matched["left_id"] == matched["right_id"]).sum()
    precision = correct / len(matched) if len(matched) else 0
    recall    = correct / len(sample)
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0

    print(f"\n--- GeoNames Multilingual Results ---")
    print(f"Pairs evaluated : {len(sample)}")
    print(f"Matched : {len(matched)}, Correct : {correct}")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    by_label = result.groupby("reliability_label").size()
    print("By label:\n", by_label.to_string())

    # Show some interesting examples
    interesting = result[result["final_decision"] == True].head(15)
    print("\nSample matched pairs:")
    print(interesting[["left_value","right_value","fuzzy_score","reliability_label"]].to_string(index=False))

    out = Path("notebooks/results_geonames_multilingual.md")
    out.write_text(f"""# GeoNames Multilingual City-Name Matching

Matching local-language city names (with diacritics/umlauts/Cyrillic) against
their ASCII transliterations and English alternate names.
{len(sample)} European cities (population > 50k).

Examples: `Zürich → Zuerich`, `München → Muenchen`, `Москва → Moskva`,
`Łódź → Lodz`, `Köln → Koeln`

| Metric | Value |
|--------|-------|
| Pairs evaluated | {len(sample)} |
| Matched | {len(matched)} |
| Correct (same GeoNames ID) | {correct} |
| Precision | {precision:.3f} |
| Recall | {recall:.3f} |
| F1 | {f1:.3f} |

## By reliability label
```
{by_label.to_string()}
```

## Sample matched pairs
{interesting[["left_value","right_value","fuzzy_score","reliability_label"]].to_markdown(index=False)}

## Data source
GeoNames cities500 (https://www.geonames.org/export/dump/cities500.zip).
License: CC BY 4.0.
""")
    print(f"\nReport saved: {out}")
    return result


if __name__ == "__main__":
    run()
