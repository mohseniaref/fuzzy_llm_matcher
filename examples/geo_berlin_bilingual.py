"""Berlin POI name matching: German (OSM name) vs English (OSM name:en).

Downloads ~2000 named places in Berlin from Overpass API and matches
the German name against the English name:en tag using fuzzy_llm_matcher.

Run:
    python examples/geo_berlin_bilingual.py
"""
from __future__ import annotations
import json, urllib.request, urllib.parse
from pathlib import Path
import pandas as pd
from fuzzy_llm_matcher import match_tables

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
QUERY = """
[out:json][timeout:60];
area["name"="Berlin"]["boundary"="administrative"]["admin_level"="4"]->.berlin;
(
  node["name"]["name:en"](area.berlin);
  way["name"]["name:en"](area.berlin);
);
out tags 1500;
"""

def fetch_berlin_bilingual(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        print(f"Using cached data: {cache_path}")
        with open(cache_path) as f:
            data = json.load(f)
    else:
        print("Querying Overpass API for Berlin bilingual POIs...")
        encoded = urllib.parse.urlencode({"data": QUERY}).encode()
        req = urllib.request.Request(OVERPASS_URL, data=encoded, headers={"User-Agent": "fuzzy_llm_matcher_demo/0.1"})
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.load(r)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(data, f)
        print(f"Cached to {cache_path}")

    rows = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        de_name = tags.get("name", "").strip()
        en_name = tags.get("name:en", "").strip()
        ptype   = tags.get("amenity") or tags.get("leisure") or tags.get("tourism") or tags.get("place") or "other"
        if de_name and en_name and de_name != en_name:
            rows.append({"id": str(el["id"]), "de_name": de_name, "en_name": en_name, "type": ptype})
    df = pd.DataFrame(rows).drop_duplicates("de_name")
    print(f"Pairs where German ≠ English: {len(df)}")
    return df


def run():
    cache = Path("data/berlin_bilingual_cache.json")
    df = fetch_berlin_bilingual(cache)

    # Sample up to 300 for speed (keep distribution)
    sample = df.sample(min(300, len(df)), random_state=42).reset_index(drop=True)

    left  = sample[["id","de_name"]].rename(columns={"de_name":"name"})
    right = df[["id","en_name"]].rename(columns={"id":"id","en_name":"name"}).reset_index(drop=True)

    result = match_tables(
        left, right,
        left_on="name", right_on="name",
        left_id="id", right_id="id",
        top_k=5, use_llm=False, n_jobs=-1,
    )

    # Evaluate: correct match = same OSM id
    matched = result[result["final_decision"] == True]
    correct = (matched["left_id"] == matched["right_id"]).sum()
    precision = correct / len(matched) if len(matched) else 0
    recall    = correct / len(sample)
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0

    print(f"\n--- Berlin Bilingual Results ---")
    print(f"Pairs evaluated : {len(sample)}")
    print(f"Matched          : {len(matched)}")
    print(f"Correct          : {correct}")
    print(f"Precision        : {precision:.3f}")
    print(f"Recall           : {recall:.3f}")
    print(f"F1               : {f1:.3f}")

    by_label = result.groupby("reliability_label").size()
    print("\nBy reliability label:\n", by_label.to_string())

    out = Path("notebooks/results_berlin_bilingual.md")
    out.write_text(f"""# Berlin Bilingual POI Name Matching

German OSM `name` vs English `name:en` tag — {len(sample)} sampled pairs.

| Metric | Value |
|--------|-------|
| Pairs evaluated | {len(sample)} |
| Matched | {len(matched)} |
| Correct (same OSM id) | {correct} |
| Precision | {precision:.3f} |
| Recall | {recall:.3f} |
| F1 | {f1:.3f} |

## By reliability label
```
{by_label.to_string()}
```

## Example matches
{matched[["left_id","right_id","fuzzy_score","reliability_label"]].head(10).to_markdown(index=False)}

## Data source
OpenStreetMap via Overpass API — `name` (German) vs `name:en` (English) tags,
Berlin administrative area. License: ODbL.
""")
    print(f"\nReport saved: {out}")
    return result

if __name__ == "__main__":
    run()
