"""OpenStreetMap / GeoNames place-name fuzzy matching benchmark.

This is a real geo-entity matching use case: the same place often appears
under different spellings in different data sources (OSM vs GeoNames, local
language vs English, abbreviations, diacritics).

Examples of the problem:
    "Köln"         vs  "Cologne"           (local vs English name)
    "NYC"          vs  "New York City"      (abbreviation)
    "Saint-Denis"  vs  "St Denis"           (abbreviation + punctuation)
    "Москва"       vs  "Moscow"             (transliteration)
    "Al Qahirah"   vs  "Cairo"              (transliteration)

This script uses a **built-in hand-crafted dataset** of 40 real city-name
pairs that represent hard matching cases. This dataset is self-contained —
no download required — so the benchmark always runs offline.

An optional section (commented out) shows how to pull live data from the
OpenStreetMap Nominatim API or from GeoNames if you want to extend it.

Run:
    python examples/osm_geonames_place_matching.py
"""

from __future__ import annotations

import time

import pandas as pd

from fuzzy_llm_matcher import evaluate_matches, match_tables

# ---------------------------------------------------------------------------
# Built-in hand-crafted geo dataset (offline, no download required)
# Columns: id, name, country_code
# left  = noisy/alternative spellings (as they appear in dirty data / OSM tags)
# right = canonical GeoNames English names
# ---------------------------------------------------------------------------

LEFT_PLACES = [
    (1,  "Köln",                  "DE"),
    (2,  "München",               "DE"),
    (3,  "Nürnberg",              "DE"),
    (4,  "Düsseldorf",            "DE"),
    (5,  "NYC",                   "US"),
    (6,  "LA",                    "US"),
    (7,  "San Fran",              "US"),
    (8,  "Philly",                "US"),
    (9,  "Saint-Denis",           "FR"),
    (10, "Lyon France",           "FR"),
    (11, "Marseilles",            "FR"),
    (12, "Strasbourg-Alsace",     "FR"),
    (13, "Al Qahirah",            "EG"),
    (14, "Aleksandria",           "EG"),
    (15, "Ispahan",               "IR"),
    (16, "Teheran",               "IR"),
    (17, "Moskva",                "RU"),
    (18, "Sankt-Peterburg",       "RU"),
    (19, "Peking",                "CN"),
    (20, "Canton",                "CN"),
    (21, "Bombay",                "IN"),
    (22, "Calcutta",              "IN"),
    (23, "Madras",                "IN"),
    (24, "Bangalore",             "IN"),
    (25, "Rio de Jan.",           "BR"),
    (26, "Sao Paolo",             "BR"),
    (27, "Bogotá",                "CO"),
    (28, "Buenos Ayres",          "AR"),
    (29, "Ciudad de Mexico",      "MX"),
    (30, "Guadalahara",           "MX"),
    (31, "Instanbul",             "TR"),
    (32, "Izmir Turkey",          "TR"),
    (33, "Soul",                  "KR"),
    (34, "Busan Korea",           "KR"),
    (35, "Tokio",                 "JP"),
    (36, "Osaca",                 "JP"),
    (37, "Djakarta",              "ID"),
    (38, "Djokdjakarta",          "ID"),
    (39, "Kairo",                 "DE"),   # hard negative: German word for Cairo, country=DE
    (40, "Frankfurt am Main",     "DE"),
]

RIGHT_PLACES = [
    (101, "Cologne",              "DE"),
    (102, "Munich",               "DE"),
    (103, "Nuremberg",            "DE"),
    (104, "Dusseldorf",           "DE"),
    (105, "Frankfurt",            "DE"),
    (106, "New York City",        "US"),
    (107, "Los Angeles",          "US"),
    (108, "San Francisco",        "US"),
    (109, "Philadelphia",         "US"),
    (110, "Saint-Denis",          "FR"),
    (111, "Lyon",                 "FR"),
    (112, "Marseille",            "FR"),
    (113, "Strasbourg",           "FR"),
    (114, "Cairo",                "EG"),
    (115, "Alexandria",           "EG"),
    (116, "Isfahan",              "IR"),
    (117, "Tehran",               "IR"),
    (118, "Moscow",               "RU"),
    (119, "Saint Petersburg",     "RU"),
    (120, "Beijing",              "CN"),
    (121, "Guangzhou",            "CN"),
    (122, "Mumbai",               "IN"),
    (123, "Kolkata",              "IN"),
    (124, "Chennai",              "IN"),
    (125, "Bengaluru",            "IN"),
    (126, "Rio de Janeiro",       "BR"),
    (127, "Sao Paulo",            "BR"),
    (128, "Bogota",               "CO"),
    (129, "Buenos Aires",         "AR"),
    (130, "Mexico City",          "MX"),
    (131, "Guadalajara",          "MX"),
    (132, "Istanbul",             "TR"),
    (133, "Izmir",                "TR"),
    (134, "Seoul",                "KR"),
    (135, "Busan",                "KR"),
    (136, "Tokyo",                "JP"),
    (137, "Osaka",                "JP"),
    (138, "Jakarta",              "ID"),
    (139, "Yogyakarta",           "ID"),
]

# Ground truth: left_id -> right_id
GROUND_TRUTH = [
    (1, 101), (2, 102), (3, 103), (4, 104), (5, 106), (6, 107),
    (7, 108), (8, 109), (9, 110), (10, 111), (11, 112), (12, 113),
    (13, 114), (14, 115), (15, 116), (16, 117), (17, 118), (18, 119),
    (19, 120), (20, 121), (21, 122), (22, 123), (23, 124), (24, 125),
    (25, 126), (26, 127), (27, 128), (28, 129), (29, 130), (30, 131),
    (31, 132), (32, 133), (33, 134), (34, 135), (35, 136), (36, 137),
    (37, 138), (38, 139), (40, 105),
    # 39 (Kairo/DE) is a hard negative — intentionally not matched
]


def main() -> None:
    left = pd.DataFrame(LEFT_PLACES, columns=["id", "name", "country_code"])
    right = pd.DataFrame(RIGHT_PLACES, columns=["id", "name", "country_code"])
    true_matches = pd.DataFrame(GROUND_TRUTH, columns=["left_id", "right_id"])

    strategies = {
        "No blocking (all vs all)": dict(block_on=None, use_llm=False),
        "Block on country_code": dict(block_on="country_code", use_llm=False),
        "Block + LLM review": dict(block_on="country_code", use_llm=True),
    }

    for label, kwargs in strategies.items():
        t = time.perf_counter()
        result = match_tables(
            left, right,
            left_on="name", right_on="name",
            left_id="id", right_id="id",
            top_k=5, n_jobs=1,
            **kwargs,
        )
        elapsed = time.perf_counter() - t
        ev = evaluate_matches(result, true_matches, runtime_seconds=elapsed)
        print(f"\n{label}")
        for k, v in ev.to_dict().items():
            print(f"  {k}: {v}")

    # Show hard cases (medium_review pairs) for the blocked+LLM run
    t = time.perf_counter()
    result_full = match_tables(
        left, right,
        left_on="name", right_on="name",
        left_id="id", right_id="id",
        top_k=5, block_on="country_code", use_llm=True,
        keep_all_candidates=False,
    )
    print("\nSample output (all rows):")
    print(result_full[["left_value", "right_value", "fuzzy_score",
                        "reliability_label", "final_decision"]].to_string(index=False))


if __name__ == "__main__":
    main()
