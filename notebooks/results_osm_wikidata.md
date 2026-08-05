# OSM ↔ Wikidata City-Name Linking

Matching city names from Wikidata English labels (with parenthetical disambiguation
stripped) against the full Wikidata label. 300 European cities.

| Metric | Value |
|--------|-------|
| Pairs evaluated | 300 |
| Matched | 298 |
| Correct (same Wikidata QID) | 298 |
| Precision | 1.000 |
| Recall | 0.993 |
| F1 | 0.997 |

## By reliability label
```
reliability_label
high             298
medium_review      2
```

## Data source
Wikidata SPARQL endpoint (https://query.wikidata.org).
License: CC0.
