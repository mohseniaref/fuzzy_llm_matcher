# Berlin Bilingual POI Name Matching

German OSM `name` vs English `name:en` tag — 300 sampled pairs.

| Metric | Value |
|--------|-------|
| Pairs evaluated | 300 |
| Matched | 8 |
| Correct (same OSM id) | 8 |
| Precision | 1.000 |
| Recall | 0.027 |
| F1 | 0.052 |

## By reliability label
```
reliability_label
high               8
low              129
medium_review    133
reject            30
```

## Example matches
|     left_id |    right_id |   fuzzy_score | reliability_label   |
|------------:|------------:|--------------:|:--------------------|
| 10829070905 | 10829070905 |       95      | high                |
| 11161925005 | 11161925005 |       96      | high                |
| 11200058937 | 11200058937 |       96.2963 | high                |
|    16872940 |    16872940 |       94.1176 | high                |
|  3870914569 |  3870914569 |       95      | high                |
|  5201598623 |  5201598623 |       95      | high                |
|  5827057985 |  5827057985 |       99.0099 | high                |
|     9393789 |     9393789 |       95      | high                |

## Data source
OpenStreetMap via Overpass API — `name` (German) vs `name:en` (English) tags,
Berlin administrative area. License: ODbL.
