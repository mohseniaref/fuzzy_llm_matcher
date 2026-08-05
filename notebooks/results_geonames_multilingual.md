# GeoNames Multilingual City-Name Matching

Matching local-language city names (with diacritics/umlauts/Cyrillic) against
their ASCII transliterations and English alternate names.
400 European cities (population > 50k).

Examples: `Zürich → Zuerich`, `München → Muenchen`, `Москва → Moskva`,
`Łódź → Lodz`, `Köln → Koeln`

| Metric | Value |
|--------|-------|
| Pairs evaluated | 400 |
| Matched | 23 |
| Correct (same GeoNames ID) | 22 |
| Precision | 0.957 |
| Recall | 0.055 |
| F1 | 0.104 |

## By reliability label
```
reliability_label
high              23
low              121
medium_review    256
```

## Sample matched pairs
| left_value                | right_value               |   fuzzy_score | reliability_label   |
|:--------------------------|:--------------------------|--------------:|:--------------------|
| Kholodna Hora             | Kholodna Gora             |       92.3077 | high                |
| San Bartolomé de Tirajana | San Bartolome de Tirajana |       96      | high                |
| Stoke-on-Trent            | Stok-on-Trehnt            |       92.8571 | high                |
| Stafford                  | Staefford                 |       94.1176 | high                |
| Kettering                 | Ketering                  |       94.1176 | high                |
| Islington                 | Islingtono                |       94.7368 | high                |
| Horsham                   | Khorsham                  |       93.3333 | high                |
| Purmerend                 | Pjurmerend                |       94.7368 | high                |
| Haarlem                   | Chaarlem                  |       93.3333 | high                |
| Plauen                    | Plauehn                   |       92.3077 | high                |
| Lippstadt                 | Lippshtadt                |       94.7368 | high                |
| Saint-Maur-des-Fossés     | Saint-Maur-des-Fosses     |       95.2381 | high                |
| Rivas-Vaciamadrid         | Rivas Vaciamadrid         |       94.1176 | high                |
| Portici                   | Portichi                  |       93.3333 | high                |
| Andria                    | Andrija                   |       92.3077 | high                |

## Data source
GeoNames cities500 (https://www.geonames.org/export/dump/cities500.zip).
License: CC BY 4.0.
