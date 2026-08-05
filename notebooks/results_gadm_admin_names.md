# GADM Administrative Boundary Name Matching

**Real challenge**: local-language region names vs genuine English translations.

Examples: `Bayern → Bavaria`, `Niedersachsen → Lower Saxony`,
`Thüringen → Thuringia`, `Toscana → Tuscany`, `Bretagne → Brittany`,
`Andalucía → Andalusia`, `Dolnośląskie → Lower Silesian`

Countries: Germany (16), France (13), Italy (20), Spain (17), Poland (16)
Total pairs: **70** regions

| Metric | Value |
|--------|-------|
| Pairs | 70 |
| Matched | 10 |
| Correct (same GADM GID) | 10 |
| Precision | 1.000 |
| Recall | 0.143 |
| F1 | 0.250 |

## By reliability label
```
reliability_label
high             10
low              32
medium_review    20
reject            8
```

## Matched pairs
| country   | local                   | english                 |   fuzzy_score | reliability_label   |
|:----------|:------------------------|:------------------------|--------------:|:--------------------|
| FRA       | Auvergne-Rhône-Alpes    | Auvergne-Rhone-Alpes    |       95      | high                |
| FRA       | Bourgogne-Franche-Comté | Bourgogne-Franche-Comte |       95.6522 | high                |
| FRA       | Centre-ValdeLoire       | Centre-Val de Loire     |       94.4444 | high                |
| FRA       | GrandEst                | Grand Est               |       94.1176 | high                |
| FRA       | Île-de-France           | Ile-de-France           |       92.3077 | high                |
| ITA       | Marche                  | Marches                 |       92.3077 | high                |
| ITA       | Piemonte                | Piemont                 |       93.3333 | high                |
| ITA       | Trentino-AltoAdige      | Trentino-Alto Adige     |       97.2973 | high                |
| ITA       | Emilia-Romagna          | Emilia Romagna          |       92.8571 | high                |
| ITA       | Friuli-VeneziaGiulia    | Friuli Venezia Giulia   |       92.6829 | high                |

## Data source
GADM version 4.1 (https://gadm.org). Free for academic and non-commercial use.
