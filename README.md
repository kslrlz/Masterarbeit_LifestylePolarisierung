# Daten und Code zur Masterarbeit

Lifestyle-Polarisierung auf Reddit, 2016 bis 2024

- alles zum Nachrechnen der Auswertung: Vektoren, abgeleitete Tabellen, Code
- Rohdaten des Pushshift-Korpus nicht enthalten und nicht nötig
- Vektoren ohne Beitragstexte und ohne Nutzer:innennamen

## Auswertung

- Arbeitsverzeichnis `Code/`, Reihenfolge der Nummern

| | | Laufzeit |
|---|---|---|
| `01_ff1.ipynb` | Konsolidierung der Landschaft | gut 3 h |
| `02_ff2.ipynb` | politische Aufladung der Cluster | rund 1 h |
| `03_ff3.ipynb` | Beitrag der einzelnen Bündel | Sekunden |
| `04_validierung.ipynb` | Zirkularität der politischen Achse | Sekunden |
| `05_abbildungen.ipynb` | alle Abbildungen des Ergebnisteils | Sekunden |

- R 4.5 mit `tidyverse`, `data.table`, `igraph`, `dbscan`, `ineq`, `spdep`, `ggrepel`
- `setup.R` setzt Pfade, Panel und Lebensstil-Etikett, `cluster_namen.R` die Clusternamen
- Ausgabe nach `Auswertung_CSV/` und `Abbildungen/`, die Dateien des letzten Laufs
  liegen dort bereits
- ohne `MA_PROJ` werden die Daten eine Ebene über `Code/` gesucht

## Woher die Daten kommen

| Code | erzeugt | Inhalt |
|---|---|---|
| nicht im Paket | `Data/Vektoren/vektoren_JAHR.txt` | Vektoren je Jahr, nativ |
| `ausrichtung/` | `SeNSe-main/…/projected_JJ_onto_16_FULL.txt` | dieselben, auf 2016 gedreht |
| `ausrichtung/` | `final_anker_16_JAHR_full.csv` | die Anker dieser Drehung |
| `achse/` | `politischer_wandel_2016_2024.csv` | Partisan-Score je Subreddit und Jahr |
| `achse/` | `Data/Vektoren/all_scores.csv` | Rohwerte des Partisan-Score |
| `cluster/` | `subreddit_clusters_aligned_2016_2024_LABEL2016.csv` | Clusterlösung je Jahr |
| `einstufung/` | `all_cluster_results_political.csv` | Einstufung des Sprachmodells |
| `woerter/` | `Data/ctfidf_top50_LABEL2016.csv` | kennzeichnende Wörter je Cluster |
| `geometrie/` | `ff1_geo_full.csv` | Kosinus-Kontrast über alle Vektorpaare |

## Die Vorstufen

- Arbeitsdateien, dokumentiert und nicht lauffertig. Liefen teils auf dem
  Rechencluster und setzen die Rohdaten voraus

- `ausrichtung/` Anker und Drehung auf 2016 (Kap. 4.3), mit den angepassten
  SeNSe-Fassungen, siehe `HERKUNFT.md`
- `achse/` politische Achse und Partisan-Score (Kap. 4.4), mit `dimen_generation.py`
  von Waller und Anderson, siehe `HERKUNFT.md`
- `cluster/` Clusterlösung auf der 2016er-Partition (Kap. 4.7)
- `einstufung/` politisch oder nicht politisch (Kap. 4.5): `llm_input_pipeline_safe`
  → `multi_llm_pipeline` → `kuerze_prompts` → `hpc_klassifikation` → `Extract_LLM`
- `woerter/` kennzeichnende Wörter (Kap. 5.5): `compute_tfidf_monthly` →
  `_ctfidf_parts` → `_ctfidf_merge` → `compute_ctfidf_label2016` →
  `_ctfidf_top50_label2016` → `label_cluster`
- `geometrie/` dieselben Größen wie in `01_ff1`, aber exakt über alle Vektorpaare
  statt über eine Stichprobe
- `validierung/` Etiketten der 120 handkodierten Subreddit-Jahre und die Rechnung
  daraus (Kap. 4.5). Die kodierten Texte liegen aus Datenschutzgründen nicht bei
- `robustheit/` Übereinstimmung der Clusterläufe (Kap. 4.7) und die auf
  residualisierten Vektoren wiederholte Clusterpipeline (Kap. 7)

## Nicht im Paket

- der Weg vom Rohkorpus zu den Vektoren: Parsen, Sprach- und Kontenfilter, Training
  der Embeddings (Kap. 3). Die Vektoren selbst liegen bei
- Zugangsdaten. `hpc_klassifikation.py` erwartet `HF_TOKEN` in der Umgebung
