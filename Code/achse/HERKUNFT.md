# Politische Achse

`Political_Axis_Analysis.ipynb` bildet die politische Achse aus den zehn Ankerpaaren
und berechnet daraus den Partisan-Score je Subreddit und Jahr (vgl. Kapitel 4.4).
Ergebnis ist die mitgelieferte Datei `politischer_wandel_2016_2024.csv`.

`dimen_generation.py` stammt aus dem Code zu Waller und Anderson (2021),
Repositorium `CSSLab/social-dimensions`, Datei `full_code/dimen_generation.py`.
Die Datei ist unverändert übernommen; Urheber sind die Autoren. Das Notebook nutzt
daraus `DimenGenerator` und `score_embedding`.

Die Zeile `sys.path.append(...)` im Notebook zeigt auf die Ordnerstruktur des
ursprünglichen Repositoriums. Hier liegen beide Dateien nebeneinander.
