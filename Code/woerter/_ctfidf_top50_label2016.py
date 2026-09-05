"""
Reparatur-Export für den LABEL2016-Re-Pool vom 21.07.2026.

Die Aggregation ist fertig und liegt in /var/tmp/label2016_agg.duckdb
(108/108 Monate, 1,618 Mrd. Zeilen cyc, 2016-2024). Gescheitert ist nur der
Schluss-Schritt in compute_ctfidf_label2016.py: ein einziges COPY mit globalem
ORDER BY über alle neun Jahre, das ~34 GB Temp-Spill erzeugt hat und dann
gestorben ist.

Dieses Skript macht denselben Export JAHRESWEISE. Die c-TF-IDF-Formel ist
wörtlich aus dem Originalskript übernommen. Da dort jede CTE ohnehin nach
year gruppiert (cc, totals, f_t, avg_words), ist die Jahres-Schleife
rechnerisch identisch und keine Näherung.

Unterschiede zum Original, alle bewusst:
  - kein globales ORDER BY  -> der Grund des Absturzes
  - schreibt NUR die Top-50-CSV, nicht das 596-Mio-Zeilen-Parquet.
    Für die FF3-Cluster-Themen ist die Top-50-Liste das benötigte Produkt.
    Wer die volle Datei für Wortschatz-Drift braucht: VOLL_PARQUET = True.
  - öffnet die 19-GB-Datenbank per ATTACH READ_ONLY, schreibt also nie hinein
  - Teildateien pro Jahr, damit ein Abbruch nicht alles verliert (Wiederaufnahme)
  - liest NICHT /mnt/e/... /monthly/ ein, nur die Datenbank im vhdx
    -> keine 9p-/ENOMEM-Gefahr, kein erneutes Lesen der 198 GB

Aufruf (als root, wie die übrige Pipeline):
  wsl -u root -- env PYTHONPATH=/home/nutzer/.local/lib/python3.10/site-packages \
      python3 -u /mnt/e/Uni/Masterarbeit/_ctfidf_top50_label2016.py
"""
import time
from pathlib import Path

import duckdb

DB = "/var/tmp/label2016_agg.duckdb"
OUT_DIR = Path("/mnt/e/Uni/Masterarbeit/Data/tfidf")
PARTS = OUT_DIR / "_top50_parts_label2016"
OUT_TOP50 = OUT_DIR / "ctfidf_top50_LABEL2016.csv"
OUT_PARQUET = OUT_DIR / "ctfidf_cluster_yearly_LABEL2016.parquet"

TOPK = 50
YEARS = range(2016, 2025)
VOLL_PARQUET = False  # True = zusaetzlich die vollen Jahres-Parquets (~8 GB auf E:)

TEMP_DIR = "/var/tmp/duckdb_temp_label2016"
MEM_LIMIT = "6GB"
THREADS = 1
TEMP_CAP = "45GiB"  # unter dem freien Platz im vhdx, damit der Lauf FAILT statt E: zu fluten

PARTS.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()  # In-Memory, damit die 19-GB-Datei unangetastet bleibt
con.execute(f"ATTACH '{DB}' AS agg (READ_ONLY)")
con.execute(f"PRAGMA memory_limit='{MEM_LIMIT}'")
con.execute(f"PRAGMA threads={THREADS}")
con.execute(f"PRAGMA temp_directory='{TEMP_DIR}'")
con.execute(f"PRAGMA max_temp_directory_size='{TEMP_CAP}'")

print(f"DB angehängt (read-only): {DB}", flush=True)
print(f"memory_limit={MEM_LIMIT} threads={THREADS} temp={TEMP_DIR} cap={TEMP_CAP}\n", flush=True)


def ctfidf_cte(year):
    """Wörtlich die Formel aus compute_ctfidf_label2016.py, auf ein Jahr eingeschränkt."""
    return f"""
    WITH
    cc        AS (SELECT cluster, year, token, SUM(count) AS count
                  FROM agg.cyc WHERE year = {year} GROUP BY cluster, year, token),
    totals    AS (SELECT cluster, year, SUM(count)::FLOAT AS total
                  FROM cc GROUP BY cluster, year),
    f_t       AS (SELECT year, token, SUM(count)::FLOAT AS ft
                  FROM cc GROUP BY year, token),
    avg_words AS (SELECT year, SUM(total)::FLOAT / COUNT(*)::FLOAT AS A
                  FROM totals GROUP BY year),
    scored    AS (
        SELECT cc.cluster, cc.year, cc.token,
               (cc.count::FLOAT / t.total * ln(1.0 + a.A / f.ft))::FLOAT AS ctfidf
        FROM cc
        JOIN totals    t USING (cluster, year)
        JOIN f_t       f USING (year, token)
        JOIN avg_words a USING (year)
    )
    """


t_start = time.time()
for year in YEARS:
    part = PARTS / f"top50_{year}.csv"
    if part.exists() and part.stat().st_size > 0:
        print(f"{year}: Teildatei existiert, übersprungen", flush=True)
        continue

    t0 = time.time()
    con.execute(f"""
        COPY (
            {ctfidf_cte(year)}
            SELECT cluster, year, token, ctfidf, rang FROM (
                SELECT cluster, year, token, ctfidf,
                       ROW_NUMBER() OVER (PARTITION BY cluster ORDER BY ctfidf DESC) AS rang
                FROM scored
            ) WHERE rang <= {TOPK}
            ORDER BY cluster, rang
        ) TO '{part}' (HEADER, DELIMITER ',')
    """)
    n_cl = con.execute(f"SELECT COUNT(DISTINCT cluster) FROM read_csv_auto('{part}')").fetchone()[0]
    print(f"{year}: {n_cl} Cluster -> {part.name}  ({time.time() - t0:.0f}s)", flush=True)

    if VOLL_PARQUET:
        pq = PARTS / f"ctfidf_{year}.parquet"
        con.execute(f"""
            COPY ({ctfidf_cte(year)} SELECT cluster, year, token, ctfidf FROM scored)
            TO '{pq}' (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)
        print(f"       + volles Parquet {pq.name}", flush=True)

print(f"\nJahre fertig in {(time.time() - t_start) / 60:.1f} min. Fuege zusammen ...", flush=True)

con.execute(f"""
    COPY (
        SELECT cluster, year, token, ctfidf, rang
        FROM read_csv_auto('{PARTS}/top50_*.csv', union_by_name=true)
        ORDER BY cluster, year, rang
    ) TO '{OUT_TOP50}' (HEADER, DELIMITER ',')
""")
print(f"-> {OUT_TOP50}", flush=True)

if VOLL_PARQUET:
    con.execute(f"""
        COPY (SELECT cluster, year, token, ctfidf FROM read_parquet('{PARTS}/ctfidf_*.parquet'))
        TO '{OUT_PARQUET}' (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)
    print(f"-> {OUT_PARQUET}", flush=True)

# ---------------------------------------------------------------------------
# Prüfsteine, aus dem Originalskript übernommen
# ---------------------------------------------------------------------------
print("\n--- Pruefsteine ---", flush=True)
r = con.execute(f"""SELECT COUNT(*), COUNT(DISTINCT cluster), MIN(year), MAX(year),
                    COUNT(DISTINCT year), SUM(CASE WHEN ctfidf IS NULL THEN 1 ELSE 0 END)
                    FROM read_csv_auto('{OUT_TOP50}')""").fetchone()
print(f"  Zeilen={r[0]:,}  Cluster={r[1]}  Jahre={r[2]}..{r[3]} ({r[4]})  NULLs={r[5]}", flush=True)

c2016 = con.execute(f"SELECT COUNT(DISTINCT cluster) FROM read_csv_auto('{OUT_TOP50}') WHERE year=2016").fetchone()[0]
print(f"  Cluster 2016 (erwartet 95): {c2016}", flush=True)

neg = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{OUT_TOP50}') WHERE cluster = -1").fetchone()[0]
print(f"  Zeilen mit cluster=-1 (erwartet 0): {neg}", flush=True)

print("\n  Themen-Proben 2016 (labelunabhaengig):", flush=True)
for marker, theme in [("lane", "LoL"), ("mana", "magicTCG"), ("crochet", "crochet")]:
    row = con.execute(f"""SELECT cluster FROM read_csv_auto('{OUT_TOP50}')
                          WHERE year=2016 AND token='{marker}'
                          ORDER BY ctfidf DESC LIMIT 1""").fetchone()
    if not row:
        print(f"    [{theme}] Marker '{marker}' nicht in den Top-50", flush=True)
        continue
    cl = row[0]
    toks = con.execute(f"""SELECT token FROM read_csv_auto('{OUT_TOP50}')
                           WHERE year=2016 AND cluster={cl}
                           ORDER BY rang LIMIT 8""").fetchall()
    print(f"    [{theme}] '{marker}' stärkster Cluster = {cl}: {', '.join(t[0] for t in toks)}", flush=True)

NATIVE = OUT_DIR / "ctfidf_top50_native.csv"
if NATIVE.exists():
    same = con.execute(f"""
        WITH
        new10 AS (SELECT cluster, list(token ORDER BY rang)[1:10] AS toks
                  FROM read_csv_auto('{OUT_TOP50}') WHERE year=2016 AND rang<=10 GROUP BY cluster),
        old10 AS (SELECT cluster, list(token ORDER BY rang)[1:10] AS toks
                  FROM read_csv_auto('{NATIVE}') WHERE year=2016 AND rang<=10 GROUP BY cluster)
        SELECT SUM(CASE WHEN n.toks = o.toks THEN 1 ELSE 0 END), COUNT(*)
        FROM new10 n JOIN old10 o USING (cluster)
    """).fetchone()
    print(f"\n  2016er-Cluster mit identischen Top-10 gegen nativ (per ID): {same[0]}/{same[1]}", flush=True)

print("\nFERTIG", flush=True)
