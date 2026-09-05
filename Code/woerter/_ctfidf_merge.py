"""Phase 2: merge per-year c-TF-IDF parts (in vhdx) into the single target file
on E:. Pure streaming scan+write (low memory, no temp). Then verify + sanity."""
import duckdb
from pathlib import Path

AGG_DB     = "/var/tmp/monthly_agg.duckdb"
PARTS_DIR  = "/var/tmp/ctfidf_parts"
OUT_CTFIDF = "/mnt/e/Uni/Masterarbeit/Data/tfidf/ctfidf_cluster_yearly.parquet"

con = duckdb.connect(AGG_DB)
con.execute("PRAGMA memory_limit='6GB'")
con.execute("PRAGMA threads=2")
con.execute("PRAGMA preserve_insertion_order=false")

print("merging parts -> single file ...", flush=True)
con.execute(f"""
    COPY (SELECT cluster, year, token, ctfidf
          FROM read_parquet('{PARTS_DIR}/*.parquet'))
    TO '{OUT_CTFIDF}' (FORMAT PARQUET, COMPRESSION SNAPPY)
""")
sz = Path(OUT_CTFIDF).stat().st_size / 1e6
print(f"  -> {OUT_CTFIDF}  ({sz:.0f} MB)", flush=True)

# verify structure
n = con.execute(f"""SELECT COUNT(*), COUNT(DISTINCT cluster), MIN(year), MAX(year),
                    COUNT(DISTINCT year), MIN(ctfidf), MAX(ctfidf),
                    SUM(CASE WHEN ctfidf IS NULL THEN 1 ELSE 0 END)
                    FROM read_parquet('{OUT_CTFIDF}')""").fetchone()
print(f"  rows={n[0]:,}  clusters={n[1]}  years={n[2]}..{n[3]} ({n[4]})  "
      f"ctfidf min/max={n[5]:.5f}/{n[6]:.5f}  nulls={n[7]}", flush=True)
# no cluster -1 (noise must be excluded)
neg = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT_CTFIDF}') WHERE cluster = -1").fetchone()[0]
print(f"  rows with cluster=-1 (should be 0): {neg}", flush=True)

# sanity: top tokens for a few clusters in 2020 (lifestyle corpus)
for cl in (0, 1, 2):
    rows = con.execute(f"""
        SELECT token, ctfidf FROM read_parquet('{OUT_CTFIDF}')
        WHERE cluster = {cl} AND year = 2020 ORDER BY ctfidf DESC LIMIT 8
    """).fetchall()
    toks = ", ".join(f"{t}({v:.3f})" for t, v in rows)
    print(f"  cluster {cl} / 2020 top: {toks}", flush=True)
print("MERGE_DONE", flush=True)
