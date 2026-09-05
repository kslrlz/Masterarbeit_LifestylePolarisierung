"""Phase 1: compute per-year c-TF-IDF parts INSIDE the WSL vhdx only.
Dös NOT write to /mnt/e (E: has almost no free space). Prints part sizes so
we can size the final merge. Memory-conservative to avoid OS OOM in 12GB WSL.

c-TF-IDF is per-year by definition, so per-year parts are exactly equivalent.
No global ORDER BY (row order isn't semantically meaningful; consumers re-sort).
"""
import duckdb, os
from pathlib import Path

AGG_DB      = "/var/tmp/monthly_agg.duckdb"
DUCKDB_TEMP = "/var/tmp/duckdb_temp"
PARTS_DIR   = "/var/tmp/ctfidf_parts"

os.makedirs(DUCKDB_TEMP, exist_ok=True)
os.makedirs(PARTS_DIR, exist_ok=True)
for f in Path(PARTS_DIR).glob("*.parquet"):
    f.unlink()

con = duckdb.connect(AGG_DB)
con.execute(f"PRAGMA temp_directory='{DUCKDB_TEMP}'")
con.execute("PRAGMA memory_limit='7GB'")
con.execute("PRAGMA threads=1")
con.execute("PRAGMA preserve_insertion_order=false")
con.execute("PRAGMA max_temp_directory_size='60GiB'")  # guard: fail before filling E:

years = [r[0] for r in con.execute("SELECT DISTINCT year FROM cyc ORDER BY year").fetchall()]
print(f"years: {years}", flush=True)

total_mb = 0
for y in years:
    # 1) one big aggregate, materialized once (spills to temp, then cheap)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE cc AS
        SELECT cluster, token, SUM(count) AS count
        FROM cyc WHERE year = {y}
        GROUP BY cluster, token
    """)
    nrows = con.execute("SELECT COUNT(*) FROM cc").fetchone()[0]
    # 2) cheap derived stats + join, write part (no ORDER BY)
    part = f"{PARTS_DIR}/ctfidf_{y}.parquet"
    con.execute(f"""
        COPY (
            WITH
            totals    AS (SELECT cluster, SUM(count)::FLOAT AS total FROM cc GROUP BY cluster),
            f_t       AS (SELECT token, SUM(count)::FLOAT AS ft FROM cc GROUP BY token),
            avg_words AS (SELECT SUM(total)::FLOAT / COUNT(*)::FLOAT AS A FROM totals)
            SELECT
                cc.cluster,
                {y} AS year,
                cc.token,
                (cc.count::FLOAT / t.total * ln(1.0 + a.A / f.ft))::FLOAT AS ctfidf
            FROM cc
            JOIN totals t USING (cluster)
            JOIN f_t    f USING (token)
            CROSS JOIN avg_words a
        )
        TO '{part}' (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)
    con.execute("DROP TABLE IF EXISTS cc")
    mb = Path(part).stat().st_size / 1e6
    total_mb += mb
    print(f"  {y}: {nrows:,} rows, {mb:.0f} MB  (kumuliert {total_mb:.0f} MB)", flush=True)

print(f"TOTAL_PARTS_MB={total_mb:.0f}", flush=True)
print("PARTS_DONE", flush=True)
