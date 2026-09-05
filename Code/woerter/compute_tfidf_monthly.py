"""
Computes two TF-IDF variants from monthly subreddit token counts:

  1. tfidf_subreddit_monthly/  (one parquet part per month)
     — TF-IDF per (subreddit, month); read via
       read_parquet('.../tfidf_subreddit_monthly/*.parquet')
     Document = one subreddit in one month.
     TF(t,s,m)  = count(t,s,m) / total_words(s,m)
     Two IDF variants side by side:
       tfidf        : IDF(t)   = ln(N / df(t))     over the whole 2016-2024 corpus
       tfidf_yearly : IDF(t,y) = ln(N_y / df_y(t)) restricted to year y
     N = #(subreddit,month) pairs; df = pairs where t appears.

  2. ctfidf_cluster_yearly.parquet   — c-TF-IDF per (cluster, year)
     Subreddit counts pooled via subreddit_clusters_aligned_2016_2024.csv.
     cluster=-1 (noise) excluded.
     c-TF-IDF(t,c,y) = (count/total) * ln(1 + A(y) / f(t,y))
       A   = average total words per cluster in year y
       f   = total count of t across all clusters in year y

Run in WSL as root (root is needed to drop the page cache between months;
without it, reads via the 9p driver on /mnt/e eventually fail with ENOMEM).
duckdb/pyarrow are installed per-user, so root needs PYTHONPATH:
    wsl -u root -- env PYTHONPATH=/home/nutzer/.local/lib/python3.10/site-packages \
        python3 -u /mnt/e/Uni/Masterarbeit/compute_tfidf_monthly.py

The script is resumable: progress is tracked in the DuckDB file, so after a
crash it picks up at the first unprocessed month.
"""

import duckdb
import re
import os
import shutil
from pathlib import Path
from collections import defaultdict
import pyarrow.parquet as pq

DATA_DIR    = Path("/mnt/e/Uni/Masterarbeit/Data/tfidf/monthly")
CLUSTER_CSV = "/mnt/e/Uni/Masterarbeit/Data/subreddit_clusters_aligned_2016_2024.csv"
AGG_DB      = "/var/tmp/monthly_agg.duckdb"
DUCKDB_TEMP = "/var/tmp/duckdb_temp"
OUT_TFIDF   = Path("/mnt/e/Uni/Masterarbeit/Data/tfidf/tfidf_subreddit_monthly")  # dir of per-month parquet parts
OUT_CTFIDF  = "/mnt/e/Uni/Masterarbeit/Data/tfidf/ctfidf_cluster_yearly.parquet"

# ---------------------------------------------------------------------------
# WSL memory reclaim: streaming ~200 GB through the 9p driver on /mnt/e fills
# and fragments the VM page cache until the driver's buffer allocation fails
# with "Cannot allocate memory". Dropping caches + compacting between months
# prevents this. Requires root; warns once and continues otherwise.
# ---------------------------------------------------------------------------
_reclaim_warned = False

def reclaim():
    global _reclaim_warned
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("1\n")
        with open("/proc/sys/vm/compact_memory", "w") as f:
            f.write("1\n")
    except OSError:
        if not _reclaim_warned:
            print("  WARNING: cannot drop page cache (not root?). "
                  "Long runs may die with ENOMEM — run via `wsl -u root`.")
            _reclaim_warned = True

# ---------------------------------------------------------------------------
# Stopwords: NLTK English + Reddit-specific extras
# ---------------------------------------------------------------------------
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "you're", "you've", "you'll", "you'd", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "she's", "her",
    "hers", "herself", "it", "it's", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom",
    "this", "that", "that'll", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do",
    "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
    "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "will", "just", "don", "don't", "should", "should've", "now", "d",
    "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn",
    "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't",
    "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn",
    "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't",
    "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won",
    "won't", "wouldn", "wouldn't",
    "reddit", "subreddit", "subreddits", "upvote", "downvote", "upvoted",
    "downvoted", "upvotes", "downvotes", "karma", "mod", "mods",
    "moderator", "moderators", "edit", "edited", "op", "deleted",
    "removed", "bot", "comment", "comments", "post", "posts", "thread",
    "threads", "crosspost", "repost", "nbsp", "http", "https", "www",
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
files_by_month = defaultdict(list)
for f in DATA_DIR.glob("counts_R*_*.parquet"):
    month = re.search(r"(\d{4}-\d{2})", f.name).group(1)
    files_by_month[month].append(f)

months = sorted(files_by_month)
N_months = len(months)
n_files  = sum(len(v) for v in files_by_month.values())
print(f"Found {n_files} files, {N_months} months.")

os.makedirs(DUCKDB_TEMP, exist_ok=True)
con = duckdb.connect(AGG_DB)
con.execute(f"PRAGMA temp_directory='{DUCKDB_TEMP}'")
con.execute("PRAGMA memory_limit='9GB'")
con.execute("PRAGMA threads=2")
con.execute("PRAGMA preserve_insertion_order=false")

con.execute("CREATE TABLE IF NOT EXISTS sw (token VARCHAR)")
con.execute("DELETE FROM sw")
con.executemany("INSERT INTO sw VALUES (?)", [[w] for w in STOPWORDS])

# Per-month progress tracking for crash-safe resume.
con.execute("CREATE TABLE IF NOT EXISTS progress (pass VARCHAR, month VARCHAR)")

def done_months(p):
    return {r[0] for r in con.execute(
        "SELECT month FROM progress WHERE pass = ?", [p]).fetchall()}

# ---------------------------------------------------------------------------
# Local month cache + subreddit-hash partitioning.
# Months from 2021 on are too big for one in-memory aggregate, so heavy
# queries run in P disjoint slices (WHERE hash(subreddit) % P = p). Results
# are exact because every subreddit lands entirely in one slice. To avoid
# re-reading /mnt/e P times, each month is first copied to WSL-local disk.
# ---------------------------------------------------------------------------
LOCAL_CACHE = Path("/var/tmp/month_cache")
LOCAL_CACHE.mkdir(exist_ok=True)

def localize(month):
    paths = []
    for f in files_by_month[month]:
        dst = LOCAL_CACHE / f.name
        if not dst.exists() or dst.stat().st_size != f.stat().st_size:
            shutil.copyfile(f, dst)
        paths.append(dst)
    return "[" + ", ".join(f"'{p.as_posix()}'" for p in paths) + "]"

def unlocalize():
    for f in LOCAL_CACHE.glob("*"):
        f.unlink()

def n_partitions(month):
    size = sum(f.stat().st_size for f in files_by_month[month])
    return min(12, max(2, -(-size // int(4e8))))  # ~1 slice per 400MB input

unlocalize()  # clear leftovers from a crashed run

# ---------------------------------------------------------------------------
# Pass 1: document frequencies for IDF
# df(t) = number of (subreddit, month) pairs where token t appears
# N     = total number of (subreddit, month) pairs
# ---------------------------------------------------------------------------
print("\nPass 1/3: computing document frequencies ...")
con.execute("CREATE TABLE IF NOT EXISTS df_acc (token VARCHAR, year INTEGER, n BIGINT)")
con.execute("CREATE TABLE IF NOT EXISTS n_acc (year INTEGER, n BIGINT)")

done1 = done_months("pass1")
if not done1:
    # Backfill from a run before progress tracking existed: months were
    # processed in order and each one added exactly one row to n_acc.
    k = con.execute("SELECT COUNT(*) FROM n_acc").fetchone()[0]
    if k:
        con.executemany("INSERT INTO progress VALUES ('pass1', ?)",
                        [[m] for m in months[:k]])
        done1 = set(months[:k])
        print(f"  resuming: {k} months already done")
    else:
        con.execute("DELETE FROM df_acc")
        con.execute("DELETE FROM n_acc")
elif done1:
    print(f"  resuming: {len(done1)} months already done")

for i, month in enumerate(months):
    if month in done1:
        continue
    path_list = localize(month)
    year = int(month[:4])
    P = n_partitions(month)

    con.execute("BEGIN")
    # per token: how many distinct subreddits have it this month
    # (two-stage GROUP BY instead of COUNT(DISTINCT) — spills to disk);
    # doc_freq/doc_freq_y later SUM over df_acc, so partial per-partition
    # rows are fine
    for p in range(P):
        con.execute(f"""
            INSERT INTO df_acc
            SELECT token, {year} AS year, COUNT(*) AS n
            FROM (
                SELECT token, subreddit
                FROM read_parquet({path_list})
                WHERE hash(subreddit) % {P} = {p}
                  AND token NOT IN (SELECT token FROM sw)
                GROUP BY token, subreddit
            )
            GROUP BY token
        """)
    # how many distinct subreddits are active this month
    con.execute(f"""
        INSERT INTO n_acc
        SELECT {year}, COUNT(DISTINCT subreddit)
        FROM read_parquet({path_list})
        WHERE token NOT IN (SELECT token FROM sw)
    """)
    con.execute("INSERT INTO progress VALUES ('pass1', ?)", [month])
    con.execute("COMMIT")
    unlocalize()
    reclaim()
    if (i + 1) % 12 == 0 or (i + 1) == N_months:
        print(f"  {i+1}/{N_months}")

con.execute("DROP TABLE IF EXISTS doc_freq")
con.execute("CREATE TABLE doc_freq AS SELECT token, SUM(n)::FLOAT AS df FROM df_acc GROUP BY token")
con.execute("DROP TABLE IF EXISTS doc_freq_y")
con.execute("CREATE TABLE doc_freq_y AS SELECT token, year, SUM(n)::FLOAT AS df FROM df_acc GROUP BY token, year")
n_docs = con.execute("SELECT SUM(n)::FLOAT FROM n_acc").fetchone()[0]
n_docs_by_year = dict(con.execute("SELECT year, SUM(n)::FLOAT FROM n_acc GROUP BY year").fetchall())
n_tokens = con.execute("SELECT COUNT(*) FROM doc_freq").fetchone()[0]
print(f"  N={n_docs:.0f} (subreddit×month pairs), {n_tokens:,} unique tokens")
for y in sorted(n_docs_by_year):
    print(f"    N_{y}={n_docs_by_year[y]:.0f}")

# ---------------------------------------------------------------------------
# Pass 2: TF-IDF per (subreddit, month) — one parquet part file per month
# (per-month parts make the pass resumable: existing parts are skipped).
# Writes directly via PyArrow (no mmap issues on /mnt/e/ for writes).
# ---------------------------------------------------------------------------
print("\nPass 2/3: computing TF-IDF per (subreddit, month) ...")
OUT_TFIDF.mkdir(parents=True, exist_ok=True)
for stale in OUT_TFIDF.glob("*.tmp"):
    stale.unlink()

for i, month in enumerate(months):
    P = n_partitions(month)
    parts = [OUT_TFIDF / f"tfidf_{month}_p{p}.parquet" for p in range(P)]
    if all(pt.exists() for pt in parts):
        continue
    path_list = localize(month)
    year = int(month[:4])
    n_docs_y = n_docs_by_year[year]

    for p, part in enumerate(parts):
        if part.exists():
            continue
        # materialize the slice first: with known cardinality the planner
        # builds the hash joins on this side, not on the 150M+-row
        # doc_freq tables (which OOMs)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE m_raw AS
            SELECT subreddit, token, SUM("count") AS count
            FROM read_parquet({path_list})
            WHERE hash(subreddit) % {P} = {p}
              AND token NOT IN (SELECT token FROM sw)
            GROUP BY subreddit, token
        """)
        # stream result in 1M-row batches instead of materializing everything
        reader = con.execute(f"""
            WITH
            totals AS (SELECT subreddit, SUM(count)::FLOAT AS total FROM m_raw GROUP BY subreddit)
            SELECT
                r.subreddit,
                '{month}'                                                    AS month,
                r.token,
                (r.count::FLOAT / t.total * ln({n_docs} / d.df))::FLOAT      AS tfidf,
                (r.count::FLOAT / t.total * ln({n_docs_y} / dy.df))::FLOAT   AS tfidf_yearly
            FROM m_raw r
            JOIN totals     t USING (subreddit)
            JOIN doc_freq   d USING (token)
            JOIN doc_freq_y dy ON dy.token = r.token AND dy.year = {year}
            ORDER BY r.subreddit, tfidf DESC
        """).fetch_record_batch(1_000_000)

        # write to a .tmp and rename, so a crash never leaves a truncated part
        tmp = OUT_TFIDF / f"tfidf_{month}_p{p}.parquet.tmp"
        writer = pq.ParquetWriter(tmp, reader.schema, compression="snappy")
        for rb in reader:
            writer.write_batch(rb)
        writer.close()
        os.replace(tmp, part)
        con.execute("DROP TABLE IF EXISTS m_raw")

    unlocalize()
    reclaim()
    if (i + 1) % 12 == 0 or (i + 1) == N_months:
        print(f"  {i+1}/{N_months}")

tfidf_mb = sum(f.stat().st_size for f in OUT_TFIDF.glob("*.parquet")) / 1e6
print(f"  → {OUT_TFIDF}/  ({tfidf_mb:.0f} MB in {len(list(OUT_TFIDF.glob('*.parquet')))} parts)")

# ---------------------------------------------------------------------------
# Pass 3: c-TF-IDF per (cluster, year)
# Accumulate cluster×year counts month-by-month, then compute in one query.
# ---------------------------------------------------------------------------
print("\nPass 3/3: computing c-TF-IDF per (cluster, year) ...")

con.execute("DROP TABLE IF EXISTS cluster_map")
con.execute("CREATE TABLE cluster_map (subreddit VARCHAR, jahr INTEGER, cluster INTEGER)")
con.execute(f"INSERT INTO cluster_map SELECT * FROM read_csv_auto('{CLUSTER_CSV}')")

con.execute("CREATE TABLE IF NOT EXISTS cyc (cluster INTEGER, year INTEGER, token VARCHAR, count BIGINT)")

done3 = done_months("pass3")
if done3:
    print(f"  resuming: {len(done3)} months already done")
else:
    con.execute("DELETE FROM cyc")

for i, month in enumerate(months):
    if month in done3:
        continue
    path_list = localize(month)
    year = int(month[:4])

    con.execute("BEGIN")
    con.execute(f"""
        INSERT INTO cyc
        SELECT cm.cluster, {year}, r.token, SUM(r."count") AS count
        FROM read_parquet({path_list}) r
        JOIN cluster_map cm ON r.subreddit = LOWER(cm.subreddit) AND cm.jahr = {year}
        WHERE cm.cluster != -1
          AND r.token NOT IN (SELECT token FROM sw)
        GROUP BY cm.cluster, r.token
    """)
    con.execute("INSERT INTO progress VALUES ('pass3', ?)", [month])
    con.execute("COMMIT")
    unlocalize()
    reclaim()
    if (i + 1) % 12 == 0 or (i + 1) == N_months:
        print(f"  {i+1}/{N_months}")

con.execute(f"""
COPY (
    WITH
    cc        AS (SELECT cluster, year, token, SUM(count) AS count
                  FROM cyc GROUP BY cluster, year, token),
    totals    AS (SELECT cluster, year, SUM(count)::FLOAT AS total
                  FROM cc GROUP BY cluster, year),
    f_t       AS (SELECT year, token, SUM(count)::FLOAT AS ft
                  FROM cc GROUP BY year, token),
    avg_words AS (SELECT year, SUM(total)::FLOAT / COUNT(*)::FLOAT AS A
                  FROM totals GROUP BY year)
    SELECT
        cc.cluster,
        cc.year,
        cc.token,
        (cc.count::FLOAT / t.total * ln(1.0 + a.A / f.ft))::FLOAT AS ctfidf
    FROM cc
    JOIN totals    t USING (cluster, year)
    JOIN f_t       f USING (year, token)
    JOIN avg_words a USING (year)
    ORDER BY cc.cluster, cc.year, ctfidf DESC
)
TO '{OUT_CTFIDF}' (FORMAT PARQUET, COMPRESSION SNAPPY)
""")
print(f"  → {OUT_CTFIDF}  ({Path(OUT_CTFIDF).stat().st_size / 1e6:.0f} MB)")

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
print("\n--- Sanity check: TF-IDF (r/politics, 2020-11) ---")
print(con.execute(f"""
    SELECT token, tfidf, tfidf_yearly FROM read_parquet('{OUT_TFIDF.as_posix()}/*.parquet')
    WHERE subreddit = 'politics' AND month = '2020-11'
    ORDER BY tfidf DESC LIMIT 10
""").df().to_string(index=False))

print("\n--- Sanity check: c-TF-IDF (cluster 0, 2020) ---")
print(con.execute(f"""
    SELECT token, ctfidf FROM read_parquet('{OUT_CTFIDF}')
    WHERE cluster = 0 AND year = 2020
    ORDER BY ctfidf DESC LIMIT 10
""").df().to_string(index=False))
