"""Pass 2 only: re-pool c-TF-IDF per (cluster, year) on the NEW LABEL2016 partition.

Re-computes c-TF-IDF from the finished monthly token-count parquets
(Data/tfidf/monthly/counts_{RC,RS}_YYYY-MM.parquet) but pools the per-subreddit
counts via the NEW clustering `subreddit_clusters_aligned_2016_2024_LABEL2016.csv`
instead of the old `subreddit_clusters_aligned_2016_2024.csv`.

Pass 1 (the expensive dump scan) is NOT touched — the monthly counts are per
(subreddit, year, token) and independent of the clustering. The old outputs
(ctfidf_cluster_yearly.parquet, ctfidf_top50_native.csv) are NOT overwritten;
everything here writes *_LABEL2016.* next to them.

This mirrors Pass 3 of compute_tfidf_monthly.py exactly (same stopwords, same
formula, same case-insensitive year+name join, same WSL memory discipline) but
uses a SEPARATE DuckDB file so the 65 GB monthly_agg.duckdb and its progress
table stay intact.

Run in WSL as root (root drops the page cache between months; without it the 9p
reads on /mnt/e eventually die with ENOMEM). duckdb/pyarrow are per-user, so
root needs PYTHONPATH:

    wsl -u root -- env PYTHONPATH=/home/nutzer/.local/lib/python3.10/site-packages \
        python3 -u /mnt/e/Uni/Masterarbeit/compute_ctfidf_label2016.py

Resumable: per-month progress is tracked in the DuckDB file, so after a crash it
picks up at the first unprocessed month.
"""

import duckdb
import re
import os
import shutil
from pathlib import Path
from collections import defaultdict

DATA_DIR    = Path("/mnt/e/Uni/Masterarbeit/Data/tfidf/monthly")
CLUSTER_CSV = "/mnt/e/Uni/Masterarbeit/subreddit_clusters_aligned_2016_2024_LABEL2016.csv"
AGG_DB      = "/var/tmp/label2016_agg.duckdb"          # separate from monthly_agg.duckdb
DUCKDB_TEMP = "/var/tmp/duckdb_temp_label2016"
LOCAL_CACHE = Path("/var/tmp/month_cache_label2016")
OUT_CTFIDF  = "/mnt/e/Uni/Masterarbeit/Data/tfidf/ctfidf_cluster_yearly_LABEL2016.parquet"
OUT_TOP50   = "/mnt/e/Uni/Masterarbeit/Data/tfidf/ctfidf_top50_LABEL2016.csv"
TOPK        = 50

# ---------------------------------------------------------------------------
# WSL memory reclaim (see compute_tfidf_monthly.py). Requires root; warns once.
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
# Stopwords: identical to compute_tfidf_monthly.py (comparability with the
# native run). Keep in sync if that list ever changes.
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
print(f"Found {n_files} files, {N_months} months.", flush=True)

os.makedirs(DUCKDB_TEMP, exist_ok=True)
LOCAL_CACHE.mkdir(exist_ok=True)

con = duckdb.connect(AGG_DB)
con.execute(f"PRAGMA temp_directory='{DUCKDB_TEMP}'")
con.execute("PRAGMA memory_limit='9GB'")
con.execute("PRAGMA threads=2")
con.execute("PRAGMA preserve_insertion_order=false")
con.execute("PRAGMA max_temp_directory_size='200GiB'")

con.execute("CREATE TABLE IF NOT EXISTS sw (token VARCHAR)")
con.execute("DELETE FROM sw")
con.executemany("INSERT INTO sw VALUES (?)", [[w] for w in STOPWORDS])

con.execute("DROP TABLE IF EXISTS cluster_map")
con.execute("CREATE TABLE cluster_map (subreddit VARCHAR, jahr INTEGER, cluster INTEGER)")
con.execute(f"INSERT INTO cluster_map SELECT subreddit, jahr, cluster FROM read_csv_auto('{CLUSTER_CSV}')")
n_panel = con.execute("SELECT COUNT(DISTINCT lower(subreddit)) FROM cluster_map WHERE cluster <> -1").fetchone()[0]
print(f"LABEL2016 panel: {n_panel:,} distinct non-noise subreddits", flush=True)

# per-month progress tracking for crash-safe resume
con.execute("CREATE TABLE IF NOT EXISTS progress (pass VARCHAR, month VARCHAR)")
con.execute("CREATE TABLE IF NOT EXISTS cyc (cluster INTEGER, year INTEGER, token VARCHAR, count BIGINT)")
con.execute("CREATE TABLE IF NOT EXISTS seen_subs (subreddit VARCHAR)")  # panel subs actually seen in counts

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

unlocalize()  # clear leftovers from a crashed run

# ---------------------------------------------------------------------------
# Accumulate cluster×year token counts from the monthly counts, pooled via the
# LABEL2016 partition. Case-insensitive name match + year match (monthly=year,
# csv=jahr). Noise (cluster=-1) and stopwords dropped. Also records which panel
# subreddits were actually seen, for the coverage anti-join below.
# ---------------------------------------------------------------------------
print("\nPooling monthly counts on LABEL2016 -> cyc ...", flush=True)
done = {r[0] for r in con.execute("SELECT month FROM progress WHERE pass='pool'").fetchall()}
if done:
    print(f"  resuming: {len(done)} months already done", flush=True)

for i, month in enumerate(months):
    if month in done:
        continue
    path_list = localize(month)
    year = int(month[:4])

    con.execute("BEGIN")
    con.execute(f"""
        INSERT INTO cyc
        SELECT cm.cluster, {year}, r.token, SUM(r."count") AS count
        FROM read_parquet({path_list}) r
        JOIN cluster_map cm ON r.subreddit = LOWER(cm.subreddit) AND cm.jahr = {year}
        WHERE cm.cluster <> -1
          AND r.token NOT IN (SELECT token FROM sw)
        GROUP BY cm.cluster, r.token
    """)
    con.execute(f"""
        INSERT INTO seen_subs
        SELECT DISTINCT LOWER(r.subreddit)
        FROM read_parquet({path_list}) r
        WHERE LOWER(r.subreddit) IN (SELECT LOWER(subreddit) FROM cluster_map WHERE cluster <> -1)
    """)
    con.execute("INSERT INTO progress VALUES ('pool', ?)", [month])
    con.execute("COMMIT")
    unlocalize()
    reclaim()
    if (i + 1) % 12 == 0 or (i + 1) == N_months:
        print(f"  {i+1}/{N_months}", flush=True)

# ---------------------------------------------------------------------------
# Step 0 (post-hoc): panel coverage anti-join. Every LABEL2016 non-noise
# subreddit should appear in the monthly counts (LABEL2016 is a subset of the
# old panel that produced these counts) -> expect 0 missing.
# ---------------------------------------------------------------------------
missing = con.execute("""
    SELECT COUNT(*) FROM (
        SELECT DISTINCT LOWER(subreddit) AS s FROM cluster_map WHERE cluster <> -1
    ) p
    LEFT JOIN (SELECT DISTINCT subreddit AS s FROM seen_subs) q USING (s)
    WHERE q.s IS NULL
""").fetchone()[0]
print(f"\n[coverage] panel non-noise subs missing from monthly counts: {missing} (expect 0)", flush=True)
if missing:
    ex = con.execute("""
        SELECT p.s FROM (SELECT DISTINCT LOWER(subreddit) AS s FROM cluster_map WHERE cluster <> -1) p
        LEFT JOIN (SELECT DISTINCT subreddit AS s FROM seen_subs) q USING (s)
        WHERE q.s IS NULL LIMIT 15
    """).fetchall()
    print("  examples:", ", ".join(r[0] for r in ex), flush=True)

# ---------------------------------------------------------------------------
# c-TF-IDF per (cluster, year), identical formula to the native run:
#   c-TF-IDF(t,c) = tf(t,c) * ln(1 + A / f(t))       (per year)
#   tf = count(t,c)/total(c) ; A = avg cluster word count ; f(t) = sum over clusters
# ---------------------------------------------------------------------------
print("\nWriting c-TF-IDF parquet ...", flush=True)
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
print(f"  -> {OUT_CTFIDF}  ({Path(OUT_CTFIDF).stat().st_size / 1e6:.0f} MB)", flush=True)

# ---------------------------------------------------------------------------
# Top-K tokens per (cluster, year) -> CSV. Same columns as ctfidf_top50_native.csv:
#   cluster, year, token, ctfidf, rang
# ---------------------------------------------------------------------------
print(f"\nWriting top-{TOPK} CSV ...", flush=True)
con.execute(f"""
COPY (
    SELECT cluster, year, token, ctfidf, rang FROM (
        SELECT cluster, year, token, ctfidf,
               ROW_NUMBER() OVER (PARTITION BY cluster, year ORDER BY ctfidf DESC) AS rang
        FROM read_parquet('{OUT_CTFIDF}')
    ) WHERE rang <= {TOPK}
    ORDER BY cluster, year, rang
)
TO '{OUT_TOP50}' (HEADER, DELIMITER ',')
""")
print(f"  -> {OUT_TOP50}", flush=True)

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
print("\n--- Pruefsteine ---", flush=True)
n = con.execute(f"""SELECT COUNT(*), COUNT(DISTINCT cluster), MIN(year), MAX(year),
                    COUNT(DISTINCT year), SUM(CASE WHEN ctfidf IS NULL THEN 1 ELSE 0 END)
                    FROM read_parquet('{OUT_CTFIDF}')""").fetchone()
print(f"  rows={n[0]:,}  clusters={n[1]}  years={n[2]}..{n[3]} ({n[4]})  nulls={n[5]}", flush=True)
c2016 = con.execute(f"SELECT COUNT(DISTINCT cluster) FROM read_parquet('{OUT_CTFIDF}') WHERE year=2016 AND cluster <> -1").fetchone()[0]
print(f"  clusters in 2016 (expect 95): {c2016}", flush=True)
neg = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT_CTFIDF}') WHERE cluster = -1").fetchone()[0]
print(f"  rows with cluster=-1 (expect 0): {neg}", flush=True)

# label-independent topic probes: for a marker token, find the 2016 cluster where
# it scores highest and show that cluster's top tokens.
print("\n  topic probes (2016, label-independent):", flush=True)
for marker, theme in [("lane", "LoL"), ("mana", "magicTCG"), ("crochet", "crochet")]:
    row = con.execute(f"""
        SELECT cluster FROM read_parquet('{OUT_CTFIDF}')
        WHERE year=2016 AND token='{marker}' ORDER BY ctfidf DESC LIMIT 1
    """).fetchone()
    if not row:
        print(f"    [{theme}] marker '{marker}' not found", flush=True)
        continue
    cl = row[0]
    toks = con.execute(f"""
        SELECT token FROM read_parquet('{OUT_CTFIDF}')
        WHERE year=2016 AND cluster={cl} ORDER BY ctfidf DESC LIMIT 8
    """).fetchall()
    print(f"    [{theme}] top cluster for '{marker}' = {cl}: {', '.join(t[0] for t in toks)}", flush=True)

# id-based diff vs native (only meaningful if cluster labels are aligned across
# the two partitions; if low, that's a label permutation, not a content problem).
NATIVE = "/mnt/e/Uni/Masterarbeit/Data/tfidf/ctfidf_cluster_yearly.parquet"
if Path(NATIVE).exists():
    same = con.execute(f"""
        WITH
        new10 AS (SELECT cluster, list(token ORDER BY ctfidf DESC)[1:10] AS toks FROM (
                    SELECT cluster, token, ctfidf, ROW_NUMBER() OVER (PARTITION BY cluster ORDER BY ctfidf DESC) rn
                    FROM read_parquet('{OUT_CTFIDF}') WHERE year=2016) WHERE rn<=10 GROUP BY cluster),
        old10 AS (SELECT cluster, list(token ORDER BY ctfidf DESC)[1:10] AS toks FROM (
                    SELECT cluster, token, ctfidf, ROW_NUMBER() OVER (PARTITION BY cluster ORDER BY ctfidf DESC) rn
                    FROM read_parquet('{NATIVE}') WHERE year=2016) WHERE rn<=10 GROUP BY cluster)
        SELECT SUM(CASE WHEN n.toks = o.toks THEN 1 ELSE 0 END), COUNT(*)
        FROM new10 n JOIN old10 o USING (cluster)
    """).fetchone()
    print(f"\n  2016 clusters with identical top-10 vs native (by id): {same[0]}/{same[1]}", flush=True)

print("\nDONE", flush=True)
