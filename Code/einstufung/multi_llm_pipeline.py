import os
import orjson
import zstandard as zstd
import duckdb
import json
import re
import concurrent.futures
from functools import partial
from pathlib import Path

# ====================================================================
# 1. KONFIGURATION & PFADE (BITTE ANPASSEN)
# ====================================================================
WHITELIST_PFAD = Path("top_bis_95_prozent_kumulativ.csv")
NVME_ROHDATEN = Path("Data/reddit/submissions")
HDD_ROHDATEN = Path("/mnt/d/hold")   # Ablage der Rohdaten des Jahres 2024
TEMP_DUCKDB = "Data/temp_duckdb"

# Der Systemprompt des Produktivlaufs steht in system_prompt.txt daneben.
# Früher stand hier eine ältere Fassung als Literal; sie ist nicht die,
# mit der die verwendeten Etiketten erzeugt wurden.
SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.txt").read_text(encoding="utf-8")

os.makedirs(TEMP_DUCKDB, exist_ok=True)

# ====================================================================
# 2. HILFSFUNKTIONEN
# ====================================================================
_URL_ONLY = re.compile(r'^\s*https?://\S+\s*$')
_AFTER_URLS = re.compile(r'https?://\S+')


def is_valid_selftext(text: str) -> bool:
    if _URL_ONLY.match(text):
        return False
    cleaned = _AFTER_URLS.sub("", text).strip()
    return len(cleaned) >= 20


def read_and_decode(reader, chunk_size, max_window_size):
    chunk = reader.read(chunk_size)
    if not chunk:
        return ""
    bytes_read = len(chunk)
    while bytes_read <= max_window_size:
        try:
            return chunk.decode("utf-8")
        except UnicodeDecodeError:
            extra = reader.read(8)
            if not extra:
                break
            chunk += extra
            bytes_read += len(extra)
    return chunk.decode("utf-8", errors="replace")


def read_lines_zst(file_name):
    with open(file_name, 'rb') as file_handle:
        buffer = ''
        reader = zstd.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(file_handle)
        while True:
            chunk = read_and_decode(reader, 2 ** 27, (2 ** 29) * 2)
            if not chunk:
                break
            lines = (buffer + chunk).split("\n")
            for line in lines[:-1]:
                yield line.strip()
            buffer = lines[-1]
        if buffer.strip():
            yield buffer.strip()
        reader.close()


BOT_KEYWORDS = {"bot", "automoderator", "auto_moderator"}


def is_bot(author: str) -> bool:
    if not author:
        return False
    a = author.lower()
    return a in BOT_KEYWORDS or a.endswith("bot") or a.startswith("bot_")


# ====================================================================
# SCHRITT A: WORKER FUNKTION FÜR MULTIPROCESSING
# ====================================================================
def process_single_zst(datei, whitelist_set, temp_dir, output_ordner):
    """Verarbeitet exakt EINE .zst Datei in einem eigenen CPU-Prozess."""
    # JEDER Worker braucht seine eigene DB Connection!
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='2GB';")  # RAM pro Worker begrenzen
    con.execute(f"PRAGMA temp_directory='{temp_dir}';")

    con.execute("""
        CREATE OR REPLACE TABLE current_month (
            subreddit VARCHAR,
            text      VARCHAR,
            score     INTEGER
        )
    """)
    batch = []
    batch_size = 100_000
    total_saved = 0
    parse_errors = 0

    for line in read_lines_zst(datei):
        if not line:
            continue
        try:
            obj = orjson.loads(line)
            sub = obj.get("subreddit")
            if not sub or sub not in whitelist_set:
                continue

            author = obj.get("author", "")
            if is_bot(author):
                continue

            title = obj.get("title", "").strip()
            selftext = obj.get("selftext", "").strip()

            if selftext in ("[deleted]", "[removed]"):
                selftext = ""

            if not selftext or not is_valid_selftext(selftext):
                continue

            raw_text = f"{title} {selftext}".strip() if title else selftext
            score = obj.get("score", 0)
            batch.append((sub, raw_text, score))

            if len(batch) >= batch_size:
                con.executemany("INSERT INTO current_month VALUES (?, ?, ?)", batch)
                total_saved += len(batch)
                batch = []

        except Exception:
            parse_errors += 1

    if batch:
        con.executemany("INSERT INTO current_month VALUES (?, ?, ?)", batch)
        total_saved += len(batch)

    msg = ""
    if total_saved > 0:
        out_pfad = output_ordner / datei.name.replace(".zst", "_texts.parquet")
        con.execute(f"COPY current_month TO '{out_pfad}' (FORMAT PARQUET)")
        msg = f"    -> Gesichert: {out_pfad.name} ({total_saved:,} Texte)"
    else:
        msg = f"    -> Keine verwertbaren Submissions in {datei.name}"

    con.close()
    return msg, parse_errors


# ====================================================================
# SCHRITT B: MULTIPROCESSING ORCHESTRIERUNG
# ====================================================================
def extract_submissions_with_duckdb(jahr):
    print(f"  [Schritt 1] Extrahiere Submissions für {jahr}...")
    OUTPUT_ORDNER = Path(f"Data/Cleaned_Submissions/{jahr}")
    OUTPUT_ORDNER.mkdir(parents=True, exist_ok=True)

    # Whitelist nur EINMAL im Hauptprozess laden
    con = duckdb.connect()
    whitelist_set = set(
        row[0] for row in con.execute(
            f"SELECT subreddit FROM read_csv_auto('{WHITELIST_PFAD}')"
        ).fetchall()
    )
    con.close()

    ROHDATEN_ORDNER = NVME_ROHDATEN if jahr <= 2023 else HDD_ROHDATEN
    zst_dateien = sorted(ROHDATEN_ORDNER.rglob(f"*{jahr}*.zst"))

    if not zst_dateien:
        print(f"  ❌ Keine ZST-Dateien für {jahr} in {ROHDATEN_ORDNER} gefunden!")
        return False

    # SMART WORKER LIMIT: 6 Kerne für NVMe, 3 Kerne für mechanische HDD
    WORKERS = 3 if jahr <= 2023 else 3
    print(f"    🚀 Starte Multiprocessing mit {WORKERS} Workern...")

    # Partial "friert" die statischen Variablen ein, sodass map() nur die Datei durchwechselt
    worker_func = partial(process_single_zst, whitelist_set=whitelist_set, temp_dir=TEMP_DUCKDB,
                          output_ordner=OUTPUT_ORDNER)

    # Parallele Ausführung
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as executor:
        results = executor.map(worker_func, zst_dateien)

        for msg, errors in results:
            print(msg)
            if errors:
                print(f"    ⚠️  {errors:,} fehlerhafte JSON-Zeilen übersprungen.")

    return True


# ====================================================================
# SCHRITT C: PROMPT-ERSTELLUNG
# ====================================================================
def build_jsonl_with_duckdb(jahr):
    print(f"  [Schritt 2] Baue JSONL-Prompts für {jahr}...")
    INPUT_ORDNER = Path(f"Data/Cleaned_Submissions/{jahr}")
    OUTPUT_PFAD = Path(f"Data/cluster_prompts_{jahr}.jsonl")

    if not list(INPUT_ORDNER.glob("*.parquet")):
        print(f"  ⚠️  Keine Parquet-Dateien in {INPUT_ORDNER}, überspringe...")
        return

    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='5GB';")
    con.execute(f"PRAGMA temp_directory='{TEMP_DUCKDB}';")

    query = f"""
        SELECT subreddit, list(text) AS posts
        FROM (
            SELECT
                subreddit,
                text,
                ROW_NUMBER() OVER (
                    PARTITION BY subreddit
                    ORDER BY score DESC, text ASC
                ) AS rn
            FROM read_parquet('{INPUT_ORDNER}/*.parquet')
            WHERE text IS NOT NULL
              AND TRIM(text) != ''
              AND text NOT IN ('[deleted]', '[removed]')
        )
        WHERE rn <= 20
        GROUP BY subreddit
        ORDER BY subreddit
    """

    try:
        cursor = con.execute(query)
    except Exception as e:
        print(f"  ❌ DuckDB-Fehler: {e}")
        con.close()
        return

    geschriebene_subs = 0
    uebersprungen = 0

    with open(OUTPUT_PFAD, "w", encoding="utf-8") as f:
        while True:
            chunk = cursor.fetchmany(1000)
            if not chunk:
                break

            for row in chunk:
                subreddit, posts = row[0], row[1]
                if not posts:
                    uebersprungen += 1
                    continue

                post_snippets = "\n".join(
                    f"{i + 1}. {str(p).replace(chr(10), ' ')}"
                    for i, p in enumerate(posts)
                )
                content_string = (
                    f"Input:\n"
                    f"Subreddit Name: r/{subreddit}\n"
                    f"Top {len(posts)} Posts:\n"
                    f"{post_snippets}\n"
                    f"Output:"
                )

                json_obj = {
                    "system_prompt": SYSTEM_PROMPT.strip(),
                    "content": content_string,
                }
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
                geschriebene_subs += 1

    con.close()
    print(
        f"  ✅ JSONL fertig! {geschriebene_subs:,} Subreddits → {OUTPUT_PFAD.name}"
        + (f" ({uebersprungen} übersprungen)" if uebersprungen else "")
    )


# ====================================================================
# 3. MASTER-PIPELINE
# ====================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 STARTE DIE MASTER-PIPELINE (Submissions) [MULTIPROCESSED]")
    print("=" * 50)

    # Hinweis: Falls du 2016-2020 schon fertig hast, kannst du
    # range(2016, 2025) auf range(2021, 2025) ändern!
    for aktuelles_jahr in range(2020, 2025):
        print(f"\n{'=' * 16} JAHR {aktuelles_jahr} {'=' * 16}")

        erfolg = extract_submissions_with_duckdb(aktuelles_jahr)
        if erfolg:
            build_jsonl_with_duckdb(aktuelles_jahr)

    print("\n🎉 ALLE JAHRE ABGESCHLOSSEN!")