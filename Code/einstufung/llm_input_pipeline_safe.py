import os
import zstandard as zstd
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import io
import json
import re
import shutil
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, as_completed

# ====================================================================
# 1. KONFIGURATION & PFADE
# ====================================================================
WHITELIST_PFAD  = Path("top_bis_95_prozent_kumulativ.csv")
NVME_ROHDATEN   = Path("Data/reddit/submissions")
OUTPUT_ROOT     = Path("Data/Cleaned_Submissions")

MAX_WORKERS     = 2  # Sicher für 16GB RAM (2x ~4GB = 8GB + Puffer)

# ... (rest of configuration stays the same)

# Filter-Einstellungen
MIN_TEXT_LENGTH = 20
CHUNK_SIZE      = 100_000 

_AFTER_URLS = re.compile(r'https?://\S+')

def is_valid_text(text: str) -> bool:
    if not text: return False
    cleaned = _AFTER_URLS.sub("", text).strip()
    return len(cleaned) >= MIN_TEXT_LENGTH

BOT_KEYWORDS = {"bot", "automoderator", "auto_moderator"}
def is_bot(author: str) -> bool:
    if not author: return False
    a = author.lower()
    return a in BOT_KEYWORDS or a.endswith("bot") or a.startswith("bot_")

# ====================================================================
# EXTRAKTION (ZST -> PARQUET) - STABILE VERSION
# ====================================================================
def process_single_zst_safe(input_path, output_path, whitelist_set):
    print(f"  -> Verarbeite: {input_path.name}")
    
    schema = pa.schema([
        ('subreddit', pa.string()),
        ('text', pa.string()),
        ('score', pa.int64())
    ])
    
    writer = None
    rows = {"subreddit": [], "text": [], "score": []}
    
    # Der Decompressor mit großem Window für 2023/24
    dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
    
    try:
        with open(input_path, 'rb') as fh:
            with dctx.stream_reader(fh) as reader:
                buffered_reader = io.BufferedReader(reader)
                
                for line in buffered_reader:
                    try:
                        obj = orjson.loads(line)
                        sub = obj.get("subreddit")
                        
                        if not sub or sub not in whitelist_set:
                            continue
                            
                        author = obj.get("author", "")
                        if is_bot(author) or author in ("[deleted]", "[removed]"):
                            continue
                            
                        title = obj.get("title", "").strip()
                        selftext = obj.get("selftext", "").strip()
                        
                        if selftext in ("[deleted]", "[removed]"):
                            selftext = ""
                            
                        # Validierung wie in der Turbo-Version
                        if not is_valid_text(selftext):
                            continue
                            
                        full_text = f"{title} {selftext}".strip() if title else selftext
                        score = obj.get("score", 0)
                        
                        rows["subreddit"].append(sub)
                        rows["text"].append(full_text)
                        rows["score"].append(score)
                        
                        if len(rows["subreddit"]) >= CHUNK_SIZE:
                            table = pa.Table.from_arrays([
                                pa.array(rows["subreddit"]),
                                pa.array(rows["text"]),
                                pa.array(rows["score"])
                            ], schema=schema)
                            if writer is None:
                                writer = pq.ParquetWriter(output_path, schema)
                            writer.write_table(table)
                            rows = {"subreddit": [], "text": [], "score": []}
                            
                    except orjson.JSONDecodeError:
                        continue
        
        # Rest schreiben
        if rows["subreddit"]:
            table = pa.Table.from_arrays([
                pa.array(rows["subreddit"]),
                pa.array(rows["text"]),
                pa.array(rows["score"])
            ], schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(output_path, schema)
            writer.write_table(table)
            
    except Exception as e:
        print(f"    ❌ Fehler bei {input_path.name}: {e}")
    finally:
        if writer: writer.close()

# ====================================================================
# PROMPT-BUILDER (Bleibt Turbo mit DuckDB, da Daten hier klein sind)
# ====================================================================
def build_jsonl_safe(jahr):
    print(f"  [Schritt 2] Baue JSONL-Prompts für {jahr}...")
    INPUT_ORDNER = OUTPUT_ROOT / str(jahr)
    OUTPUT_PFAD  = Path(f"Data/cluster_prompts_{jahr}.jsonl")
    if not list(INPUT_ORDNER.glob("*.parquet")): return

    import duckdb
    con = duckdb.connect()
    
    # --- DUCKDB SPEICHER-KONFIGURATION ---
    # Limitiert RAM-Nutzung und aktiviert Spilling auf Festplatte
    con.execute("SET memory_limit = '10GB'")
    con.execute("SET threads = 2")
    con.execute("PRAGMA temp_directory = 'duckdb_temp_storage'")
    con.execute("SET preserve_insertion_order = false")
    # ------------------------------------

    query = f"""
        SELECT subreddit, list(text) AS posts
        FROM (
            SELECT subreddit, text,
            ROW_NUMBER() OVER (PARTITION BY subreddit ORDER BY score DESC) AS rn
            FROM read_parquet('{INPUT_ORDNER}/*.parquet')
        )
        WHERE rn <= 20 GROUP BY subreddit ORDER BY subreddit
    """
    try:
        cursor = con.execute(query)
        with open(OUTPUT_PFAD, "w", encoding="utf-8") as f:
            while True:
                chunk = cursor.fetchmany(1000)
                if not chunk: break
                for row in chunk:
                    subreddit, posts = row[0], row[1]
                    post_snippets = "\n".join(f"{i + 1}. {str(p).replace(chr(10), ' ')}" for i, p in enumerate(posts))
                    content_string = f"Input:\nSubreddit Name: r/{subreddit}\nTop {len(posts)} Posts:\n{post_snippets}\nOutput:"
                    f.write(json.dumps({"system_prompt": "...", "content": content_string}, ensure_ascii=False) + "\n")
        print(f"  ✅ JSONL fertig: {OUTPUT_PFAD.name}")
    finally:
        con.close()

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 START SAFE MASTER-PIPELINE (2023-2024)")
    print("=" * 50)

    import pandas as pd
    df_white = pd.read_csv(WHITELIST_PFAD)
    whitelist_set = set(df_white["subreddit"].tolist())

    for jahr in [2023, 2024]:
        print(f"\n{'=' * 16} JAHR {jahr} {'=' * 16}")
        OUT_DIR = OUTPUT_ROOT / str(jahr)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        
        zst_files = sorted(NVME_ROHDATEN.rglob(f"*{jahr}*.zst"))
        
        # Multiprocessing für die Extraktion
        tasks = []
        for f in zst_files:
            out_file = OUT_DIR / f.name.replace(".zst", "_texts.parquet")
            # Prüfe ob Datei existiert UND nicht leer ist
            if not out_file.exists() or out_file.stat().st_size == 0:
                tasks.append((f, out_file, whitelist_set))

        if tasks:
            print(f"  🚀 Starte Extraktion mit {MAX_WORKERS} Workern...")
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(process_single_zst_safe, *t) for t in tasks]
                for future in as_completed(futures):
                    future.result()  # Um Fehler abzufangen
        
        build_jsonl_safe(jahr)

    print("\n🎉 PIPELINE ABGESCHLOSSEN!")
