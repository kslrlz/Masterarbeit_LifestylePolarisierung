import json
import os
import re
from pathlib import Path

# Konfiguration
INPUT_DIR = Path("Data")
OUTPUT_DIR = Path("Data/Fixed_Prompts")
MAX_CHARS_PER_POST = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)

def fix_jsonl(file_path):
    print(f"🔧 Repariere: {file_path.name}")
    output_path = OUTPUT_DIR / file_path.name
    
    with open(file_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            try:
                data = json.loads(line)
                content = data.get("content", "")
                
                # Wir splitten den Content in Header (Name/Count) und die nummerierte Liste
                # Beispiel: "Input:\nSubreddit Name: r/xyz\nTop 20 Posts:\n1. text\n2. text..."
                
                parts = content.split("\n")
                new_parts = []
                
                for part in parts:
                    # Prüfen ob die Zeile mit einer Nummer beginnt (z.B. "1. ", "20. ")
                    if re.match(r'^\d+\. ', part):
                        # Nummer und Text trennen
                        match = re.match(r'^(\d+\. )(.*)', part)
                        if match:
                            prefix, text = match.groups()
                            # Text kürzen auf 500 Zeichen und Markierung setzen, falls gekürzt
                            if len(text) > MAX_CHARS_PER_POST:
                                new_parts.append(prefix + text[:MAX_CHARS_PER_POST] + " [...]")
                            else:
                                new_parts.append(prefix + text)
                        else:
                            new_parts.append(part)
                    else:
                        # Header-Zeilen einfach übernehmen
                        new_parts.append(part)
                
                data["content"] = "\n".join(new_parts)
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                
            except Exception as e:
                print(f"  ⚠️ Fehler in Zeile: {e}")
                continue

    print(f"✅ Fertig: {output_path.name}")

if __name__ == "__main__":
    jsonl_files = list(INPUT_DIR.glob("cluster_prompts_*.jsonl"))
    for f in jsonl_files:
        fix_jsonl(f)
