"""
Klassifikation der Subreddits als politisch oder nicht politisch.

Dies ist der Lauf, aus dem die in der Arbeit verwendeten Etiketten stammen:
vLLM mit meta-llama/Meta-Llama-3.1-8B-Instruct, temperature 0.0 und festem Seed,
ausgeführt auf dem Rechencluster der Universität Leipzig.

Eingabe  Data/cluster_prompts_<jahr>.jsonl   (aus multi_llm_pipeline.py)
Ausgabe  Data/cluster_results_<jahr>.jsonl

Der Systemprompt steht in jeder Zeile der Eingabedatei und liegt zusätzlich
als system_prompt.txt daneben. HF_TOKEN wird vor dem Start exportiert.
"""

import json
import os
from vllm import LLM, SamplingParams

# Der absolut echte, saubere Pfad auf dem Cluster!
BASE_PATH = "/work2/sc45vuvu-ma_classification"

# CUDA & HF Environments
os.environ["LD_LIBRARY_PATH"] = "/software/all/CUDA/12.8.0/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["CUDA_HOME"] = "/software/all/CUDA/12.8.0"
os.environ["PATH"] = f"{BASE_PATH}/llm_env/bin:/software/all/CUDA/12.8.0/bin:" + os.environ.get("PATH", "")
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["HF_HOME"]  = f"{BASE_PATH}/hf_cache_fresh"
# HF_TOKEN wird vor dem Start exportiert, nicht im Skript hinterlegt

LOCAL_DIR = f"{BASE_PATH}/llama3_local"

print("Initialisiere vLLM Engine (nur EINMAL für alle Jahre!)...")
sampling_params = SamplingParams(temperature=0.0, seed=42, max_tokens=600)

llm = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    tokenizer=LOCAL_DIR,
    dtype="bfloat16",
    gpu_memory_utilization=0.90,
    max_model_len=30000,
    disable_log_stats=True,
    tokenizer_mode="auto", 
)

# Schleife über alle Jahre von 2016 bis 2024
jahre = range(2016, 2025)

for year in jahre:
    input_file  = f"{BASE_PATH}/Data/cluster_prompts_{year}.jsonl"
    output_file = f"{BASE_PATH}/Data/cluster_results_{year}.jsonl"
    
    if not os.path.exists(input_file):
        print(f"\n[WARNUNG] Datei für {year} nicht gefunden. Überspringe...")
        continue

    print(f"\n" + "="*40)
    print(f"Bearbeite Jahr: {year}")
    print("="*40)
    
    prompts = []
    original_data = []

    # Daten laden
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            original_data.append(data)
            system_prompt = data.get("system_prompt", "") + "\nOutput EXCLUSIVELY valid JSON. Start directly with {."
            user_content  = data.get("content", "")
            prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_prompt}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_content}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            prompts.append(prompt)

    print(f"{len(prompts)} Prompts geladen. Starte Generierung...")
    outputs = llm.generate(prompts, sampling_params)

    # Ergebnisse speichern
    print("Schreibe Ergebnisse auf Platte...")
    with open(output_file, "w", encoding="utf-8") as outfile:
        for i, output in enumerate(outputs):
            result_dict = original_data[i]
            result_dict["model_output"] = output.outputs[0].text
            outfile.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
            
            # Alle 1000 Schritte den Puffer flushen
            if (i + 1) % 1000 == 0 or (i + 1) == len(prompts):
                outfile.flush()
                os.fsync(outfile.fileno())

    print(f"[ERFOLG] Jahr {year} komplett abgeschlossen und gespeichert in {output_file}!")

print("\nAlle Jahre wurden erfolgreich verarbeitet. Programm beendet.")
