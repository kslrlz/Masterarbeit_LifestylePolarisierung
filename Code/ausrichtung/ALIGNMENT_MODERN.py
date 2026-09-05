import pandas as pd
import numpy as np
import gensim
from tqdm import tqdm
import pickle
import collections
import copy
from gensim.models import KeyedVectors
import math
import time
import multiprocessing
from multiprocessing import Pool
import statistics
import os

# --- Hilfsfunktionen der Autoren ---

def ndcg_score(list_ranks, list_scores):
    """Berechnet den Semantic NDCG Score für ein Anker-Paar."""
    score = 0
    for s, r in zip(list_scores, list_ranks):
        add = s / (math.log((r + 2), 2))
        score += add
    return score

def smart_procrustes_align_gensim(base_vectors, other_vectors):
    """
    Berechnet die orthogonale Rotationsmatrix (SVD).
    """
    # Normalisierung
    base_vecs = base_vectors / np.linalg.norm(base_vectors, axis=1, keepdims=True)
    other_vecs = other_vectors / np.linalg.norm(other_vectors, axis=1, keepdims=True)

    # Matrix-Punktprodukt
    m = other_vecs.T.dot(base_vecs)
    # Singulärwertzerlegung (SVD)
    u, _, v = np.linalg.svd(m)
    # Die Rotationsmatrix Q
    ortho = u.dot(v)
    return ortho

# --- Hauptfunktion ---

def run_sense_alignment(
    lang_src, lang_trg, 
    path_vec_src, path_vec_trg,
    root_dir='SeNSe-main/SeNSe-main/',
    top_similar=35, 
    tolerance_limit=0.13,
    perform_alignment=True
):
    """
    Führt das komplette SeNSe Alignment durch oder gibt nur die Anker zurück.
    
    Args:
        ...
        perform_alignment (bool): Wenn False, wird nach der Anker-Berechnung gestoppt
                                  und nur der DataFrame der Anker zurückgegeben.
    """
    data_dir = os.path.join(root_dir, 'data')
    output_dir = os.path.join(root_dir, 'output')
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # 1. Laden der Modelle und Dictionaries
    print("Lade Vektoren und Dictionaries...")
    model_src = KeyedVectors.load_word2vec_format(path_vec_src, binary=False, unicode_errors='ignore')
    model_trg = KeyedVectors.load_word2vec_format(path_vec_trg, binary=False, unicode_errors='ignore')
    
    with open(os.path.join(data_dir, f'translation_dictionary_{lang_src}_{lang_trg}.pkl'), "rb") as f:
        dict_trans_src_trg = pickle.load(f)
    with open(os.path.join(data_dir, f'translation_dictionary_{lang_trg}_{lang_src}.pkl'), "rb") as f:
        dict_trans_trg_src = pickle.load(f)
    with open(os.path.join(data_dir, f'most_similar_dictionary_{lang_src}.pkl'), "rb") as f:
        dict_sim_src = pickle.load(f)
    with open(os.path.join(data_dir, f'most_similar_dictionary_{lang_trg}.pkl'), "rb") as f:
        dict_sim_trg = pickle.load(f)

    vocab_src = model_src.index_to_key
    vocab_trg = model_trg.index_to_key

    # 2. Schnittmenge (Common Words) bilden
    dict_common = {}
    for word in vocab_src:
        trans = dict_trans_src_trg.get(word)
        if trans and trans in model_trg:
            dict_common[word] = trans

    # 3. Anker-Stabilität (SNDCG) berechnen
    print("Berechne semantische Stabilität (SNDCG)...")
    dict_anchors_scores = {}
    for src_word, trg_word in tqdm(dict_common.items()):
        sim_src = dict_sim_src[src_word][:top_similar]
        list_ranks, list_scores = [], []
        for n_word, _ in sim_src:
            n_trans = dict_trans_src_trg.get(n_word)
            if n_trans in model_trg:
                score = model_trg.similarity(n_trans, trg_word)
                list_scores.append(score)
                list_ranks.append(0) # Vereinfacht für dieses Modul
        
        final_score = ndcg_score(list_ranks, list_scores)
        if final_score > 0:
            dict_anchors_scores[(src_word, trg_word)] = final_score

    # 4. Filterung und Auswahl der besten Anker
    print(f"Filterung der Anker (Toleranz: {tolerance_limit})...")
    df_anchors = pd.DataFrame([{'src': k[0], 'trg': k[1], 'score': v} for k, v in dict_anchors_scores.items()])
    df_anchors = df_anchors.sort_values('score', ascending=False).drop_duplicates('src').drop_duplicates('trg')
    
    # Kürzen der schlechtesten Anker
    num_to_keep = int(len(df_anchors) * (1 - tolerance_limit))
    df_anchors = df_anchors.head(num_to_keep)
    
    if not perform_alignment:
        print(f"Vorgang gestoppt. Gebe {len(df_anchors)} berechnete Anker zurück.")
        return df_anchors

    # 5. Finale Transformation (Alignment)
    print(f"Führe Alignment mit {len(df_anchors)} Ankern durch...")
    src_anchors_vecs = np.array([model_src[row['src']] for _, row in df_anchors.iterrows()])
    trg_anchors_vecs = np.array([model_trg[row['trg']] for _, row in df_anchors.iterrows()])
    
    # Rotationsmatrix berechnen
    ortho_matrix = smart_procrustes_align_gensim(trg_anchors_vecs, src_anchors_vecs)
    
    # Gesamten Quell-Raum projizieren
    all_vectors_src = model_src.vectors
    projected_vectors = all_vectors_src.dot(ortho_matrix)
    
    # Speichern der Ergebnisse
    out_path = os.path.join(output_dir, f'projected_{lang_src}_onto_{lang_trg}.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"{len(model_src)} {model_src.vector_size}\n")
        for i, word in enumerate(model_src.index_to_key):
            vec_str = " ".join([f"{v:.6f}" for v in projected_vectors[i]])
            f.write(f"{word} {vec_str}\n")
            
    print(f"Alignment abgeschlossen! Datei gespeichert unter: {out_path}")
    return df_anchors, out_path
