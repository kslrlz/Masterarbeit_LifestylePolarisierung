import pandas as pd
import numpy as np
import gensim
from tqdm import tqdm
import pickle
import math
import time
import os
import statistics
from gensim.models import KeyedVectors

# --- Core Math Functions (Identical to Authors) ---

def ndcg_score(list_ranks, list_scores):
    """Calculates the Semantic NDCG score exactly as defined in the paper."""
    score = 0
    for s, r in zip(list_scores, list_ranks):
        # r+2 because rank 0 is position 1, and log2(2)=1
        add = s / (math.log((r + 2), 2))
        score += add
    return score

def smart_procrustes_align_gensim(base_vectors, other_vectors):
    """
    Orthogonal Procrustes using Singular Value Decomposition (SVD).
    Ensures an orthogonal transformation (rotation/reflection).
    """
    # Length normalization (Unit length vectors)
    base_vecs = base_vectors / np.linalg.norm(base_vectors, axis=1, keepdims=True)
    other_vecs = other_vectors / np.linalg.norm(other_vectors, axis=1, keepdims=True)

    m = other_vecs.T.dot(base_vecs)
    u, _, v = np.linalg.svd(m)
    ortho = u.dot(v)
    return ortho

# --- Main Alignment Process ---

def run_full_sense_alignment(
    lang_src, lang_trg, 
    path_vec_src, path_vec_trg,
    root_dir='SeNSe-main/SeNSe-main/',
    top_similar=35, 
    tolerance_limit=0.13,
    perform_alignment=True
):
    """
    Full implementation of the SeNSe algorithm including Step 3: Dispersion.
    """
    data_dir = os.path.join(root_dir, 'data')
    output_dir = os.path.join(root_dir, 'output')
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    print("Step 0: Loading models and dictionaries...")
    model_src = KeyedVectors.load_word2vec_format(path_vec_src, binary=False, unicode_errors='ignore')
    model_trg = KeyedVectors.load_word2vec_format(path_vec_trg, binary=False, unicode_errors='ignore')
    
    with open(os.path.join(data_dir, f'translation_dictionary_{lang_src}_{lang_trg}.pkl'), "rb") as f:
        dict_trans_src_trg = pickle.load(f)
    with open(os.path.join(data_dir, f'most_similar_dictionary_{lang_src}.pkl'), "rb") as f:
        dict_sim_src = pickle.load(f)
    with open(os.path.join(data_dir, f'most_similar_dictionary_{lang_trg}.pkl'), "rb") as f:
        dict_sim_trg = pickle.load(f)

    vocab_src = model_src.index_to_key
    vocab_trg = model_trg.index_to_key

    # 1. Select Common Words
    dict_common = {}
    for word in vocab_src:
        trans = dict_trans_src_trg.get(word)
        if trans and trans in model_trg:
            dict_common[word] = trans

    # 2. Compute SNDCG (Semantic Stablity)
    print("Step 1 & 2: Computing SNDCG and Selecting Best Anchors...")
    dict_anchors_scores = {}
    for src_word, trg_word in tqdm(dict_common.items()):
        # Source to Target Analysis
        most_similar_src_all = dict_sim_src[src_word][:top_similar]
        list_ranks, list_scores = [], []
        
        # Original Rank-based Logic
        for rank, (neigh_word, _) in enumerate(most_similar_src_all):
            neigh_trans = dict_trans_src_trg.get(neigh_word)
            if neigh_trans and neigh_trans in model_trg:
                score = model_trg.similarity(neigh_trans, trg_word)
                list_scores.append(score)
                list_ranks.append(rank)
        
        score = ndcg_score(list_ranks, list_scores)
        if score > 0:
            dict_anchors_scores[(src_word, trg_word)] = score

    # 3. Deduplication and Tolerance Cutting
    df_anchors = pd.DataFrame([{'src': k[0], 'trg': k[1], 'score': v} for k, v in dict_anchors_scores.items()])
    df_anchors = df_anchors.sort_values('score', ascending=False).drop_duplicates('src').drop_duplicates('trg')
    
    # Normalizing scores (Authors do this before cutting)
    min_s, max_s = df_anchors['score'].min(), df_anchors['score'].max()
    df_anchors['score'] = (df_anchors['score'] - min_s) / (max_s - min_s)
    
    # Cutting worst anchors
    num_to_keep = int(len(df_anchors) * (1 - tolerance_limit))
    df_anchors = df_anchors.head(num_to_keep)
    print(f"Anchors after tolerance filter: {len(df_anchors)}")

    # 4. Dispersion Part (Ensuring anchors are spread out)
    # This matches the logic from the authors' ALIGNMENT.py
    print("Step 3: Dispersion (Removing spatially clustered anchors)...")
    
    # Determine the similarity threshold for dispersion
    list_top_sim_values = []
    for _, row in df_anchors.iterrows():
        # Get similarity of the very first neighbor
        list_top_sim_values.append(dict_sim_trg[row['trg']][0][1])
    
    # Clean the distribution (remove outliers) as authors did
    q1, q3 = np.percentile(list_top_sim_values, [5, 95])
    clean_vals = [v for v in list_top_sim_values if q1 <= v <= q3]
    limit_similarity = statistics.mean(clean_vals)
    
    # Iteratively remove similar anchors
    # We sort by score (best first) and remove neighbors of high-score anchors
    final_anchor_keys = []
    removed_trg_words = set()
    
    # Sort dict for dispersion
    anchor_list = df_anchors.to_dict('records')
    
    for anchor in tqdm(anchor_list):
        if anchor['trg'] in removed_trg_words:
            continue
            
        final_anchor_keys.append(anchor)
        
        # Check neighbors of this anchor in target space
        # If a neighbor is another potential anchor, remove the neighbor
        neighbors = dict_sim_trg[anchor['trg']]
        for neigh, sim_val in neighbors:
            if sim_val >= limit_similarity:
                removed_trg_words.add(neigh)
            else:
                break # Neighbors are sorted by similarity

    df_final_anchors = pd.DataFrame(final_anchor_keys)
    print(f"Anchors after Dispersion: {len(df_final_anchors)}")

    if not perform_alignment:
        return df_final_anchors

    # 5. Final Transformation (Alignment)
    print(f"Step 4 & 5: Performing Procrustes Alignment...")
    src_anchors_vecs = np.array([model_src[row['src']] for _, row in df_final_anchors.iterrows()])
    trg_anchors_vecs = np.array([model_trg[row['trg']] for _, row in df_final_anchors.iterrows()])
    
    # Calculate Q
    ortho_matrix = smart_procrustes_align_gensim(trg_anchors_vecs, src_anchors_vecs)
    
    # Project complete space
    projected_vectors = model_src.vectors.dot(ortho_matrix)
    
    # Save results
    out_path = os.path.join(output_dir, f'projected_{lang_src}_onto_{lang_trg}_FULL.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"{len(model_src)} {model_src.vector_size}\n")
        for i, word in enumerate(model_src.index_to_key):
            vec_str = " ".join([f"{v:.6f}" for v in projected_vectors[i]])
            f.write(f"{word} {vec_str}\n")
            
    print(f"Alignment complete! Saved to: {out_path}")
    return df_final_anchors, out_path
