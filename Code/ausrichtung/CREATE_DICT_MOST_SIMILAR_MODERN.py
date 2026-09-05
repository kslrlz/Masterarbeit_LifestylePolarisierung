import pandas as pd
import numpy as np
from tqdm import tqdm
import gensim
import gensim.models
from gensim.models import KeyedVectors
import pickle
import time
import os

def run_most_similar(src_lang, trg_lang, src_path, trg_path, output_dir='SeNSe-main/SeNSe-main/data'):
    """
    Executes the SeNSe 'Most Similar Dictionary' creation logic.
    Updated for Gensim 4.0+ compatibility.
    
    Args:
        src_lang (str): Label for source year (e.g., '16')
        trg_lang (str): Label for target year (e.g., '17')
        src_path (str): File path to source word2vec .txt embeddings
        trg_path (str): File path to target word2vec .txt embeddings
        output_dir (str): Directory where the .pkl dictionaries will be saved
    """
    output_most_similar_dict_src = os.path.join(output_dir, f'most_similar_dictionary_{src_lang}.pkl')
    output_most_similar_dict_trg = os.path.join(output_dir, f'most_similar_dictionary_{trg_lang}.pkl')

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Starting process for labels: {src_lang} and {trg_lang}")
    print(f"Source Path: {src_path}")
    print(f"Target Path: {trg_path}")
    
    start_time = time.time()

    # 1. Load models
    print("Loading source model...")
    model_src = KeyedVectors.load_word2vec_format(src_path, binary=False, unicode_errors='ignore')
    
    print("Loading target model...")
    model_trg = KeyedVectors.load_word2vec_format(trg_path, binary=False, unicode_errors='ignore')

    # 2. Access vocabulary (Gensim 4.0+ compatibility: index_to_key replaces vocab)
    vocab_src = model_src.index_to_key
    vocab_trg = model_trg.index_to_key

    dict_src = {}
    dict_trg = {}

    # 3. Compute most similar for source (Author's logic: top 200)
    print(f"Computing Top-200 neighbors for {src_lang}...")
    for word in tqdm(vocab_src):
        most_similar_src = model_src.most_similar(word, topn=200)
        dict_src[word] = most_similar_src

    # Save source
    print(f"Saving {output_most_similar_dict_src}...")
    with open(output_most_similar_dict_src, "wb") as f:
        pickle.dump(dict_src, f)

    # 4. Compute most similar for target
    print(f"Computing Top-200 neighbors for {trg_lang}...")
    for word in tqdm(vocab_trg):
        most_similar_trg = model_trg.most_similar(word, topn=200)
        dict_trg[word] = most_similar_trg

    # Save target
    print(f"Saving {output_most_similar_dict_trg}...")
    with open(output_most_similar_dict_trg, "wb") as f:
        pickle.dump(dict_trg, f)

    end_time = time.time()
    print(f'Total elapsed time: {end_time - start_time:.2f} seconds')
    return output_most_similar_dict_src, output_most_similar_dict_trg
