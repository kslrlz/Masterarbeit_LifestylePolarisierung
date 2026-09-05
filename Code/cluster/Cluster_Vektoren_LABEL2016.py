"""
=======================================================================
 KOPIE der Clustering-Pipeline mit der 2016er-LABEL-REGEL (Robustheitscheck)
=======================================================================
 ZWECK
 -----
 Das Original (Cluster_Vektoren.ipynb, Zelle 19) filtert Lifestyle-Subreddits
 über die EINTRITTSJAHR-Regel:
     is_political == False im ERSTEN Jahr, in dem ein Label existiert.
 Das R-Auswertungs-Notebook nutzt dagegen die 2016er-LABEL-Regel:
     2016er-Label == "false"; fehlt es, Imputation aus 2017-2024, ABER nur wenn
     eindeutig (immer true / immer false). Uneindeutige -> raus.
 Differenz: 691 Subreddits in der gescorten Landschaft (2,4 %), davon 65 in fix2016.

 In der AUSWERTUNG ist diese Differenz nachweislich folgenlos (eta^2 nativ 2024:
 0.3689 ungefiltert vs. 0.3690 gefiltert). OFFEN ist, ob sie auch die CLUSTERLOESUNG
 unberührt lässt -- denn die 691 Subs waren beim Schätzen der Cluster dabei.
 Genau das prüft dieses Skript.

 WAS ES TUT
 ----------
 Identische Pipeline (gleiche Vektoren, gleiche AlignedUMAP-/HDBSCAN-Parameter,
 gleicher random_state), NUR der Lifestyle-Filter ist ausgetauscht. Am Ende:
 Vergleich mit der bestehenden Lösung (Clusterzahl, Rauschanteil, ARI, AMI).

 SICHERHEIT
 ----------
 - Schreibt AUSSCHLIESSLICH nach subreddit_clusters_aligned_2016_2024_LABEL2016.csv
   und cluster_vergleich_LABEL2016.csv. Das Original wird NICHT angefasst.
 - Laufzeit: Stunden (AlignedUMAP über 9 Jahre). RAM-hungrig -> nichts Grosses
   parallel laufen lassen.

 START
 -----
   python Cluster_Vektoren_LABEL2016.py > log_cluster_label2016.txt 2>&1
=======================================================================
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import umap
import hdbscan
from gensim.models import KeyedVectors
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

PROJ = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJ)

POL_CSV   = "all_cluster_results_political.csv"
ORIGINAL  = "subreddit_clusters_aligned_2016_2024.csv"          # nur LESEN
OUT_CSV   = "subreddit_clusters_aligned_2016_2024_LABEL2016.csv"
CMP_CSV   = "cluster_vergleich_LABEL2016.csv"
JAHRE     = list(range(2016, 2025))

t0 = time.time()


def log(msg):
    print(msg, flush=True)


# =====================================================================
# 0. Lifestyle-Filter nach der 2016er-LABEL-REGEL (= Logik des R-Notebooks)
# =====================================================================
log("Baue Lifestyle-Set nach der 2016er-Label-Regel (mit Imputation)...")

pol = pd.read_csv(POL_CSV, dtype=str)
pol["subreddit"] = pol["subreddit"].str.replace(r"^r/", "", regex=True)
pol = pol.drop_duplicates(subset="subreddit")

jahr_cols = [str(j) for j in JAHRE]
lab = pol[jahr_cols].apply(lambda s: s.str.strip().str.lower())

lab16   = lab["2016"]
andere  = lab[[c for c in jahr_cols if c != "2016"]]
n_true  = (andere == "true").sum(axis=1)
n_false = (andere == "false").sum(axis=1)

# Imputation nur bei Eindeutigkeit; sonst NaN -> fällt raus
imput = np.where((n_true == 0) & (n_false > 0), "false",
        np.where((n_false == 0) & (n_true > 0), "true", None))

leer    = lab16.isna() | (lab16 == "")
lab_fix = np.where(leer, imput, lab16)

allowed_lifestyles = set(pol.loc[lab_fix == "false", "subreddit"])
n_unklar = int(((lab_fix == None) | pd.isna(lab_fix)).sum())  # noqa: E711

log(f"  Lifestyle-Subs (2016er-Regel):        {len(allowed_lifestyles)}")
log(f"  uneindeutig / ohne Label -> raus:     {n_unklar}")

# Zum Vergleich: was die EINTRITTSJAHR-Regel (Original) auswählen würde
df_long = pd.read_csv(POL_CSV).melt(id_vars="subreddit", var_name="jahr",
                                    value_name="is_political").dropna()
df_long["jahr"] = df_long["jahr"].astype(int)
df_first = df_long.sort_values("jahr").groupby("subreddit").first()
entry_rule = set(df_first[df_first["is_political"] == False]  # noqa: E712
                 .index.str.replace("r/", "", regex=False))

log(f"  Zum Vergleich, Eintrittsjahr-Regel:   {len(entry_rule)}")
log(f"  nur in Eintrittsjahr-Regel (= die strittigen): {len(entry_rule - allowed_lifestyles)}")
log(f"  nur in 2016er-Regel:                  {len(allowed_lifestyles - entry_rule)}\n")


# =====================================================================
# 1. Vektoren laden (identisch zum Original)
# =====================================================================
data_list, subreddits_list, geladene_jahre = [], [], []

for jahr in JAHRE:
    if jahr == 2016:
        path = "Data/Vektoren/vektoren_2016.txt"
    else:
        path = f"SeNSe-main/SeNSe-main/output/projected_{str(jahr)[-2:]}_onto_16_FULL.txt"

    if not os.path.exists(path):
        log(f"WARNUNG: {path} fehlt -> Jahr {jahr} übersprungen.")
        continue

    model = KeyedVectors.load_word2vec_format(path, binary=False)
    valid_subs = [w for w in model.index_to_key if w in allowed_lifestyles]
    valid_idx = [model.key_to_index[w] for w in valid_subs]

    subreddits_list.append(valid_subs)
    data_list.append(model.vectors[valid_idx])
    geladene_jahre.append(jahr)
    log(f"Jahr {jahr}: {len(valid_subs)} Lifestyle-Vektoren (von {len(model.index_to_key)}).")

if len(data_list) != 9:
    sys.exit(f"ABBRUCH: nur {len(data_list)} Jahre geladen, erwartet 9.")


# =====================================================================
# 2. Relations (identisch zum Original)
# =====================================================================
relation_dicts = []
for i in range(len(data_list) - 1):
    idx_next = {sub: j for j, sub in enumerate(subreddits_list[i + 1])}
    relation_dicts.append({j: idx_next[sub]
                           for j, sub in enumerate(subreddits_list[i])
                           if sub in idx_next})
log(f"\n{len(relation_dicts)} Relations-Dicts erstellt.")


# =====================================================================
# 3. AlignedUMAP (identische Parameter, identischer random_state)
# =====================================================================
log("\nStarte AlignedUMAP (dauert Stunden)...")
aligned_mapper = umap.AlignedUMAP(
    n_neighbors=30,
    n_components=15,
    min_dist=0.0,
    metric="cosine",
    alignment_regularisation=0.03,
    alignment_window_size=1,
    random_state=42,
)
aligned_mapper.fit(data_list, relations=relation_dicts)
log(f"AlignedUMAP fertig nach {(time.time()-t0)/60:.1f} min.")


# =====================================================================
# 4. HDBSCAN je Jahr (identische Parameter)
# =====================================================================
all_results = []
for i, jahr in enumerate(geladene_jahre):
    labels = hdbscan.HDBSCAN(
        min_cluster_size=30,
        min_samples=10,
        metric="euclidean",
        prediction_data=True,
    ).fit_predict(aligned_mapper.embeddings_[i])

    all_results.append(pd.DataFrame({"subreddit": subreddits_list[i],
                                     "jahr": jahr,
                                     "cluster": labels}))
    k = len(set(labels)) - (1 if -1 in labels else 0)
    log(f"Jahr {jahr}: {k} Cluster | Rauschen: {list(labels).count(-1)}")

df_new = pd.concat(all_results, ignore_index=True)
df_new.to_csv(OUT_CSV, index=False)
log(f"\nGeschrieben: {OUT_CSV}  (Original unberuehrt)")


# =====================================================================
# 5. VERGLEICH mit der bestehenden Lösung
# =====================================================================
log("\n" + "=" * 70)
log("VERGLEICH: 2016er-Label-Regel vs. bestehende Lösung (Eintrittsjahr-Regel)")
log("=" * 70)

df_old = pd.read_csv(ORIGINAL)
rows = []

for jahr in geladene_jahre:
    a = df_old[df_old.jahr == jahr].set_index("subreddit")["cluster"]
    b = df_new[df_new.jahr == jahr].set_index("subreddit")["cluster"]
    gemeinsam = a.index.intersection(b.index)

    # ARI/AMI nur auf geclusterten Subs (Rauschen ist keine Klasse)
    both = pd.DataFrame({"alt": a.loc[gemeinsam], "neu": b.loc[gemeinsam]})
    core = both[(both.alt != -1) & (both.neu != -1)]

    rows.append({
        "jahr": jahr,
        "k_alt": a[a != -1].nunique(),
        "k_neu": b[b != -1].nunique(),
        "n_alt": len(a),
        "n_neu": len(b),
        "rausch_alt": round((a == -1).mean(), 4),
        "rausch_neu": round((b == -1).mean(), 4),
        "n_gemeinsam_geclustert": len(core),
        "ARI": round(adjusted_rand_score(core.alt, core.neu), 4) if len(core) else np.nan,
        "AMI": round(adjusted_mutual_info_score(core.alt, core.neu), 4) if len(core) else np.nan,
    })

cmp = pd.DataFrame(rows)
cmp.to_csv(CMP_CSV, index=False)
log(cmp.to_string(index=False))
log(f"\nGeschrieben: {CMP_CSV}")

log("\nLESEHILFE:")
log("  ARI/AMI nahe 1.00  -> praktisch dieselbe Partition; der Filter ist folgenlos,")
log("                        die bestehenden FF2-Zahlen bleiben gültig.")
log("  ARI ab ca. 0.90    -> gleiche Struktur, Randverschiebungen. Unkritisch, in einem")
log("                        Satz berichtbar ('Clusterlösung robust gegen die Filterwahl').")
log("  ARI < ca. 0.80     -> die Partition hängt an der Filterwahl. Dann müsste die")
log("                        Auswertung auf dieser Lösung wiederholt werden.")
log(f"\nGesamtlaufzeit: {(time.time()-t0)/60:.1f} min")
