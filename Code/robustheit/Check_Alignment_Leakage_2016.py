

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

POL_CSV  = "all_cluster_results_political.csv"
ALIGNED  = "subreddit_clusters_aligned_2016_2024_LABEL2016.csv"   # nur LESEN
VEC_2016 = "Data/Vektoren/vektoren_2016.txt"
OUT_CSV  = "check_alignment_leakage_2016.csv"
JAHRE    = list(range(2016, 2025))

SEED_A = 42      # derselbe Seed wie im produktiven Lauf
SEED_B = 1234    # zweiter Seed, liefert das Nullniveau

t0 = time.time()


def log(msg):
    print(msg, flush=True)


def _ver(paketname, modul):
    """Version robust bestimmen. Manche hdbscan-Builds haben kein __version__."""
    try:
        from importlib.metadata import version
        return version(paketname)
    except Exception:
        return getattr(modul, "__version__", "unbekannt")


log("Versionen: umap-learn %s | hdbscan %s | numpy %s | python %s"
    % (_ver("umap-learn", umap), _ver("hdbscan", hdbscan),
       np.__version__, sys.version.split()[0]))


# =====================================================================
# 1. Lifestyle-Filter, wörtlich aus Cluster_Vektoren_LABEL2016.py
# =====================================================================
log("\nBaue Lifestyle-Set nach der 2016er-Label-Regel (mit Imputation)...")

pol = pd.read_csv(POL_CSV, dtype=str)
pol["subreddit"] = pol["subreddit"].str.replace(r"^r/", "", regex=True)
pol = pol.drop_duplicates(subset="subreddit")

jahr_cols = [str(j) for j in JAHRE]
lab = pol[jahr_cols].apply(lambda s: s.str.strip().str.lower())

lab16  = lab["2016"]
andere = lab[[c for c in jahr_cols if c != "2016"]]
n_true  = (andere == "true").sum(axis=1)
n_false = (andere == "false").sum(axis=1)

imput = np.where((n_true == 0) & (n_false > 0), "false",
        np.where((n_false == 0) & (n_true > 0), "true", None))

leer    = lab16.isna() | (lab16 == "")
lab_fix = np.where(leer, imput, lab16)

allowed_lifestyles = set(pol.loc[lab_fix == "false", "subreddit"])
log(f"  Lifestyle-Subs (2016er-Regel): {len(allowed_lifestyles)}")


# =====================================================================
# 2. Vektoren 2016 laden, auf dieselbe Menge einschränken
# =====================================================================
if not os.path.exists(VEC_2016):
    sys.exit(f"ABBRUCH: {VEC_2016} fehlt.")

model = KeyedVectors.load_word2vec_format(VEC_2016, binary=False)
valid_subs = [w for w in model.index_to_key if w in allowed_lifestyles]
valid_idx  = [model.key_to_index[w] for w in valid_subs]
vectors    = model.vectors[valid_idx]

log(f"  2016: {len(valid_subs)} Lifestyle-Vektoren (von {len(model.index_to_key)}).")

# Gegenprobe gegen die produktive Lösung. Muss übereinstimmen,
# sonst vergleichen wir zwei verschiedene Mengen.
df_aligned = pd.read_csv(ALIGNED)
a16 = df_aligned[df_aligned.jahr == 2016].set_index("subreddit")["cluster"]
log(f"  produktive Lösung 2016: {len(a16)} Subreddits")
if len(a16) != len(valid_subs):
    log("  ⚠ WARNUNG: Mengen unterschiedlich gross. Verglichen wird nur die "
        "Schnittmenge, das Ergebnis ist dann mit Vorsicht zu lesen.")


# =====================================================================
# 3. SOLO-Reduktionen, gewöhnliches UMAP ohne Kopplung
# =====================================================================
def solo_lauf(seed):
    log(f"\nUMAP solo, random_state={seed} ...")
    t = time.time()
    reducer = umap.UMAP(
        n_neighbors=30,
        n_components=15,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
    )
    emb = reducer.fit_transform(vectors)
    labels = hdbscan.HDBSCAN(
        min_cluster_size=30,
        min_samples=10,
        metric="euclidean",
        prediction_data=True,
    ).fit_predict(emb)
    k = len(set(labels)) - (1 if -1 in labels else 0)
    rausch = float((labels == -1).mean())
    log(f"  fertig nach {(time.time()-t)/60:.1f} min | {k} Cluster | "
        f"Rauschen {rausch:.4f}")
    return pd.Series(labels, index=valid_subs), k, rausch


solo_a, k_a, r_a = solo_lauf(SEED_A)
solo_b, k_b, r_b = solo_lauf(SEED_B)

k_al = int(a16[a16 != -1].nunique())
r_al = float((a16 == -1).mean())
log(f"\nproduktive Loesung 2016: {k_al} Cluster | Rauschen {r_al:.4f}")


# =====================================================================
# 4. Vergleich. Konvention wie in cluster_vergleich_LABEL2016.csv:
#    ARI/AMI nur auf Subreddits, die in BEIDEN Lösungen geclustert sind
#    (Rauschen ist keine Klasse).
# =====================================================================
def vergleich(name, s1, s2):
    gem = s1.index.intersection(s2.index)
    both = pd.DataFrame({"x": s1.loc[gem], "y": s2.loc[gem]})
    core = both[(both.x != -1) & (both.y != -1)]
    return {
        "vergleich": name,
        "n_gemeinsam_geclustert": len(core),
        "ARI": round(adjusted_rand_score(core.x, core.y), 4) if len(core) else np.nan,
        "AMI": round(adjusted_mutual_info_score(core.x, core.y), 4) if len(core) else np.nan,
    }


rows = [
    vergleich("aligned vs solo_a", a16, solo_a),
    vergleich("aligned vs solo_b", a16, solo_b),
    vergleich("solo_a vs solo_b (NULLNIVEAU)", solo_a, solo_b),
]

cmp = pd.DataFrame(rows)
cmp["k_aligned"] = k_al
cmp["k_solo_a"] = k_a
cmp["k_solo_b"] = k_b
cmp["rausch_aligned"] = round(r_al, 4)
cmp["rausch_solo_a"] = round(r_a, 4)
cmp["rausch_solo_b"] = round(r_b, 4)
cmp.to_csv(OUT_CSV, index=False)

log("\n" + "=" * 70)
log(cmp.to_string(index=False))
log("=" * 70)
log(f"\nGeschrieben: {OUT_CSV}")

log("""
LESEHILFE
  Entscheidend ist NICHT die Höhe von 'aligned vs solo', sondern der
  ABSTAND zum Nullniveau 'solo_a vs solo_b'.

  Liegen die beiden aligned-Zeilen auf dem Niveau der solo-Zeile,
  verschiebt die Kopplung die 2016er-Partition nicht stärker, als das
  Verfahren ohnehin zwischen zwei Läufen streut. Dann genügt im Text
  ein Satz.

  Liegen sie deutlich darunter, trägt die 2016er-Partition Struktur der
  Folgejahre. Dann ist die Beschreibung anzupassen und der Punkt gehört
  in die Limitationen, gemeinsam mit dem Hinweis zur Cluster-Verkettung
  in FF3.

  Vorsicht bei der Clusterzahl: weicht k_solo stark von k_aligned ab,
  sinkt ARI schon deshalb. Die Zahlen also zusammen lesen.
""")

log(f"Gesamtlaufzeit: {(time.time()-t0)/60:.1f} min")
