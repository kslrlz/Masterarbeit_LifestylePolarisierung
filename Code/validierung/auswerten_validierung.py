"""
Auswertung der LLM-Validierung. Start nach dem Labeln:  python auswerten_validierung.py
Vergleicht deine Hand-Labels (meine_labels.csv) gegen das LLM (schluessel.csv).
Rechnet: Confusion-Matrix, Accuracy, Precision/Recall/F1 (pro Klasse), Cohen's kappa,
         Auswertung pro Schicht, prevalenz-gewichtete Accuracy, + Fehlerliste.
"""
import csv, os, sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

BASE = os.path.dirname(os.path.abspath(__file__))
def _find(name, dirs):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p): return p
    return os.path.join(dirs[0], name)
# Script darf im Projekt-Root ODER im LLM_Validierung-Ordner liegen
DIR = os.path.dirname(_find("schluessel.csv", [BASE, os.path.join(BASE,"LLM_Validierung"), os.path.dirname(BASE)]))
POL_CSV = _find("all_cluster_results_political.csv",
                [BASE, os.path.dirname(BASE), os.path.dirname(DIR), os.path.dirname(os.path.dirname(BASE))])
EXCLUDE = {"r/the_donald","r/latestagecapitalism","r/worldnews","r/audiophile","r/vegan","r/personalfinance"}

def load(p): return list(csv.DictReader(open(p, encoding="utf-8")))
key   = {r["id"]: r for r in load(os.path.join(DIR,"schluessel.csv"))}
mine  = {r["id"]: r["mein_label"] for r in load(os.path.join(DIR,"meine_labels.csv"))}

def tb(x): return True if str(x).strip().lower()=="true" else (False if str(x).strip().lower()=="false" else None)

# gepaarte Fälle (ohne 'unsicher'/leer)
rows=[]; n_unsure=0
for vid, ml in mine.items():
    if vid not in key: continue
    g = tb(ml); p = tb(key[vid]["llm_label"])
    if g is None: n_unsure += 1; continue
    rows.append((vid, g, p, key[vid]["stratum"]))

if not rows:
    print("Noch keine (eindeutigen) Labels in meine_labels.csv. Erst labeln (python label_llm.py)."); sys.exit()

# Confusion: positiv = politisch (true). gold=du, pred=LLM
TP=sum(1 for _,g,p,_ in rows if g and p)
FP=sum(1 for _,g,p,_ in rows if not g and p)
FN=sum(1 for _,g,p,_ in rows if g and not p)
TN=sum(1 for _,g,p,_ in rows if not g and not p)
n=len(rows)
acc=(TP+TN)/n
def f1(tp,fp,fn):
    pr=tp/(tp+fp) if tp+fp else 0.0; rc=tp/(tp+fn) if tp+fn else 0.0
    f=2*pr*rc/(pr+rc) if pr+rc else 0.0; return pr,rc,f
pr_p,rc_p,f1_p=f1(TP,FP,FN)   # Klasse politisch
pr_l,rc_l,f1_l=f1(TN,FN,FP)   # Klasse lifestyle
# Cohen's kappa
po=acc
p_g_pos=(TP+FN)/n; p_p_pos=(TP+FP)/n
pe=p_g_pos*p_p_pos+(1-p_g_pos)*(1-p_p_pos)
kappa=(po-pe)/(1-pe) if (1-pe) else 0.0

print(f"\n=== LLM-Validierung — {n} gepaarte Faelle ({n_unsure} 'unsicher' ausgeschlossen) ===\n")
print("Confusion-Matrix (Zeile = DU/Gold, Spalte = LLM):")
print(f"                 LLM:politisch   LLM:lifestyle")
print(f"  DU:politisch        {TP:4d}            {FN:4d}")
print(f"  DU:lifestyle        {FP:4d}            {TN:4d}\n")
print(f"Accuracy (Stichprobe):           {acc:.3f}")
print(f"Cohen's kappa:                   {kappa:.3f}   ({'sehr gut' if kappa>=.8 else 'gut' if kappa>=.6 else 'maessig' if kappa>=.4 else 'schwach'})")
print(f"Klasse POLITISCH  Precision={pr_p:.3f}  Recall={rc_p:.3f}  F1={f1_p:.3f}")
print(f"Klasse LIFESTYLE  Precision={pr_l:.3f}  Recall={rc_l:.3f}  F1={f1_l:.3f}")
print(f"Macro-F1:                        {(f1_p+f1_l)/2:.3f}")

# pro Schicht
print("\n--- Accuracy pro Schicht ---")
acc_by={}
for st in ["stabil_politisch","stabil_lifestyle","grenz_flacker"]:
    sub=[r for r in rows if r[3]==st]
    if sub:
        a=sum(1 for _,g,p,_ in sub if g==p)/len(sub); acc_by[st]=a
        print(f"  {st:18s}: {a:.3f}  (n={len(sub)})")

# prevalenz-gewichtete Accuracy (Schicht-Größen der Population aus der CSV)
pop={"stabil_politisch":0,"stabil_lifestyle":0,"grenz_flacker":0}
with open(POL_CSV,encoding="utf-8") as fh:
    r=csv.reader(fh); next(r)
    for row in r:
        if row[0].lower() in EXCLUDE: continue
        vals={tb(v) for v in row[1:] if tb(v) is not None}
        if len(vals)<1: continue
        if sum(1 for v in row[1:] if tb(v) is not None)<2: continue
        if vals=={True}: pop["stabil_politisch"]+=1
        elif vals=={False}: pop["stabil_lifestyle"]+=1
        else: pop["grenz_flacker"]+=1
tot=sum(pop.values())
if all(st in acc_by for st in pop) and tot:
    wacc=sum(acc_by[st]*pop[st]/tot for st in pop)
    print(f"\nPrevalenz-gewichtete Accuracy (auf Population hochgerechnet): {wacc:.3f}")
    print(f"  Populationsanteile: " + ", ".join(f"{st}={pop[st]/tot:.1%}" for st in pop))

# Fehlerliste
errs=[(vid,g,p,st) for vid,g,p,st in rows if g!=p]
print(f"\n--- {len(errs)} Abweichungen (LLM != du) ---")
fout=os.path.join(DIR,"fehleranalyse.csv")
with open(fout,"w",encoding="utf-8",newline="") as fh:
    w=csv.writer(fh); w.writerow(["id","subreddit","jahr","stratum","dein_label","llm_label","llm_reasoning"])
    for vid,g,p,st in errs:
        k=key[vid]
        w.writerow([vid,k["subreddit"],k["jahr"],st,g,p,k["llm_reasoning"]])
        print(f"  {vid} {k['subreddit']} ({k['jahr']}) [{st}]: du={g} LLM={p}")
        print(f"      LLM-reasoning: {k['llm_reasoning'][:160]}")
print(f"\nFehlerliste gespeichert: {fout}")
print("-> Fuer die Bias-Diskussion: schau, ob die Fehler thematisch clustern (z.B. Identitaet/LGBT/Feminismus ueber-, rechts-Lifestyle unterflaggt).")
