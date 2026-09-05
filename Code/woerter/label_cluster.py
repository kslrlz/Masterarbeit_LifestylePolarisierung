#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interaktives Labeln der nativen Cluster für FF3.

    python label_cluster.py                      # loslegen
    python label_cluster.py --jahre 2016:2024    # mehr Jahre in den Pool
    python label_cluster.py --stand              # nur Fortschritt anzeigen
    python label_cluster.py --export             # Pool als CSV rausschreiben

WARUM SO
--------
Auf der nativen Partition wird jedes Jahr neu geclustert. Die Cluster-IDs sind
über die Jahre nicht vergleichbar, Cluster 26 in 2016 hat nichts mit Cluster 26
in 2024 zu tun. Analyseeinheit wird deshalb die KATEGORIE, die bleibt stabil,
auch wenn ein Cluster sich aufspaltet.

Drei Regeln stecken im Aufbau:

  BLIND      Angezeigt werden nur Größe, Top-Woerter und Mitglieder. Kein Jahr,
             keine Cluster-ID, kein Partisan-Score, und die Reihenfolge ist
             gemischt. Sähe man das Ergebnis beim Labeln, wäre die Rechnung
             zirkulär.

  UNABHAENGIG  Jeder Cluster wird für sich gelabelt, auch wenn er in mehreren
             Jahren fast gleich aussieht. Das kostet Zeit, liefert aber eine
             Test-Retest-Reliabilität: die Übereinstimmung zwischen inhaltlich
             fast identischen Clustern aus verschiedenen Jahren (Notebook 7.4).
             Beim Vererben von Jahr zu Jahr gäbe es diese Zahl nicht und Fehler
             würden sich fortpflanzen.

  ZWEI EBENEN  Freies Label (was es konkret ist) plus Kategorie (Analyseeinheit).
             Feiner labeln als nötig, nach oben zusammenfassen geht später
             jederzeit, aufspalten nicht.

Gespeichert wird an `nr_key` = "jahr_cluster", nicht an der laufenden Nummer.
Änderst du später die Jahresauswahl, bleiben die Labels korrekt zugeordnet.
Geschrieben wird nach jedem Eintrag, ein Absturz kostet also nichts.
"""

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- Konfiguration
PROJ       = os.path.dirname(os.path.abspath(__file__))
PARTITION  = os.path.join(PROJ, "subreddit_clusters_aligned_2016_2024_LABEL2016.csv")
CTFIDF     = os.path.join(PROJ, "Data", "ctfidf_top50_LABEL2016.csv")
SCORES     = os.path.join(PROJ, "politischer_wandel_2016_2024.csv")
LABELDATEI = os.path.join(PROJ, "Auswertung_CSV", "ff3_labels.csv")
POOLDATEI  = os.path.join(PROJ, "Auswertung_CSV", "ff3_labelpool.csv")

SEED       = 20260723
N_WOERTER  = 25    # in der Standardansicht, ueber --woerter aenderbar
N_SUBS     = 25    # dito, ueber --subs

FUELLER = {
    "get", "like", "would", "one", "people", "really", "also", "much", "think",
    "know", "good", "even", "still", "well", "time", "make", "want", "way",
    "things", "see", "could", "thanks", "thank", "please", "new", "use", "using",
    "need", "first", "back", "shit", "fuck", "lol", "amp", "yes", "love", "got",
    "going", "said", "say", "dont",
}

HILFE = """
  LABEL     = was dieser Cluster konkret ist, frei formuliert, so genau wie du
              magst ("League of Legends", "Heimwerken"). Material für die
              Fallvignetten. Muss sich nicht wiederholen.
  KATEGORIE = die Analyseeinheit. Sie fasst Cluster zu Domänen zusammen, die
              über die Jahre verglichen werden ("Gaming", "Region", "Sport").
              Nur was sich WIEDERHOLT, taugt als Kategorie. Faustregel: über
              alle Cluster hinweg höchstens ein paar Dutzend verschiedene.

  <text>              freies Label setzen und weiter
  <text> | <kat>      Label und Kategorie setzen
                      Kategorie darf abgekürzt werden, ein eindeutiger Präfix
                      wird zur bestehenden Kategorie ausgeschrieben (Gam -> Gaming)
  <leer>              überspringen, später nochmal

  w                   ALLE Top-Woerter dieses Clusters
  m                   ALLE Mitglieder dieses Clusters
  k                   alle bisher vergebenen Kategorien
  l                   alle bisher vergebenen freien Labels
  s                   Fortschritt anzeigen
  z                   letzten Eintrag zurücknehmen
  !                   Jahr und Cluster-ID dieses Clusters aufdecken (sparsam!)
  ?                   diese Hilfe
  q                   speichern und beenden
"""


# ----------------------------------------------------------------- Daten
def jahre_parsen(text):
    """'2016,2020,2024' oder '2016:2024' -> Liste."""
    if ":" in text:
        a, b = text.split(":")
        return list(range(int(a), int(b) + 1))
    return [int(j.strip()) for j in text.split(",")]


def pool_bauen(jahre):
    """Baut die Vorlage: ein Eintrag je nativem Cluster und Jahr."""
    for pfad in (PARTITION, CTFIDF, SCORES):
        if not os.path.exists(pfad):
            sys.exit(f"Datei fehlt: {pfad}")

    part = pd.read_csv(PARTITION)
    part = part[(part.cluster != -1) & (part.jahr.isin(jahre))]

    roh = pd.read_csv(SCORES)
    roh = roh.rename(columns={roh.columns[0]: "subreddit"})
    scores = roh.melt(id_vars="subreddit", var_name="jahr", value_name="score")
    scores["jahr"] = scores.jahr.str.replace("score_", "", regex=False).astype(int)
    scores = scores.dropna(subset=["score"])

    # z-Referenz = voller 2016er Embeddingraum, wie im Notebook seit 23.07.2026
    ref = scores[scores.jahr == 2016].score
    scores["z"] = (scores.score - ref.mean()) / ref.std(ddof=1)

    nat = part.merge(scores[["subreddit", "jahr", "z"]], on=["subreddit", "jahr"])
    if nat.empty:
        sys.exit("Keine Subreddits mit Score in den gewählten Jahren.")

    # Beitrag jedes Clusters zu between, wie im Notebook Teil 3.
    # Nur für die Fortschrittsanzeige, im Labeln taucht er nirgends auf.
    jahr_stat = nat.groupby("jahr").z.agg(["mean", "size"]).rename(
        columns={"mean": "gm", "size": "N"})
    grp = nat.groupby(["jahr", "cluster"]).z.agg(["size", "mean"]).rename(
        columns={"size": "n_c", "mean": "mittel"}).reset_index()
    grp = grp.merge(jahr_stat, on="jahr")
    grp["b"] = (grp.n_c / grp.N) * (grp.mittel - grp.gm) ** 2
    grp["nr_key"] = grp.jahr.astype(str) + "_" + grp.cluster.astype(str)

    # Top-Woerter.
    # na_filter=False, weil sonst echte Tokens zu Fehlwerten werden: "null" steht
    # 7x in der Datei, dazu kämen "nan", "na", "none", "inf". Der Preis ist, dass
    # rang als Text ankommt und explizit umgewandelt werden muss.
    tf = pd.read_csv(CTFIDF, na_filter=False)
    # feste int64, nicht die nullable Int64: sonst passt der Join-Index unten
    # nicht zu dem aus der Partitionsdatei und die Woerter kämen leer zurück.
    tf["rang"]    = pd.to_numeric(tf["rang"], errors="coerce")
    tf["year"]    = pd.to_numeric(tf["year"], errors="coerce").astype("int64")
    tf["cluster"] = pd.to_numeric(tf["cluster"], errors="coerce").astype("int64")
    tf = tf[tf.year.isin(jahre) & ~tf.token.isin(FUELLER)]
    # VOLLSTAENDIG behalten und erst bei der Anzeige kürzen, damit "w" alles zeigen kann
    tf = tf.sort_values("rang")
    woerter = tf.groupby(["year", "cluster"]).token.apply(", ".join).rename("woerter_alle")
    woerter.index.names = ["jahr", "cluster"]

    # Mitglieder vollständig, aber deterministisch gemischt. Die ersten N sind
    # damit eine Zufallsstichprobe und nicht die alphabetisch ersten, "m" zeigt
    # trotzdem die ganze Liste.
    rng = random.Random(SEED)

    def gemischt(s):
        v = sorted(s)
        rng.shuffle(v)
        return ", ".join(v)

    mitglieder = nat.groupby(["jahr", "cluster"]).subreddit.apply(gemischt).rename("subs_alle")

    pool = (grp.set_index(["jahr", "cluster"])
               .join(woerter).join(mitglieder).reset_index())
    pool["woerter_alle"] = pool.woerter_alle.fillna("(keine Top-Woerter in der c-TF-IDF-Datei)")

    # deterministisch sortieren, DANN mischen -> Nummerierung reproduzierbar
    pool = pool.sort_values(["jahr", "cluster"]).reset_index(drop=True)
    pool = pool.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    pool["nr"] = [f"{i + 1:04d}" for i in range(len(pool))]
    return pool[["nr", "nr_key", "jahr", "cluster", "n_c", "b", "woerter_alle", "subs_alle"]]


def kopf(text, n):
    """Die ersten n kommagetrennten Einträge."""
    teile = text.split(", ")
    return ", ".join(teile[:n]), len(teile)


def zu_linien(pool, schwelle):
    """Fasst Cluster verschiedener Jahre mit fast gleicher Mitgliedschaft zu
    LINIEN zusammen, damit derselbe Cluster nicht bis zu neunmal gelabelt wird.

    Verbunden wird über Jaccard der Mitgliedermengen, nur über Jahresgrenzen
    hinweg, danach transitive Hülle. Bei 0.5 umfasst die größte Linie genau
    neun Cluster, also höchstens einen je Jahr: es werden keine verschiedenen
    Cluster desselben Jahres zusammengeworfen. Das ist eine mechanische,
    dokumentierte Regel und keine Einzelfallentscheidung.

    PREIS: die Test-Retest-Zahl aus Notebook 7.4 wird WERTLOS. Sie misst die
    Übereinstimmung zwischen Clustern mit Jaccard >= 0.5 über Jahresgrenzen,
    und genau die bilden hier eine Linie mit einem gemeinsamen Label. Das
    Ergebnis wäre 100 Prozent per Konstruktion. Zelle 7.4 gehört im
    Linienbetrieb NICHT in den Methodenteil, Ersatz wäre ein bewusster
    Doppeldurchgang über eine Zufallsstichprobe von Linien.
    """
    mengen = {r.nr_key: set(r.subs_alle.split(", ")) for r in pool.itertuples()}
    jahr = dict(zip(pool.nr_key, pool.jahr))

    # Kandidatenpaare über invertierten Index statt alle Paare zu prüfen
    inv = {}
    for k, s in mengen.items():
        for sub in s:
            inv.setdefault(sub, []).append(k)
    gemeinsam = {}
    for ks in inv.values():
        if len(ks) < 2:
            continue
        ks = sorted(ks)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                if jahr[ks[i]] != jahr[ks[j]]:
                    paar = (ks[i], ks[j])
                    gemeinsam[paar] = gemeinsam.get(paar, 0) + 1

    eltern = {k: k for k in mengen}

    def finde(x):
        while eltern[x] != x:
            eltern[x] = eltern[eltern[x]]
            x = eltern[x]
        return x

    for (a, b), n in gemeinsam.items():
        if n / (len(mengen[a]) + len(mengen[b]) - n) >= schwelle:
            ra, rb = finde(a), finde(b)
            if ra != rb:
                eltern[ra] = rb

    p = pool.copy()
    p["linie"] = [finde(k) for k in p.nr_key]

    # Vertreter je Linie = größter Cluster. Seine Woerter und Mitglieder
    # werden angezeigt, weil sie am wenigsten von Zufall geprägt sind.
    p = p.sort_values("n_c", ascending=False)
    vertreter = p.groupby("linie", as_index=False).first()
    zusatz = p.groupby("linie").agg(
        n_cluster=("nr_key", "size"),
        n_jahre=("jahr", "nunique"),
        b_summe=("b", "sum"),
        n_c_summe=("n_c", "sum"),
        nr_keys=("nr_key", list)).reset_index()

    linien = vertreter.drop(columns=["b", "n_cluster"], errors="ignore").merge(zusatz, on="linie")
    linien = linien.sort_values("nr").reset_index(drop=True)
    linien["nr"] = [f"{i + 1:04d}" for i in range(len(linien))]
    linien = linien.rename(columns={"b_summe": "b", "n_c_summe": "n_c_gesamt"})
    return linien


def labels_laden():
    spalten = ["nr", "nr_key", "label", "kategorie"]
    if not os.path.exists(LABELDATEI):
        return pd.DataFrame(columns=spalten)
    df = pd.read_csv(LABELDATEI, dtype=str).fillna("")
    for s in spalten:
        if s not in df.columns:
            df[s] = ""
    return df[spalten]


def labels_speichern(df):
    os.makedirs(os.path.dirname(LABELDATEI), exist_ok=True)
    df.sort_values("nr_key").to_csv(LABELDATEI, index=False, encoding="utf-8")


# ----------------------------------------------------------------- Anzeige
def trennlinie(zeichen="-", breite=78):
    print(zeichen * breite)


def umbrechen(text, praefix, breite=78):
    """Wortumbruch, damit lange Mitgliederlisten lesbar bleiben.
    Folgezeilen werden auf Präfixbreite eingerückt statt es zu wiederholen."""
    einzug = " " * len(praefix)
    zeilen, akt = [], praefix
    for wort in text.split(" "):
        if akt not in (praefix, einzug) and len(akt) + 1 + len(wort) > breite:
            zeilen.append(akt)
            akt = einzug + wort
        else:
            akt = akt + wort if akt.endswith(" ") else akt + " " + wort
    zeilen.append(akt)
    return "\n".join(zeilen)


def fortschritt(pool, labels):
    fertig = set(labels[labels.label != ""].nr_key)
    p = pool.copy()
    p["fertig"] = p.nr_key.isin(fertig)
    n, ges = int(p.fertig.sum()), len(p)
    print()
    trennlinie("=")
    print(f" Cluster gelabelt    : {n} von {ges}  ({100 * n / ges:.1f} %)")
    print(f" Subreddits abgedeckt: {100 * p.n_c[p.fertig].sum() / p.n_c.sum():.1f} %")
    print(f" between abgedeckt   : {100 * p.b[p.fertig].sum() / p.b.sum():.1f} %")
    print()
    print(" je Jahr:")
    for jahr, g in p.groupby("jahr"):
        print(f"   {jahr}   {int(g.fertig.sum()):>4} / {len(g):<4} Cluster"
              f"   {100 * g.n_c[g.fertig].sum() / g.n_c.sum():>5.1f} % Subs"
              f"   {100 * g.b[g.fertig].sum() / g.b.sum():>5.1f} % between")
    trennlinie("=")
    print()


def kategorien_zeigen(labels):
    k = labels[labels.kategorie != ""].kategorie.value_counts()
    print()
    if k.empty:
        print("  noch keine Kategorien vergeben")
    else:
        print(f"  {len(k)} Kategorien bisher:")
        for name, n in k.items():
            print(f"    {n:>4}x  {name}")
    print()


def labels_zeigen(labels):
    """Alle bisher vergebenen freien Labels, nach Kategorie gruppiert."""
    df = labels[labels.label != ""]
    print()
    if df.empty:
        print("  noch nichts gelabelt")
        print()
        return
    print(f"  {df.label.nunique()} verschiedene Labels bisher:")
    for kat, g in df.groupby(df.kategorie.replace("", "(ohne Kategorie)")):
        namen = sorted(set(g.label))
        print(f"\n    {kat}  ({len(namen)})")
        print(umbrechen(", ".join(namen), "      "))
    print()


def kategorien_kompakt(labels, max_zeilen=2, breite=78):
    """Einzeiler über dem Prompt, damit die Benennung konsistent bleibt.

    Zeigt nur die Kategorien-VOKABULAR, nicht welcher Cluster welche bekommen
    hat. Die Blindheit bleibt damit unberührt, es hilft nur gegen
    Synonyme wie 'Gaming' neben 'Games'.
    """
    k = labels[labels.kategorie != ""].kategorie.value_counts()
    if k.empty:
        return "  bisher  : noch keine Kategorien vergeben"
    teile = [f"{name} ({n})" for name, n in k.items()]
    zeilen, akt, gezeigt = [], "  bisher  : ", 0
    for t in teile:
        kandidat = akt + ("" if akt.endswith(": ") else "  ") + t
        if len(kandidat) > breite:
            zeilen.append(akt)
            if len(zeilen) >= max_zeilen:
                rest = len(teile) - gezeigt
                zeilen[-1] += f"  +{rest} weitere"
                return "\n".join(zeilen)
            akt = "            " + t
        else:
            akt = kandidat
        gezeigt += 1
    zeilen.append(akt)
    return "\n".join(zeilen)


def kategorie_ergaenzen(eingabe, bekannt):
    """Eindeutigen Präfix zu einer bestehenden Kategorie ausschreiben."""
    if not eingabe or eingabe in bekannt:
        return eingabe, False
    treffer = [k for k in bekannt if k.lower().startswith(eingabe.lower())]
    if len(treffer) == 1:
        return treffer[0], True
    return eingabe, False


def linien_auffuellen(einheiten, labels):
    """Labels, die an einzelnen Clustern hängen, auf ihre ganze Linie ausweiten.

    Nötig für alles, was vor dem Linienbetrieb gelabelt wurde: sonst gilt die
    Linie als erledigt, die übrigen Cluster hätten aber keine Zeile und
    landeten in der Auswertung unter "(ungelabelt)".
    Bei widersprüchlichen Labels innerhalb einer Linie wird nichts ergänzt,
    sondern gemeldet.
    """
    vorhanden = labels[labels.label != ""].drop_duplicates("nr_key").set_index("nr_key")
    neu, konflikte = [], []
    for r in einheiten.itertuples():
        drin = [k for k in r.nr_keys if k in vorhanden.index]
        if not drin or len(drin) == len(r.nr_keys):
            continue
        paare = {(vorhanden.at[k, "label"], vorhanden.at[k, "kategorie"]) for k in drin}
        if len(paare) > 1:
            konflikte.append((r.nr, sorted(p[0] for p in paare)))
            continue
        lab, kat = paare.pop()
        neu += [{"nr": r.nr, "nr_key": k, "label": lab, "kategorie": kat}
                for k in r.nr_keys if k not in vorhanden.index]

    for nr, labs in konflikte:
        print(f"  Linie [{nr}] hat widersprüchliche Labels: {', '.join(labs)} "
              "-> nichts ergänzt, bitte prüfen")
    if neu:
        labels = pd.concat([labels, pd.DataFrame(neu)], ignore_index=True)
        labels_speichern(labels)
        print(f"  {len(neu)} Cluster aus bereits gelabelten Linien ergänzt")
    return labels


# ----------------------------------------------------------------- Hauptschleife
def labeln(einheiten, pool, labels):
    """einheiten = Linien (oder einzelne Cluster bei --einzeln).
    pool = die Cluster-Ebene, nur für die Fortschrittsanzeige."""
    verlauf = []          # zuletzt gesetzte Linien, fuer z
    uebersprungen = set()

    def erledigt(ks, fertig):
        return any(k in fertig for k in ks)

    while True:
        fertig = set(labels[labels.label != ""].nr_key)
        maske = [not erledigt(ks, fertig) and nr not in uebersprungen
                 for ks, nr in zip(einheiten.nr_keys, einheiten.nr)]
        offen = einheiten[maske]
        if offen.empty:
            if uebersprungen:
                print(f"\nAlles bearbeitet, {len(uebersprungen)} uebersprungen. "
                      "Neustart des Skripts zeigt sie wieder.\n")
            else:
                print("\nAlles gelabelt.\n")
            return labels

        zeile = offen.iloc[0]
        n_fertig = sum(erledigt(ks, fertig) for ks in einheiten.nr_keys)
        n_ges = len(einheiten)
        anteil_b = 100 * pool.b[pool.nr_key.isin(fertig)].sum() / pool.b.sum()

        spanne = (f"   in {int(zeile.n_jahre)} Jahren" if int(zeile.n_cluster) > 1 else "")
        print()
        trennlinie()
        print(f" [{zeile.nr}]   n = {int(zeile.n_c)} Subreddits{spanne}"
              f"          {n_fertig}/{n_ges} gelabelt   |   between {anteil_b:.1f} %")
        trennlinie()
        w_text, w_ges = kopf(zeile.woerter_alle, N_WOERTER)
        s_text, s_ges = kopf(zeile.subs_alle, N_SUBS)
        print(umbrechen(w_text, "  Woerter : "))
        if w_ges > N_WOERTER:
            print(f"            ... {w_ges} insgesamt, 'w' zeigt alle")
        print(umbrechen(s_text, "  Subs    : "))
        if s_ges > N_SUBS:
            print(f"            ... {s_ges} insgesamt, 'm' zeigt alle")
        trennlinie()
        print(kategorien_kompakt(labels))

        try:
            eingabe = input("  Label | Kategorie > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nabgebrochen, Stand ist gespeichert.\n")
            return labels

        if eingabe == "q":
            print("\nbeendet, Stand ist gespeichert.\n")
            return labels
        if eingabe == "?":
            print(HILFE)
            continue
        if eingabe == "w":
            print()
            print(umbrechen(zeile.woerter_alle, f"  alle {w_ges} Woerter : "))
            print()
            continue
        if eingabe == "m":
            print()
            print(umbrechen(zeile.subs_alle, f"  alle {s_ges} Subs : "))
            print()
            continue
        if eingabe == "k":
            kategorien_zeigen(labels)
            continue
        if eingabe == "l":
            labels_zeigen(labels)
            continue
        if eingabe == "s":
            fortschritt(pool, labels)
            continue
        if eingabe == "!":
            print(f"  -> {int(zeile.n_cluster)} Cluster: {', '.join(sorted(zeile.nr_keys))}   "
                  "(jede Aufdeckung schwächt die Blindheit)")
            continue
        if eingabe == "z":
            if not verlauf:
                print("  nichts zurückzunehmen")
                continue
            weg = verlauf.pop()
            labels = labels[~labels.nr_key.isin(weg)].copy()
            labels_speichern(labels)
            print(f"  zurückgenommen: {len(weg)} Cluster")
            continue
        if eingabe == "":
            uebersprungen.add(zeile.nr)
            continue

        # --- Label und Kategorie trennen ---
        if "|" in eingabe:
            label, kategorie = [t.strip() for t in eingabe.split("|", 1)]
        else:
            label, kategorie = eingabe, ""
        if not label:
            print("  leeres Label, nichts gesetzt")
            continue

        bekannt = sorted(set(labels[labels.kategorie != ""].kategorie))
        kategorie, ergaenzt = kategorie_ergaenzen(kategorie, bekannt)
        if ergaenzt:
            print(f"  Kategorie ergänzt zu: {kategorie}")

        # Das Label gilt für ALLE Cluster der Linie. Geschrieben wird weiterhin
        # eine Zeile je Cluster, damit das Notebook nichts davon wissen muss.
        ks = list(zeile.nr_keys)
        labels = labels[~labels.nr_key.isin(ks)].copy()
        labels = pd.concat([labels, pd.DataFrame(
            [{"nr": zeile.nr, "nr_key": k, "label": label, "kategorie": kategorie}
             for k in ks])], ignore_index=True)
        labels_speichern(labels)
        verlauf.append(ks)
        if len(ks) > 1:
            print(f"  gesetzt für {len(ks)} Cluster der Linie")


# ----------------------------------------------------------------- Einstieg
def main():
    global N_WOERTER, N_SUBS
    ap = argparse.ArgumentParser(description="Interaktives Labeln der nativen Cluster (FF3)")
    ap.add_argument("--jahre", default="2016,2020,2024",
                    help="z.B. '2016,2020,2024' oder '2016:2024' (Standard: drei Stützjahre)")
    ap.add_argument("--woerter", type=int, default=N_WOERTER,
                    help=f"Top-Woerter in der Standardansicht (Vorgabe {N_WOERTER})")
    ap.add_argument("--subs", type=int, default=N_SUBS,
                    help=f"Mitglieder in der Standardansicht (Vorgabe {N_SUBS})")
    ap.add_argument("--einzeln", action="store_true",
                    help="jeden Cluster einzeln labeln statt Linien zusammenzufassen")
    ap.add_argument("--jaccard", type=float, default=0.5,
                    help="Schwelle, ab der Cluster verschiedener Jahre eine Linie bilden (0.5)")
    ap.add_argument("--stand", action="store_true", help="nur Fortschritt anzeigen")
    ap.add_argument("--export", action="store_true", help="Pool nach ff3_labelpool.csv schreiben")
    ap.add_argument("--labeldatei", default=None,
                    help="abweichende Labeldatei, z.B. zum gefahrlosen Ausprobieren")
    args = ap.parse_args()
    N_WOERTER, N_SUBS = args.woerter, args.subs
    if args.labeldatei:
        global LABELDATEI
        LABELDATEI = os.path.abspath(args.labeldatei)
        print(f"Labeldatei: {LABELDATEI}")

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    jahre = jahre_parsen(args.jahre)
    print(f"\nbaue Pool fuer {jahre} ...")
    pool = pool_bauen(jahre)
    labels = labels_laden()
    print(f"Pool: {len(pool)} Cluster   |   Labeldatei: {len(labels)} Einträge")

    waisen = set(labels[labels.label != ""].nr_key) - set(pool.nr_key)
    if waisen:
        print(f"Hinweis: {len(waisen)} Label(s) gehören zu Jahren ausserhalb der Auswahl. "
              "Sie bleiben erhalten und zählen hier nur nicht mit.")

    if args.export:
        os.makedirs(os.path.dirname(POOLDATEI), exist_ok=True)
        pool.drop(columns=["b"]).to_csv(POOLDATEI, index=False, encoding="utf-8")
        print(f"-> {POOLDATEI}")
        return

    if args.stand:
        fortschritt(pool, labels)
        kategorien_zeigen(labels)
        return

    # --- Einheiten bestimmen: Linien oder einzelne Cluster ---
    if args.einzeln:
        einheiten = pool.copy()
        einheiten["nr_keys"] = [[k] for k in einheiten.nr_key]
        einheiten["n_cluster"] = 1
        einheiten["n_jahre"] = 1
    else:
        print(f"fasse Cluster zu Linien zusammen (Jaccard >= {args.jaccard}) ...")
        einheiten = zu_linien(pool, args.jaccard)
        mehrfach = int((einheiten.n_cluster > 1).sum())
        print(f"Linien: {len(einheiten)} statt {len(pool)} Clustern"
              f"   ({mehrfach} davon über mehrere Jahre,"
              f" größte {int(einheiten.n_cluster.max())} Cluster)")
        print("Ein Label gilt für alle Cluster seiner Linie. '--einzeln' schaltet das ab.")
        labels = linien_auffuellen(einheiten, labels)

    fortschritt(pool, labels)
    print(HILFE)
    labels = labeln(einheiten, pool, labels)
    fortschritt(pool, labels)
    print(f"gespeichert nach {LABELDATEI}")
    print("Weiter im Notebook mit Zelle 7.3 (Zerlegung) und 7.4 (Reliabilitaet).\n")


if __name__ == "__main__":
    main()
