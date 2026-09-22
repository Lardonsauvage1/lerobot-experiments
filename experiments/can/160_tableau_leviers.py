#!/usr/bin/env python
"""LE TABLEAU DES LEVIERS — ce qui marche, mesuré en ÉCHECS SUPPRIMÉS.

⚠️ `--md` RÉÉCRIT docs/TABLEAU_LEVIERS.md de zéro : ne jamais y ajouter de texte à la main,
il serait effacé. L'analyse narrative vit dans docs/BALAYAGE_UNET.md. — ce qui marche, mesuré en ÉCHECS SUPPRIMÉS.

Règle d'or : une ligne n'entre ici que si les deux bras diffèrent par UNE SEULE chose,
sur les MÊMES 500 états, et si possible à graine égale. Pas de comparaison bancale.

Pourquoi les échecs et pas les points : passer de 60 à 61 % est bien plus facile que de
90 à 91 %. La réduction d'échecs est comparable d'un niveau à l'autre, et elle révèle que
les leviers se composent MULTIPLICATIVEMENT (cf. docs/METHODOLOGIE.md §6).

⭐ POUR ÉTENDRE : ajouter un tuple dans PAIRES. Rien d'autre.
   ("nom du levier", "bras_avant", "bras_après", "note d'appariement")
Les noms sont les clés des JSON de résultats (cam2_eval500, porte_dure, pas_coupes,
balayage_unet...). `MANUELS` sert aux références dont le bras « avant » vient d'un autre
fichier ou d'une moyenne de graines.

Usage : venv312/bin/python experiments/can/160_tableau_leviers.py [--md]
"""
import argparse, json, math
from pathlib import Path

N = 500
RES = ["cam2_eval500.json", "porte_dure.json", "pas_coupes.json", "balayage_unet.json", "pas_debruitage.json",
       "refs_birdview.json",
       "boucle_fermee.json", "vision_500.json"]

PAIRES = [
    # --- banc BIRDVIEW (la vue réellement occultée, 57,7 % des pas) ---
    ("⭐ Poignet seul PLUTÔT que vue de dessus seule", "ref_birdview_seul", "ref_poignet_seul", "même recette, seule la caméra change"),
    ("Ajouter le poignet à la vue de dessus", "ref_birdview_seul", "bi_birdview_poignet", "meilleure méthode : départ du mono, aucun gel"),
    ("Ajouter la vue de dessus au poignet", "ref_poignet_seul", "bi_birdview_poignet", "même modèle bi-caméra, autre référence"),
    ("⭐ 5 pas de débruitage au lieu de 10 (gros décodeur)", "unet_128256512_gele_s42", "pas_5", "MÊME modèle, seul le réglage d'exécution change"),
    ("⭐ 5 pas de débruitage au lieu de 10 (petit décodeur)", "gel_A_sans_gel_s42", "pas_5_petit", "MÊME modèle"),
    ("20 pas de débruitage au lieu de 10",   "unet_128256512_gele_s42", "pas_20",  "MÊME modèle — au-delà du point de rendement"),
    ("20 pas de débruitage au lieu de 10 (petit décodeur)", "gel_A_sans_gel_s42", "pas_20_petit", "MÊME modèle — l'excès de pas COÛTE"),
    ("2 pas de débruitage au lieu de 10",    "unet_128256512_gele_s42", "pas_2",   "MÊME modèle — la falaise"),
    ("3 pas de débruitage au lieu de 10 (petit décodeur)", "gel_A_sans_gel_s42", "pas_3_petit", "MÊME modèle"),
    ("⭐ Décodeur ×2 · graine 42 APPARIÉE",  "J_fresh_wrist",      "unet_128256512_gele_s42", "graine 42, poignet NEUF des 2 côtés"),
    ("⭐ Décodeur ×2 · graine 43 APPARIÉE",  "ref_fresh_s43",      "unet_128256512_gele_s43", "graine 43, poignet NEUF des 2 côtés"),
    ("Poignet REPRIS vs NEUF (graine 42)",   "J_fresh_wrist",      "I_frozen",                "graine 42, décodeur identique"),
    ("Poignet REPRIS vs NEUF (graine 43)",   "ref_fresh_s43",      "I2_frozen_s43",           "graine 43, décodeur identique"),
    ("Décodeur ×2 (mono, graine 42)",       "A2_fixstats",        "unet_128256512_MONO_s42", "graine 42"),
    ("Décodeur ×2 (mono, graine 43)",       "A_side",             "unet_128256512_MONO_s43", "2e réplique de chaque côté"),
    ("Décodeur ×4 vs ×2 (mono)",            "unet_128256512_MONO_s42", "unet_2565121024_mono_s42", "graine 42 — le 2e doublement"),
    ("Décodeur ×4 vs ×2 (bi-caméra)",       "unet_128256512_gele_s42", "unet_2565121024_gele_s42", "graine 42 — le 2e doublement"),
    ("Décodeur ×2 (bi-caméra, graine 43)",  "I2_frozen_s43",      "unet_128256512_gele_s43", "graine 43"),
    ("Décodeur ×2 (bi-caméra gelé)",        "I_frozen",           "unet_128256512_gele_s42", "graine 42"),
    ("Encodeur GELÉ pour intégrer le poignet","A2_fixstats",      "I_frozen",                "graine 42"),
    ("Alterner plutôt que figer (porte)",   "L42_fixe_globale",   "L42_entrainee",           "MÊME modèle"),
    ("Rattrapages COUPÉS vs complets",      "R1_rattrapage",      "R3_correction",           "graine 42"),
    ("Encodeur poignet NEUF vs repris",     "J_fresh_wrist",      "I_frozen",                "graine 42"),
    ("Porte DURE vs porte souple",          "douce",              "dure",                    "MÊME modèle K"),
    ("2e caméra EXTÉRIEURE (dessus)",       "A2_fixstats",        "B2_fixstats",             "graine 42"),
    ("Dropout de caméra (poignet naïf)",    "C2_wrist_fixstats",  "C3_camdrop",              "graine 42"),
    ("Porte apprise vs TIRAGE AU SORT",     "L42_entrainee",      "L42_hasard",              "MÊME modèle L42"),
    ("+200 épisodes experts",               "A2_fixstats",        "R2_200expert",            "graine 42"),
    ("Routeur souple sur modèle gelé",      "I_frozen",           "K_frozen_router",         "graine 42"),
    ("Routeur souple sur poignet naïf",     "C2_wrist_fixstats",  "H_router",                "graine 42"),
    ("Rattrapages COMPLETS",                "R2_200expert",       "R1_rattrapage",           "graine 42, données égales"),
    ("Recadrage aléatoire (poignet naïf)",  "C2_wrist_fixstats",  "F_wrist_crop",            "graine 42"),
    ("Poignet en concaténation NAÏVE",      "A2_fixstats",        "C2_wrist_fixstats",       "graine 42"),
    ("Recadrage aléatoire (mono, 96 px)",   "A2_fixstats",        "G_side_crop",             "graine 42"),
]
# bras dont le taux vient d'ailleurs (moyenne de graines, autre campagne...)
MANUELS = {}


def charger():
    S, PE = {}, {}
    for f in RES:
        p = Path("results/runs/can") / f
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        items = ({r["name"]: r for r in d["results"]} if isinstance(d, dict) and "results" in d
                 else {k: v for k, v in d.items() if isinstance(v, dict) and "success_rate" in v})
        for k, v in items.items():
            S[k] = v["success_rate"]
            if v.get("per_episode"):
                PE[k] = [e["success"] if isinstance(e, dict) else e for e in v["per_episode"]]
    S.update(MANUELS)
    return S, PE


def mcnemar(A, B):
    """Test exact apparié. Compare deux MODÈLES ; il ne voit pas le hasard d'entraînement."""
    n01 = sum(1 for x, y in zip(A, B) if x and not y)
    n10 = sum(1 for x, y in zip(A, B) if y and not x)
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="écrit docs/TABLEAU_LEVIERS.md")
    a = ap.parse_args()
    S, PE = charger()
    lignes, manquants = [], []
    for nom, av, ap_, note in PAIRES:
        if av not in S or ap_ not in S:
            manquants.append((nom, av if av not in S else ap_)); continue
        ea, eb = (1 - S[av]) * N, (1 - S[ap_]) * N
        p = mcnemar(PE[av], PE[ap_]) if av in PE and ap_ in PE else None
        lignes.append({"nom": nom, "avant": S[av], "apres": S[ap_], "ea": ea, "eb": eb,
                       "red": (1 - eb / ea) * 100, "p": p, "note": note})
    lignes.sort(key=lambda x: -x["red"])

    def fmt(l):
        ps = "n/d" if l["p"] is None else (f"{l['p']:.4f}" if l["p"] >= 1e-4 else "<1e-4")
        return (l["nom"], f"{l['avant']*100:.1f} → {l['apres']*100:.1f} %",
                f"{l['ea']:.0f} → {l['eb']:.0f}", f"{l['red']:+.1f} %", ps, l["note"])

    print(f"\n  {'levier':40s} {'réussite':>16s} {'échecs':>11s} {'échecs -':>9s}  {'McNemar':>8s}")
    print("  " + "-" * 96)
    for l in lignes:
        n, r, e, d, ps, _ = fmt(l)
        print(f"  {n:40s} {r:>16s} {e:>11s} {d:>9s}  {ps:>8s}")
    if manquants:
        print("\n  (pas encore mesuré : " + ", ".join(f"{n} [{k}]" for n, k in manquants) + ")")

    if a.md:
        pos = [l for l in lignes if l["red"] > 0]
        neg = [l for l in lignes if l["red"] <= 0]
        out = ["# Le tableau des leviers — ce qui marche, en échecs supprimés", "",
               "*Généré par `experiments/can/160_tableau_leviers.py`. Ne pas éditer à la main.*", "",
               "Une ligne n'entre ici que si les deux bras diffèrent par **une seule chose**, sur les",
               "**mêmes 500 états**, et si possible à graine égale. Les gains sont donnés en **réduction",
               "du nombre d'échecs** et pas seulement en points : passer de 60 à 61 % est bien plus facile",
               "que de 90 à 91 % (cf. `METHODOLOGIE.md` §6).", "",
               "⚠️ McNemar compare deux **modèles**, pas deux **configurations** : il ne voit pas le hasard",
               "d'entraînement. Une ligne à p < 0,05 sur une seule graine reste à confirmer.", ""]
        for titre, gr in (("## Ce qui améliore", pos), ("## Ce qui dégrade", neg)):
            out += [titre, "", "| levier | réussite | échecs /500 | échecs en moins | McNemar | appariement |",
                    "|---|---|---|---|---|---|"]
            for l in gr:
                n, r, e, d, ps, note = fmt(l)
                out.append(f"| {n} | {r} | {e} | **{d}** | {ps} | {note} |")
            out.append("")
        Path("docs/TABLEAU_LEVIERS.md").write_text("\n".join(out))
        print("\n  -> docs/TABLEAU_LEVIERS.md")


if __name__ == "__main__":
    main()
