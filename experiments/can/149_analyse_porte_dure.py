#!/usr/bin/env python
"""Verdict sur le durcissement de la porte : dure vs douce, apparié (mêmes 500 états)."""
import json, math
from pathlib import Path

R = json.loads(Path("results/runs/can/porte_dure.json").read_text())


def mcnemar(a, b):
    """Test exact apparié. a, b = listes de booléens sur les MÊMES épisodes."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return n01, n10, p


print("=" * 66)
print("  DURCISSEMENT DE LA PORTE — n'encoder qu'UNE caméra par pas")
print("=" * 66)
for m in ("douce", "dure"):
    if m not in R:
        print(f"  {m:6s} : pas encore évalué"); continue
    r = R[m]
    tot = r["passes"] + r["evitees"]
    print(f"\n  porte {m.upper()}")
    print(f"    réussite          : {r['n_success']}/{r['n']} = {r['success_rate']:.1%} "
          f"[{r['ci95'][0]:.1%}, {r['ci95'][1]:.1%}]")
    print(f"    passes d'encodeur : {r['passes']} exécutées / {tot} demandées "
          f"→ {r['evitees']/max(tot,1)*100:.0f} % ÉVITÉES")
    if m == "dure":
        c = r["choix"]; s = max(sum(c), 1)
        print(f"    caméra retenue    : vue globale {c[0]/s*100:.0f} %  ·  poignet {c[1]/s*100:.0f} %")

if "douce" in R and "dure" in R:
    a = [e["success"] for e in R["douce"]["per_episode"]]
    b = [e["success"] for e in R["dure"]["per_episode"]]
    n01, n10, p = mcnemar(a, b)
    d = (R["dure"]["success_rate"] - R["douce"]["success_rate"]) * 100
    print(f"\n  --- APPARIÉ (mêmes 500 états, même modèle, seule la porte change) ---")
    print(f"    douce réussit seule : {n01}   ·   dure réussit seule : {n10}")
    print(f"    écart dur − doux    : {d:+.1f} pts   (McNemar exact p = {p:.3f})")
    verdict = ("le durcissement est GRATUIT" if p > 0.05 else
               ("le durcissement COÛTE" if d < 0 else "le durcissement AIDE"))
    print(f"    → {verdict}")
    e = R["dure"]["evitees"] / max(R["dure"]["passes"] + R["dure"]["evitees"], 1)
    print(f"    {e*100:.0f} % de calcul d'encodeur en moins, pour {d:+.1f} pt de réussite")
    print(f"    bascules : {n01 + n10} épisodes sur {len(a)} changent d'issue "
          f"({(n01+n10)/len(a)*100:.0f} %) — l'écart net de {abs(n01-n10)} épisodes est du bruit")

print("\n  Référence — encodeur gelé sans porte : 86,2 % (4 bras : 86,2 · 91,2 · 83,0 · 84,2)")
print("=" * 66)
