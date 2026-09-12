#!/usr/bin/env python
"""Graphe comparatif du verdict : baseline vs CAMP vs témoin, sur le banc corrigé.

Trois barres avec IC95 de Wilson, plus les cycles descente/remontée par échec — car si la
mémoire sert à quelque chose, elle doit d'abord **briser la boucle**, et cela se voit sur
les cycles avant même de se voir sur le succès.
"""
import sys
sys.path.insert(0, ".")
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OCC = Path("results/runs/can/occluded")
SPECS = [("temoin_bancfix", "témoin\n(entraîné sur CLAIR)", "tab:gray"),
         ("occ03_baseline", "baseline 9D\n(entraînée sur OCCLUS)", "tab:blue"),
         ("occ03_camp", "CAMP-lite 41D\n(+ mémoire)", "tab:green")]

res = []
for tag, lab, col in SPECS:
    f = OCC / f"eval_{tag}.json"
    if f.exists():
        d = json.load(open(f)); d["_lab"], d["_col"] = lab, col
        res.append(d)
if not res:
    sys.exit("aucun résultat à tracer")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
x = range(len(res))
sr = [r["success_rate"] * 100 for r in res]
err = [[(r["success_rate"] - r["ci95"][0]) * 100 for r in res],
       [(r["ci95"][1] - r["success_rate"]) * 100 for r in res]]
ax[0].bar(x, sr, color=[r["_col"] for r in res])
ax[0].errorbar(x, sr, yerr=err, fmt="none", ecolor="black", capsize=5)
for i, r in enumerate(res):
    ax[0].annotate(f"{r['success_rate']:.1%}\n({r['n_success']}/{r['n']})",
                   (i, sr[i]), ha="center", va="bottom", fontsize=9)
ax[0].set_xticks(list(x)); ax[0].set_xticklabels([r["_lab"] for r in res], fontsize=9)
ax[0].set_ylabel("succès (%)"); ax[0].set_ylim(0, max(sr) * 1.35 + 5)
ax[0].set_title("Succès sous occlusion 3 cm")

cyc = [r.get("z_cycles_failed") or 0 for r in res]
ax[1].bar(x, cyc, color=[r["_col"] for r in res])
for i, c in enumerate(cyc):
    ax[1].annotate(f"{c:.2f}", (i, c), ha="center", va="bottom", fontsize=9)
ax[1].set_xticks(list(x)); ax[1].set_xticklabels([r["_lab"] for r in res], fontsize=9)
ax[1].set_ylabel("cycles descente/remontée par échec")
ax[1].set_title("La boucle est-elle brisée ?\n(plus bas = moins de tentatives répétées)")

fig.suptitle("Verdict — la mémoire CAMP aide-t-elle sous occlusion ? "
             f"(banc corrigé, {res[0]['n']} rollouts/barre)", fontsize=12)
fig.tight_layout(); fig.savefig(OCC / "verdict_ab.png", dpi=110)
print(f"✓ {OCC / 'verdict_ab.png'}")
for r in res:
    print(f"  {r['tag']:>16} : {r['n_success']}/{r['n']} = {r['success_rate']:6.1%} "
          f"[{r['ci95'][0]:.1%}-{r['ci95'][1]:.1%}] | cycles/échec {r.get('z_cycles_failed')}")
