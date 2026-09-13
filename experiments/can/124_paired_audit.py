#!/usr/bin/env python
"""Audit : recalcule TOUTES les comparaisons appariées depuis les trajectoires sur disque.

Pourquoi ce script existe. Le bilan du chantier annonçait un coût d'occlusion de +16,7 pts
(p = 0,002) avec un plafond clair de 72,7 % — soit 109 succès sur 150. Or aucun fichier de
résultats ne contient 109 succès : la référence claire vaut 104/150 = 69,3 %. Le chiffre
publié avait été transcrit à la main, et l'IC95 semblait recopié de la ligne oracle.

Le remède n'est pas de recorriger à la main : c'est de recalculer depuis la source. Chaque
`traj_<tag>.npz` contient `ep_meta`, le succès épisode par épisode. Les rollouts partagent
les mêmes états initiaux → les comparaisons sont appariées, et McNemar s'applique.

Usage : venv312/bin/python experiments/can/124_paired_audit.py [--ref b3_baseline]
"""
import argparse
import json
import math
from math import comb
from pathlib import Path

import numpy as np

OCC = Path("results/runs/can/occluded")


def successes(tag):
    """{index d'épisode: succès} depuis traj_<tag>.npz."""
    meta = json.loads(str(np.load(OCC / f"traj_{tag}.npz", allow_pickle=True)["ep_meta"]))
    return {e["ep"]: bool(e["success"]) for e in meta}


def mcnemar_exact(b, c):
    """p bilatéral exact (binomiale) sur les seuls discordants. Pas d'approximation chi2 :
    à n=150 les effectifs discordants sont petits (20-60) et chi2 y est optimiste."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(tag, ref):
    A, B = successes(tag), successes(ref)
    common = sorted(set(A) & set(B))
    a = np.array([A[e] for e in common])
    b = np.array([B[e] for e in common])
    n = len(common)
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    d = a.mean() - b.mean()
    # variance de la différence de deux proportions APPARIÉES (pas deux Wilson indépendants)
    var = (only_a + only_b - (only_a - only_b) ** 2 / n) / n ** 2
    h = 1.96 * math.sqrt(var)
    return {"tag": tag, "n": n, "k": int(a.sum()), "k_ref": int(b.sum()),
            "delta": d, "lo": d - h, "hi": d + h,
            "p": mcnemar_exact(only_a, only_b),
            "disc": (only_a, only_b)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="b3_baseline")
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="paires explicites tag:ref (ex. small64:kp64)")
    a = ap.parse_args()

    jobs = ([tuple(p.split(":")) for p in a.pairs] if a.pairs else
            [(f.stem[5:], a.ref) for f in sorted(OCC.glob("traj_*.npz"))
             if f.stem[5:] != a.ref])

    print(f"{'comparaison':34s} {'succès':>12s} {'écart':>8s} {'IC95':>18s} {'p':>9s}")
    print("-" * 88)
    rows = []
    for tag, ref in jobs:
        if not (OCC / f"traj_{tag}.npz").exists() or not (OCC / f"traj_{ref}.npz").exists():
            continue
        try:
            r = compare(tag, ref)
        except Exception as e:
            print(f"  {tag}: {e}")
            continue
        rows.append(r)
        star = " ★" if r["p"] < 0.05 else ""
        print(f"{tag + ' vs ' + ref:34s} {r['k']:3d}/{r['n']} vs {r['k_ref']:3d}   "
              f"{r['delta']*100:+6.1f}  [{r['lo']*100:+6.1f};{r['hi']*100:+6.1f}]  "
              f"{r['p']:8.4g}{star}")
    print("-" * 88)
    print(f"{sum(r['p'] < 0.05 for r in rows)} significative(s) sur {len(rows)} comparaisons")
    print("★ = p < 0,05 (McNemar exact, bilatéral)")


if __name__ == "__main__":
    main()
