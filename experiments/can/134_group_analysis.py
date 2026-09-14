#!/usr/bin/env python
"""Comparaison de GROUPES de répliques — le seul test valide pour juger une CONFIGURATION.

Pourquoi ce script existe. McNemar apparié répond à « ces deux MODÈLES diffèrent-ils ? ».
Il ne répond PAS à « ce réglage sert-il à quelque chose ? ». On l'a appris à nos dépens :
deux entraînements STRICTEMENT identiques (bras C et C2) ont donné 54,2 % et 61,6 %, écart
que McNemar déclare significatif à p = 0,0007. Le plancher de bruit inter-run de ce banc est
de 4 à 8 points sur 500 rollouts.

Ici chaque RUN est une observation, pas chaque épisode. On compare les moyennes de groupes par
un test de Welch (variances inégales, petits effectifs). La puissance est faible à n = 3 — c'est
le prix de l'honnêteté : un effet sous ~8 points ne sera pas détectable, et c'est le bon verdict.

Usage : venv312/bin/python experiments/can/134_group_analysis.py
"""
import json
import math
from pathlib import Path

OUT = Path("results/runs/can/cam2_eval500.json")

GROUPS = {
    "poignet — référence": ["C_side_wrist", "C2_wrist_fixstats", "ref_s43"],
    "poignet — tête auxiliaire": ["aux_s42", "aux_s43", "aux_s44"],
    "caméra seule": ["A_side", "A2_fixstats"],
    "côté + dessus": ["B_side_top", "B2_fixstats"],
}


def welch(xs, ys):
    """t de Welch + degrés de liberté de Welch-Satterthwaite. Retourne (t, df, p approx)."""
    nx, ny = len(xs), len(ys)
    if nx < 2 or ny < 2:
        return None
    mx, my = sum(xs) / nx, sum(ys) / ny
    vx = sum((v - mx) ** 2 for v in xs) / (nx - 1)
    vy = sum((v - my) ** 2 for v in ys) / (ny - 1)
    se2 = vx / nx + vy / ny
    if se2 <= 0:
        return None
    t = (mx - my) / math.sqrt(se2)
    df = se2 ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    # p bilatéral via la loi de Student, par la fonction bêta incomplète régularisée
    x = df / (df + t * t)
    p = _betainc(df / 2, 0.5, x)
    return t, df, min(1.0, p)


def _betainc(a, b, x):
    """I_x(a,b) par fraction continue de Lentz — suffisant pour un p bilatéral de Student."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(200):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-10:
            break
    return front * (f - 1.0)


def main():
    if not OUT.exists():
        print(f"{OUT} absent"); return
    res = {r["name"]: r for r in json.loads(OUT.read_text())["results"]}

    rates = {}
    print("=== RÉPLIQUES PAR CONFIGURATION ===")
    for g, names in GROUPS.items():
        vals = [(n, res[n]["success_rate"] * 100) for n in names if n in res]
        if not vals:
            continue
        rates[g] = [v for _, v in vals]
        detail = "  ".join(f"{v:.1f}" for _, v in vals)
        m = sum(rates[g]) / len(rates[g])
        if len(rates[g]) > 1:
            sd = math.sqrt(sum((v - m) ** 2 for v in rates[g]) / (len(rates[g]) - 1))
            print(f"  {g:30s} n={len(vals)}  [{detail}]  moyenne {m:5.1f} %  écart-type {sd:4.1f}")
        else:
            print(f"  {g:30s} n=1   [{detail}]  (pas d'estimation de variance)")

    print("\n=== LE TEST QUI COMPTE (Welch sur les moyennes de runs) ===")
    a, b = "poignet — tête auxiliaire", "poignet — référence"
    if a in rates and b in rates:
        w = welch(rates[a], rates[b])
        d = sum(rates[a]) / len(rates[a]) - sum(rates[b]) / len(rates[b])
        if w:
            t, df, p = w
            verdict = "★ significatif" if p < 0.05 else "NON significatif"
            print(f"  {a}\n  vs {b}")
            print(f"  écart des moyennes : {d:+.1f} pts   t={t:.2f}  df={df:.1f}  p={p:.3f}  -> {verdict}")
        else:
            print(f"  écart des moyennes : {d:+.1f} pts  (pas assez de répliques pour tester)")
    else:
        manque = [g for g in (a, b) if g not in rates]
        print(f"  en attente de répliques pour : {manque}")


if __name__ == "__main__":
    main()
