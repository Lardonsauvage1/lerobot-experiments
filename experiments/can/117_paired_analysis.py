#!/usr/bin/env python
"""Analyse APPARIÉE : l'occlusion cause-t-elle l'échec, ou trie-t-elle les épisodes ?

Le problème que ça résout. À 1 cm, `recovery_rate` vaut 80,5 % alors que le succès global
n'est que de 60,4 % : les épisodes occlus réussissent MIEUX que les autres. Ce n'est pas un
paradoxe mais un biais de sélection — à petit rayon, l'occlusion ne se déclenche que si le
préhenseur passe à moins d'1 cm au-dessus de la canette, donc **seulement quand le modèle
vise juste**. « Être occlus » est un marqueur de bonne visée. La métrique est confondue.

La bonne comparaison. Tous les rayons partent des **MÊMES 250 états initiaux figés**
(`can_eval500.npy`). On apparie donc **par numéro d'épisode** : même départ, même modèle,
seule l'occlusion change. C'est un plan apparié exact.

⚠️ Ne PAS apparier sur `dxy_min` (la distance minimale atteinte) : c'est une variable
POSTÉRIEURE au traitement — l'occlusion modifie la trajectoire, donc conditionner dessus
ouvrirait un biais de collision. L'index d'épisode, lui, est fixé avant tout.

Test : McNemar sur les paires discordantes (réussi->raté vs raté->réussi).

Sortie : tableau au terminal + results/runs/can/occluded/paired.json
"""
import sys
sys.path.insert(0, ".")

import json
from math import comb
from pathlib import Path

import numpy as np

OCC = Path("results/runs/can/occluded")


def mcnemar_p(b, c):
    """p bilatéral exact (binomiale) sur les paires discordantes."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load(tag):
    f = OCC / f"traj_r{tag}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    meta = json.loads(str(d["ep_meta"]))
    return {m["ep"]: m for m in meta}


base = load("00")
if base is None:
    sys.exit("traj_r00.npz manquant — le contrôle témoin (111c_temoin_recheck.sh) n'a pas "
             "encore tourné. Sans population de référence appariée, rien à comparer.")

rows = []
for f in sorted(OCC.glob("traj_r*.npz")):
    tag = f.stem.replace("traj_r", "")
    if tag == "00":
        continue
    cur = load(tag)
    eps = sorted(set(base) & set(cur))
    b = sum(1 for e in eps if base[e]["success"] and not cur[e]["success"])   # perdus
    c = sum(1 for e in eps if not base[e]["success"] and cur[e]["success"])   # gagnés
    same = len(eps) - b - c
    # parmi les épisodes PERDUS, combien ont réellement subi une occlusion ?
    lost_occ = sum(1 for e in eps if base[e]["success"] and not cur[e]["success"]
                   and cur[e]["n_occl_events"] > 0)
    # et parmi ceux que le témoin réussissait ET qui ont été occlus : taux de survie
    exposed = [e for e in eps if base[e]["success"] and cur[e]["n_occl_events"] > 0]
    surv = (sum(cur[e]["success"] for e in exposed) / len(exposed)) if exposed else None
    rows.append({"tag": tag, "n_paires": len(eps), "perdus": b, "gagnés": c,
                 "inchangés": same, "p_mcnemar": mcnemar_p(b, c),
                 "perdus_occlus": lost_occ,
                 "n_exposés": len(exposed), "survie_exposés": surv})

print(f"Appariement par numéro d'épisode (mêmes états initiaux figés). "
      f"Témoin = {sum(m['success'] for m in base.values())}/{len(base)}\n")
print(f"{'rayon':>7}{'paires':>8}{'perdus':>8}{'gagnés':>8}{'p':>10}"
      f"{'perdus occlus':>15}{'survie exposés':>16}")
for r in rows:
    s = f"{r['survie_exposés']:.1%}" if r["survie_exposés"] is not None else "-"
    print(f"{r['tag']:>7}{r['n_paires']:>8}{r['perdus']:>8}{r['gagnés']:>8}"
          f"{r['p_mcnemar']:>10.2e}{r['perdus_occlus']:>15}{s:>16}")

print("\nLecture :")
print("  perdus         = réussis au témoin, ratés avec occlusion (l'effet cherché)")
print("  gagnés         = l'inverse (bruit ; l'occlusion ne devrait pas AIDER)")
print("  perdus occlus  = parmi les perdus, ceux ayant réellement subi une occlusion.")
print("                   Un écart avec 'perdus' = des pertes dues au hasard du rollout,")
print("                   PAS à l'occlusion -> à retrancher de l'effet.")
print("  survie exposés = parmi les épisodes que le témoin réussissait ET qui ont été")
print("                   occlus, la fraction qui réussit quand même. ⭐ C'est LA mesure")
print("                   non confondue de la robustesse à l'occlusion.")

(OCC / "paired.json").write_text(json.dumps(rows, indent=2))
print(f"\n✓ {OCC / 'paired.json'}")
