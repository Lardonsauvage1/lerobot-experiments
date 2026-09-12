#!/usr/bin/env python
"""LE test qui décide si CAMP-lite peut marcher chez nous.

Question : le code mémoire m_t DISTINGUE-T-IL la 1re approche de la 2e ?

Pourquoi c'est LA question. Toute l'utilité de CAMP ici tient à briser la boucle
« descendre -> rater -> remonter -> redescendre ». Or une policy sans état, replacée dans
le même état, refait la même chose — la boucle est une conséquence logique, pas un bug.
CAMP la brise SI et SEULEMENT SI m_t diffère entre les tentatives : c'est cette différence,
et elle seule, qui rend l'entrée du dénoiseur différente au 2e passage.

Le risque est réel : le module est pré-entraîné sur des trajectoires EXPERTES, qui ne
contiennent jamais deux tentatives. Rien ne l'a donc poussé à leur donner des codes
distincts. Si le test échoue, inutile d'entraîner quoi que ce soit — on aurait ajouté
32 dimensions qui ne portent rien, et la policy apprendrait simplement à les ignorer.

Entrée : les trajectoires de rollouts occlus (results/runs/can/occluded/traj_r*.npz),
         produites gratuitement par la courbe d'occlusion.
Sortie : results/runs/can/camp/discriminates_<tag>.png + verdict au terminal.

Lancer : venv312/bin/python -u experiments/can/114_camp_discriminates.py \
             --ckpt results/runs/can/camp/memory_L64_K32_m32.pt
"""
import sys
sys.path.insert(0, ".")

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import camp

OCC = Path("results/runs/can/occluded")
OUT = Path("results/runs/can/camp"); OUT.mkdir(parents=True, exist_ok=True)


def attempt_instants(z, min_amp=0.02):
    """Instants des TENTATIVES = minima locaux de la hauteur de l'effecteur.

    ⚠️ CORRECTION du 2026-09-09. La première version découpait les tentatives avec les
    drapeaux d'approche (`appr_flags`) — c'est-à-dire la métrique dont on a justement montré
    qu'elle est AVEUGLE : le modèle qui boucle ne sort jamais de la zone de saisie, donc
    elle ne voyait qu'un seul segment là où il y a trois tentatives. Le test ne portait
    alors que sur une poignée d'épisodes atypiques.

    On segmente désormais sur les creux de la trajectoire verticale : chaque descente suivie
    d'une remontée est une tentative, indépendamment de toute géométrie de zone.
    """
    z = np.asarray(z, dtype=float)
    if z.size < 3:
        return []
    ext, direction = [(0, float(z[0]))], 0
    for i, v in enumerate(z[1:], start=1):
        v = float(v)
        if direction >= 0 and v < ext[-1][1] - min_amp:
            ext.append((i, v)); direction = -1
        elif direction <= 0 and v > ext[-1][1] + min_amp:
            ext.append((i, v)); direction = 1
        elif (direction < 0 and v < ext[-1][1]) or (direction > 0 and v > ext[-1][1]):
            ext[-1] = (i, v)
    # les minima = un extremum sur deux, en partant du premier creux
    vals = [e[1] for e in ext]
    return [ext[k][0] for k in range(1, len(ext) - 1)
            if vals[k] < vals[k - 1] and vals[k] < vals[k + 1]] or (
           [ext[k][0] for k in range(1, len(ext)) if vals[k] < vals[k - 1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/runs/can/camp/memory_L64_K32_m32.pt")
    ap.add_argument("--traj", nargs="*", default=None, help="npz (défaut : tous les traj_r*)")
    a = ap.parse_args()

    files = [Path(f) for f in a.traj] if a.traj else sorted(OCC.glob("traj_r*.npz"))
    if not files:
        sys.exit(f"aucune trajectoire dans {OCC} — lancer d'abord 111_occlusion_curve.py")

    ck = torch.load(a.ckpt, weights_only=False)
    ca = ck["args"]
    model = None
    codes_by_rank, dists_by_rank = {}, []
    n_multi = n_ep = 0

    for f in files:
        d = np.load(f, allow_pickle=True)
        pro, act = d["proprio"], d["action"]
        starts = d["ep_start"]
        meta = json.loads(str(d["ep_meta"]))
        if model is None:
            model = camp.CampMemory(act.shape[1], pro.shape[1], n_coef=ca["K"],
                                    mem_dim=ca["mem_dim"], codebook_size=ca["codebook"])
            model.load_state_dict(ck["state_dict"]); model.eval()

        bounds = list(starts) + [len(pro)]
        for e in range(len(starts)):
            lo, hi = bounds[e], bounds[e + 1]
            if hi - lo < 5:
                continue
            n_ep += 1
            p = torch.tensor(pro[lo:hi])[None]
            ac = torch.tensor(act[lo:hi])[None]
            prev = torch.cat([torch.zeros_like(ac[:, :1]), ac[:, :-1]], dim=1)
            with torch.no_grad():
                _, m, _, idx, _ = model(p, prev)
            m, idx = m[0], idx[0]

            inst = attempt_instants(pro[lo:hi, 2])
            if len(inst) < 2:
                continue
            n_multi += 1
            # code au CREUX de chaque descente = l'instant de la tentative de saisie
            reps = [(int(idx[t]), m[t]) for t in inst]
            for rank, (code, _) in enumerate(reps[:4]):
                codes_by_rank.setdefault(rank + 1, []).append(code)
            for k in range(1, len(reps)):
                dists_by_rank.append((k + 1, float(torch.norm(reps[k][1] - reps[0][1])),
                                      reps[k][0] != reps[0][0]))

    print(f"{n_ep} épisodes | {n_multi} avec >=2 approches (la boucle d'échec)\n")
    if n_multi == 0:
        sys.exit("aucun épisode multi-approches : rien à tester. "
                 "aucune trajectoire à deux tentatives — vérifier min_amp.")

    diff = [x for _, _, x in dists_by_rank]
    dist = [x for _, x, _ in dists_by_rank]
    frac = float(np.mean(diff))
    print(f"Codes DIFFÉRENTS entre la 1re approche et les suivantes : {frac:.1%} "
          f"({sum(diff)}/{len(diff)} comparaisons)")
    print(f"Distance latente moyenne |m_k - m_1| : {np.mean(dist):.4f} "
          f"(médiane {np.median(dist):.4f})")
    for r in sorted(codes_by_rank):
        c = codes_by_rank[r]
        print(f"  approche {r} : {len(c):4d} occurrences | {len(set(c)):3d} codes distincts")

    print("\n--- VERDICT ---")
    if frac > 0.7:
        print(f"✅ La mémoire DISTINGUE les tentatives ({frac:.0%}). Le pré-entraînement "
              "expert suffit : m_t change au 2e passage, donc la boucle est brisable.")
    elif frac > 0.3:
        print(f"🟠 Discrimination PARTIELLE ({frac:.0%}). Ça peut suffire, mais élargir le "
              "pré-entraînement (rollouts ratés, échecs synthétiques) devrait aider nettement.")
    else:
        print(f"❌ La mémoire NE distingue PAS les tentatives ({frac:.0%}) : au 2e passage "
              "l'entrée du dénoiseur est quasi identique -> la boucle RESTERA fermée.\n"
              "   N'entraîne pas la policy en l'état. Élargir d'abord le pré-entraînement "
              "aux trajectoires d'ÉCHEC (rollouts enregistrés + échecs synthétiques).")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(dist, bins=40)
    ax[0].set_xlabel("|m_k - m_1|"); ax[0].set_title("distance latente entre tentatives")
    ranks = sorted(codes_by_rank)
    ax[1].bar([str(r) for r in ranks], [len(set(codes_by_rank[r])) for r in ranks])
    ax[1].set_xlabel("n° d'approche"); ax[1].set_ylabel("codes distincts")
    ax[1].set_title("diversité des codes par tentative")
    fig.suptitle(f"CAMP — la mémoire distingue-t-elle les tentatives ? "
                 f"({frac:.0%} de codes différents)")
    fig.tight_layout()
    tag = Path(a.ckpt).stem.replace("memory_", "")
    fig.savefig(OUT / f"discriminates_{tag}.png", dpi=110)
    print(f"\n✓ {OUT / f'discriminates_{tag}.png'}")


if __name__ == "__main__":
    main()
