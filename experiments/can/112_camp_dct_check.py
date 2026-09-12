#!/usr/bin/env python
"""CAMP-lite, contrôle préalable : la DCT tronquée retient-elle vraiment le GESTE ?

Tout CAMP repose sur une hypothèse jamais vérifiée chez nous : qu'on peut résumer une
trajectoire d'actions par ses K premières fréquences sans perdre ce qui compte. Si c'est
faux, le module mémoire a un objectif de pré-entraînement inatteignable et rien de la
suite ne marchera — autant le savoir maintenant, en numpy, plutôt qu'après un entraînement.

On prend les VRAIES actions des démos Can, on projette sur K coefficients, on reconstruit,
et on mesure l'erreur relative (RMSE / écart-type de la dimension). Balaye L et K.

Sortie : results/runs/can/occluded/camp_dct_check.png + chiffres au terminal.
"""
import sys
sys.path.insert(0, ".")

from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src import camp

HDF5 = "data_cache/robomimic_can_ph/low_dim_v141.hdf5"
OUT = Path("results/runs/can/occluded"); OUT.mkdir(parents=True, exist_ok=True)
DIM_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "pince"]
LS = [32, 64, 118]
KS = [4, 8, 16, 32, 64]

with h5py.File(HDF5, "r") as f:
    demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
    acts = [np.asarray(f["data"][d]["actions"]) for d in demos[:40]]
print(f"{len(acts)} démos | longueurs {min(len(a) for a in acts)}-{max(len(a) for a in acts)} "
      f"(médiane {int(np.median([len(a) for a in acts]))}) | action {acts[0].shape[1]}D\n")

# ------------------------------------------------------------------ balayage L x K
print(f"Erreur de reconstruction relative (RMSE / écart-type), moyenne sur les 7 dimensions :")
print(f"{'L \\\\ K':>8}" + "".join(f"{k:>9}" for k in KS))
grid = np.zeros((len(LS), len(KS)))
for i, L in enumerate(LS):
    wins = np.stack([a[s:s + L] for a in acts for s in range(0, len(a) - L, L // 2)])  # (N,L,7)
    sd = wins.reshape(-1, wins.shape[-1]).std(0) + 1e-8
    row = ""
    for j, K in enumerate(KS):
        D = camp.dct_matrix(L, min(K, L)).numpy()
        rec = np.einsum("kl,nka->nla", D, np.einsum("kl,nla->nka", D, wins))
        err = float(np.mean(np.sqrt(((rec - wins) ** 2).mean(axis=(0, 1))) / sd))
        grid[i, j] = err
        row += f"{err:>9.1%}"
    print(f"{L:>8}" + row)

# ------------------------------------------------------------------ tracé qualitatif
L, K = 64, 32
D = camp.dct_matrix(L, K).numpy()
a = acts[0][:L]
rec = np.einsum("kl,ka->la", D, np.einsum("kl,la->ka", D, a))

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
for d in range(7):
    ax = axes[d // 4, d % 4]
    ax.plot(a[:, d], lw=2, label="vraie action")
    ax.plot(rec[:, d], lw=1.6, ls="--", label=f"reconstruite (K={K})")
    ax.set_title(DIM_NAMES[d], fontsize=10)
    if d == 0:
        ax.legend(fontsize=8)
ax = axes[1, 3]
for i, L_ in enumerate(LS):
    ax.plot(KS, grid[i] * 100, marker="o", label=f"L={L_}")
ax.set_xscale("log", base=2); ax.set_xlabel("K (coefficients gardés)")
ax.set_ylabel("erreur relative (%)"); ax.axhline(10, color="grey", ls=":", lw=1)
ax.set_title("erreur vs compression"); ax.legend(fontsize=8)
fig.suptitle("CAMP — la DCT tronquée retient-elle le geste ? (actions réelles Can)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "camp_dct_check.png", dpi=110)
print(f"\n✓ {OUT/'camp_dct_check.png'}")

# ------------------------------------------------- la pince : le cas qui pourrait casser
print("\nDimension PINCE seule (signal quasi binaire = très haute fréquence, "
      "c'est elle qui résiste le plus à la troncature) :")
for L in LS:
    wins = np.stack([a[s:s + L] for a in acts for s in range(0, len(a) - L, L // 2)])
    sd = wins[..., 6].std() + 1e-8
    line = f"{'L=' + str(L):>8}"
    for K in KS:
        D = camp.dct_matrix(L, min(K, L)).numpy()
        rec = np.einsum("kl,nk->nl", D, np.einsum("kl,nl->nk", D, wins[..., 6]))
        line += f"{float(np.sqrt(((rec - wins[..., 6]) ** 2).mean()) / sd):>9.1%}"
    print(line)
