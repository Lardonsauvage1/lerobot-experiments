#!/usr/bin/env python
"""HIL-SERL phase 1 : entraîne le classifieur de récompense sur les démos Push-T.

Positifs = frames coverage>0.8 (T sur cible), négatifs = coverage<0.6.
Split par ÉPISODE (pas de fuite). Sauvegarde results/runs/hilserl_pusht/reward_clf.pt.
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from reward_classifier import RewardCNN

from lerobot.datasets.lerobot_dataset import LeRobotDataset

POS_T, NEG_T = 0.8, 0.6
NEG_PER_POS = 2.0          # ratio négatifs:positifs (sous-échantillonne les négatifs)
EPOCHS = 12
BATCH = 128
LR = 1e-3
OUT = "results/runs/hilserl_pusht/reward_clf.pt"
DEV = "cpu"   # MPS instable (erreur Metal) sur ce petit CNN M1 ; CPU fiable et suffisant


def main():
    ds = LeRobotDataset("lerobot/pusht")
    n = ds.num_frames
    rew = np.array([float(ds.hf_dataset[i]["next.reward"]) for i in range(n)])
    epi = np.array([int(ds.hf_dataset[i]["episode_index"]) for i in range(n)])

    pos_idx = np.where(rew > POS_T)[0]
    neg_all = np.where(rew < NEG_T)[0]
    rng = np.random.default_rng(0)
    neg_idx = rng.choice(neg_all, size=min(len(neg_all), int(len(pos_idx) * NEG_PER_POS)), replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    lab = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))]).astype(np.float32)
    print(f"positifs {len(pos_idx)}  négatifs {len(neg_idx)}  total {len(idx)}")

    # split par épisode : 10% des épisodes en validation
    all_ep = np.unique(epi[idx])
    val_ep = set(rng.choice(all_ep, size=max(1, len(all_ep) // 10), replace=False).tolist())
    is_val = np.array([epi[i] in val_ep for i in idx])

    # préchargement des images (uint8) une seule fois -> pas de re-décodage vidéo
    print("préchargement des images...")
    imgs = np.zeros((len(idx), 3, 96, 96), np.uint8)
    for k, i in enumerate(idx):
        im = ds[int(i)]["observation.image"]           # (3,96,96) float [0,1]
        imgs[k] = (np.asarray(im) * 255).clip(0, 255).astype(np.uint8)
        if k % 1000 == 0:
            print(f"  {k}/{len(idx)}")
    X = torch.from_numpy(imgs)                          # uint8 CHW
    Y = torch.from_numpy(lab)
    tr, va = ~is_val, is_val
    Xtr, Ytr = X[tr], Y[tr]
    Xva, Yva = X[va].float().div(255).to(DEV), Y[va].to(DEV)
    print(f"train {tr.sum()}  val {va.sum()}  (val épisodes {sorted(val_ep)})")

    model = RewardCNN().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    pos_weight = torch.tensor([NEG_PER_POS], device=DEV)   # rééquilibrage
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ntr = int(tr.sum())
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(ntr)
        tot = 0.0
        for b in range(0, ntr, BATCH):
            sel = perm[b:b + BATCH]
            xb = Xtr[sel].float().div(255).to(DEV)
            yb = Ytr[sel].to(DEV)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(sel)
        # val
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xva))
            pred = (pv >= 0.5).float()
            acc = (pred == Yva).float().mean().item()
            tp = ((pred == 1) & (Yva == 1)).sum().item()
            fp = ((pred == 1) & (Yva == 0)).sum().item()
            fn = ((pred == 0) & (Yva == 1)).sum().item()
            prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        print(f"ep {ep+1:2d}  loss {tot/ntr:.4f}  val_acc {acc:.3f}  P {prec:.3f}  R {rec:.3f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "thresh": 0.5,
                "pos_t": POS_T, "neg_t": NEG_T}, OUT)
    print(f"-> sauvegardé {OUT}")


if __name__ == "__main__":
    main()
