#!/usr/bin/env python
"""CAMP-lite, phase 1 : pré-entraînement du module mémoire (auto-supervisé, sans images).

Le module apprend à RECONSTRUIRE sa propre trajectoire d'actions passée, en coefficients
DCT pondérés en fréquence. S'il en est capable, son état caché contient l'histoire du
geste — et c'est cet état, compressé puis quantifié en 32 nombres, qui ira nourrir la
policy (phase 2).

POURQUOI ÇA PEUT TOURNER MAINTENANT, avant même de connaître le rayon d'occlusion :
le module ne voit AUCUNE image. Ses entrées (proprio + action précédente) sont
rigoureusement identiques quel que soit le niveau d'occlusion — occlure change ce que la
POLICY voit, pas ce que le bras a fait. Ce pré-entraînement est donc réutilisable tel quel
pour tous les rayons.

⚠️ LIMITE ASSUMÉE, à surveiller : le module est entraîné sur des trajectoires EXPERTES,
qui ne contiennent jamais d'échec ni de boucle. En déploiement il devra résumer exactement
cela — des tentatives ratées. C'est un décalage de distribution réel. Deux mitigations
ici : (1) bruit gaussien sur les actions d'entrée (`--noise`), (2) inversions temporelles
locales et gels de segments (`--augment`) qui fabriquent des « hésitations » absentes des
démos. Ça ne remplace pas de vrais rollouts ratés — c'est noté comme dette (cf. §9 du doc).

Sortie : results/runs/can/camp/memory_pretrain.pt (+ courbes)
Lancer : venv312/bin/python -u experiments/can/113_camp_pretrain.py
"""
import sys
sys.path.insert(0, ".")

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import camp

HDF5 = "data_cache/robomimic_can_ph/low_dim_v141.hdf5"
OUT = Path("results/runs/can/camp"); OUT.mkdir(parents=True, exist_ok=True)
N_TRAIN_DEMOS = 150          # même split que tout le projet (150 train / 50 val)
PROPRIO_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]  # 9D vision pure


def load_episodes(n_train=N_TRAIN_DEMOS):
    """Retourne (train, val), listes de (proprio 9D, actions 7D) par épisode."""
    eps = []
    with h5py.File(HDF5, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        for d in demos:
            g = f["data"][d]
            p = np.concatenate([np.asarray(g["obs"][k]) for k in PROPRIO_KEYS], axis=1)
            eps.append((p.astype(np.float32), np.asarray(g["actions"]).astype(np.float32)))
    return eps[:n_train], eps[n_train:]


def make_batch(eps, idxs, T, device, *, noise=0.0, augment=False, rng=None):
    """Segments contigus de longueur T, un état caché par épisode (jamais mélangés)."""
    P, A = [], []
    for i in idxs:
        p, a = eps[i]
        s = 0 if len(a) <= T else int(rng.integers(0, len(a) - T))
        p_, a_ = p[s:s + T], a[s:s + T]
        if len(a_) < T:                                   # padding par répétition de la fin
            pad = T - len(a_)
            p_ = np.concatenate([p_, np.repeat(p_[-1:], pad, 0)])
            a_ = np.concatenate([a_, np.repeat(a_[-1:], pad, 0)])
        if augment:
            # fabrique des "hésitations" absentes des démos expertes : on gèle un segment
            # (le bras marque un temps d'arrêt, comme lors d'un échec de saisie)
            if rng.random() < 0.3:
                lo = int(rng.integers(0, max(T - 8, 1))); ln = int(rng.integers(3, 9))
                a_ = a_.copy(); a_[lo:lo + ln] = a_[lo]
        P.append(p_); A.append(a_)
    P = torch.tensor(np.stack(P), device=device)
    A = torch.tensor(np.stack(A), device=device)
    if noise > 0:
        A = A + noise * torch.randn_like(A)
    return P, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=64, help="fenêtre d'actions résumée (pas)")
    ap.add_argument("--K", type=int, default=32, help="coefficients DCT gardés")
    ap.add_argument("--T", type=int, default=96, help="longueur du segment déroulé (BPTT)")
    ap.add_argument("--hidden", type=int, default=64, help="taille de l'état caché du LSTM")
    ap.add_argument("--no-vq", dest="no_vq", action="store_true",
                    help="court-circuite la quantification (isole son effet)")
    ap.add_argument("--mem-dim", dest="mem_dim", type=int, default=32)
    ap.add_argument("--codebook", type=int, default=128)
    ap.add_argument("--gamma", type=float, default=3.0, help="pente de la pondération fréquentielle")
    ap.add_argument("--lam-cons", dest="lam_cons", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)      # batch 16 = celui du papier (table 5)
    ap.add_argument("--lr", type=float, default=1e-4)     # 1e-4 = LR du projet
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--augment", action="store_true", default=True)
    ap.add_argument("--patience", type=int, default=15,
                    help="relevés (x100 steps) sans amélioration de la val avant arrêt")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    train, val = load_episodes()
    print(f"{len(train)} épisodes train / {len(val)} val | proprio {train[0][0].shape[1]}D "
          f"| action {train[0][1].shape[1]}D | device {device}", flush=True)

    model = camp.CampMemory(action_dim=train[0][1].shape[1], state_dim=train[0][0].shape[1],
                            n_coef=a.K, hidden=a.hidden, mem_dim=a.mem_dim,
                            codebook_size=a.codebook).to(device)
    if a.no_vq:   # diagnostic : la quantification est-elle le goulot ?
        model.vq.forward = lambda z: (z, torch.zeros((), device=z.device),
                                      torch.zeros(z.shape[:-1], dtype=torch.long, device=z.device))
        print("  (quantification DÉSACTIVÉE)", flush=True)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"CampMemory : {n_par:,} paramètres "
          f"({n_par / 263e6:.3%} du dénoiseur 263M — le surcoût d'inférence est là)\n", flush=True)

    dct = camp.dct_matrix(a.L, a.K).to(device)
    w = camp.freq_weights(a.K, a.gamma).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)   # AdamW = optimiseur du papier

    # ⚠️ SURAPPRENTISSAGE. Ici la val-loss EST le juge (contrairement à la policy, où
    # loss != succès) : l'objectif du module est de reconstruire des trajectoires JAMAIS
    # vues. On garde donc le meilleur checkpoint au sens val, on surveille l'écart
    # train/val, et on arrête si la val remonte durablement.
    hist = []
    best_val, best_state, best_step, patience = float("inf"), None, 0, 0
    for step in range(1, a.steps + 1):
        idxs = rng.integers(0, len(train), a.batch)
        P, A = make_batch(train, idxs, a.T, device, noise=a.noise, augment=a.augment, rng=rng)
        # ZÉRO au 1er pas (et non a_0) : en ligne, l'action du pas 0 n'est pas encore
        # décidée quand la mémoire doit produire m_0. Cohérence train/inférence.
        prev = torch.cat([torch.zeros_like(A[:, :1]), A[:, :-1]], dim=1)
        coef, m, l_vq, idx, _ = model(P, prev)
        loss, parts = camp.camp_losses(coef, A, dct, w, lam_cons=a.lam_cons)
        loss = loss + l_vq
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        if step % 100 == 0 or step == 1:
            ppl = model.vq.perplexity(idx)
            with torch.no_grad():
                vi = rng.integers(0, len(val), a.batch)
                Pv, Av = make_batch(val, vi, a.T, device, rng=rng)
                pv = torch.cat([torch.zeros_like(Av[:, :1]), Av[:, :-1]], dim=1)
                cv, _, _, iv, _ = model(Pv, pv)
                vloss, vparts = camp.camp_losses(cv, Av, dct, w, lam_cons=a.lam_cons)
            rec = {"step": step, "loss": float(loss.detach()), "rec": parts["rec"],
                   "cons": parts["cons"], "vq": float(l_vq.detach()), "ppl": ppl,
                   "val_rec": vparts["rec"], "val_ppl": model.vq.perplexity(iv)}
            hist.append(rec)
            print(f"  step {step:5d} | rec {rec['rec']:.4f} (val {rec['val_rec']:.4f}) "
                  f"| cons {rec['cons']:.4f} | vq {rec['vq']:.4f} "
                  f"| codes actifs {ppl:5.1f}/{a.codebook}", flush=True)
            # suivi du meilleur + arrêt anticipé
            _v = vparts["rec"]
            if _v < best_val * 0.999:
                best_val, best_step, patience = _v, step, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
            _gap = (_v - parts["rec"]) / max(parts["rec"], 1e-9)
            if patience >= a.patience:
                print(f"  ARRÊT : val sans amélioration depuis {patience} relevés "
                      f"(meilleure {best_val:.4f} au step {best_step})", flush=True)
                break
            if _gap > 0.5:
                print(f"  ⚠️ écart train/val {_gap:+.0%} — surapprentissage qui démarre", flush=True)
            if step > 500 and ppl < 3:
                print("  ⚠️ COLLAPSE du codebook (perplexité < 3) : la mémoire ne transporte "
                      "presque plus rien. Baisser beta, ou augmenter mem_dim.", flush=True)

    if best_state is not None:                      # on sauve le MEILLEUR, pas le dernier
        model.load_state_dict(best_state)
        print(f"\nmeilleur checkpoint retenu : step {best_step}, val_rec {best_val:.4f}", flush=True)
    tag = a.tag or f"L{a.L}_K{a.K}_m{a.mem_dim}"
    ckpt = OUT / f"memory_{tag}.pt"
    torch.save({"state_dict": model.state_dict(), "args": vars(a),
                "proprio_keys": PROPRIO_KEYS, "hist": hist}, ckpt)
    (OUT / f"memory_{tag}_hist.json").write_text(json.dumps(hist, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    st = [h["step"] for h in hist]
    ax[0].plot(st, [h["rec"] for h in hist], label="train")
    ax[0].plot(st, [h["val_rec"] for h in hist], label="val")
    ax[0].set_yscale("log"); ax[0].set_title("reconstruction DCT"); ax[0].legend()
    ax[1].plot(st, [h["cons"] for h in hist]); ax[1].set_yscale("log")
    ax[1].set_title("cohérence temporelle")
    ax[2].plot(st, [h["ppl"] for h in hist], label="train")
    ax[2].plot(st, [h["val_ppl"] for h in hist], label="val")
    ax[2].axhline(3, color="red", ls=":", lw=1, label="seuil collapse")
    ax[2].set_title(f"codes actifs / {a.codebook}"); ax[2].legend()
    for x in ax: x.set_xlabel("step")
    fig.suptitle(f"CAMP-lite — pré-entraînement mémoire ({tag})")
    fig.tight_layout(); fig.savefig(OUT / f"memory_{tag}.png", dpi=110)
    print(f"\n✓ {ckpt}\n✓ {OUT / f'memory_{tag}.png'}")


if __name__ == "__main__":
    main()
