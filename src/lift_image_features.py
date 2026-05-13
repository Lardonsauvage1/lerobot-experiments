"""Extraction et cache des features ResNet18 pour le dataset Robomimic Lift.

Pipeline (one-shot) :
  1. Pour chaque démo du dataset Lift PH, parcourir les états enregistrés.
  2. Pour chaque frame, set sim state, render image (agentview).
  3. Passer l'image dans ResNet18 (gelé, pré-entraîné ImageNet).
  4. Cache : features (N, 512), states (N, 19), actions (N, 7), episodes (N,).

Le cache est ensuite chargé par les scripts d'expé pour entraîner uniquement
la tête de prédiction (équivalent du pipeline PushT runs 24/29-32).
"""

import os
import re
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Import des helpers existants
from src.lift_data import STATE_KEYS, STATE_DIM_TOTAL


def extract_lift_image_features(
    hdf5_path="data_cache/robomimic_lift_ph/low_dim_v15.hdf5",
    cache_path="data_cache/lift_ph_image_features.pt",
    image_size=96,
    camera_name="agentview",
    batch_size=128,
):
    """Génère (ou recharge) le cache des features ResNet18 sur le dataset Lift.

    Retourne un dict :
      - features : (N, 512) torch.float32 — features ResNet18 par frame
      - states   : (N, 19) torch.float32 — état low-dim concaténé
      - actions  : (N, 7)  torch.float32 — actions du dataset
      - episodes : (N,) np.int64 — episode_index par frame
      - ep_lengths : (n_demos,) np.int64
      - stats : dict mean/std pour states + actions (sur tout le dataset)
      - meta  : dict avec image_size, camera_name, etc.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"  Cache features image : {cache_path}", flush=True)
        return torch.load(cache_path, weights_only=False)

    print(f"  Extraction des features image (cible : {cache_path})...", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup env Robosuite Lift pour le rendu
    import robosuite as rs
    env = rs.make(
        env_name="Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=False, camera_names=camera_name,
        camera_heights=image_size, camera_widths=image_size,
        control_freq=20, reward_shaping=False,
    )

    # ResNet18 gelé
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.DEFAULT
    resnet = resnet18(weights=weights)
    resnet.eval()
    backbone = nn.Sequential(*list(resnet.children())[:-1])  # GAP, pas la fc finale

    # Normalisation ImageNet (mean/std qu'attend ResNet18)
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    all_features, all_states, all_actions, all_eps_idx, ep_lengths = [], [], [], [], []

    with h5py.File(hdf5_path, "r") as f:
        demo_names = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        print(f"  {len(demo_names)} démos à traiter...", flush=True)

        for d_idx, demo_name in enumerate(demo_names):
            demo = f["data"][demo_name]
            states_demo = demo["states"][:]                  # (n, state_dim_sim)
            actions_demo = demo["actions"][:]                # (n, 7)
            obs_grp = demo["obs"]
            state_low = np.concatenate([obs_grp[k][:] for k in STATE_KEYS], axis=1)  # (n, 19)
            n = states_demo.shape[0]

            # Reset env puis pour chaque frame : set state → render
            env.reset()
            frames = []
            for i in range(n):
                env.sim.set_state_from_flattened(states_demo[i])
                env.sim.forward()
                img = env.sim.render(height=image_size, width=image_size, camera_name=camera_name)[::-1]
                frames.append(img)
            frames_np = np.stack(frames)  # (n, H, W, 3) uint8

            # Tensor (n, 3, H, W) float [0, 1]
            imgs = torch.tensor(frames_np, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
            # Normalisation ImageNet
            imgs = (imgs - imagenet_mean) / imagenet_std

            # Pass dans ResNet par batches
            feats = []
            with torch.no_grad():
                for j in range(0, n, batch_size):
                    chunk = imgs[j:j + batch_size]
                    out = backbone(chunk).flatten(1)  # (b, 512)
                    feats.append(out)
            feats = torch.cat(feats)

            all_features.append(feats)
            all_states.append(torch.tensor(state_low, dtype=torch.float32))
            all_actions.append(torch.tensor(actions_demo, dtype=torch.float32))
            all_eps_idx.extend([d_idx] * n)
            ep_lengths.append(n)

            if (d_idx + 1) % 25 == 0:
                print(f"    {d_idx + 1}/{len(demo_names)} démos traitées", flush=True)

    env.close()

    features = torch.cat(all_features)         # (N, 512)
    states = torch.cat(all_states)             # (N, 19)
    actions = torch.cat(all_actions)           # (N, 7)
    episodes = np.array(all_eps_idx, dtype=np.int64)
    ep_lengths = np.array(ep_lengths, dtype=np.int64)

    stats = {
        "state_mean": states.mean(dim=0),
        "state_std": states.std(dim=0).clamp(min=1e-6),
        "action_mean": actions.mean(dim=0),
        "action_std": actions.std(dim=0).clamp(min=1e-6),
        "feature_mean": features.mean(dim=0),
        "feature_std": features.std(dim=0).clamp(min=1e-6),
    }
    meta = {
        "n_demos": len(ep_lengths),
        "n_frames": len(features),
        "feature_dim": features.shape[1],
        "state_dim": STATE_DIM_TOTAL,
        "action_dim": actions.shape[1],
        "image_size": image_size,
        "camera_name": camera_name,
        "resnet": "resnet18_imagenet1k_v1",
        "dataset": "robomimic/lift/ph/low_dim_v15",
    }
    print(f"  Features shape : {features.shape}, states {states.shape}, actions {actions.shape}", flush=True)

    data = {
        "features": features,
        "states": states,
        "actions": actions,
        "episodes": episodes,
        "ep_lengths": ep_lengths,
        "stats": stats,
        "meta": meta,
    }
    torch.save(data, cache_path)
    print(f"  Cache sauvegardé : {cache_path} ({cache_path.stat().st_size / 1e6:.0f} Mo)", flush=True)
    return data
