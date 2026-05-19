"""Extraction et cache des features DINOv2-small pour le dataset Robomimic Lift.

DINOv2 est un Vision Transformer auto-supervisé (Meta 2023), pré-entraîné sur
142M images. Beaucoup plus puissant comme backbone visuel que ResNet18-ImageNet
sur les downstream tasks robotiques (cf. literature 2024).

Pipeline (one-shot) :
  1. Pour chaque démo, set sim state à chaque frame
  2. Render image 224×224 (taille standard DINOv2)
  3. Normalisation ImageNet → DINOv2 forward (frozen)
  4. Extraction du CLS token (384D image-level feature)
  5. Cache : features (N, 384), states (N, 19), actions (N, 7), episodes (N,)
"""

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from src.lift_data import STATE_KEYS, STATE_DIM_TOTAL


def extract_lift_dinov2_features(
    hdf5_path="data_cache/robomimic_lift_ph/low_dim_v15.hdf5",
    cache_path="data_cache/lift_ph_dinov2_features.pt",
    image_size=224,  # taille standard DINOv2 (multiple de 14)
    camera_name="agentview",
    batch_size=32,
    device=None,
):
    """Génère le cache des features DINOv2-small sur le dataset Lift.

    Returns dict :
      - features : (N, 384) torch.float32 — CLS token DINOv2
      - states   : (N, 19) torch.float32
      - actions  : (N, 7)  torch.float32
      - episodes : (N,) np.int64
      - ep_lengths : (n_demos,) np.int64
      - stats / meta
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"  Cache features DINOv2 : {cache_path}", flush=True)
        return torch.load(cache_path, weights_only=False)

    print(f"  Extraction features DINOv2 → {cache_path}...", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Device : MPS si dispo
    if device is None:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"  Device : {device}", flush=True)

    # Setup env Robosuite
    import robosuite as rs
    env = rs.make(
        env_name="Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=False, camera_names=camera_name,
        camera_heights=image_size, camera_widths=image_size,
        control_freq=20, reward_shaping=False,
    )

    # DINOv2-small (frozen, eval mode)
    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    for p in dinov2.parameters():
        p.requires_grad = False
    feat_dim = dinov2.config.hidden_size  # 384

    # Normalisation ImageNet (DINOv2 utilise les mêmes stats)
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    all_features, all_states, all_actions, all_eps_idx, ep_lengths = [], [], [], [], []

    with h5py.File(hdf5_path, "r") as f:
        demo_names = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        print(f"  {len(demo_names)} démos × {image_size}×{image_size} à traiter...", flush=True)

        for d_idx, demo_name in enumerate(demo_names):
            demo = f["data"][demo_name]
            states_demo = demo["states"][:]
            actions_demo = demo["actions"][:]
            obs_grp = demo["obs"]
            state_low = np.concatenate([obs_grp[k][:] for k in STATE_KEYS], axis=1)
            n = states_demo.shape[0]

            env.reset()
            frames = []
            for i in range(n):
                env.sim.set_state_from_flattened(states_demo[i])
                env.sim.forward()
                img = env.sim.render(height=image_size, width=image_size, camera_name=camera_name)[::-1]
                frames.append(img)
            frames_np = np.stack(frames)  # (n, H, W, 3) uint8

            # Convert to (n, 3, H, W) float, normalize, pass through DINOv2
            imgs = torch.from_numpy(frames_np.copy()).permute(0, 3, 1, 2).float() / 255.0
            imgs = imgs.to(device)
            imgs = (imgs - imagenet_mean) / imagenet_std

            feats = []
            with torch.no_grad():
                for j in range(0, n, batch_size):
                    chunk = imgs[j:j + batch_size]
                    out = dinov2(chunk)
                    # CLS token = features image-level
                    cls = out.last_hidden_state[:, 0]  # (batch, 384)
                    feats.append(cls.cpu())
            feats = torch.cat(feats)

            all_features.append(feats)
            all_states.append(torch.tensor(state_low, dtype=torch.float32))
            all_actions.append(torch.tensor(actions_demo, dtype=torch.float32))
            all_eps_idx.extend([d_idx] * n)
            ep_lengths.append(n)

            if (d_idx + 1) % 20 == 0:
                print(f"    {d_idx + 1}/{len(demo_names)} démos", flush=True)
    env.close()

    features = torch.cat(all_features)   # (N, 384)
    states = torch.cat(all_states)
    actions = torch.cat(all_actions)
    episodes = np.array(all_eps_idx, dtype=np.int64)
    ep_lengths = np.array(ep_lengths, dtype=np.int64)

    stats = {
        "state_mean": states.mean(dim=0),
        "state_std": states.std(dim=0).clamp(min=1e-6),
        "action_mean": actions.mean(dim=0),
        "action_std": actions.std(dim=0).clamp(min=1e-6),
    }
    meta = {
        "n_demos": len(ep_lengths),
        "n_frames": len(features),
        "feature_dim": feat_dim,
        "state_dim": STATE_DIM_TOTAL,
        "action_dim": actions.shape[1],
        "image_size": image_size,
        "camera_name": camera_name,
        "encoder": "facebook/dinov2-small",
        "encoder_params": 22_056_576,
        "feature_extraction": "CLS token last_hidden_state[:, 0]",
        "dataset": "robomimic/lift/ph/low_dim_v15",
    }
    print(f"  Features shape : {features.shape}", flush=True)

    data = {
        "features": features, "states": states, "actions": actions,
        "episodes": episodes, "ep_lengths": ep_lengths,
        "stats": stats, "meta": meta,
    }
    torch.save(data, cache_path)
    print(f"  Cache sauvegardé : {cache_path} ({cache_path.stat().st_size / 1e6:.0f} Mo)", flush=True)
    return data
