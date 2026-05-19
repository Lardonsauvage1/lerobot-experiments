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


def extract_lift_raw_images(
    hdf5_path="data_cache/robomimic_lift_ph/low_dim_v15.hdf5",
    cache_path="data_cache/lift_ph_raw_images.pt",
    image_size=96,
    camera_name="agentview",
):
    """Cache des images brutes (uint8) du dataset Lift — pour entraîner backbone trainable.

    Retourne dict :
      - images : (N, 3, H, W) uint8 (à diviser par 255 en training)
      - states : (N, 19) float32
      - actions: (N, 7)  float32
      - episodes: (N,) int64
      - ep_lengths: (n_demos,) int64
      - stats : mean/std states + actions
      - meta : info
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"  Cache images brutes : {cache_path}", flush=True)
        return torch.load(cache_path, weights_only=False)

    print(f"  Extraction des images brutes → {cache_path}...", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    import robosuite as rs
    env = rs.make(
        env_name="Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=False, camera_names=camera_name,
        camera_heights=image_size, camera_widths=image_size,
        control_freq=20, reward_shaping=False,
    )

    all_images, all_states, all_actions, all_eps_idx, ep_lengths = [], [], [], [], []

    with h5py.File(hdf5_path, "r") as f:
        demo_names = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        print(f"  {len(demo_names)} démos à traiter...", flush=True)
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
            # Permute en (n, 3, H, W) et stocker en uint8
            imgs = torch.from_numpy(frames_np.copy()).permute(0, 3, 1, 2)
            all_images.append(imgs)

            all_states.append(torch.tensor(state_low, dtype=torch.float32))
            all_actions.append(torch.tensor(actions_demo, dtype=torch.float32))
            all_eps_idx.extend([d_idx] * n)
            ep_lengths.append(n)

            if (d_idx + 1) % 25 == 0:
                print(f"    {d_idx + 1}/{len(demo_names)} démos", flush=True)
    env.close()

    images = torch.cat(all_images)
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
        "n_frames": len(images),
        "state_dim": STATE_DIM_TOTAL,
        "action_dim": actions.shape[1],
        "image_size": image_size,
        "image_dtype": "uint8",
        "camera_name": camera_name,
        "dataset": "robomimic/lift/ph/low_dim_v15",
    }
    print(f"  Images shape : {images.shape} (uint8)", flush=True)

    data = {
        "images": images, "states": states, "actions": actions,
        "episodes": episodes, "ep_lengths": ep_lengths,
        "stats": stats, "meta": meta,
    }
    torch.save(data, cache_path)
    print(f"  Cache sauvegardé : {cache_path} ({cache_path.stat().st_size / 1e6:.0f} Mo)", flush=True)
    return data


def extract_lift_image_features_augmented(
    hdf5_path="data_cache/robomimic_lift_ph/low_dim_v15.hdf5",
    cache_path="data_cache/lift_ph_image_features_aug.pt",
    image_size=96,
    camera_name="agentview",
    n_versions=8,
    batch_size=128,
    seed=42,
):
    """Version augmentée : extrait n_versions features par frame.

    Pour chaque frame :
      - v0 : image originale, no augmentation
      - v1..v_{n-1} : random crop + color jitter (différents)

    Cache shape : features (N, n_versions, 512). Au training : random select.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"  Cache features augmentées : {cache_path}", flush=True)
        return torch.load(cache_path, weights_only=False)

    print(f"  Extraction features AUGMENTÉES (n_versions={n_versions}) → {cache_path}...", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    import robosuite as rs
    env = rs.make(
        env_name="Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=False, camera_names=camera_name,
        camera_heights=image_size, camera_widths=image_size,
        control_freq=20, reward_shaping=False,
    )

    from torchvision.models import resnet18, ResNet18_Weights
    from torchvision.transforms import v2

    weights = ResNet18_Weights.DEFAULT
    resnet = resnet18(weights=weights)
    resnet.eval()
    backbone = nn.Sequential(*list(resnet.children())[:-1])

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # Augmentations légères : crop + jitter (pas de flip car gauche/droite compte)
    augment = v2.Compose([
        v2.RandomResizedCrop(size=(image_size, image_size), scale=(0.85, 1.0),
                              ratio=(0.95, 1.05), antialias=True),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
    ])

    torch.manual_seed(seed)

    all_features = []        # liste de (n, n_versions, 512)
    all_states, all_actions, all_eps_idx, ep_lengths = [], [], [], []

    with h5py.File(hdf5_path, "r") as f:
        demo_names = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        print(f"  {len(demo_names)} démos × {n_versions} versions à traiter...", flush=True)

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
            frames_np = np.stack(frames)
            imgs = torch.tensor(frames_np.copy(), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0  # (n, 3, H, W)

            # n_versions features par frame
            versions_feats = []
            for v in range(n_versions):
                if v == 0:
                    aug_imgs = imgs  # original
                else:
                    aug_imgs = augment(imgs)
                norm_imgs = (aug_imgs - imagenet_mean) / imagenet_std
                with torch.no_grad():
                    feats = []
                    for j in range(0, n, batch_size):
                        chunk = norm_imgs[j:j + batch_size]
                        out = backbone(chunk).flatten(1)
                        feats.append(out)
                    versions_feats.append(torch.cat(feats))  # (n, 512)
            # (n, n_versions, 512)
            feats_stack = torch.stack(versions_feats, dim=1)
            all_features.append(feats_stack)

            all_states.append(torch.tensor(state_low, dtype=torch.float32))
            all_actions.append(torch.tensor(actions_demo, dtype=torch.float32))
            all_eps_idx.extend([d_idx] * n)
            ep_lengths.append(n)

            if (d_idx + 1) % 20 == 0:
                print(f"    {d_idx + 1}/{len(demo_names)} démos traitées", flush=True)

    env.close()

    features = torch.cat(all_features)   # (N, n_versions, 512)
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
        "n_versions": n_versions,
        "feature_dim": features.shape[-1],
        "state_dim": STATE_DIM_TOTAL,
        "action_dim": actions.shape[1],
        "image_size": image_size,
        "camera_name": camera_name,
        "resnet": "resnet18_imagenet1k_v1",
        "augmentation": "RandomResizedCrop(scale=0.85-1.0) + ColorJitter(brt=0.2,ctr=0.2,sat=0.15)",
        "dataset": "robomimic/lift/ph/low_dim_v15",
    }
    print(f"  Features shape : {features.shape} (frames, versions, dim)", flush=True)

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
