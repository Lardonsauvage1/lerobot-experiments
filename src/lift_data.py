"""Chargement et cache des données Robomimic Lift.

Le dataset hdf5 contient 200 démos de "Pick up the cube and lift it",
chacune avec un état low-dim (eef_pos, eef_quat, gripper_qpos, object)
et une action 7D (delta EEF + gripper).
"""

import numpy as np
import torch
import h5py
from pathlib import Path


STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")
STATE_DIMS = {"robot0_eef_pos": 3, "robot0_eef_quat": 4, "robot0_gripper_qpos": 2, "object": 10}
STATE_DIM_TOTAL = sum(STATE_DIMS.values())  # = 19


def build_state_vector(obs_dict):
    """Concatène les features d'observation en un vecteur 19D dans l'ordre canonique.

    Tolère les deux conventions de nommage :
    - dataset Robomimic hdf5 : 'object' (10D = cube_pos + cube_quat + gripper_to_cube_pos)
    - env Robosuite live    : 'object-state' (même contenu, autre clé)
    """
    parts = []
    for k in STATE_KEYS:
        if k in obs_dict:
            parts.append(np.asarray(obs_dict[k]).flatten())
        elif k == "object" and "object-state" in obs_dict:
            parts.append(np.asarray(obs_dict["object-state"]).flatten())
        else:
            raise KeyError(f"State key '{k}' not found in obs_dict (keys: {sorted(obs_dict.keys())})")
    return np.concatenate(parts)


def load_lift_cached(
    hdf5_path="data_cache/robomimic_lift_ph/low_dim_v15.hdf5",
    cache_path="data_cache/lift_ph_state.pt",
):
    """Charge le dataset Lift PH, met en cache torch.

    Returns dict avec :
      - states: (N, 19) torch.float32
      - actions: (N, 7) torch.float32
      - episodes: (N,) np.int64 — episode_index par frame
      - ep_lengths: (n_demos,) np.int64 — longueur de chaque démo
      - stats: {state_mean, state_std, action_mean, action_std} torch.float32
      - meta: {n_demos, state_dim, action_dim, dataset}
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"  Lift cache : {cache_path}", flush=True)
        return torch.load(cache_path, weights_only=False)

    print(f"  Chargement hdf5 : {hdf5_path}", flush=True)
    all_states, all_actions, all_eps_idx, ep_lengths = [], [], [], []

    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        for ep_idx, demo_name in enumerate(demos):
            demo = f["data"][demo_name]
            n = int(demo.attrs["num_samples"])
            obs_grp = demo["obs"]
            state = np.concatenate([obs_grp[k][:] for k in STATE_KEYS], axis=1)  # (n, 19)
            actions = demo["actions"][:]  # (n, 7)
            all_states.append(state)
            all_actions.append(actions)
            all_eps_idx.extend([ep_idx] * n)
            ep_lengths.append(n)

    states = torch.tensor(np.concatenate(all_states), dtype=torch.float32)
    actions = torch.tensor(np.concatenate(all_actions), dtype=torch.float32)
    episodes = np.array(all_eps_idx, dtype=np.int64)
    ep_lengths = np.array(ep_lengths, dtype=np.int64)

    stats = {
        "state_mean": states.mean(dim=0),
        "state_std": states.std(dim=0).clamp(min=1e-6),
        "action_mean": actions.mean(dim=0),
        "action_std": actions.std(dim=0).clamp(min=1e-6),
    }
    meta = {
        "n_demos": len(demos),
        "state_dim": STATE_DIM_TOTAL,
        "action_dim": actions.shape[1],
        "dataset": "robomimic/lift/ph/low_dim",
        "state_keys": list(STATE_KEYS),
        "state_dims": dict(STATE_DIMS),
    }
    print(f"  Demos : {meta['n_demos']}, Frames : {len(states)}, state={STATE_DIM_TOTAL}D, action={actions.shape[1]}D", flush=True)

    data = {
        "states": states,
        "actions": actions,
        "episodes": episodes,
        "ep_lengths": ep_lengths,
        "stats": stats,
        "meta": meta,
    }
    Path(cache_path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(data, cache_path)
    print(f"  Cache sauvegardé : {cache_path} ({cache_path.stat().st_size/1024:.0f} ko)", flush=True)
    return data
