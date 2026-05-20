"""Harnais d'éval propre — Phase 4 (compression).

Socle de comparaison FIGÉ, identique pour tous les modèles (baseline, rétrécis, quantifiés) :
  - split train 150 / val 50 déterministe (seed fixe), aligné HDF5 ↔ LeRobot
    (la conversion `lift_to_lerobot.py` et `lift_data.py` trient les démos par numéro)
  - init states de rollout = ceux du VAL set (départs jamais entraînés → généralisation)
  - métriques CONTINUES par épisode (pas juste le succès binaire qui sature à 100%) :
      * success        : la tâche a-t-elle réussi à un moment
      * t_success      : 1er step où is_success (proxy "décision" — se dégrade avant le succès)
      * max_z          : hauteur max du cube (marge de levage au-dessus du seuil)
      * hold_fraction  : fraction des steps maintenus en succès après le 1er succès (stabilité)

Voir docs/COMPRESSION.md pour le protocole complet.
"""

import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from src.lift_data import STATE_KEYS

HDF5_PATH = "data_cache/robomimic_lift_ph/low_dim_v15.hdf5"
SPLIT_PATH = "results/runs/lift/phase4_split.json"
N_TOTAL_DEMOS = 200
N_VAL = 50
SPLIT_SEED = 42  # cohérent avec .claude/standard_metrics.md


# --------------------------------------------------------------------------- split
def load_or_make_split(path=SPLIT_PATH, n_total=N_TOTAL_DEMOS, n_val=N_VAL, seed=SPLIT_SEED):
    """Retourne {'train': [...150], 'val': [...50], 'seed':..} ; génère et sauve si absent.

    Le split est une permutation déterministe des indices de démo 0..n_total-1.
    Les indices valent À LA FOIS pour `dataset.episodes` (lerobot-train) et pour
    les init states HDF5 (éval) car l'ordre des démos est identique des deux côtés.
    """
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    val = sorted(int(i) for i in perm[:n_val])
    train = sorted(int(i) for i in perm[n_val:])
    split = {"seed": seed, "n_total": n_total, "n_val": n_val, "val": val, "train": train}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(split, indent=2))
    return split


# ----------------------------------------------------------------------- env Robomimic
def patch_controller_config(env_meta):
    """Adapte les controller_configs Robomimic 1.4 au format composite Robosuite 1.5."""
    old = env_meta["env_kwargs"]["controller_configs"]
    arm = dict(old)
    if "damping" in arm:
        arm["damping_ratio"] = arm.pop("damping")
    if "damping_limits" in arm:
        arm["damping_ratio_limits"] = arm.pop("damping_limits")
    arm.setdefault("input_type", "delta")
    arm.setdefault("input_ref_frame", "base")
    arm["gripper"] = {"type": "GRIP"}
    env_meta["env_kwargs"]["controller_configs"] = {"type": "BASIC", "body_parts": {"right": arm}}


def fix_obs_sign(obs):
    """Robomimic 1.4 stockait eef_pos - cube_pos ; Robosuite 1.5 renvoie l'inverse → flip."""
    obs = dict(obs)
    if "object" in obs:
        obj = np.asarray(obs["object"]).copy()
        obj[7:10] = -obj[7:10]
        obs["object"] = obj
    return obs


def make_env(hdf5_path=HDF5_PATH):
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": list(STATE_KEYS), "rgb": []}})
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=hdf5_path)
    patch_controller_config(env_meta)
    return EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=False
    )


def load_init_states(episode_indices, hdf5_path=HDF5_PATH):
    """Charge l'état initial (flattened mujoco state) des démos données, dans l'ordre fourni."""
    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        return np.stack([f["data"][demos[i]]["states"][0] for i in episode_indices])


def build_state_vector(obs):
    parts = [np.asarray(obs[k]).flatten() for k in STATE_KEYS]
    return np.concatenate(parts).astype(np.float32)


# --------------------------------------------------------------------------- rollout
def rollout_eval(policy, pre, post, env, init_states, *, device, max_steps=200,
                 image_size=96, num_inference_steps=10, verbose=True):
    """Évalue `policy` sur les init_states fournis. Retourne (per_episode, aggregate)."""
    policy.diffusion.num_inference_steps = num_inference_steps
    per_ep = []
    n = init_states.shape[0]
    t0 = time.time()
    for ep in range(n):
        obs = fix_obs_sign(env.reset_to(dict(states=init_states[ep])))
        policy.reset()
        max_z = 0.0
        succ_flags = []
        t_success = None
        for step_i in range(max_steps):
            img = env.env.sim.render(height=image_size, width=image_size, camera_name="agentview")[::-1]
            img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
            state_t = torch.from_numpy(build_state_vector(obs))
            obs_dict = {"observation.image": img_t.unsqueeze(0).to(device),
                        "observation.state": state_t.unsqueeze(0).to(device)}
            obs_dict = pre(obs_dict)
            with torch.no_grad():
                a = policy.select_action(obs_dict)
            a = np.clip(post(a).squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)
            obs, _, done, _ = env.step(a)
            obs = fix_obs_sign(obs)
            max_z = max(max_z, float(np.asarray(obs["object"])[2]))
            is_succ = bool(env.is_success()["task"])
            succ_flags.append(is_succ)
            if is_succ and t_success is None:
                t_success = step_i
            if done:
                break
        success = t_success is not None
        # stabilité du maintien : fraction de succès parmi les steps qui suivent le 1er succès
        hold = float(np.mean(succ_flags[t_success:])) if success else 0.0
        m = {"success": success, "t_success": t_success if success else None,
             "max_z": max_z, "hold_fraction": hold}
        per_ep.append(m)
        if verbose:
            run = sum(e["success"] for e in per_ep) / len(per_ep)
            ts = f"{t_success:3d}" if success else "  -"
            print(f"  ep {ep+1:2d}/{n} : {'✓' if success else '✗'} "
                  f"t_succ={ts} max_z={max_z:.3f} hold={hold:.2f} | running={run:.0%}", flush=True)
    agg = aggregate(per_ep)
    agg["elapsed_s"] = time.time() - t0
    return per_ep, agg


def aggregate(per_ep):
    n = len(per_ep)
    succ = [e for e in per_ep if e["success"]]
    return {
        "n": n,
        "success_rate": len(succ) / n if n else 0.0,
        # temps-au-succès : médiane sur les épisodes réussis (plus bas = plus décisif)
        "t_success_median": float(np.median([e["t_success"] for e in succ])) if succ else None,
        "t_success_mean": float(np.mean([e["t_success"] for e in succ])) if succ else None,
        "max_z_mean": float(np.mean([e["max_z"] for e in per_ep])) if n else None,
        "hold_fraction_mean": float(np.mean([e["hold_fraction"] for e in succ])) if succ else 0.0,
    }
