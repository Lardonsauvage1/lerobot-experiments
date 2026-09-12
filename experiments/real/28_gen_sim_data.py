#!/usr/bin/env python
"""Génère les données du viewer 3D (robot_sim) pour le MODÈLE PROPRE (cooldown gb10, dataset nettoyé).
Pour 6 points de départ (ep0 f0 + 5 autres répartis) : chunk prédit (1 image -> 15 pas) + vraie trajectoire.
Écrit pred_ep0.json (PRED), preds5.json (PREDS), real6.json (REAL), demo.json (DEMO = ep0 complet)."""
import json, numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

MODEL = "results/runs/real/apple_clean_gb10/cooldown_030000/checkpoints/005000/pretrained_model"
REPO, ROOT = "local/apple_joint_224_clean_img", "data_cache/lerobot_apple_joint_224_clean_img"
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

p = DiffusionPolicy.from_pretrained(MODEL); p.eval().to(DEV)
pre, post = make_pre_post_processors(policy_cfg=p.config, pretrained_path=MODEL,
    preprocessor_overrides={"device_processor": {"device": DEV.type}})
IMG = [k for k in p.config.input_features if "image" in k]
NCH = p.config.horizon - p.config.n_obs_steps + 1

def predict_chunk(ds, start):
    p.config.n_action_steps = NCH; p.reset()
    f0 = ds[start]
    obs = {k: f0[k].unsqueeze(0).to(DEV) for k in IMG}
    obs["observation.state"] = f0["observation.state"].unsqueeze(0).to(DEV)
    with torch.no_grad():
        first = p.select_action(pre({k: v.clone() for k, v in obs.items()}))
        q = [first] + [p._queues["action"].popleft() for _ in range(len(p._queues["action"]))]
    return np.stack([post(a).squeeze(0).cpu().numpy() for a in q])

def rnd(a): return [round(float(x), 4) for x in a]

# points de départ : ep0 f0 (début) + 5 répartis (fraction de la longueur d'épisode -> poses en plein geste)
starts = [(0, 0.0), (3, 0.4), (12, 0.35), (20, 0.45), (30, 0.4), (40, 0.5)]
PRED = None; DEMO = None; PREDS = []; REAL = {}
for k, (ep, frac) in enumerate(starts):
    ds = LeRobotDataset(REPO, root=ROOT, episodes=[ep])
    start = int(frac * (ds.num_frames - 1))
    chunk = predict_chunk(ds, start)
    gt = np.stack([ds[min(start + i, ds.num_frames - 1)]["action"].numpy() for i in range(len(chunk))])
    label = f"ep{ep} f{start}"
    REAL[label] = [rnd(r) for r in gt]
    poses = [rnd(r) for r in chunk]
    err = float(np.abs(chunk[:, :5] - gt[:, :5]).max())
    print(f"{label:12} | {len(chunk)} pas | err max {err:.3f} rad ({np.degrees(err):.1f} deg)")
    if k == 0:
        PRED = poses
        DEMO = [rnd(ds[i]["action"].numpy()) for i in range(ds.num_frames)]  # ep0 complet
    else:
        PREDS.append({"label": label, "ep": ep, "frame": start, "poses": poses})

json.dump(PRED, open("robot_sim/pred_ep0.json", "w"))
json.dump(PREDS, open("robot_sim/preds5.json", "w"))
json.dump(REAL, open("robot_sim/real6.json", "w"))
json.dump(DEMO, open("robot_sim/demo.json", "w"))
print(f"-> écrit : PRED({len(PRED)}) PREDS({len(PREDS)}) REAL({len(REAL)}) DEMO({len(DEMO)})")
