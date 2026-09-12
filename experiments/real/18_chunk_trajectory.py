#!/usr/bin/env python
"""Genere la trajectoire complete predite depuis UNE image de depart, et la compare a la demo.
On force n_action_steps = horizon-n_obs_steps+1 pour recuperer tous les pas futurs d'un seul chunk
(via le chemin select_action qui gere l'empilage des 2 cameras + la normalisation)."""
import sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

MODEL = sys.argv[1] if len(sys.argv) > 1 else "results/runs/real/apple_joint_224_r34/cooldown_012000/checkpoints/005000/pretrained_model"
EP = int(sys.argv[2]) if len(sys.argv) > 2 else 8
START = int(sys.argv[3]) if len(sys.argv) > 3 else 0
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

REPO = sys.argv[4] if len(sys.argv) > 4 else "local/apple_joint_224"
ROOT = sys.argv[5] if len(sys.argv) > 5 else "data_cache/lerobot_apple_joint_224"
ds = LeRobotDataset(REPO, root=ROOT, episodes=[EP])
p = DiffusionPolicy.from_pretrained(MODEL); p.eval().to(DEV)
pre, post = make_pre_post_processors(policy_cfg=p.config, pretrained_path=MODEL,
    preprocessor_overrides={"device_processor": {"device": DEV.type}})
IMG = [k for k in p.config.input_features if "image" in k]

# recupere tout le chunk futur : n_action_steps = horizon - n_obs_steps + 1
NCH = p.config.horizon - p.config.n_obs_steps + 1
p.config.n_action_steps = NCH
p.reset()

f0 = ds[START]
obs = {k: f0[k].unsqueeze(0).to(DEV) for k in IMG}
obs["observation.state"] = f0["observation.state"].unsqueeze(0).to(DEV)
with torch.no_grad():
    first = p.select_action(pre({k: v.clone() for k, v in obs.items()}))      # remplit la file
    q = [first] + [p._queues["action"].popleft() for _ in range(len(p._queues["action"]))]
chunk = np.stack([post(a).squeeze(0).cpu().numpy() for a in q])                 # [NCH, 6] denormalise

# verite terrain : actions demo START..START+NCH-1
gt = np.stack([ds[min(START + i, ds.num_frames - 1)]["action"].numpy() for i in range(len(chunk))])
err = np.abs(chunk[:, :5] - gt[:, :5]).max()

print(f"ep{EP} depart frame {START} | {len(chunk)} pas futurs predits | err max joints vs demo = {err:.3f} rad ({np.degrees(err):.1f}deg)")
print(f"\n{'pas':>3} | {'PREDIT (5 joints + grip)':^46} | {'DEMO (5 joints + grip)':^46}")
for i in range(len(chunk)):
    print(f"{i:>3} | {str(np.round(chunk[i],3)):^46} | {str(np.round(gt[i],3)):^46}")

fig, ax = plt.subplots(1, 2, figsize=(14, 5)); t = np.arange(len(chunk))
for d in range(5):
    ax[0].plot(t, gt[:, d], "--", color=f"C{d}", alpha=0.55)
    ax[0].plot(t, chunk[:, d], "-o", color=f"C{d}", ms=3, label=f"joint_{d+1}")
ax[0].set_title("Joints : — predit  vs  -- demo"); ax[0].set_xlabel("pas futur"); ax[0].set_ylabel("rad"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].plot(t, gt[:, 5], "--", color="k", alpha=0.6, label="demo"); ax[1].plot(t, chunk[:, 5], "-o", color="purple", ms=3, label="predit")
ax[1].set_title("Gripper"); ax[1].set_ylim(-0.1, 1.1); ax[1].set_xlabel("pas futur"); ax[1].legend(); ax[1].grid(alpha=0.3)
fig.suptitle(f"Trajectoire predite (1 image) vs demo — ep{EP} frame {START} (err max {np.degrees(err):.1f}deg)", fontweight="bold")
import os as _os; _os.makedirs("results/runs/real/witness", exist_ok=True)
fig.tight_layout(); out = f"results/runs/real/witness/chunk_ep{EP}_f{START}.png"; fig.savefig(out, dpi=120)
print(f"-> {out}")
