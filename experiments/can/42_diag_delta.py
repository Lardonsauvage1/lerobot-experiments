"""DIAGNOSTIC delta articulaire — pourquoi 0% ? Le modèle prédit-il des deltas utiles ?

Charge le checkpoint delta évalué, fait N rollouts, et compare :
  - échelle des deltas PRÉDITS (post-unnormalize, avant +jpos) vs deltas du DATASET
  - mouvement réel du bras (déplacement articulaire cumulé / net)
  - vs ORACLE (deltas vrais du dataset rejoués = 100%)
Tranche : modèle figé (~0) ? bon ordre de grandeur mais mauvaise direction ? bras immobile ?
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# réutilise la machinerie de l'éval
import importlib.util
spec = importlib.util.spec_from_file_location("ev37",
    str(Path(__file__).parent / "37_eval_joint_birdview.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "results/runs/can/joint_r34_delta_coh_P/cooldown_40k/checkpoints/005000/pretrained_model"
N_EP = int(sys.argv[2]) if len(sys.argv) > 2 else 3
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

# --- stats deltas du dataset (référence) ---
root = "data_cache/lerobot_can_ph_joint_delta_coh_birdview"
ds = json.load(open(f"{root}/meta/stats.json"))["action"]
ds_std = np.array(ds["std"])[:7]; ds_mean = np.array(ds["mean"])[:7]
np.set_printoptions(precision=4, suppress=True, linewidth=120)

print(f"=== DIAGNOSTIC 2 — deltas PRÉDITS par le modèle ({Path(CKPT).parts[-3]}) ===\n")
policy, pre, post = ev.load(CKPT, DEV)

pred_deltas = []   # 8D actions prédites (dim 0-6 = delta joint)
jpos_traj = []     # jpos réel visité
import robosuite  # noqa
env = ev.build_joint_env(50)
states = ev.load_states()[:N_EP] if hasattr(ev, "load_states") else None

# charge les états initiaux depuis le HDF5 (comme l'éval)
import h5py
with h5py.File(ev.HDF5, "r") as f:
    demos = sorted(f["data"].keys(), key=lambda d: int(d.split("_")[1]))[:N_EP]
    init_states = [f[f"data/{d}/states"][0] for d in demos]

for ep, st0 in enumerate(init_states):
    obs = env.reset_to({"states": st0}); policy.reset()
    j0 = np.asarray(obs["robot0_joint_pos"]).flatten()[:7].copy()
    ep_jpos = [j0.copy()]
    for t in range(ev.MAX_STEPS):
        images = {}
        for cam, key in ev.IMAGE_KEYS.items():
            img = env.env.sim.render(height=96, width=96, camera_name=cam)[::-1]
            images[key] = torch.from_numpy(img.copy()).permute(2, 0, 1).float().unsqueeze(0).to(DEV) / 255.0
        stt = torch.from_numpy(ev.state_joint(obs))
        od = pre({**images, "observation.state": stt.unsqueeze(0).to(DEV)})
        od = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in od.items()}
        with torch.no_grad():
            a = policy.select_action(od)
        a = post(a).squeeze(0).cpu().numpy().astype(np.float32)  # 8D : dim0-6 = DELTA prédit
        pred_deltas.append(a[:7].copy())
        a2 = a.copy(); a2[:7] = np.asarray(obs["robot0_joint_pos"]).flatten()[:7] + a[:7]
        obs = env.step(a2)[0]
        ep_jpos.append(np.asarray(obs["robot0_joint_pos"]).flatten()[:7].copy())
        if env.is_success()["task"]:
            break
    ep_jpos = np.array(ep_jpos)
    jpos_traj.append(ep_jpos)
    net = np.linalg.norm(ep_jpos[-1] - ep_jpos[0])
    path = np.linalg.norm(np.diff(ep_jpos, axis=0), axis=1).sum()
    print(f"  ép {ep}: {len(ep_jpos)-1} steps | déplacement NET={net:.3f} rad | chemin CUMULÉ={path:.3f} rad "
          f"| succès={'OUI' if env.is_success()['task'] else 'non'}")

pred = np.array(pred_deltas)  # (T,7)
print(f"\n=== ÉCHELLE des deltas prédits vs dataset (rad, par joint) ===")
print(f"  {'joint':<7}{'pred_std':>10}{'data_std':>10}{'ratio':>8}   {'pred_mean':>10}{'data_mean':>10}")
for i in range(7):
    ps, ds_ = pred[:, i].std(), ds_std[i]
    print(f"  j{i:<6}{ps:>10.4f}{ds_:>10.4f}{ps/ds_ if ds_ else 0:>8.2f}   {pred[:,i].mean():>10.4f}{ds_mean[i]:>10.4f}")

print(f"\n=== VERDICT diag ===")
ratio = np.array([pred[:, i].std() / ds_std[i] for i in range(7)])
print(f"  ratio std pred/dataset moyen = {ratio.mean():.2f}")
avg_net = np.mean([np.linalg.norm(t[-1]-t[0]) for t in jpos_traj])
avg_path = np.mean([np.linalg.norm(np.diff(t,axis=0),axis=1).sum() for t in jpos_traj])
print(f"  déplacement bras moyen : net={avg_net:.3f} rad, chemin cumulé={avg_path:.3f} rad")
if ratio.mean() < 0.3:
    print("  -> modèle QUASI FIGÉ : prédit des deltas ~0 (collapse mode-averaging) -> bras immobile")
elif avg_net < 0.2:
    print("  -> le bras BOUGE (bonne échelle) mais N'AVANCE PAS (annule / mauvaise direction)")
else:
    print("  -> le bras bouge ET se déplace : l'échec vient d'ailleurs (précision/timing)")
np.save("/tmp/diag_pred_deltas.npy", pred)
print("\n(deltas prédits sauvés dans /tmp/diag_pred_deltas.npy)")
