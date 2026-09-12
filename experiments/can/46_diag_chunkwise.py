"""DIAGNOSTIC delta CHUNK-WISE — pourquoi seulement 6% ?
Charge le checkpoint cooldown chunk-wise, fait N rollouts (ancre fixe par replan),
enregistre les deltas RELATIFS prédits (post output, avant +ancre) vs stats chunk-wise du dataset,
+ déplacement du bras. Tranche : magnitude collapse ? direction ? offset systématique ?
"""
import json, importlib.util, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
spec = importlib.util.spec_from_file_location("ev37", str(Path(__file__).parent / "37_eval_joint_birdview.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "results/runs/can/joint_r34_reljoint_cw/cooldown_40k/checkpoints/005000/pretrained_model"
N_EP = int(sys.argv[2]) if len(sys.argv) > 2 else 3
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

rs = json.load(open("data_cache/lerobot_can_ph_joint_birdview/meta/relstats_chunkwise.json"))
rel_std_data = np.array(rs["std"])[:7]
np.set_printoptions(precision=4, suppress=True, linewidth=120)

print(f"=== DIAG CHUNK-WISE ({Path(CKPT).parts[-3]}) ===\n")
policy, pre, post = ev.load(CKPT, DEV)
n_act = int(policy.config.n_action_steps)

import h5py
with h5py.File(ev.HDF5, "r") as f:
    demos = sorted(f["data"].keys(), key=lambda d: int(d.split("_")[1]))[:N_EP]
    init_states = [f[f"data/{d}/states"][0] for d in demos]

pred_rel = []; jpos_traj = []
env = ev.build_joint_env(50)
for ep, st0 in enumerate(init_states):
    obs = env.reset_to({"states": st0}); policy.reset()
    ep_j = [np.asarray(obs["robot0_joint_pos"]).flatten()[:7].copy()]; anchor = None
    for t in range(ev.MAX_STEPS):
        if t % n_act == 0:
            anchor = np.asarray(obs["robot0_joint_pos"]).flatten()[:7].copy()
        images = {}
        for cam, key in ev.IMAGE_KEYS.items():
            img = env.env.sim.render(height=96, width=96, camera_name=cam)[::-1]
            images[key] = torch.from_numpy(img.copy()).permute(2,0,1).float().unsqueeze(0).to(DEV)/255.0
        stt = torch.from_numpy(ev.state_joint(obs))
        od = pre({**images, "observation.state": stt.unsqueeze(0).to(DEV)})
        od = {k:(v.to(DEV) if torch.is_tensor(v) else v) for k,v in od.items()}
        with torch.no_grad():
            a = policy.select_action(od)
        a = post(a).squeeze(0).cpu().numpy().astype(np.float32)  # joints RELATIFS + gripper abs
        pred_rel.append(a[:7].copy())
        a2 = a.copy(); a2[:7] = anchor + a[:7]
        obs = env.step(a2)[0]
        ep_j.append(np.asarray(obs["robot0_joint_pos"]).flatten()[:7].copy())
        if env.is_success()["task"]: break
    ep_j = np.array(ep_j); jpos_traj.append(ep_j)
    print(f"  ép {ep}: {len(ep_j)-1} steps | NET={np.linalg.norm(ep_j[-1]-ep_j[0]):.3f} rad "
          f"| chemin={np.linalg.norm(np.diff(ep_j,axis=0),axis=1).sum():.3f} rad "
          f"| succès={'OUI' if env.is_success()['task'] else 'non'}")

pred = np.array(pred_rel)
print(f"\n=== échelle deltas RELATIFS prédits vs dataset chunk-wise ===")
print(f"  {'joint':<6}{'pred_std':>10}{'data_std':>10}{'ratio':>8}")
for i in range(7):
    ps=pred[:,i].std(); print(f"  j{i:<5}{ps:>10.4f}{rel_std_data[i]:>10.4f}{ps/rel_std_data[i]:>8.2f}")
ratio=np.array([pred[:,i].std()/rel_std_data[i] for i in range(7)])
avg_path=np.mean([np.linalg.norm(np.diff(t,axis=0),axis=1).sum() for t in jpos_traj])
print(f"\n  ratio magnitude moyen = {ratio.mean():.2f}  (séquentiel était 0.39)")
print(f"  chemin bras moyen = {avg_path:.3f} rad")
if ratio.mean() < 0.5:
    print("  -> encore un EFFONDREMENT de magnitude (le chunk-wise n'a pas suffi)")
elif avg_path < 1.5:
    print("  -> bonne magnitude mais bras sous-actionné / mauvaise direction")
else:
    print("  -> bras bouge à la bonne échelle : échec = précision/offset systématique")
