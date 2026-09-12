"""Vérifie que le checkpoint RELJOINT dé-normalise les JOINTS avec les stats RELATIVES.

Test : construit une action normalisée n (joints via stats relatives, gripper via stats absolues),
la passe dans post() du checkpoint -> doit récupérer les valeurs brutes attendues.
Puis confirme : brut_relatif + ancre = action absolue (par construction).
"""
import json, importlib.util, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
spec = importlib.util.spec_from_file_location("ev37", str(Path(__file__).parent / "37_eval_joint_birdview.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)

CKPT = sys.argv[1] if len(sys.argv) > 1 else "results/runs/can/reljoint_mini/checkpoints/000300/pretrained_model"
_, _, post = ev.load(CKPT, "cpu")

rs = json.load(open("data_cache/lerobot_can_ph_joint_birdview/meta/relstats_chunkwise.json"))
rmin, rmax = np.array(rs["min"]), np.array(rs["max"])            # relatif joints 0-6
ds = json.load(open("data_cache/lerobot_can_ph_joint_birdview/meta/stats.json"))["action"]
gmin, gmax = np.array(ds["min"])[7], np.array(ds["max"])[7]      # gripper absolu (dim 7)

# valeurs brutes cibles : un relatif "milieu" par joint + un gripper
raw_rel = (rmin + rmax) / 2 + 0.3 * (rmax - rmin) / 2            # dans la plage relative
raw_grip = (gmin + gmax) / 2

# normalise MIN_MAX -> [-1,1]
n = np.zeros(8, dtype=np.float32)
n[:7] = 2 * (raw_rel - rmin) / (rmax - rmin) - 1
n[7] = 2 * (raw_grip - gmin) / (gmax - gmin) - 1

out = post(torch.tensor(n).unsqueeze(0)).squeeze(0).cpu().numpy()

print("=== ROUND-TRIP : post(normalisé) doit rendre le brut attendu ===")
err_j = np.abs(out[:7] - raw_rel).max()
err_g = abs(out[7] - raw_grip)
print(f"  joints : max|out - raw_rel| = {err_j:.2e}   (attendu ~0 si post utilise stats RELATIVES)")
print(f"  gripper: |out - raw_grip|   = {err_g:.2e}   (attendu ~0 si gripper absolu)")
print(f"  out[:7]     = {np.array2string(out[:7], precision=4)}")
print(f"  raw_rel[:7] = {np.array2string(raw_rel, precision=4)}")

OK = err_j < 1e-3 and err_g < 1e-3
print(f"\n  {'✅ POST utilise bien les stats RELATIVES pour les joints' if OK else '❌ ÉCHEC : stats mal bakées'}")

# sanity : brut_relatif + ancre = absolu (par construction, montré numériquement)
anchor = np.array([0.1, -0.2, 0.3, -1.5, 0.0, 1.0, 0.5])
print(f"\n=== + ancre -> absolu (par construction) ===")
print(f"  ancre + out[:7] = {np.array2string(anchor + out[:7], precision=4)}  (= cible articulaire absolue)")
sys.exit(0 if OK else 1)
