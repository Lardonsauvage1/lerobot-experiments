"""Éval du delta CHUNK-WISE (relatif) — ancre FIXE par replan.

La politique prédit des joints RELATIFS (post() les dé-normalise avec les stats relatives sauvées
dans le checkpoint). On ajoute l'ancre = jpos capturée au moment du REPLAN (tous les n_action_steps),
la même pour tout le chunk -> chunk-wise correct (≠ re-ancrage par pas = séquentiel).
Gripper reste absolu.

Usage : venv312/bin/python experiments/can/44_eval_reljoint.py --run-dir DIR --steps S --n N [--kp 50]
"""
import argparse, csv, importlib.util, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
spec = importlib.util.spec_from_file_location("ev37", str(Path(__file__).parent / "37_eval_joint_birdview.py"))
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)


def eval_ckpt_rel(ckpt, states, device, kp, max_steps, img_size=96):
    policy, pre, post = ev.load(ckpt, device)
    n_act = int(policy.config.n_action_steps)
    succ = []; t0 = __import__("time").time()
    for i in range(0, len(states), ev.CHUNK):
        env = ev.build_joint_env(kp)
        for ep in range(i, min(i + ev.CHUNK, len(states))):
            obs = env.reset_to({"states": states[ep]}); policy.reset(); ok = False
            anchor = None
            for t in range(max_steps):
                if t % n_act == 0:  # REPLAN -> capture l'ancre (jpos courante), fixe pour le chunk
                    anchor = np.asarray(obs["robot0_joint_pos"]).flatten()[:7].copy()
                images = {}
                for cam, key in ev.IMAGE_KEYS.items():
                    img = env.env.sim.render(height=img_size, width=img_size, camera_name=cam)[::-1]
                    images[key] = torch.from_numpy(img.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                st = torch.from_numpy(ev.state_joint(obs))
                od = pre({**images, "observation.state": st.unsqueeze(0).to(device)})
                od = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in od.items()}
                with torch.no_grad():
                    a = policy.select_action(od)
                a = post(a).squeeze(0).cpu().numpy().astype(np.float32)  # joints RELATIFS + gripper absolu
                a[:7] = anchor + a[:7]                                    # + ancre FIXE du replan
                obs = env.step(a)[0]
                if env.is_success()["task"]:
                    ok = True; break
            succ.append(ok)
        try: env.env.close()
        except Exception: pass
    k = sum(succ); n = len(succ); lo, hi = ev.wilson_ci(k, n)
    return {"n_success": k, "n": n, "success_rate": k / n, "ci95_low": lo, "ci95_high": hi,
            "elapsed_sec": __import__("time").time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--steps", default="")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--kp", type=float, default=50)
    ap.add_argument("--max-steps", type=int, default=ev.MAX_STEPS)
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rd = Path(a.run_dir)
    states = ev.load_states()[:a.n] if hasattr(ev, "load_states") else None
    if states is None:
        import h5py
        with h5py.File(ev.HDF5, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda d: int(d.split("_")[1]))[:a.n]
            states = [f[f"data/{d}/states"][0] for d in demos]
    steps = [int(s) for s in a.steps.split(",")] if a.steps.strip() else \
        sorted(int(p.name) for p in (rd / "checkpoints").glob("*") if p.name.isdigit())
    new = not Path(a.out).exists()
    with open(a.out, "a", newline="") as fo:
        w = csv.writer(fo)
        if new: w.writerow(["step", "n_success", "n", "success_rate", "ci95_low", "ci95_high", "elapsed_sec"])
        for s in steps:
            ck = rd / "checkpoints" / f"{s:06d}" / "pretrained_model"
            if not ck.exists(): ck = rd / "checkpoints" / f"{s:06d}"
            r = eval_ckpt_rel(str(ck), states, device, a.kp, a.max_steps, a.img_size)
            w.writerow([s, r["n_success"], r["n"], r["success_rate"], r["ci95_low"], r["ci95_high"], r["elapsed_sec"]])
            fo.flush()
            print(f"step {s}: {r['n_success']}/{r['n']} = {r['success_rate']*100:.1f}% "
                  f"IC95[{r['ci95_low']*100:.0f}-{r['ci95_high']*100:.0f}%]")


if __name__ == "__main__":
    main()
