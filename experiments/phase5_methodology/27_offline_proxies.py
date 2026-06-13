"""Proxies OFFLINE du succes (sans simulation) pour checkpoints mini-CNN :
  3) erreur d'action GENEREE (diffusion complete) vs chunk expert — MSE + decomposition par pas d'horizon (compounding)
  2) MMD (Maximum Mean Discrepancy) entre actions echantillonnees (K tirages) et actions expertes
     -> equivalent sample-based de Hellinger pour un modele generatif.

Sous-echantillonne : N fenetres val fixes (memes pour tous les checkpoints), tous les --interval steps.
Sortie CSV : step, action_mse, mmd, mse_h0..mse_h{n_action_steps-1}.
"""
import argparse, csv, json, sys, time
from pathlib import Path
sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite; silence_robosuite()
import torch
from torch.utils.data import DataLoader
from src import can_eval
from src.mini_cnn import load_minicnn_from_ckpt

PHASE = Path("results/runs/phase5_methodology")
DATASET_REPO = "local/can_ph_proprio_birdview"
DATASET_ROOT = "data_cache/lerobot_can_ph_proprio_birdview"
OBS_IMAGES, OBS_STATE, ACTION = "observation.images", "observation.state", "action"


def build_delta_timestamps(cfg, fps):
    dt = {k: [i / fps for i in cfg.observation_delta_indices] for k in cfg.input_features}
    dt["action"] = [i / fps for i in cfg.action_delta_indices]
    return dt


def prep_images(policy, b):
    b = dict(b)
    if policy.config.image_features:
        for key in policy.config.image_features:
            if policy.config.n_obs_steps == 1 and b[key].ndim == 4:
                b[key] = b[key].unsqueeze(1)
        b[OBS_IMAGES] = torch.stack([b[key] for key in policy.config.image_features], dim=-4)
    return b


def patch_device_cpu(ckpt):
    for fn in ("policy_preprocessor.json", "policy_postprocessor.json"):
        p = Path(ckpt) / fn
        if not p.exists(): continue
        c = json.loads(p.read_text()); ch = False
        for st in c.get("steps", []):
            if st.get("registry_name") == "device_processor" and st.get("config", {}).get("device") != "cpu":
                st["config"]["device"] = "cpu"; ch = True
        if ch: p.write_text(json.dumps(c, indent=2))


def mmd_rbf(x, y):
    """MMD^2 (kernel RBF, bandwidth = heuristique mediane) entre 2 nuages (n,d) et (m,d)."""
    def pdist2(a, b):
        return (a.pow(2).sum(1, keepdim=True) - 2 * a @ b.t() + b.pow(2).sum(1)[None, :]).clamp_min(0)
    xx, yy, xy = pdist2(x, x), pdist2(y, y), pdist2(x, y)
    with torch.no_grad():
        med = torch.median(torch.cat([xx.flatten(), yy.flatten(), xy.flatten()]))
        sig = med.clamp_min(1e-8)
    kxx, kyy, kxy = torch.exp(-xx / sig), torch.exp(-yy / sig), torch.exp(-xy / sig)
    return (kxx.mean() + kyy.mean() - 2 * kxy.mean()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--interval", type=int, default=2000)
    ap.add_argument("--n-windows", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device(a.device if (a.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch

    val_idx = can_eval.load_or_make_split()["val"]
    fps = json.load(open(Path(DATASET_ROOT) / "meta" / "info.json"))["fps"]
    done = set()
    if Path(a.out).exists():
        for r in csv.DictReader(open(a.out)): done.add(int(r["step"]))

    targets = {}
    for rd in a.run_dirs.split(","):
        ckroot = PHASE / rd / "checkpoints"
        if not ckroot.exists(): continue
        for p in sorted(ckroot.iterdir()):
            if p.is_dir() and p.name.isdigit():
                s = int(p.name)
                if s % a.interval == 0 and s not in done and (p / "pretrained_model" / "model.safetensors").exists():
                    targets[s] = p / "pretrained_model"
    targets = sorted(targets.items())
    print(f"[27] device={device}, {len(targets)} ckpts, N={a.n_windows} fenetres, K={a.k}", flush=True)

    subset_batches = None  # fenetres val figees (memes pour tous les ckpts)
    for step, ckdir in targets:
        ckpt = str(ckdir)
        try:
            patch_device_cpu(ckpt)
            policy = load_minicnn_from_ckpt(ckpt, device); policy.eval()
            pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
                to_transition=batch_to_transition, to_output=transition_to_batch)
            n_obs = policy.config.n_obs_steps; n_act = policy.config.n_action_steps
            start = n_obs - 1; end = start + n_act
            if subset_batches is None:
                ds = LeRobotDataset(DATASET_REPO, root=DATASET_ROOT, episodes=val_idx,
                                    delta_timestamps=build_delta_timestamps(policy.config, fps))
                g = torch.Generator(); g.manual_seed(a.seed)
                dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=0, generator=g, drop_last=True)
                subset_batches = []
                for b in dl:
                    subset_batches.append(b)
                    if len(subset_batches) * a.batch >= a.n_windows: break

            t0 = time.time()
            se_step = torch.zeros(n_act); nb = 0; mmd_acc = 0.0; mmd_n = 0; cov_acc = 0.0
            with torch.no_grad():
                for b in subset_batches:
                    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
                    pb = prep_images(policy, pre(b))
                    expert = pb[ACTION][:, start:end]                       # (B, n_act, act_dim) normalise
                    samples = []
                    for _ in range(a.k):
                        gen = policy.diffusion.generate_actions(pb)         # (B, n_act, act_dim)
                        samples.append(gen)
                    S = torch.stack(samples, 0)                             # (K, B, n_act, act_dim)
                    mean_pred = S.mean(0)                                   # (B, n_act, act_dim) estimateur ponctuel
                    se_step += ((mean_pred - expert) ** 2).mean(dim=(0, 2)).cpu()  # (n_act,)
                    # COUVERTURE par etat : min sur les K tirages de la MSE a l'expert (multimodalite-conscient)
                    d2 = ((S - expert.unsqueeze(0)) ** 2).mean(dim=(2, 3))  # (K, B) MSE de chaque tirage a l'expert
                    cov_acc += float(d2.min(dim=0).values.mean())          # moyenne_etats( min_K )
                    nb += 1
                    Bsz = expert.shape[0]
                    pol = S.permute(1, 0, 2, 3).reshape(Bsz * a.k, -1)      # (B*K, n_act*act_dim)
                    exp = expert.reshape(Bsz, -1)                           # (B, n_act*act_dim)
                    mmd_acc += mmd_rbf(pol, exp); mmd_n += 1
            mse_step = (se_step / nb)
            row = {"step": step, "action_mse": float(mse_step.mean()), "coverage": cov_acc / nb, "mmd": mmd_acc / mmd_n,
                   **{f"mse_h{i}": float(mse_step[i]) for i in range(n_act)}, "elapsed_sec": time.time() - t0}
            wh = not Path(a.out).exists()
            with open(a.out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if wh: w.writeheader()
                w.writerow(row)
            print(f"[27] step={step}: action_mse={row['action_mse']:.4f} mmd={row['mmd']:.4f} ({row['elapsed_sec']:.0f}s)", flush=True)
            del policy
        except Exception as e:
            import traceback; print(f"[27] erreur step={step}: {e}\n{traceback.format_exc()[:500]}", flush=True)
    print("[27] DONE", flush=True)


if __name__ == "__main__":
    main()
