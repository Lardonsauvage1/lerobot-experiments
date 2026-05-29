"""Phase 5 (Can) — éval du modèle vision-only AVEC 2 caméras (agentview + wrist).

Pendant de 03_eval_proprio.py mais avec 2 images : à chaque step on rend agentview ET
robot0_eye_in_hand (96px), on les passe sous les clés observation.images.agentview /
observation.images.wrist (mêmes clés que dans le dataset).

Sortie : results/runs/can/06_proprio_wrist_eval.json
"""
import sys
sys.path.insert(0, ".")

from src.quiet_robosuite import silence_robosuite
silence_robosuite()

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import can_eval

DATASET_REPO, DATASET_ROOT = "local/can_ph_proprio_wrist", "data_cache/lerobot_can_ph_proprio_wrist"
STEPS = 10
MAX_STEPS = can_eval.MAX_STEPS
IMAGE_SIZE = 96


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def state_proprio(obs):
    return np.concatenate([
        np.asarray(obs["robot0_eef_pos"]).flatten(),
        np.asarray(obs["robot0_eef_quat"]).flatten(),
        np.asarray(obs["robot0_gripper_qpos"]).flatten(),
    ]).astype(np.float32)


def render_pair(env, hd=IMAGE_SIZE):
    """Rend les 2 caméras à la fois et renvoie (agentview, wrist) sous format (3,H,W) torch."""
    ag = env.env.sim.render(height=hd, width=hd, camera_name="agentview")[::-1]
    wr = env.env.sim.render(height=hd, width=hd, camera_name="robot0_eye_in_hand")[::-1]
    ag_t = torch.from_numpy(ag.copy()).permute(2, 0, 1).float() / 255.0
    wr_t = torch.from_numpy(wr.copy()).permute(2, 0, 1).float() / 255.0
    return ag_t, wr_t


def build_delta_timestamps(cfg, fps):
    dt = {k: [i / fps for i in cfg.observation_delta_indices] for k in cfg.input_features}
    dt["action"] = [i / fps for i in cfg.action_delta_indices]
    return dt


def rollout_multi_cam(policy, pre, post, env, init_states, device, steps_pas):
    """Rollout avec 2 caméras. 50 départs val → un seul env (sous le seuil de dégradation)."""
    policy.diffusion.num_inference_steps = steps_pas
    per = []; t0 = time.time()
    for ep in range(len(init_states)):
        obs = env.reset_to(dict(states=init_states[ep])); policy.reset()
        t_success = None
        for step_i in range(MAX_STEPS):
            ag_t, wr_t = render_pair(env)
            state_t = torch.from_numpy(state_proprio(obs))
            obs_dict = pre({
                "observation.images.agentview": ag_t.unsqueeze(0).to(device),
                "observation.images.wrist": wr_t.unsqueeze(0).to(device),
                "observation.state": state_t.unsqueeze(0).to(device),
            })
            with torch.no_grad():
                a = policy.select_action(obs_dict)
            a = np.clip(post(a).squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)
            obs = env.step(a)[0]
            if env.is_success()["task"]:
                t_success = step_i; break
        per.append({"success": t_success is not None, "t_success": t_success})
        if (ep + 1) % 10 == 0:
            run = sum(e["success"] for e in per) / len(per)
            print(f"  ep {ep + 1}/{len(init_states)} : running={run:.0%}", flush=True)
    return per, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/runs/can/06_proprio_wrist")
    ap.add_argument("--out", default="results/runs/can/06_proprio_wrist_eval.json")
    args = ap.parse_args()
    RUN_DIR = Path(args.run_dir)
    OUT = Path(args.out)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    val_idx = can_eval.load_or_make_split()["val"]
    init_states = can_eval.load_init_states(val_idx)
    fps = json.load(open(Path(DATASET_ROOT) / "meta" / "info.json"))["fps"]

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action,
    )

    def load(ckpt):
        policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
        pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
            to_transition=batch_to_transition, to_output=transition_to_batch)
        post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
        return policy, pre, post

    ckpt_root = RUN_DIR / "checkpoints"
    steps_ck = sorted(int(p.name) for p in ckpt_root.iterdir() if p.name.isdigit())
    print(f"Checkpoints: {steps_ck}", flush=True)

    val_ds = None
    val_by_step = {}
    for s in steps_ck:
        ckpt = str(ckpt_root / f"{s:06d}" / "pretrained_model")
        policy, pre, _ = load(ckpt)
        if val_ds is None:
            val_ds = LeRobotDataset(DATASET_REPO, root=DATASET_ROOT, episodes=val_idx,
                                    delta_timestamps=build_delta_timestamps(policy.config, fps))
        dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
        tot, n = 0.0, 0
        with torch.no_grad():
            for batch in dl:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                bs = batch["action"].shape[0]
                loss, _ = policy.forward(pre(batch)); tot += float(loss) * bs; n += bs
        val_by_step[s] = tot / n
        print(f"  step {s}: val_loss={val_by_step[s]:.4f}", flush=True)
        del policy
    best = min(val_by_step, key=val_by_step.get)
    print(f"-> meilleur checkpoint {best} (val_loss={val_by_step[best]:.4f})", flush=True)

    ckpt = str(ckpt_root / f"{best:06d}" / "pretrained_model")
    policy, pre, post = load(ckpt)
    env = can_eval.make_env()
    per, elapsed = rollout_multi_cam(policy, pre, post, env, init_states, device, STEPS)
    k = sum(e["success"] for e in per)
    n = len(per)
    lo, hi = wilson_ci(k, n)
    t_succs = [e["t_success"] for e in per if e["success"]]
    t_med = float(np.median(t_succs)) if t_succs else None
    print(f"\n=== CAN 2-cams (agentview+wrist, 9D proprio) : {k}/{n} = {k/n:.1%} "
          f"[{lo:.1%}, {hi:.1%}]  t_succ_med={t_med} (best {best}, @{STEPS} pas, {elapsed/60:.1f}min) ===",
          flush=True)
    OUT.write_text(json.dumps({"best_step": best, "val_loss": val_by_step[best], "steps": STEPS,
                               "n": n, "n_success": k, "success_rate": k/n, "ci95": [lo, hi],
                               "t_success_median": t_med, "val_by_step": val_by_step}, indent=2))
    print(f"JSON: {OUT}")


if __name__ == "__main__":
    main()
