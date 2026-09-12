"""Rend des vidéos de rollout du modèle JOINT (40k) sur Can — pour VOIR pourquoi il échoue.

Env JOINT_POSITION absolu (kp=50), state articulaire, action 8D brute (comme 37_eval_joint).
Sauve une mp4 par épisode (agentview + birdview côte à côte, haute rés pour l'œil ; la policy
voit toujours du 96x96). Nom de fichier = succès/échec.

Usage : venv312/bin/python -u experiments/can/44_joint_videos.py --ckpt <pretrained_model> --n 6 --out <dir>
"""
import argparse, math, sys
from pathlib import Path
sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite
silence_robosuite()
import numpy as np, cv2, torch

STATES_PATH = "results/runs/can/can_eval500.npy"
HDF5 = "data_cache/robomimic_can_ph/low_dim_v141.hdf5"
IMAGE_KEYS = {"agentview": "observation.images.agentview", "birdview": "observation.images.birdview"}
MAX_STEPS = 300
INFER_STEPS = 10
VID = 256  # résolution d'affichage


def state_joint(obs):
    return np.concatenate([np.asarray(obs["robot0_joint_pos"]).flatten(),
                           np.asarray(obs["robot0_gripper_qpos"]).flatten()]).astype(np.float32)


def build_joint_env(kp=50):
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils
    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": ["robot0_joint_pos", "robot0_gripper_qpos", "object"], "rgb": []}})
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=HDF5)
    arm = {"type": "JOINT_POSITION", "input_type": "absolute",
           "kp": kp, "damping_ratio": 1, "impedance_mode": "fixed",
           "kp_limits": [0, 1000], "damping_ratio_limits": [0, 10],
           "qpos_limits": None, "interpolation": None, "ramp_ratio": 0.2,
           "input_max": 1, "input_min": -1, "output_max": 1, "output_min": -1,
           "gripper": {"type": "GRIP"}}
    env_meta["env_kwargs"]["controller_configs"] = {"type": "BASIC", "body_parts": {"right": arm}}
    return EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=False)


def load(ckpt, device):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action)
    policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
    policy.diffusion.num_inference_steps = INFER_STEPS
    pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    return policy, pre, post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--kp", type=float, default=50)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--out", default="results/runs/can/joint_r34_bigunet/videos")
    a = ap.parse_args()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    states = np.load(STATES_PATH)[: a.n]
    policy, pre, post = load(a.ckpt, device)
    env = build_joint_env(a.kp)
    print(f"[44] {a.n} épisodes, device={device}, kp={a.kp}", flush=True)
    for ep in range(a.n):
        obs = env.reset_to({"states": states[ep]}); policy.reset()
        frames = []; ok = False
        for t in range(a.max_steps):
            images = {}
            for cam, key in IMAGE_KEYS.items():
                img96 = env.env.sim.render(height=96, width=96, camera_name=cam)[::-1]
                images[key] = torch.from_numpy(img96.copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            st = torch.from_numpy(state_joint(obs))
            od = pre({**images, "observation.state": st.unsqueeze(0).to(device)})
            od = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in od.items()}
            with torch.no_grad():
                act = policy.select_action(od)
            act = post(act).squeeze(0).cpu().numpy().astype(np.float32)  # 8D joint brut
            # rendu HD pour la vidéo (la policy a déjà vu du 96)
            av = env.env.sim.render(height=VID, width=VID, camera_name="agentview")[::-1]
            bv = env.env.sim.render(height=VID, width=VID, camera_name="birdview")[::-1]
            frames.append(np.concatenate([av, bv], axis=1))
            obs = env.step(act)[0]
            if env.is_success()["task"]: ok = True; break
        tag = "OK" if ok else "FAIL"
        fn = out / f"joint40k_ep{ep:02d}_{tag}.mp4"
        h, w, _ = frames[0].shape
        vw = cv2.VideoWriter(str(fn), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"[44] ep{ep}: {tag} ({len(frames)} frames) -> {fn}", flush=True)
    try: env.env.close()
    except Exception: pass
    print("[44] DONE", flush=True)


if __name__ == "__main__":
    main()
