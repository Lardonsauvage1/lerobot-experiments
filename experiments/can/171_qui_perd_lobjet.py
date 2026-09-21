"""QUELLE CAMERA PERD VRAIMENT LA CANETTE ?

La question decide du banc d'essai : si le vrai robot travaille avec une vue que le
bras masque, s'entrainer sur une vue degagee en simulation ne transfere pas la
difficulte. Encore faut-il savoir laquelle des deux vues exterieures est occultee.

Intuition A : agentview (trois-quarts) — le bras passe devant en approchant.
Intuition B : birdview (dessus)       — la pince descend DANS L'AXE de la camera.

On mesure, pas on suppose : on deroule une politique qui marche, et a chaque pas on
compte les pixels de canette visibles dans chaque vue, a la resolution REELLE (96 px).

Sortie : results/runs/can/occlusion_cameras.json
"""
import sys; sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite; silence_robosuite()
import json, argparse
import numpy as np, torch
from pathlib import Path
from src import can_eval

CAMS = ["agentview", "birdview", "robot0_eye_in_hand"]
CKPT = "results/runs/can/unet_128256512_gele_s42/cooldown/checkpoints/005000/pretrained_model"


def pixels_canette(img):
    """La canette est le seul objet franchement rouge de la scene."""
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    return int(((r > 90) & (r - g > 40) & (r - b > 40)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--px", type=int, default=96)
    a = ap.parse_args()

    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (batch_to_transition, transition_to_batch,
                                              policy_action_to_transition, transition_to_policy_action)
    pol = DiffusionPolicy.from_pretrained(CKPT).to(dev).eval()
    pol.diffusion.num_inference_steps = 5
    pre = PolicyProcessorPipeline.from_pretrained(CKPT, config_filename="policy_preprocessor.json",
            to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(CKPT, config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    states = np.load("results/runs/can/can_eval500.npy")[: a.n]
    env = can_eval.make_env()
    compte = {c: [] for c in CAMS}
    n_reussi = 0

    for ep, st in enumerate(states):
        obs = env.reset_to(dict(states=st)); pol.reset()
        for _ in range(can_eval.MAX_STEPS):
            imgs = {c: env.env.sim.render(height=a.px, width=a.px, camera_name=c)[::-1] for c in CAMS}
            for c in CAMS:
                compte[c].append(pixels_canette(imgs[c]))
            t = {k: torch.from_numpy(imgs[c].copy()).permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255.0
                 for c, k in [("agentview", "observation.images.agentview"),
                              ("robot0_eye_in_hand", "observation.images.wrist")]}
            s = np.concatenate([np.asarray(obs["robot0_eef_pos"]).flatten(),
                                np.asarray(obs["robot0_eef_quat"]).flatten(),
                                np.asarray(obs["robot0_gripper_qpos"]).flatten()]).astype(np.float32)
            od = pre({**t, "observation.state": torch.from_numpy(s).unsqueeze(0).to(dev)})
            with torch.no_grad():
                act = pol.select_action(od)
            obs = env.step(np.clip(post(act).squeeze(0).cpu().numpy(), -1, 1).astype(np.float32))[0]
            if env.is_success()["task"]:
                n_reussi += 1; break
        print(f"  ep {ep+1}/{a.n}", flush=True)

    res = {"n_episodes": a.n, "n_reussis": n_reussi, "px": a.px, "par_camera": {}}
    print(f"\n{'caméra':22s} {'invisible':>10s} {'≤2 px':>8s} {'médiane':>8s} {'max':>6s}")
    for c in CAMS:
        v = np.array(compte[c]); n = len(v)
        d = {"pas": n, "part_invisible": float((v == 0).mean()), "part_2px": float((v <= 2).mean()),
             "mediane_px": float(np.median(v)), "max_px": int(v.max())}
        res["par_camera"][c] = d
        print(f"{c:22s} {d['part_invisible']:9.1%} {d['part_2px']:7.1%} {d['mediane_px']:8.0f} {d['max_px']:6d}")
    Path("results/runs/can").mkdir(parents=True, exist_ok=True)
    Path("results/runs/can/occlusion_cameras.json").write_text(json.dumps(res, indent=2))
    print(f"\n{n_reussi}/{a.n} épisodes réussis · -> results/runs/can/occlusion_cameras.json")


if __name__ == "__main__":
    main()
