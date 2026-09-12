"""AUDIT cohérence train/déploiement du joint : teacher-forcing.
On nourrit le modèle avec de VRAIES frames du dataset (images + state enregistrés) et on compare
son action PRÉDITE à l'action ENREGISTRÉE. Si proche -> chaîne cohérente (échec = boucle fermée).
Si aberrant (mauvaise échelle/ordre) -> bug de pipeline.
"""
import sys, glob
sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite; silence_robosuite()
import numpy as np, pandas as pd, cv2, torch

CK = "results/runs/can/joint_r34_bigunet/checkpoints/040000/pretrained_model"
DS = "data_cache/lerobot_can_ph_joint_birdview"
N = 40  # frames de l'épisode 0


def load(ckpt, device):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (batch_to_transition, transition_to_batch,
        policy_action_to_transition, transition_to_policy_action)
    policy = DiffusionPolicy.from_pretrained(ckpt).to(device).eval()
    policy.diffusion.num_inference_steps = 10
    pre = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(ckpt, config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    return policy, pre, post


def main():
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    df = pd.read_parquet(sorted(glob.glob(f"{DS}/data/**/*.parquet", recursive=True))[0])
    S = np.stack(df["observation.state"].values); A = np.stack(df["action"].values)
    caps = {c: cv2.VideoCapture(f"{DS}/videos/observation.images.{c}/chunk-000/file-000.mp4")
            for c in ["agentview", "birdview"]}
    policy, pre, post = load(CK, dev)
    print(f"[46] device={dev} | teacher-forcing {N} frames de l'épisode 0", flush=True)
    errs = []
    for t in range(N):
        images = {}
        for c, key in [("agentview", "observation.images.agentview"), ("birdview", "observation.images.birdview")]:
            caps[c].set(cv2.CAP_PROP_POS_FRAMES, t); ok, f = caps[c].read()
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            images[key] = torch.from_numpy(f.copy()).permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255.0
        st = torch.from_numpy(S[t].astype(np.float32))
        od = pre({**images, "observation.state": st.unsqueeze(0).to(dev)})
        od = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in od.items()}
        with torch.no_grad():
            a = policy.select_action(od)
        pred = post(a).squeeze(0).cpu().numpy().astype(np.float32)
        rec = A[t]
        errs.append(np.abs(pred[:7] - rec[:7]))
        if t < 5 or t == N - 1:
            print(f" t={t:2d} rec_joints={np.round(rec[:7],3)}", flush=True)
            print(f"      pred_joints={np.round(pred[:7],3)}  |err|={np.round(np.abs(pred[:7]-rec[:7]),3)}", flush=True)
    errs = np.array(errs)
    print(f"\n[46] MAE par joint (rad) sur {N} frames : {np.round(errs.mean(0),4)}", flush=True)
    print(f"[46] MAE globale joints = {errs.mean():.4f} rad  (~{np.degrees(errs.mean()):.2f}°)", flush=True)
    print(f"[46] amplitude des cibles joints (rec) = {np.round(A[:,:7].min(0),2)} .. {np.round(A[:,:7].max(0),2)}", flush=True)
    print("[46] DONE — si MAE << amplitude -> chaîne cohérente (échec = boucle fermée, pas un bug)", flush=True)


if __name__ == "__main__":
    main()
