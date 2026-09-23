"""birdview est-elle VRAIMENT moins occultée dans les démos qu'au déploiement ?

LE DOUTE. La comparaison « 97,7 % dans les démos contre 42,3 % en rollout » mélangeait
trois choses : une politique qui n'avait jamais vu birdview, des longueurs d'épisode
incomparables (démos = que des réussites ~116 pas ; rollouts d'un modèle faible = des
échecs de 300 pas), et un possible renversement de causalite — un bras qui ECHOUE
stationne au-dessus de la canette, donc la masque. L'occlusion serait alors le SYMPTOME
de l'échec, pas sa cause.

CE QUE FAIT CE SCRIPT. Il compare à protocole égal :
  - la politique évaluée est celle du banc (entraînée SUR birdview) ;
  - on sépare les épisodes REUSSIS des RATES ;
  - on compare les réussites aux démos, qui sont toutes des réussites ;
  - on regarde aussi la phase : les 60 premiers pas (approche) contre le reste.

Si les rollouts REUSSIS sont aussi dégagés que les démos, alors il n'y a pas de décalage
train/déploiement : il y a seulement des échecs qui masquent la caméra en stationnant.

Sortie : results/runs/can/occlusion_honnete.json
"""
import sys; sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite; silence_robosuite()

import argparse, json
from pathlib import Path
import numpy as np, torch
from src import can_eval

def pix(img):
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    return int(((r > 90) & (r - g > 40) & (r - b > 40)).sum())

def part_invisible(serie):
    v = np.array(serie)
    return float((v == 0).mean()) if len(v) else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/runs/can/ref_birdview_seul/cooldown/checkpoints/005000/pretrained_model")
    ap.add_argument("--tag", default="ref_birdview_seul")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--approche", type=int, default=60, help="nb de pas comptés comme phase d'approche")
    a = ap.parse_args()

    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    for f in ("policy_preprocessor.json", "policy_postprocessor.json"):
        q = Path(a.ckpt) / f; s = q.read_text()
        if '"xpu"' in s or '"cuda"' in s:
            q.write_text(s.replace('"xpu"', f'"{dev.type}"').replace('"cuda"', f'"{dev.type}"'))

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (batch_to_transition, transition_to_batch,
                                              policy_action_to_transition, transition_to_policy_action)
    cfg = json.loads((Path(a.ckpt) / "config.json").read_text())
    keys = sorted(k for k in cfg["input_features"] if "image" in k)
    assert len(keys) == 1, f"ce script attend un mono, trouvé {keys}"
    pol = DiffusionPolicy.from_pretrained(a.ckpt).to(dev).eval(); pol.diffusion.num_inference_steps = 5
    pre = PolicyProcessorPipeline.from_pretrained(a.ckpt, config_filename="policy_preprocessor.json",
            to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(a.ckpt, config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    states = np.load("results/runs/can/can_eval500.npy")[: a.n]
    env = can_eval.make_env()
    eps = []
    for i, st in enumerate(states):
        obs = env.reset_to(dict(states=st)); pol.reset()
        serie, ok = [], False
        for _ in range(can_eval.MAX_STEPS):
            img = env.env.sim.render(height=96, width=96, camera_name="birdview")[::-1]
            serie.append(pix(img))
            t = torch.from_numpy(img.copy()).permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255.0
            s = np.concatenate([np.asarray(obs["robot0_eef_pos"]).flatten(),
                                np.asarray(obs["robot0_eef_quat"]).flatten(),
                                np.asarray(obs["robot0_gripper_qpos"]).flatten()]).astype(np.float32)
            od = pre({keys[0]: t, "observation.state": torch.from_numpy(s).unsqueeze(0).to(dev)})
            with torch.no_grad(): act = pol.select_action(od)
            obs = env.step(np.clip(post(act).squeeze(0).cpu().numpy(), -1, 1).astype(np.float32))[0]
            if env.is_success()["task"]: ok = True; break
        eps.append({"succes": ok, "n_pas": len(serie), "serie": serie})
        print(f"  ép {i+1}/{a.n} : {'réussi' if ok else 'raté':6s} {len(serie):3d} pas · "
              f"invisible {part_invisible(serie):.0%}", flush=True)

    R, E = [e for e in eps if e["succes"]], [e for e in eps if not e["succes"]]
    def agg(g, tr=None):
        s = [x for e in g for x in (e["serie"][:tr] if tr else e["serie"])]
        return part_invisible(s), len(s)
    res = {"tag": a.tag, "n_episodes": len(eps), "n_reussis": len(R),
           "tous": agg(eps), "reussis": agg(R), "rates": agg(E),
           "reussis_approche": agg(R, a.approche), "rates_approche": agg(E, a.approche),
           "episodes": [{"succes": e["succes"], "n_pas": e["n_pas"]} for e in eps]}
    print(f"\n  {'':26s} {'invisible':>10s} {'pas':>8s}")
    for lbl, k in [("TOUS les épisodes", "tous"), ("RÉUSSIS seulement", "reussis"),
                   ("RATÉS seulement", "rates"),
                   (f"réussis, {a.approche} 1ers pas", "reussis_approche"),
                   (f"ratés, {a.approche} 1ers pas", "rates_approche")]:
        p, n = res[k]
        print(f"  {lbl:26s} {p:9.1%} {n:8d}")
    print(f"\n  rappel démos (toutes réussies) : 2,3 % invisible")
    Path("results/runs/can/occlusion_honnete.json").write_text(json.dumps(res, indent=2))
    print(f"  -> results/runs/can/occlusion_honnete.json")

if __name__ == "__main__":
    main()
