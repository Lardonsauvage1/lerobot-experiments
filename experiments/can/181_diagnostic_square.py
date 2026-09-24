"""OÙ échouent les 232 épisodes de square ? Saisie, ou insertion ?

POURQUOI. Monter la résolution multiplie la grille du softmax spatial (3×3 à 96 px, 7×7 à
224 px) et coûte 4,2× plus cher à entraîner. Ça ne vaut la peine QUE si le goulot est la
PRÉCISION de positionnement. Ce diagnostic le dit sans rien réentraîner.

CE QU'IL MESURE, par épisode :
  - l'écrou a-t-il été SOULEVÉ (hauteur > seuil) : la saisie a-t-elle réussi ?
  - distance minimale écrou -> pion pendant l'épisode : à quel point on s'est approché du but
  - hauteur de l'écrou au moment le plus proche : au-dessus du pion, ou à côté ?

TROIS SIGNATURES, TROIS REMÈDES :
  jamais soulevé            -> la SAISIE est le goulot (résolution plausible, mais pas seule)
  soulevé, loin du pion     -> le TRANSPORT / la localisation, pas la précision fine
  soulevé, proche mais rate -> l'INSERTION : la précision est le goulot -> RÉSOLUTION

Usage : TACHE=square venv312/bin/python experiments/can/181_diagnostic_square.py --ckpt ... --n 100
Sortie : results/runs/can/diagnostic_square.json
"""
import sys; sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite; silence_robosuite()

import argparse, json
from pathlib import Path
import numpy as np, torch
from src import can_eval

OUT = Path("results/runs/can/diagnostic_square.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--pas", type=int, default=5)
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
    SUF = {"birdview": "birdview", "wrist": "robot0_eye_in_hand", "agentview": "agentview"}
    if len(keys) > 1:
        image_keys = {SUF[k.split(".")[-1]]: k for k in keys}
    else:
        repo = json.loads((Path(a.ckpt) / "train_config.json").read_text())["dataset"]["repo_id"]
        cam = "robot0_eye_in_hand" if repo.endswith("wristonly") else "birdview"
        image_keys = {cam: keys[0]}
    print(f"  {a.tag} : caméras = {image_keys}", flush=True)

    pol = DiffusionPolicy.from_pretrained(a.ckpt).to(dev).eval()
    pol.diffusion.num_inference_steps = a.pas
    pre = PolicyProcessorPipeline.from_pretrained(a.ckpt, config_filename="policy_preprocessor.json",
            to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(a.ckpt, config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    states = np.load(can_eval.ETATS_PATH)[: a.n]
    env = can_eval.make_env()
    sim = env.env.sim
    adr = sim.model.get_joint_qpos_addr("SquareNut_joint0")
    d0 = adr[0] if isinstance(adr, tuple) else adr
    # pion cible de NutAssemblySquare = peg1
    pid = sim.model.body_name2id("peg1")
    peg = np.array(sim.model.body_pos[pid], copy=True)
    print(f"  pion peg1 en {np.round(peg,4)}", flush=True)

    eps = []
    for i, st in enumerate(states):
        obs = env.reset_to(dict(states=st)); pol.reset()
        sim = env.env.sim
        z0 = float(sim.data.qpos[d0 + 2])
        hmax, dmin, z_at_dmin, ok = z0, 1e9, z0, False
        for _ in range(can_eval.MAX_STEPS):
            imgs = {k: torch.from_numpy(sim.render(height=96, width=96, camera_name=c)[::-1].copy())
                    .permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255.0
                    for c, k in image_keys.items()}
            s = np.concatenate([np.asarray(obs["robot0_eef_pos"]).flatten(),
                                np.asarray(obs["robot0_eef_quat"]).flatten(),
                                np.asarray(obs["robot0_gripper_qpos"]).flatten()]).astype(np.float32)
            od = pre({**imgs, "observation.state": torch.from_numpy(s).unsqueeze(0).to(dev)})
            with torch.no_grad(): act = pol.select_action(od)
            obs = env.step(np.clip(post(act).squeeze(0).cpu().numpy(), -1, 1).astype(np.float32))[0]
            p = np.array(sim.data.qpos[d0:d0 + 3])
            hmax = max(hmax, float(p[2]))
            d = float(np.linalg.norm(p[:2] - peg[:2]))       # distance HORIZONTALE au pion
            if d < dmin: dmin, z_at_dmin = d, float(p[2])
            if env.is_success()["task"]: ok = True; break
        eps.append({"succes": ok, "souleve": hmax - z0 > 0.03,
                    "hauteur_max": hmax - z0, "dist_min_pion": dmin, "z_au_plus_pres": z_at_dmin - z0})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{a.n}", flush=True)

    R = [e for e in eps if e["succes"]]; E = [e for e in eps if not e["succes"]]
    sl = [e for e in E if e["souleve"]]; ns = [e for e in E if not e["souleve"]]
    print(f"\n  === {a.tag} · {len(eps)} épisodes ===")
    print(f"  réussis                         {len(R):3d}  ({len(R)/len(eps):.0%})")
    print(f"  ÉCHECS                          {len(E):3d}")
    print(f"    dont écrou JAMAIS soulevé     {len(ns):3d}  ({len(ns)/max(1,len(E)):.0%} des échecs)  -> SAISIE")
    print(f"    dont soulevé mais raté        {len(sl):3d}  ({len(sl)/max(1,len(E)):.0%} des échecs)")
    if sl:
        d = np.array([e["dist_min_pion"] for e in sl])
        print(f"      distance min. au pion : médiane {100*np.median(d):.1f} cm · "
              f"p25 {100*np.percentile(d,25):.1f} · p75 {100*np.percentile(d,75):.1f}")
        proche = (d < 0.02).mean()
        print(f"      arrivés à moins de 2 cm du pion : {100*proche:.0f} % -> INSERTION ratée de peu")
    if R:
        d = np.array([e["dist_min_pion"] for e in R])
        print(f"  (réussis, distance min. au pion : médiane {100*np.median(d):.1f} cm)")
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res[a.tag] = {"n": len(eps), "n_succes": len(R), "n_non_souleve": len(ns),
                  "n_souleve_rate": len(sl), "episodes": eps, "ckpt": a.ckpt}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
