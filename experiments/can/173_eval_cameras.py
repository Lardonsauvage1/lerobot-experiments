"""Évaluation 500 rollouts qui DÉDUIT les caméras au lieu de les supposer.

POURQUOI CE SCRIPT EXISTE. `161_eval_mono.py` code en dur `{"agentview": cams[0]}` :
sur un modèle entraîné en vue de dessus, il aurait rendu agentview — silencieusement,
sans erreur, avec un score qui ne veut rien dire. `159_eval_unet.py` code en dur le
couple agentview+poignet. Aucun des deux ne sait lire birdview.

La règle ici : la caméra n'est JAMAIS devinée.
  - multi-caméras : le nom est dans la clé (`observation.images.birdview`)
  - mono : la clé est `observation.image` et ne dit rien -> on remonte au dataset
    d'entraînement (`train_config.json`), seul endroit qui porte l'information.
Si le dataset n'est pas reconnu, on S'ARRÊTE. Pas de valeur par défaut.

Usage : venv312/bin/python experiments/can/173_eval_cameras.py --ckpt ... --tag ... --n 500
Sortie : results/runs/can/refs_birdview.json
"""
import sys; sys.path.insert(0, ".")
from src.quiet_robosuite import silence_robosuite; silence_robosuite()

import argparse, json
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location("v500", "experiments/can/10_vision_500_rollouts.py")
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)

OUT = Path("results/runs/can/refs_birdview.json")

# clé de feature multi-caméras -> nom robosuite
SUFFIXE_VERS_CAM = {"agentview": "agentview", "birdview": "birdview",
                    "wrist": "robot0_eye_in_hand", "sideview": "sideview"}
# dataset mono -> nom robosuite (la clé `observation.image` ne dit pas quelle caméra)
DATASET_VERS_CAM = {"birdviewonly": "birdview", "wristonly": "robot0_eye_in_hand"}


def cameras_du_modele(ckpt):
    cfg = json.loads((Path(ckpt) / "config.json").read_text())
    keys = sorted(k for k in cfg.get("input_features", {}) if "image" in k)
    if len(keys) > 1:
        m = {}
        for k in keys:
            suf = k.split(".")[-1]
            if suf not in SUFFIXE_VERS_CAM:
                raise SystemExit(f"ARRÊT : suffixe de caméra inconnu « {suf} » dans {k}")
            m[SUFFIXE_VERS_CAM[suf]] = k
        return m, cfg
    # mono : l'information est dans le dataset d'entraînement, pas dans la clé
    tc = json.loads((Path(ckpt) / "train_config.json").read_text())
    repo = tc["dataset"]["repo_id"]
    for motif, cam in DATASET_VERS_CAM.items():
        if repo.endswith(motif):
            return {cam: keys[0]}, cfg
    if repo.endswith("_fixstats") or repo.endswith("proprio"):
        return {"agentview": keys[0]}, cfg
    raise SystemExit(f"ARRÊT : dataset mono non reconnu « {repo} » — "
                     f"ajouter l'entrée dans DATASET_VERS_CAM plutôt que de deviner.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--pas", type=int, default=5, help="pas de débruitage (5 = optimum mesuré)")
    a = ap.parse_args()

    import torch
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import (batch_to_transition, transition_to_batch,
                                              policy_action_to_transition, transition_to_policy_action)
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # Les checkpoints viennent d'atomman et portent `device: "xpu"` (Arc). Sans cette
    # réécriture, le chargement du pipeline lève AssertionError sur torch.xpu.is_available().
    # ⚠️ Le md5 du checkpoint change donc ICI : il doit être vérifié À LA RÉCEPTION, jamais après.
    for f in ("policy_preprocessor.json", "policy_postprocessor.json"):
        q = Path(a.ckpt) / f
        s = q.read_text()
        if '"xpu"' in s or '"cuda"' in s:
            q.write_text(s.replace('"xpu"', f'"{dev.type}"').replace('"cuda"', f'"{dev.type}"'))
            print(f"  {f} : device -> {dev.type}", flush=True)
    image_keys, cfg = cameras_du_modele(a.ckpt)
    print(f"  {a.tag} : caméras déduites = {image_keys}", flush=True)

    pol = DiffusionPolicy.from_pretrained(a.ckpt).to(dev).eval()
    pol.diffusion.num_inference_steps = a.pas
    pre = PolicyProcessorPipeline.from_pretrained(a.ckpt, config_filename="policy_preprocessor.json",
            to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(a.ckpt, config_filename="policy_postprocessor.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action)

    per, el = V.rollout_can(pol, pre, post, V.make_eval_states()[:a.n], dev, image_keys)
    k, n = sum(e["success"] for e in per), len(per)
    lo, hi = V.wilson_ci(k, n)
    print(f"\n  === {a.tag} — {sorted(image_keys)} · U-Net {cfg.get('down_dims')} · {a.pas} pas ===")
    print(f"  {n - k} échecs / {n} = {k/n:.1%} [{lo:.1%}, {hi:.1%}]   ({el/60:.0f} min)")

    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res[a.tag] = {"n": n, "n_success": k, "success_rate": k/n, "ci95": [lo, hi],
                  "cameras": sorted(image_keys), "down_dims": cfg.get("down_dims"),
                  "pas_debruitage": a.pas, "ckpt": a.ckpt,
                  "per_episode": [{"ep": i, "success": bool(e["success"])} for i, e in enumerate(per)]}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
