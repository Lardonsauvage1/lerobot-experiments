#!/usr/bin/env python
"""Fabrique un checkpoint BI-CAMÉRA strictement équivalent au modèle MONO-CAMÉRA A2.

Pourquoi. Tous les mécanismes testés cette semaine échouent de la même façon : la descente
de gradient s'approprie le poignet, qui est l'entrée la plus prédictive de l'action immédiate,
et n'apprend jamais à se servir de la vue globale. Le diagnostic du routeur l'a chiffré —
il pondère le poignet à 74 % alors que la vue globale seule vaut 78,5 % et le poignet seul 0 %.

Le remède : ne pas laisser le poignet PARTICIPER à la construction de la représentation.
On part d'un modèle mono-caméra qui marche (A2, 78,5 %), on l'insère dans une architecture
bi-caméra où les colonnes du poignet sont à ZÉRO — donc sans aucun effet — et l'on gèle
l'encodeur de la vue globale. Le poignet ne peut plus que COMPLÉTER, jamais remplacer.

C'est le protocole à encodeur gelé, en deux temps (cf. BLIP-2 et travaux récents en
politiques robotiques).

Disposition du conditionnement (vérifiée dans modeling_diffusion) :
    [ embed_temps(128) | pas0: état(9) cam0(64) cam1(64) | pas1: état(9) cam0(64) cam1(64) ]
Le mono-caméra n'a pas les blocs cam1 : on réinsère ses colonnes aux bons offsets.

Usage : venv312/bin/python experiments/can/141_init_from_mono.py
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")
import torch
from safetensors.torch import load_file, save_file

MONO = Path("results/runs/can/cam2_A2_fixstats/cooldown/checkpoints/005000/pretrained_model")
BI   = Path("results/runs/can/cam2_C2_wrist_fixstats/cooldown/checkpoints/005000/pretrained_model")
OUT  = Path("results/runs/can/cam2_I_frozen_init/pretrained_model")

EMB, SD, FD, NOBS = 128, 9, 64, 2      # embed temps, état, features/caméra, pas d'obs


def main():
    for p in (MONO, BI):
        if not (p / "model.safetensors").exists():
            print(f"manque : {p}"); return 1
    mono = load_file(MONO / "model.safetensors")
    bi   = load_file(BI / "model.safetensors")       # sert de MOULE (bonnes formes)
    out = {}
    n_copied = n_mapped = n_fresh = 0

    for k, v in bi.items():
        if k in mono and mono[k].shape == v.shape:
            out[k] = mono[k].clone(); n_copied += 1
        elif k.startswith("diffusion.rgb_encoder.1."):
            # encodeur du POIGNET : on garde l'init du moule (il sera entraîné)
            out[k] = v.clone(); n_fresh += 1
        elif k.startswith("diffusion.rgb_encoder.0."):
            mk = k.replace("diffusion.rgb_encoder.0.", "diffusion.rgb_encoder.")
            out[k] = mono[mk].clone() if mk in mono and mono[mk].shape == v.shape else v.clone()
            n_copied += 1
        elif v.ndim == 2 and k in mono and mono[k].shape[1] != v.shape[1]:
            # couche FiLM : on replace les colonnes du mono aux bons offsets, reste à ZÉRO
            w = torch.zeros_like(v)
            src = mono[k]
            w[:, :EMB] = src[:, :EMB]                                     # embed temps
            for s in range(NOBS):
                d0 = EMB + s * (SD + 2 * FD)          # offset bi-caméra
                s0 = EMB + s * (SD + FD)              # offset mono-caméra
                w[:, d0:d0 + SD + FD] = src[:, s0:s0 + SD + FD]           # état + cam0
                # w[:, d0+SD+FD : d0+SD+2*FD] reste à ZÉRO -> le poignet n'a AUCUN effet
            out[k] = w; n_mapped += 1
        else:
            out[k] = v.clone(); n_fresh += 1

    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "train_config.json", "policy_preprocessor.json",
              "policy_postprocessor.json",
              "policy_preprocessor_step_3_normalizer_processor.safetensors",
              "policy_postprocessor_step_0_unnormalizer_processor.safetensors"):
        if (BI / f).exists():
            shutil.copy(BI / f, OUT / f)
    save_file(out, OUT / "model.safetensors")
    print(f"  {n_copied} tenseurs copiés · {n_mapped} couches FiLM remappées · {n_fresh} neufs")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
