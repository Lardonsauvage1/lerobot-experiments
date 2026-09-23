"""Harnais d'éval Can (PickPlaceCan) — réutilise lift_eval (générique) + spécificités Can.

⚠️ Pourquoi un state Can dédié : robosuite 1.5 (env live) RÉORDONNE l'obs `object` vs le
dataset 1.4, et sa partie relative (obj↔pince) est instable. On garde donc un state
MINIMAL mais FIABLE (vérifié identique dataset↔live sur tous les frames) :

    state Can (12D) = eef_pos(3) + eef_quat(4) + gripper_qpos(2) + can_pos(3)

- proprio (eef_pos/quat, gripper) : identique dataset↔live.
- can_pos : position absolue de la canette, déterministe. Index DIFFÉRENT selon la source :
    * dataset (conversion) : object[0:3]   (voir can_to_lerobot.py)
    * env live (rollout)   : object[7:10]  (ci-dessous)
- la pose relative fragile est abandonnée (l'image agentview porte le reste).
→ aucun fix_obs_sign nécessaire (contrairement à Lift).
"""

import os
import numpy as np

from src import lift_eval

# ⚙️ TACHE : même code d'évaluation pour plusieurs tâches robomimic (cf. src/can_to_lerobot.py).
# MAX_STEPS garde la MÊME générosité relative d'une tâche à l'autre — 2,6 × la longueur médiane
# des démos — sinon les taux de réussite ne seraient pas comparables :
#   can    démos médiane 115 (max 151) -> 300 pas
#   square démos médiane 150 (max 236) -> 400 pas
TACHES_EVAL = {
    "can":    dict(hdf5="data_cache/robomimic_can_ph/low_dim_v141.hdf5",
                   split="results/runs/can/can_split.json", max_steps=300,
                   etats="results/runs/can/can_eval500.npy"),
    "square": dict(hdf5="data_cache/robomimic_square_ph/low_dim_v141.hdf5",
                   split="results/runs/can/square_split.json", max_steps=400,
                   etats="results/runs/can/square_eval500.npy"),
}
_TE = TACHES_EVAL[os.environ.get("TACHE", "can")]
HDF5_PATH = _TE["hdf5"]
SPLIT_PATH = _TE["split"]
ETATS_PATH = _TE["etats"]
MAX_STEPS = _TE["max_steps"]
N_TOTAL_DEMOS = 200
N_VAL = 50
STATE_DIM = 12


def build_state_vector_can(obs):
    """12D depuis l'obs de l'ENV LIVE : proprio + can_pos (object[7:10])."""
    eef_pos = np.asarray(obs["robot0_eef_pos"]).flatten()
    eef_quat = np.asarray(obs["robot0_eef_quat"]).flatten()
    grip = np.asarray(obs["robot0_gripper_qpos"]).flatten()
    can_pos = np.asarray(obs["object"]).flatten()[7:10]
    return np.concatenate([eef_pos, eef_quat, grip, can_pos]).astype(np.float32)


def build_state_vector_can_dataset(obs_object, eef_pos, eef_quat, grip):
    """12D depuis le DATASET (conversion) : can_pos = object[0:3]."""
    can_pos = np.asarray(obs_object).flatten()[0:3]
    return np.concatenate([np.asarray(eef_pos).flatten(), np.asarray(eef_quat).flatten(),
                           np.asarray(grip).flatten(), can_pos]).astype(np.float32)


def make_env():
    """Env d'évaluation Can.

    ⚠️ PAR DÉFAUT, les MARQUEURS DE VISUALISATION sont DÉSACTIVÉS (CAN_VIZ=1 les rétablit) (sites de préhension, repères
    robot). Le wrapper robomimic les active par défaut, alors que la conversion du dataset
    appelle robosuite.make() directement et ne les dessine PAS. Les images d'évaluation
    portent donc un artefact absent de l'entraînement : 68 pixels verts sur 9216, soit une
    diagonale PLUS GROSSE que la canette vue de dessus (9 à 56 px). Mesuré le 2026-09-22.
    Coût mesuré sur comparaison appariée (mêmes poids, mêmes 500 états) : le mono POIGNET
    passe de 274 à 27 échecs sans les repères (p = 4e-70), birdview de 360 à 298 (p = 1,5e-6).
    Le repère est un objet rouge à pointe verte planté AU POINT DE SAISIE, donc au centre du
    champ de la caméra embarquée, en permanence. L'encodeur sortant 32 KEYPOINTS (positions),
    un objet neuf et contrasté ne bruite pas un peu le vecteur : il capture des keypoints.
    → Le rendu SANS repères est le seul correct : les données d'entraînement n'en ont pas.
    """
    env = lift_eval.make_env(HDF5_PATH)
    if os.environ.get("CAN_VIZ") != "1":       # SANS repères par DÉFAUT
        e = getattr(env, "env", env)
        vis = getattr(e, "_visualizations", None)
        assert vis, "pas de _visualizations : impossible de désactiver les marqueurs"
        e.visualize(vis_settings={k: False for k in vis})
    return env


def load_or_make_split():
    return lift_eval.load_or_make_split(path=SPLIT_PATH, n_total=N_TOTAL_DEMOS, n_val=N_VAL)


def load_init_states(episode_indices):
    return lift_eval.load_init_states(episode_indices, hdf5_path=HDF5_PATH)


def rollout_eval_can(policy, pre, post, states, *, device, num_inference_steps=10,
                     stop_on_success=False, chunk=50):
    """Rollout Can : identité pour fix_obs (pas de mismatch de signe), state_fn dédié."""
    return lift_eval.rollout_eval_chunked(
        policy, pre, post, states, device=device, make_env_fn=make_env, chunk=chunk,
        num_inference_steps=num_inference_steps, stop_on_success=stop_on_success,
        max_steps=MAX_STEPS, fix_obs_fn=lambda o: o, state_fn=build_state_vector_can)
