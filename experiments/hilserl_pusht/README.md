# Phase 7 — HIL-SERL en simulation sur Push-T (image), intervention clavier cartésienne

But : reproduire HIL-SERL (RL human-in-the-loop) sur le benchmark Push-T en SIM, avec intervention
CLAVIER cartésienne (poussoir x/y), pour valider la méthode + monter le muscle intervention/RL
avant d'envisager le réel. Push-T = reward intégré (coverage) → PAS de classifieur de récompense.

Décisions : IMAGE (64px, proche du réel) · sur le Mac (Push-T léger + clavier interactif) ·
RL démarre depuis les démos Push-T (RLPD 50/50) · poussoir en déplacement relatif au clavier.

Harnais : lerobot/rl/{actor,learner,buffer,gym_manipulator}.py + policies/sac + processor/hil_processor.py
À construire : wrapper gym_pusht au format HIL-SERL + handler clavier cartésien + config SAC image.

## État (build) — ce qui marche / reste
FAIT + validé headless :
- `pusht_hil_env.py` : wrapper gym_pusht (image 96) + clavier cartésien (flèches->delta x/y), pousse teleop_action/is_intervention dans info. OBS/reward OK.
- `gen_config.py` -> `pusht_hilserl.json` : TrainRLServerPipelineConfig (env name=gym_hil/PushT-v0, policy sac image, dataset_stats action+state en [0,512]). Charge + policy SAC build (1.12M).
- `run_actor_pusht.py` : launcher actor qui monkey-patch make_robot_env -> notre env.
- deps : `grpcio protobuf` installés.
- `make_processors(env,None,cfg.env)` construit le pipeline OK.

RESTE :
- Le LEARNER exige un DATASET de démos (RLPD) : fournir un LeRobotDataset Push-T
  features observation.image(3,96,96)+observation.state(2)+action(2)+next.reward+next.done,
  MÊME taille image (96) que l'env. -> à créer (re-render lerobot/pusht en 96, ou enregistrer au clavier).
- Run interactif (clavier + Accessibilité macOS pour pynput) : learner puis actor.

## Runbook (une fois le dataset prêt)
1. `python -m lerobot.rl.learner --config_path=experiments/hilserl_pusht/pusht_hilserl.json`  (démarre en 1er)
2. `python experiments/hilserl_pusht/run_actor_pusht.py --config_path=experiments/hilserl_pusht/pusht_hilserl.json`
3. Clavier pendant le rollout : FLÈCHES = bouger le poussoir (intervention) · `s` = succès · `q` = fin épisode.
   (macOS : autoriser le Terminal en Accessibilité + Surveillance des entrées.)
