# WRISTCAP — le poignet aide-t-il à HAUTE capacité ?

**Question** : sur Can vision-pure, ajouter la caméra poignet (eye-in-hand) à l'agentview a toujours
FAIT PERDRE (agentview seule `02` = 70,4 % ; agentview+poignet `08`/`14` = 55-68 %). MAIS tous ces runs
étaient en **petite capacité (R18)**. On a découvert *après* que Can était **bridé par la capacité**
(R34 + gros U-Net → 94,8 %, run 31). **Jamais testé : le poignet à haute capacité.**

**Hypothèse** : le poignet échouait faute de capacité pour exploiter une 2ᵉ vue. À R34 + gros U-Net,
il aide peut-être enfin. **Succès = bras B (poignet) bat bras A (agentview seule), même capacité.**

Paire matched, **seule différence = la caméra poignet** :
| Bras | Vision | Dataset (images) | Capacité |
|---|---|---|---|
| **A** baseline | agentview SEULE | `can_ph_proprio_img` | R34 + U-Net[128,256,512] |
| **B** test | agentview + poignet, encodeurs séparés | `can_ph_proprio_wrist_img` | R34 + U-Net[128,256,512] |

Résolution **96px** (= tous nos chiffres de référence ; on ne change qu'UNE variable : la capacité).
Action cartésienne 7D, état 9D, 150 ép train / 50 val, LR const 1e-4 + EMA, 30k steps.

## Pipeline (dans l'ordre)

### 0. Conversion vidéo→images (sur le Mac — torchcodec HS sur gb10 aarch64)
```bash
# agentview seule
venv312/bin/python experiments/can/95_wristcap_to_images.py \
  local/can_ph_proprio       data_cache/lerobot_can_ph_proprio \
  local/can_ph_proprio_img   data_cache/lerobot_can_ph_proprio_img
# agentview + poignet
venv312/bin/python experiments/can/95_wristcap_to_images.py \
  local/can_ph_proprio_wrist       data_cache/lerobot_can_ph_proprio_wrist \
  local/can_ph_proprio_wrist_img   data_cache/lerobot_can_ph_proprio_wrist_img
```

### 1. Transfert des 2 datasets images vers gb10
```bash
for D in can_ph_proprio_img can_ph_proprio_wrist_img; do
  rsync -az data_cache/lerobot_$D/ gb10:'~/lerobot-experiments/data_cache/lerobot_'$D'/'
done
rsync -az experiments/can/9[5-8]_wristcap_*.sh experiments/can/95_wristcap_to_images.py gb10:'~/lerobot-experiments/experiments/can/'
```

### 2. Trainings sur gb10 (dans tmux, l'un après l'autre ou 2 sessions)
```bash
ssh gb10 '~/ffmpeg-env/bin/tmux new -d -s A "bash ~/lerobot-experiments/experiments/can/96_wristcap_A_train_agentview.sh"'
ssh gb10 '~/ffmpeg-env/bin/tmux new -d -s B "bash ~/lerobot-experiments/experiments/can/97_wristcap_B_train_wrist.sh"'
```

### 3. Cooldown de chaque bras (OBLIGATOIRE avant éval)
```bash
ssh gb10 'bash ~/lerobot-experiments/experiments/can/98_wristcap_cooldown.sh wristcap_A_agentview'
ssh gb10 'bash ~/lerobot-experiments/experiments/can/98_wristcap_cooldown.sh wristcap_B_wrist'
```

### 4. Rapatriement des checkpoints cooldownés (poids inférence)
```bash
for R in wristcap_A_agentview wristcap_B_wrist; do
  mkdir -p results/runs/can/$R/cooldown/checkpoints/005000
  rsync -az gb10:'~/lerobot-experiments/results/runs/can/'$R'/cooldown/checkpoints/005000/pretrained_model/' \
        results/runs/can/$R/cooldown/checkpoints/005000/pretrained_model/
done
```
⚠️ Patcher les config.json rapatriés (quirk gb10 : `type`+`device` manquants) — cf. patch pomme.

### 5. Éval 500 rollouts sur PRINCIPAL (seule machine robosuite)
Ajouter ces 2 entrées à `MODELS` dans `10_vision_500_rollouts.py` (checkpoint = cooldown/005000) :
```python
{"name": "wristcap_A_agentview_hicap (cd)", "ckpt": "results/runs/can/wristcap_A_agentview/cooldown/checkpoints/005000/pretrained_model",
 "cams": ["agentview"], "image_keys": {"agentview": "observation.image"}},
{"name": "wristcap_B_wrist_hicap (cd)", "ckpt": "results/runs/can/wristcap_B_wrist/cooldown/checkpoints/005000/pretrained_model",
 "cams": ["agentview", "robot0_eye_in_hand"],
 "image_keys": {"agentview": "observation.images.agentview", "robot0_eye_in_hand": "observation.images.wrist"}},
```
puis `venv312/bin/python experiments/can/10_vision_500_rollouts.py` (500 rollouts, IC95 Wilson).

## Verdict
- **B > A** → le poignet aide enfin, débloqué par la capacité. 🎯
- **B ≤ A** → le poignet ne sert pas sur Can même à haute capacité → hypothèse suivante = **224px**
  (le détail fin du poignet, perdu à 96px, cf. littérature eye-in-hand). Reconvertir en 224 puis rejouer.

## Machines
- **gb10** : trainings + cooldowns (GPU Blackwell, dataset IMAGES obligatoire).
- **Principal** : éval rollouts (robosuite). ~30k steps R34+bigU-Net 2-cam 96px sur gb10 ≈ rapide.
