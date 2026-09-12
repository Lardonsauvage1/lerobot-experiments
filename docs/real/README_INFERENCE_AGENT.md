# 🍎 Note de passation — Inférer les modèles « prise de pomme » (sur ce PC Linux / atomman)

> **Pour un agent Claude qui lit ce fichier en local sur atomman.** Tout ce qu'il faut pour
> retrouver les 3 modèles déployables et les faire tourner est ici. Tout est déjà installé.
> Rédigé le 2026-07-13. Machine : `sam-AtomMan`, Ubuntu 24.04, **CPU only (Intel Core Ultra 9 185H,
> PAS de GPU/CUDA)**.

---

## 1. Où sont les modèles

Répertoire : `~/lerobot-experiments/deployable_models/apple/`

| Dossier | Ce que c'est | Contient |
|---|---|---|
| `B_cd5k_from12000/` | cooldown 5k depuis le checkpoint tardif 12000 (river-valley) | `brut/`, `ema/` |
| `A_cd5k_from2000/` | cooldown 5k depuis le checkpoint 2000 (1er déployable) | `brut/`, `ema/` |
| `C_cd1k_from2000_short/` | cooldown court 1k depuis 2000 | `brut/`, `ema/` |

Chaque `brut/` ou `ema/` est un checkpoint LeRobot complet (`model.safetensors`, `config.json`,
`train_config.json`, fichiers `policy_preprocessor_*` / `policy_postprocessor_*`). **Autonome** :
la normalisation est sauvée dedans, pas besoin du dataset pour inférer.

### ⭐ Ordre recommandé (val-loss moyennée mesurée, val ép. 40-45, 5 seeds)
1. **`B_cd5k_from12000/brut`** = **0,01069** ← 🏆 MEILLEUR / déployable actuel (checkpoint tardif river-valley + cooldown)
2. `A_cd5k_from2000/brut` = 0,01684 (1er déployable, battu de −36 % par B)
3. `C_cd1k_from2000_short/brut` = ~0,027 (le moins bon)

Commencer par **B brut**. (Les variantes EMA ont été retirées : sur ces runs courts, le brut les bat
toujours — l'EMA n'est utile que sur des entraînements longs >10k pas.)
⚠️ Ces chiffres sont de la **val-loss** (débruitage), PAS du succès → **tester B et A sur le bras**
pour trancher (cf. encadré ci-dessous).

> ⚠️ **La val-loss ne prédit PAS le succès réel.** Sur une tâche sœur (mini-CNN Can) on a mesuré une
> corrélation loss↔succès de **+0,27** (quasi nulle, voire inversée). **Le seul juge = le robot.**
> Donc : tester les 3 sur le bras, ne pas trancher sur la val-loss seule.

---

## 2. Environnement (déjà installé)

- venv : `~/lerobot-experiments/venv/bin/python` (Python 3.12)
- `lerobot` **0.5.1**, `torch` **2.10.0+cpu**, `cv2`, `numpy` présents.
- ⚠️ Les modèles ont été **entraînés sur Mac (MPS) avec lerobot 0.5.0** puis chargés ici en 0.5.1 —
  compatible. Un futur upgrade de lerobot pourrait casser le chargement (API processors) : si ça
  plante, épingler lerobot 0.5.x.

---

## 3. Comment charger + inférer (script vérifié)

Script prêt : **`~/lerobot-experiments/experiments/real/14_infer_test.py`** (fabrique une observation
factice et sort une action 6D — sert de test ET de modèle de code).

```bash
cd ~/lerobot-experiments
venv/bin/python experiments/real/14_infer_test.py deployable_models/apple/B_cd5k_from12000/brut
# -> [infer] ACTION produite (dim 6) = [ ... 5 joints ... , gripper ]   +   "OK — le modele se charge et infere."
```

**Les 3 pièges à connaître (sinon ça plante) :**
1. Charger avec **`DiffusionPolicy.from_pretrained(chemin)`** — PAS `make_policy` (qui exige les
   métadonnées d'un dataset qu'on n'a pas ici).
   ```python
   from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
   policy = DiffusionPolicy.from_pretrained("deployable_models/apple/B_cd5k_from12000/brut")
   policy.eval().to("cpu"); policy.reset()
   ```
2. Le preprocessor sauvé contient `device: mps` → **forcer le device local** sinon AssertionError :
   ```python
   from lerobot.policies.factory import make_pre_post_processors
   pre, post = make_pre_post_processors(policy_cfg=policy.config,
       pretrained_path="deployable_models/apple/B_cd5k_from12000/brut",
       preprocessor_overrides={"device_processor": {"device": "cpu"}})
   ```
3. Appeler **`policy.reset()`** au début de chaque épisode (vide la file du chunking).
   Inférence d'un pas : `action = post(policy.select_action(pre(obs)))`.

---

## 4. Entrées / Sorties du modèle

**Entrées** (dict, batch dim = 1) :
| clé | forme | contenu |
|---|---|---|
| `observation.images.left` | `[1,3,224,224]` | caméra gauche, RGB float [0,1], CHW |
| `observation.images.right` | `[1,3,224,224]` | caméra droite, RGB float [0,1], CHW |
| `observation.state` | `[1,5]` | 5 angles articulaires (rad), ordre joint_1..joint_5 |

Prétraitement image au déploiement : JPEG des caméras → `cv2.imdecode` → **resize 640×480 → 224×224**
(cv2.INTER_AREA) → RGB → CHW → /255 (float). La normalisation (mean/std) est faite par le
`preprocessor`, ne PAS la refaire à la main.

### Sortie EXACTE du réseau

- **Une passe du réseau prédit un chunk = tenseur `[16, 6]`** (`horizon=16` pas × 6 dims).
  Chaque pas = `[joint_1, joint_2, joint_3, joint_4, joint_5 (positions absolues, rad), gripper]`.
- On **exécute seulement les `n_action_steps=8` premiers pas** (`[8, 6]`), on jette les 8 suivants,
  puis on ré-observe et on re-prédit (receding horizon → boucle fermée).
- **`policy.select_action(obs)` masque tout ça** : il gère une file interne et renvoie **1 seule
  action `[1, 6]` par appel** (il ne re-lance le réseau qu'une fois toutes les 8 actions demandées).
  → dans la boucle de contrôle, appeler `select_action` **à chaque pas** (15 fois/s) ; il ne fait le
  gros calcul diffusion qu'1 fois sur 8.
- **gripper** = 6ᵉ valeur (float à l'entraînement) → au déploiement **seuiller ~0,5** : `>0,5 ⇒ FERMER`
  (saisir), sinon OUVRIR ; publier un `std_msgs/Bool`.
- Les 5 joints sont des **positions absolues cibles** (pas des deltas) → publier directement comme
  consigne de `JointTrajectory`.

### Fréquences (toutes)

| grandeur | valeur | note |
|---|---|---|
| **Cadence de contrôle** (publication consignes) | **15 Hz** (1 pas = 0,0667 s) | = fps des données ; à respecter |
| Pas prédits par inférence (`horizon`) | 16 (≈ 1,07 s de futur) | |
| Pas exécutés avant re-plan (`n_action_steps`) | 8 (≈ 0,53 s) | |
| **Fréquence d'inférence réseau réelle** | **~1,9 Hz** (1 inférence / 8 pas) | le lourd (diffusion) tourne ~2×/s |
| Obs passées utilisées (`n_obs_steps`) | 2 (espacées de 1/15 s) | fournir un flux régulier à 15 Hz |
| Débruitage diffusion (`num_inference_steps`) | 100 par inférence | cf. §5 latence (réductible) |

⚠️ La cadence 15 Hz doit être **régulière** : le réseau suppose des obs espacées de 1/15 s
(`delta_timestamps`). Un flux irrégulier (jitter/latence) dégrade la prédiction. Aligner par
timestamp comme à l'enregistrement.

> 🚫 **MALENTENDU À ÉVITER** : NON, le robot n'est PAS commandé à 2 Hz.
> - `select_action` renvoie **1 action par appel** ET on l'appelle **15 fois/s** → le bras reçoit
>   des consignes à **15 Hz** (une par pas de 0,0667 s).
> - Le **réseau diffusion** (le gros calcul, ~540 ms) ne tourne que **~1,9 fois/s** (1 fois toutes les
>   8 actions), et chaque fois il produit **16 actions d'un coup** (pas 1).
> - Donc : « 1 valeur par appel de select_action à 15 Hz » ✅, mais « 16 valeurs par inférence réseau,
>   ~2 fois/s » ✅. Le « 2 Hz » = fréquence du réseau, PAS la fréquence de commande du robot.
> - Deux implémentations valides : (a) appeler `select_action` à 15 Hz (il gère la file tout seul) ;
>   (b) appeler le réseau, récupérer les 16, en jouer 8 à 15 Hz, re-inférer. Même résultat.

---

## 5. Latence — MESURÉE sur atomman (CPU)

Bench réel (`experiments/real/15_latency_test.py`, B brut, atomman CPU) :
- **1 inférence réseau ≈ 540 ms** (100 pas de débruitage DDPM ; l'encodage image 2×ResNet34 se fait 1 fois).
- 1 lecture de file (queue) ≈ 1 ms (négligeable).
- **Débit soutenu ≈ 14,6-14,9 Hz** → **essentiellement la cible 15 Hz** (à ~3 % près, OK en pratique).

> ⚠️ **`num_inference_steps=16` NE suffit PAS à accélérer** : le scheduler DDPM par défaut ignore ce
> réglage (testé : latence inchangée). Pour vraiment réduire les pas de débruitage il faut passer la
> policy en **DDIM** (changement de code, non fait). Pas nécessaire ici puisque ~15 Hz tient déjà.
> Si un jour on veut plus de marge : DDIM + ~16 pas ≈ ×6 plus rapide.

---

## 6. Déploiement ROS (chemin cible)

Le robot publie/attend ces topics (cf. dataset source `Demis_dataset/README_MODELE_IO.md`) :
- **S'abonner** : `/head_camera/left/image_raw/compressed` + `/head_camera/right/.../compressed`
  (`sensor_msgs/CompressedImage`, JPEG 640×480), `/joint_states` (`sensor_msgs/JointState`, 5 joints rad).
- **Publier** : `/arm_controller/joint_trajectory` (`trajectory_msgs/JointTrajectory`, 5 consignes de
  position) + `/gripper` (`std_msgs/Bool`, true=FERME).
- Boucle : aligner les 3 flux par timestamp (nearest), construire `obs`, `action = post(select_action(pre(obs)))`,
  publier les 5 joints + le gripper seuillé, à ~15 Hz. `policy.reset()` au début de chaque essai.
- Un rejeu « faux réseau » validant le chemin ROS existe côté robot : `~/roby_replay.py`.

---

## 7. Origine / traçabilité

- Archi : Diffusion Policy, **ResNet34 ×2** (encodeur séparé par caméra) + U-Net décodeur
  `[128,256,512]`, images **224px**, action **articulaire absolue 6D**.
- Dataset : 46 démos réelles (oracle) « pomme blanche → dépose en D », converti mcap→LeRobot 15 Hz.
- Entraîné sur Mac (Principal), cooldowns LR 1e-4→0. Détails complets côté Mac :
  `results/runs/real/apple_joint_224_r34/` + `RUNS_EN_COURS.md`.
- **Petit dataset (40 épisodes distincts) → overfit modéré** ; le vrai remède = plus de démos.
