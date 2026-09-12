# 🛑 Robot mauvais / erratique ? → 99 % du temps c'est le PRÉTRAITEMENT, pas le modèle

> Doc de dépannage pour l'agent qui déploie. Établi le 2026-07-13 après diagnostic.
> Complément de `README_INFERENCE_AGENT.md` (même dossier).

---

## 1. Le symptôme

Le bras fait un mouvement **quasi constant, ~1 rad (≈55°) à côté**, insensible à l'image /
à la position de la pomme → échec systématique.

## 2. La cause (DIAGNOSTIQUÉE)

**L'observation est envoyée au modèle SANS la normalisation** (le `preprocessor`), ou avec un
**mauvais format d'image**. Le modèle reçoit alors des entrées hors échelle → l'encodeur vision sort
des features aberrantes → le modèle retombe sur une **pose « moyenne » constante**, ignorant l'entrée.

**Preuve mesurée (modèle B, teacher-forcing sur de vraies frames du dataset) :**

| Prétraitement | Erreur action vs vérité terrain |
|---|---|
| ❌ **sans `preprocessor`** (obs brute) | **~1,0 rad (55°)**, sortie constante ← = le bug robot |
| ✅ **avec `preprocessor`** | **0,01–0,02 rad (<1°)**, gripper correct |

→ **Le modèle est bon.** Le problème est 100 % dans le pipeline d'entrée au déploiement.

## 3. Le correctif

Toujours passer l'observation par le **preprocessor du checkpoint** avant `select_action`, et le
**postprocessor** après :

```python
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

MODEL = "/home/sam/lerobot-experiments/deployable_models/apple/B_cd5k_from12000/brut"
policy = DiffusionPolicy.from_pretrained(MODEL); policy.eval().to("cpu"); policy.reset()
pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=MODEL,
    preprocessor_overrides={"device_processor": {"device": "cpu"}})   # <- override device (checkpoint = mps)

# À CHAQUE pas de contrôle (15 Hz) :
#   obs = { "observation.images.left":  tensor [1,3,224,224] float [0,1] RGB CHW,
#           "observation.images.right": tensor [1,3,224,224] float [0,1] RGB CHW,
#           "observation.state":        tensor [1,5] float (joints rad) }
action = post(policy.select_action(pre(obs)))   # <-- pre() et post() OBLIGATOIRES
```

Le `preprocessor` applique la mean/std d'entraînement (sauvées dans le checkpoint). **Ne PAS
normaliser à la main en plus** — juste construire `obs` en float [0,1] / rad et laisser `pre()` faire.

## 4. Checklist des pièges (par ordre de probabilité)

1. **`pre()` oublié** → LE bug. Sortie constante. (cf. §2)
2. **Image BGR au lieu de RGB** : `cv2.imdecode` renvoie du **BGR** → il FAUT
   `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`. (C'est ainsi que le dataset a été construit.)
3. **Échelle image** : doit être **float [0,1]** (diviser par 255), PAS [0,255] uint8, PAS [-1,1].
4. **Layout** : **CHW** `[3,224,224]` (pas HWC `[224,224,3]`).
5. **Resize** : 640×480 → **224×224** (cv2, INTER_AREA). Mauvaise taille = crash ou features fausses.
6. **Gauche/droite inversées** : `observation.images.left` = caméra gauche, `.right` = droite. Ne pas swap.
7. **Ordre des joints** : `observation.state` = joint_1..joint_5 dans CET ordre, en **radians**.
8. **`policy.reset()`** au début de chaque essai (vide la file du chunking).
9. **Cadence** : appeler `select_action` à **15 Hz** régulier (cf. README §4, malentendu "2 Hz").

## 5. Se vérifier AVANT de rebrancher le robot (test décisif)

Un script de diagnostic existe : **`experiments/real/17_debug_inference.py`** (côté Mac ; à copier ici
si besoin). Il fait du **teacher-forcing** : nourrit le modèle avec de vraies frames du dataset et
compare l'action prédite à la vérité terrain.

- **Attendu si OK** : erreur **< 0,05 rad** (quelques degrés) à chaque frame, gripper correct.
- **Si erreur ~1 rad + sortie constante** → ton pipeline d'entrée a l'un des bugs du §4.

Reproduire la même construction d'`obs` que dans ce script pour ton nœud ROS, puis vérifier que
`select_action(pre(obs))` redonne < 0,05 rad d'erreur sur quelques frames connues. Une fois ce test
vert, brancher le robot.

## 6. ✅ Nœud ROS2 de référence PRÊT À L'EMPLOI

Un nœud qui câble déjà TOUT correctement (décodage JPEG, BGR→RGB, resize 224, `pre()`/`post()`,
chunking, thread d'inférence + timer 15 Hz, publication trajectoire + gripper) :

**`/home/sam/lerobot-experiments/deployable_models/apple/roby_infer_node.py`**

- Souscrit : `/head_camera/left|right/image_raw/compressed`, `/joint_states`.
- Publie : `/arm_controller/joint_trajectory` (5 joints) + `/gripper` (Bool).
- Prétraitement **validé end-to-end** (JPEG caméra → action) = 0,017 rad (0,9°) vs vérité terrain.
- Config en haut du fichier (MODEL_PATH, DEVICE, seuil gripper, topics…). Modèle par défaut = B brut.
- ⚠️ Env : besoin d'un Python avec **rclpy (ROS2) ET torch+lerobot** (cf. entête du fichier). Lancer :
  `python roby_infer_node.py` (ROS2 sourcé).
- Si ça reste lent/mou : baisser `n_action_steps` (ré-observe plus souvent) — voir README §5.

Partir de ce nœud plutôt que de recoder le prétraitement à la main (c'est là que le bug se glisse).

## 7. Récap

- ✅ Modèle B (`B_cd5k_from12000/brut`) reproduit les démos à **<1°** offline → **il n'est pas en cause**.
- 🛑 Le robot rate parce que l'`obs` n'est pas préparée comme à l'entraînement (normalisation / format).
- 🔧 Fix = utiliser **`roby_infer_node.py`** (§6), ou à la main `select_action(pre(obs))` + checklist §4 + auto-vérif §5.
