# Phase 3 — Robomimic Lift : du behavior cloning raté à 100 % en Diffusion Policy

> Suite de [`JOURNEY.md`](JOURNEY.md) (PushT). On change de tâche pour se rapprocher du bras réel cible,
> et on passe d'un plafond de 70 % (behavior cloning) à **100 %** avec une Diffusion Policy.

## Contexte : pourquoi changer de tâche

Après PushT (pousser un cube en 2D), passage à **Robomimic Lift** : un bras **Panda 7-DoF** doit
**saisir un cube et le soulever**. Plus proche du futur bras 5 axes + pince. Données : **200 démos
« proficient human » (PH)** — état low-dim (pose effecteur, gripper, objet) + image `agentview` 96×96,
action **7D** (delta effecteur + pince).

## Le parcours (runs 33 → 46)

| Run | Setup principal | Best success |
|---|---|---|
| 33-34 | MLP state seul (MSE, puis CE+résidu) | **0 %** |
| 35-36 | + image (MSE / CE+résidu) | 28-34 % |
| 37 | « Robomimic-clone » (MLP[1024] + GMM K=5, chunk=1) | 28 % |
| 38-39 | + image augmentation (échec) | 16-20 % |
| 40-41 | + anti-overfit (Dropout + LayerNorm + AdamW + label smoothing) | 48-50 % |
| 42 | + **ResNet trainable** (LR 1e-5) sur MPS | 66 % (puis crash NaN) |
| 43 | + FrozenBatchNorm2d (tue le gain) | 30 % |
| 44 | run 42 + NaN guard + save-best immédiat | **70 %** (record BC) |
| 45 | DINOv2-small **gelé** au lieu de ResNet | 44 % (≈ ResNet gelé) |
| **46** | **Diffusion Policy** (lerobot-train, U-Net 1D + DDPM, 264 M params) | **100 %** ⭐ |

## Leçons clés

1. **Sans image → 0 %.** Avec des features image (ResNet18) → saut à 28 %+. Le state low-dim seul ne suffit pas avec notre formulation.
2. **Dégeler le ResNet (LR 1e-5) est LE move** sur l'archi MLP : +30 pts (34 % → 66 %). Le ResNet gelé porte les stats ImageNet → mismatch avec la scène Lift.
3. **Anti-overfit obligatoire sur petit dataset** : Dropout 0.4 + LayerNorm + AdamW(wd=5e-4) + label smoothing 0.1.
4. **MPS Apple Silicon** : `.contiguous()` après fancy indexing, sinon erreurs de vue au backward.
5. **BatchNorm trainable = breakthrough, mais NaN possible** → NaN guard + sauvegarde du best sur disque immédiatement.
6. **DINOv2 gelé ≈ ResNet gelé** : la scène Lift est visuellement simple, la richesse du pré-entraînement self-supervisé n'aide pas. C'est l'**adaptabilité** (backbone trainable) qui compte, pas la qualité du pré-entraînement.
7. **Diffusion Policy >> MLP** sur le multimodal robotique : 100 % vs 70 %. Le débruitage diffusion gère intrinsèquement la multimodalité (plusieurs trajectoires valides), là où le MLP/CE la contournait péniblement.

## Run 46 — le breakthrough Diffusion Policy, et 5 bugs d'eval cachés

Le training s'est bien passé (12K steps, train loss 0.037). **Mais l'eval initiale donnait 0 %** alors que le modèle était bon. Cinq bugs cumulés, chacun suffisant pour tout faire échouer — tous **silencieux côté loss** :

1. **DDPM 100 pas par défaut.** `num_inference_steps=None` → fallback à `num_train_timesteps=100` → ~9 s/sample, 12 h pour 200 ép. Fix : `policy.diffusion.num_inference_steps = 10`.
2. **`select_action()` ne normalise pas.** LeRobot v0.5 a externalisé les normalizers dans `PolicyProcessorPipeline`. Sans les appliquer, le modèle voit des obs brutes. Fix : `preprocessor(obs) → select_action → postprocessor(action)`.
3. **Init env hors distribution.** `rs.make("Lift")` randomise autrement que les démos. Fix : reset depuis l'état initial d'une démo HDF5 (`reset_to(states=...)`).
4. **Mismatch versions Robosuite 1.4.1 (démos) ↔ 1.5.2 (installé)** sur l'`object-state`. Fix : wrapper `EnvRobosuite` de Robomimic + patch `controller_configs` au format composite.
5. **Sign flip sur `object[7:10]`.** Robomimic 1.4 stockait `eef−cube`, Robosuite 1.5 renvoie `cube−eef` → robot qui fonce du mauvais côté. Fix : `obj[7:10] *= -1`.

**Leçon transverse** : un train loss bas ne dit rien sur l'eval. Toujours faire un smoke test 1-2 épisodes avant un eval long, et comparer les **actions brutes du modèle vs la distribution du dataset** comme premier debug.

## Sim-to-real : le piège des markers Mujoco

Robosuite rend par défaut des « debug sites » sur l'image (point rouge au centre de la pince, axe vert). **Présents au training ET à l'eval** → pas de mismatch en simu. **Mais sur un vrai robot ils n'existent pas.** Le modèle peut avoir appris « fermer la pince quand le point rouge touche le cube » au lieu de la géométrie 3D — un *sim shortcut* classique qui casse le sim-to-real.

À faire pour le transfert au bras réel : désactiver les markers (rgba=0 sur les debug sites) ou domain randomization au training.

## Suite → Phase 4 : compression

Le run 46 résout Lift mais pèse **263.7 M params** (énorme, et 927 ms par décision sur Mac). La phase suivante le **compresse pour la latence** sans perdre le 100 % → voir [`COMPRESSION.md`](COMPRESSION.md) et le tableau de résultats `results/runs/lift/51_unet_sweep_eval/SUMMARY.md`.
