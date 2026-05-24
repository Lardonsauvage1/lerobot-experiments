# Phase 3 — Robomimic Lift : du behavior cloning raté à 100 % en Diffusion Policy

> Saisir et soulever un cube avec un bras Panda 7-DoF. On passe d'un plafond de 70 % (behavior cloning) à **100 %** avec une Diffusion Policy.
> Liens : [◀ Phase 1-2 PushT](PUSHT.md) · **Phase 3 (ici)** · [Phase 4 ▶ Compression](COMPRESSION.md) · [index](README.md)

## Contexte

Après PushT (cube 2D), passage à **Robomimic Lift** : un bras **Panda 7-DoF** doit **saisir un cube et le soulever**. Plus proche du futur bras 5 axes + pince. Données : **200 démos « proficient human »** — état low-dim (pose effecteur, gripper, objet) + image `agentview` 96×96, action **7D** (delta effecteur + pince).

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

## Résultats

- **Sans image : 0 %.** Avec image : jusqu'à 28-34 % (MLP/CE), puis **70 %** en dégelant le ResNet (run 44, record behavior cloning).
- **Diffusion Policy (run 46) : 100 %** — saut décisif vs 70 %. C'est le modèle de référence de la phase suivante (compression).

## Leçons clés

1. **Sans image → 0 %.** Le state low-dim seul ne suffit pas avec notre formulation ; les features image débloquent tout.
2. **Dégeler le ResNet (LR 1e-5) est LE move** sur l'archi MLP : +30 pts (34 → 66 %). ResNet gelé = stats ImageNet → mismatch avec la scène Lift.
3. **Anti-overfit obligatoire sur petit dataset** : Dropout 0.4 + LayerNorm + AdamW(wd=5e-4) + label smoothing.
4. **DINOv2 gelé ≈ ResNet gelé** : la scène Lift est visuellement simple → la richesse du pré-entraînement n'aide pas, c'est l'**adaptabilité** (backbone entraînable) qui compte. *(Ça justifiera le mini-CNN en phase 4.)*
5. **Diffusion Policy ≫ MLP** sur le multimodal : le débruitage gère intrinsèquement les trajectoires multiples valides, là où MLP/CE les contournait à la main.

## Détails techniques

**Les 5 bugs d'eval du run 46** (training OK à 0.037 de loss, mais eval initiale à 0 % — chacun suffisant pour tout casser, tous silencieux côté loss) :
1. **DDPM 100 pas par défaut** (`num_inference_steps=None` → fallback) → fixer à 10.
2. **`select_action()` ne normalise pas** (v0.5 externalise les normalizers) → appliquer `preprocessor`/`postprocessor`.
3. **Init env hors distribution** (`rs.make` ≠ démos) → `reset_to(states=...)` depuis une démo HDF5.
4. **Mismatch Robosuite 1.4.1 ↔ 1.5.2** sur `object-state` → wrapper `EnvRobosuite` + patch `controller_configs`.
5. **Sign flip `object[7:10]`** (`eef−cube` vs `cube−eef`) → `obj[7:10] *= -1`.
> Leçon transverse : un train loss bas ne dit rien sur l'eval. Smoke test 1-2 épisodes avant un eval long.

**Sim-to-real — markers Mujoco** : Robosuite rend des « debug sites » (point rouge sur la pince) présents au training ET à l'eval, mais **absents d'un vrai robot** → risque de *sim shortcut*. À faire pour le transfert : désactiver les markers (rgba=0) ou domain randomization.

**MPS** : `.contiguous()` après fancy indexing ; BatchNorm trainable → NaN possible (NaN guard + save-best immédiat).

## Suite → [Phase 4 : Compression](COMPRESSION.md)

Le run 46 résout Lift mais pèse **263.7 M params** (938 ms/décision sur Mac). La phase suivante le **compresse** sans perdre le 100 %.
