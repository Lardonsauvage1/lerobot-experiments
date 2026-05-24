# Phase 4 — Compression du Diffusion Policy

> Le run 46 résout Lift à 100 % mais pèse **263.7 M params** (938 ms/décision). On cherche le plus petit/rapide modèle gardant les perfs. Résultat : **1.65 M, 100 %, ~44 ms** (÷160 params, ÷21 latence).
> Liens : [◀ Phase 3 Lift](LIFT.md) · **Phase 4 (ici)** · [index](README.md)

## Contexte

Point de départ (run 46) — **presque tout le poids est dans le U-Net** :

| Composant | Params | Part |
|---|---|---|
| **U-Net débruiteur** (`down_dims [512,1024,2048]`) | 252.5 M | **95.8 %** |
| Vision (ResNet18 + spatial-softmax) | 11.2 M | 4.2 % |
| **Total** | **263.7 M** | ~1.05 Go fp32 |

Décisions : **métrique = latence** (sur diffusion, une décision = *N passes U-Net* → 2 leviers **multiplicatifs** : pas de diffusion × taille U-Net) ; on garde l'image ; **critère dé-saturé** (le succès binaire sature à 100 % dès 3K steps → on ajoute des métriques **continues** : succès + temps-au-succès + marge de levage).

## Le parcours (3 leviers)

1. **Largeur du U-Net** (`down_dims`) — réentraîné sur 150 démos, du `[512,1024,2048]` au `[8,16,32]`.
2. **Pas de diffusion** (`num_inference_steps`) — gratuit, sans réentraînement, 10 → 1.
3. **Vision** — ResNet18 remplacé par un mini-CNN maison (0.03 M).

## Résultats

> **Point de fonctionnement : U-Net `[32,64,128]` + vision mini-CNN → 1.65 M params, 100 % à 4 pas de diffusion, ~44 ms/décision.** vs baseline 263.7 M @ 10 pas (938 ms) → **÷160 params, ÷21 latence, perf identique** (au plafond expert).

### Sweep U-Net (taille vs succès, à 10 pas)

| Modèle (`down_dims`) | Params | U-Net | Succès | t_succ | max_z |
|---|---|---|---|---|---|
| baseline `[512,1024,2048]` | 263.7 M | 252 M | 100 % | 43.0 | 1.033 |
| `[256,512,1024]` | 76.6 M | 65 M | 98 % | 44.0 | 0.999 |
| `[128,256,512]` | 28.7 M | 17 M | 98 % | 48.0 | 0.951 |
| `[64,128,256]` | 16.2 M | 5 M | 100 % | 47.0 | 0.969 |
| **`[32,64,128]`** ⭐ | 12.8 M | 1.6 M | 100 % | 46.5 | 0.990 |
| `[16,32,64]` | 11.8 M | 0.4 M | **2 %** 💥 | 65.0 | 0.829 |
| `[8,16,32]` | 11.5 M | 0.3 M | **0 %** 💥 | — | 0.827 |

### Grille latence × succès (lignes = pas de diffusion, colonnes = U-Net)

Cases = **latence ms · succès** (50 val ép.).

| pas \ U-Net | `[512,1024,2048]`| `[256,512,1024]` | `[128,256,512]` | `[64,128,256]` |`[32,64,128]`⭐ | `[16,32,64]` | `[8,16,32]` |
|---|---|---|---|---|---|---|---|
| **10** | 938·100% | 297·100% | 114·98% | 111·100% | 110·100% | 108·2% | 109·0% |
| **5** | 457·100% | 145·98% | 60·100% | 58·100% | 58·100% | 57·0% | 57·0% |
| **4**⭐ | 396·100% | 109·100% | 51·100% | 48·100% | **49·100%** | 48·0% | 48·0% |
| **2** | 199·56% | 67·68% | 29·76% | 29·60% | 28·12% | 28·0% | 28·0% |
| **1** | 103·0% | 28·0% | 18·0% | 17·0% | 17·0% | 17·0% | 17·0% |

### Vision mini-CNN (`[32,64,128]` +)

| Vision | Params total | Succès @4 pas | Latence @4 pas | val-loss |
|---|---|---|---|---|
| **ResNet18** | 12.8 M | 100 % | 49 ms | 0.083 |
| **mini-CNN** (0.03 M) | **1.65 M** | **100 %** | **43.7 ms** | **0.071** |

## Leçons clés

1. **U-Net surdimensionné ×160** : 252 M → 1.6 M sans perte. **Plancher de capacité = `[32,64,128]`** ; en dessous, falaise nette (`[16,32,64]` → 2 %).
2. **Pas de diffusion : plancher = 4** (universel, indépendant de la taille). 100 % de 4 à 10 pas, cassure à 2.
3. **La vision était le vrai mur** : une fois le U-Net minimal, ResNet18 (11.2 M) domine. Un **mini-CNN 0.03 M** from scratch suffit (Lift visuellement simple — cohérent avec « DINOv2 ≈ ResNet gelé » en phase 3). Gain params + vitesse d'entraînement ÷4 ; **latence ~inchangée** (le U-Net domine à l'inférence, pas l'unique encodage vision).
4. **La val-loss prédit la falaise de capacité** (0.083 → 0.243 → 0.663) — mais reste un **proxy bruité** en régime « assez gros ». Croiser avec le rollout.

## Détails techniques

**Protocole d'éval (Phase 0)** — figé, identique pour tous les modèles :
- **Split train 150 / val 50, contigu** (démos 0–149 / 150–199). Contigu *obligé* : `lerobot-train` active l'EpisodeAwareSampler (via `drop_n_last_frames`) qui indexe dans l'espace original → un train non préfixe-0 déborde (IndexError). 150 démos = simuler le régime peu-de-données du bras réel.
- **Rollout sur les 50 init states val** (jamais entraînés → généralisation), métriques continues.
- **Early-stop par modèle** via val-loss (croisée avec le rollout). Split versionné : `results/runs/lift/phase4_split.json`.

**Entraînement** : Mac MPS lent (~16 h/12K steps) ou box GPU (`ssh gpu` → 192.168.1.95, souvent offline → Mac). Le **mini-CNN** a un save/load custom (`src/mini_cnn.py` : la config dit `resnet18` → swap obligatoire avant load).

**Plafond démos** : remettre l'env sur les états enregistrés des démos → nos modèles sont **au niveau expert** sur succès + temps-au-succès (`54_demo_ceiling.py`).

**Outils** : `47` (rollout), `48` (val-loss), `52` (sweep eval), `54/55` (démos), `56` (latence), `57` (sweep pas), `58` (grille), `61/62` (mini-CNN), `50` (train + val-loss continue), `src/lift_eval.py`, `src/mini_cnn.py`. Tableau maître : [`../results/runs/lift/51_unet_sweep_eval/SUMMARY.md`](../results/runs/lift/51_unet_sweep_eval/SUMMARY.md).

## Suite

Phase compression **bouclée**. Rien d'évident à gratter côté compression (vision déjà à 0.03 M). Frontières suivantes, ailleurs : **sim-to-real** (markers Mujoco, domain randomization), **efficacité données** (combien de démos suffisent ?), **tâche plus dure** (Can/Square).
