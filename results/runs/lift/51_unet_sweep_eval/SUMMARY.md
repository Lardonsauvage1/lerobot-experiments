# Compression du Diffusion Policy sur Lift — résultats (Phase 4)

Tâche : Robomimic **Lift**. Protocole : train 150 / val 50 (split contigu figé), éval rollout sur les **50 init states val held-out**, métriques continues (succès, temps-au-succès, marge). Modèles entraînés avec `50_train_valloss.py` (val-loss continue), meilleur checkpoint = min de val-loss.

## ⭐ Point de fonctionnement final

> **U-Net `[32,64,128]` + vision mini-CNN → 1.65 M params, ~99 % de succès à 4 pas de diffusion, ~44 ms/décision.**
>
> vs baseline `[512,1024,2048]` + ResNet18 (263.7 M) à 10 pas = **938 ms**. → **÷160 en params, ÷21 en latence, performance équivalente**. En temps réel : largement dans le budget 20 Hz.
>
> ⚠️ Succès honnête = **98.6 % sur 500 rollouts** (le « 100 % » ci-dessous est mesuré sur 50 ép. → dans le bruit, voir § validation 500 rollouts juste après). Le modèle final est **à égalité statistique** avec des U-Nets 3–10× plus gros.

## Validation rigoureuse — 500 rollouts (IC95 Wilson)

⚠️ Tous les autres tableaux de ce doc sont sur **50 épisodes** → IC95 ≈ ±7 pts : à ce niveau un vrai ~98–99 % affiche souvent un parfait « 100 % » (P(50/50) ≈ 49 % pour un vrai 98.6 %). On a donc tout re-mesuré sur **500 rollouts appariés** (mêmes 500 départs figés `phase4_eval500.npy`, IC95 ≈ ±2 pts, env recréé par tranche — anti-dégradation renderer). Discussion : `docs/COMPRESSION.md`.

### Grille données × taille U-Net (mini-CNN, @ 4 pas) — succès % [IC95]

| N \ U-Net | `[32,64,128]` 1.6M | `[64,128,256]` 5M | `[128,256,512]` 17M |
|---|---|---|---|
| **150** | 98.6 [97.1–99.3] | 99.0 [97.7–99.6] | 99.8 [98.9–100] |
| **100** | 95.4 [93.2–96.9] | 98.4 [96.9–99.2] | 99.8 [98.9–100] |
| **50** | 98.2 [96.6–99.1] | 99.6 [98.6–99.9] | 99.4 [98.3–99.8] |
| **20** | **97.4** [95.6–98.5] | 92.2 [89.5–94.2] | 88.0 [84.9–90.6] |
| **10** | 85.0 [81.6–87.9] | 78.2 [74.4–81.6] | 93.8 [91.3–95.6] |

- **N ≥ 50** : la taille du U-Net ne compte quasi pas (~98–99.8 %, toutes tailles à égalité).
- **N = 20** : le gros U-Net **sur-apprend** (97.4 % > 92.2 % > 88.0 %, CIs disjoints) → le petit `[32,64,128]` est **plus robuste à données rares**.
- **N = 10** : tout décroche, variance d'entraînement dominante (non-monotone). (`grid_data_x_unet_500.{json,png}`)

### Grille pas × données (`[32,64,128]` mini-CNN) — succès %

| pas \ N | 150 | 100 | 50 | 20 | 10 |
|---|---|---|---|---|---|
| **2** | 31 | 6 | 8 | 6 | 3 |
| **4** | 98.6 | 95.4 | 98.2 | 97.4 | 85.0 |
| **10** | 98.4 | 96.2 | 98.8 | 98.8 | **95.6** |
| **20** | 98.0 | 96.8 | 99.4 | 99.0 | 94.0 |
| **50** | 96.0 | 95.6 | 98.2 | 99.0 | 93.6 |

- **2 pas s'effondre partout** (3–31 %) ; **4 pas suffit à pleines données** (~98 %, plus n'aide pas — 50 pas baisse même).
- **À N=10, 4 pas insuffisant (85 %) → ~10 pas (95.6 %)** puis plateau (~94 %, sans rejoindre le ~98 % pleines-données).
- → **plancher de pas dépendant des données** : 4 à pleines données, ~10 si peu de démos ; les pas récupèrent une partie du déficit mais ne remplacent pas les démos. (`grid_steps_x_data_500.{json,png}`)

## Vision mini-CNN — casser le mur ResNet18

Une fois le U-Net minimal (`[32,64,128]`, 1.6 M), la **vision ResNet18 (11.2 M) domine** (params, vitesse d'entraînement). On l'a remplacée par un **mini-CNN maison** (3 convs `3→16→32→64`, GroupNorm, **0.03 M**), réentraîné from scratch sur 150 démos (`61_train_minicnn.py`, `src/mini_cnn.py`).

| `[32,64,128]` + | Params | Vision | Succès @4 pas | Latence @4 pas | val-loss |
|---|---|---|---|---|---|
| **ResNet18** | 12.8 M | 11.2 M | 100 % | 49 ms | 0.083 |
| **mini-CNN** | **1.65 M** | **0.03 M** | **100 %** | **43.7 ms** | **0.071** |

- **Le mini-CNN égale (voire dépasse légèrement) le ResNet18** : succès 100 %, val-loss et max_z un poil meilleurs. La scène Lift est visuellement simple → une vision minuscule suffit.
- **Gains** : params ÷7.8 (total ÷160 vs baseline), **vitesse d'entraînement ÷4** (la vision était le goulot de calcul).
- **Latence : gain modeste** (49 → 44 ms). À l'inférence (batch 1, 4 pas), ce sont les **4 passes U-Net** qui dominent, pas l'unique encodage vision (~5-8 ms). Donc côté latence pure le mini-CNN apporte peu — mais il ne coûte rien et débloque taille + RAM + vitesse d'entraînement.

## Grille latence × succès (le tableau maître)

Cases = **latence d'un sample (ms) · succès (50 val ép.)**. Lignes = pas de diffusion, colonnes = taille U-Net.

| pas \ U-Net | 264M | 77M | 29M | 16M | **13M** ⭐ | 12M `[16,32,64]` | 12M `[8,16,32]` |
|---|---|---|---|---|---|---|---|
| **10** | 938 · 100% | 297 · 100% | 114 · 98% | 111 · 100% | 110 · 100% | 108 · 2% | 109 · 0% |
| **5** | 457 · 100% | 145 · 98% | 60 · 100% | 58 · 100% | 58 · 100% | 57 · 0% | 57 · 0% |
| **4** | 396 · 100% | 109 · 100% | 51 · 100% | 48 · 100% | **49 · 100%** | 48 · 0% | 48 · 0% |
| **2** | 199 · 56% | 67 · 68% | 29 · 76% | 29 · 60% | 28 · 12% | 28 · 0% | 28 · 0% |
| **1** | 103 · 0% | 28 · 0% | 18 · 0% | 17 · 0% | 17 · 0% | 17 · 0% | 17 · 0% |

**Deux planchers indépendants, mis en évidence par la grille :**
- **Plancher de pas = 4** (universel, indépendant de la taille) : tous les modèles capables tiennent ~100 % de 4 à 10 pas, chutent à 2, meurent à 1. (1 pas = 0 % partout, même le baseline.)
- **Plancher de capacité U-Net = `[32,64,128]`** : `[16,32,64]` (0.4 M) et `[8,16,32]` (0.3 M) sont à ~0 % à *tous* les pas → U-Net trop petit pour débruiter la trajectoire.
- La latence à pas fixe est **plate** sur les petits modèles (16M ≈ 13M ≈ 48-58 ms) → **plateau vision** (ResNet18, 11.2 M, qui tourne à chaque décision).

## Sweep U-Net — taille vs succès (à 10 pas)

| Modèle (`down_dims`) | Params | U-Net | Succès | t_success | max_z | val_loss |
|---|---|---|---|---|---|---|
| **Baseline** `[512,1024,2048]` | 263.7 M | 252 M | 100 % | 43.0 | 1.033 | — |
| `[256,512,1024]` | 76.6 M | 65 M | 98 % | 44.0 | 0.999 | 0.067 |
| `[128,256,512]` | 28.7 M | 17 M | 98 % | 48.0 | 0.951 | 0.070 |
| `[64,128,256]` | 16.2 M | 5 M | 100 % | 47.0 | 0.969 | 0.072 |
| **`[32,64,128]`** ⭐ | 12.8 M | 1.6 M | 100 % | 46.5 | 0.990 | 0.083 |
| `[16,32,64]` | 11.8 M | 0.4 M | 2 % 💥 | 65.0 | 0.829 | 0.243 |
| `[8,16,32]` | 11.5 M | 0.3 M | 0 % 💥 | — | 0.827 | 0.663 |

Le U-Net passe de **252 M → 1.6 M (÷160)** sans perte de succès — il était massivement surdimensionné. La **val-loss prédit la falaise** (0.083 → 0.243 → 0.663). Sous `[32,64,128]`, le total ne bouge plus (vision = 11.2 M domine) : descendre encore = perdre 100 % pour ~1 M. → pour aller plus loin il faut attaquer la **vision**. Voir `sweep_pareto.png`.

## Plafond théorique — démos expertes (50 scènes val)

| | Succès | t_success | max_z | hold |
|---|---|---|---|---|
| **Démos expertes (plafond)** | 100 % | 43.0 | 0.880 | 1.00 |
| Baseline 263.7 M | 100 % | 43.0 | 1.033 | 0.48 |
| Gagnant `[32,64,128]` | 100 % | 46.5 | 0.990 | — |

**Nos modèles sont AU plafond expert** sur les métriques comparables (succès 100 %, levage aussi rapide que l'humain). `max_z`/`hold` ne sont pas comparables directement : les démos sont courtes (s'arrêtent au levage) → `hold=1.00`, `max_z=0.88` par construction ; nos modèles tournent 200 steps (montent plus haut, et `hold` mesure du post-succès hors-distribution).

## Détails — val-loss propre par checkpoint

| step | `[256,512,1024]` | `[128,256,512]` | `[64,128,256]` |
|---|---|---|---|
| 1500 | 0.082 | 0.080 | 0.105 |
| 3000 | 0.069 | 0.071 | 0.083 |
| 4500 | **0.067** | **0.070** | 0.074 |
| 6000 | 0.072 | 0.074 | **0.072** |
| 7500 | 0.074 | 0.073 | 0.073 |

→ overfit léger après ~4500-6000 (val-loss remonte). Stop par modèle au min de val-loss.

## Fichiers

| Fichier | Contenu |
|---|---|
| `../grid_data_x_unet_500.{json,png}` | **grille données × U-Net, 500 rollouts** (15 cellules, IC95 Wilson) |
| `../grid_steps_x_data_500.{json,png}` | **grille pas × données, 500 rollouts** (25 cellules, IC95 Wilson) |
| `../67_dataeff_500.json` · `../69_grid_500.json` · `../71_steps_data_500.json` | données brutes des grilles 500 |
| `grid.{json,md}` | grille latence × succès (7 tailles × 5 pas, **50 ép.**/case) |
| `sweep_eval.json` + `sweep_pareto.png` | sweep U-Net (params, val-loss, métriques rollout) |
| `latency.{json,png}` | latence (3 modèles × pas, obs factices) |
| `inference_steps.{json,png}` | succès vs pas sur `[32,64,128]` |
| `demo_ceiling.json` | plafond démos expertes |
| `videos/*.mp4` | épisodes (3 scènes val × modèles) — locaux |

Mini-CNN : résultats dans `../61_minicnn/minicnn_eval.json` ; code `experiments/lift/61_train_minicnn.py` + `62_minicnn_eval.py` + `src/mini_cnn.py`.

Scripts : `52` (sweep eval), `54/55` (plafond/vidéos démos), `56` (latence), `57` (sweep pas), `58` (grille latence), `61/62` (mini-CNN), **`67` (données 500), `run_69_grid.sh`+`69` (grille données×U-Net 500), `70` (assemblage), `71`+`72` (pas×données 500)**, `src/lift_eval.py` (`rollout_eval_chunked`). Récits : `docs/COMPRESSION.md` (cadrage + grilles 500), `docs/LIFT.md` (phase 3).
