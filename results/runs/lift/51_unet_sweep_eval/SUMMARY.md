# Compression du Diffusion Policy sur Lift — résultats (Phase 4)

Tâche : Robomimic **Lift**. Protocole : train 150 / val 50 (split contigu figé), éval rollout sur les **50 init states val held-out**, métriques continues (succès, temps-au-succès, marge). Modèles entraînés avec `50_train_valloss.py` (val-loss continue), meilleur checkpoint = min de val-loss.

## ⭐ Point de fonctionnement final

> **`[32,64,128]` (12.8 M params) à 4 pas de diffusion → 100 % de succès, ~49 ms/décision.**
>
> vs baseline `[512,1024,2048]` (263.7 M) à 10 pas = **938 ms**. → **÷21 en params, ÷19 en latence, performance identique** (au niveau du plafond expert). En temps réel : latence amortie ~6 ms/step d'env (budget 20 Hz = 50 ms) → confortable. `[64,128,256]` (16.2 M) est équivalent en latence et garde un peu plus de marge si on veut être prudent.

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
| `grid.{json,md}` | grille latence × succès (7 tailles × 5 pas, 50 ép./case) |
| `sweep_eval.json` + `sweep_pareto.png` | sweep U-Net (params, val-loss, métriques rollout) |
| `latency.{json,png}` | latence (3 modèles × pas, obs factices) |
| `inference_steps.{json,png}` | succès vs pas sur `[32,64,128]` |
| `demo_ceiling.json` | plafond démos expertes |
| `videos/*.mp4` | épisodes (3 scènes val × modèles) — locaux |

Scripts : `52` (sweep eval), `54/55` (plafond/vidéos démos), `56` (latence), `57` (sweep pas), `58` (grille). Récits : `docs/COMPRESSION.md` (cadrage), `docs/LIFT.md` (phase 3).
