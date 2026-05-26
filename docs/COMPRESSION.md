# Phase 4 — Compression & limites du modèle

> Le run 46 résout Lift à 100 % mais pèse **263.7 M params** (938 ms/décision). On cherche les **limites** : jusqu'où rétrécir le modèle, réduire les pas de diffusion, et réduire les données — en gardant les perfs ? Résultat : modèle **1.65 M, ~44 ms** (÷160 params, ÷21 latence) à **~99 % de succès** (98.6 % sur 500 rollouts, statistiquement à égalité avec les gros modèles), et **~20 démos suffisent** (au lieu de 150).
> Liens : [◀ Phase 3 Lift](LIFT.md) · **Phase 4 (ici)** · [index](README.md)

## Contexte

Point de départ (run 46) — **presque tout le poids est dans le U-Net** :

| Composant | Params | Part |
|---|---|---|
| **U-Net débruiteur** (`down_dims [512,1024,2048]`) | 252.5 M | **95.8 %** |
| Vision (ResNet18 + spatial-softmax) | 11.2 M | 4.2 % |
| **Total** | **263.7 M** | ~1.05 Go fp32 |

Décisions : **métrique = latence** (sur diffusion, une décision = *N passes U-Net* → 2 leviers **multiplicatifs** : pas × taille U-Net) ; on garde l'image ; **critère dé-saturé** (le succès binaire sature à 100 % → métriques **continues** : succès + temps-au-succès + marge).

## Le parcours (4 leviers)

1. **Largeur du U-Net** (`down_dims`) — du `[512,1024,2048]` au `[8,16,32]`.
2. **Pas de diffusion** (`num_inference_steps`) — gratuit, sans réentraînement, 10 → 1.
3. **Vision** — ResNet18 remplacé par un mini-CNN maison (0.03 M).
4. **Données** — combien de démos suffisent (150 → 10).

## Résultats

> **Point de fonctionnement : U-Net `[32,64,128]` + vision mini-CNN → 1.65 M params, ~99 % à 4 pas, ~44 ms.** vs baseline 263.7 M @ 10 pas (938 ms) → **÷160 params, ÷21 latence, perf équivalente** (98.6 % sur 500 rollouts, à égalité statistique avec les U-Nets 3-10× plus gros — cf. grille plus bas).

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

### Grille taille U-Net × données — 500 rollouts appariés (mini-CNN, @ 4 pas)

> ⚠️ **Refait proprement (la 1re version était dans le bruit).** À **50 rollouts** l'IC95 du succès vaut **±~7 pts** → les écarts entre cellules étaient indiscernables (N=20 « battait » N=50 = artefact). Ici : **500 rollouts**, **mêmes 500 départs figés pour tous** (test apparié), **IC95 de Wilson ≈ ±2 pts**. La val-loss, elle, ne dit *pas* si le robot accomplit la tâche → c'est bien le **succès** qui juge, avec assez de mesures. (Méthodo + un bug d'éval corrigé : voir Détails techniques.)

Succès % [IC95 Wilson], lignes = nb de démos, colonnes = largeur du U-Net (vision mini-CNN constante) :

| N \ U-Net | `[32,64,128]` (1.6 M) | `[64,128,256]` (5 M) | `[128,256,512]` (17 M) |
|---|---|---|---|
| **150** | 98.6 [97.1–99.3] | 99.0 [97.7–99.6] | **99.8** [98.9–100] |
| **100** | 95.4 [93.2–96.9] | 98.4 [96.9–99.2] | **99.8** [98.9–100] |
| **50** | 98.2 [96.6–99.1] | **99.6** [98.6–99.9] | 99.4 [98.3–99.8] |
| **20** | **97.4** [95.6–98.5] | 92.2 [89.5–94.2] | 88.0 [84.9–90.6] |
| **10** | 85.0 [81.6–87.9] | 78.2 [74.4–81.6] | 93.8 [91.3–95.6] |

Trois régimes nets (voir `results/runs/lift/grid_data_x_unet_500.png`) :
1. **Données abondantes (N ≥ 50) : la capacité ne compte quasi pas** — toutes les tailles à ~98–99.8 % (le gros U-Net grappille à peine). Le sweet spot `[32,64,128]` ne sacrifie rien.
2. **N = 20 : le gros U-Net SUR-APPREND** — inversion franche, CIs disjoints : `[32,64,128]` **97.4 %** > `[64,128,256]` 92.2 % > `[128,256,512]` 88.0 %. À données rares, **moins de capacité généralise mieux** (`t_success` le confirme : 61 vs 74 vs 54).
3. **N = 10 : tout décroche et la variance d'ENTRAÎNEMENT domine** — non-monotone (`[128,256,512]` 93.8 % « gagne » par chance de tirage / checkpoint précoce, alors qu'il ne fait que 88 % à N=20). À 10 démos, le résultat dépend surtout du modèle qu'on a tiré, pas de la capacité.

→ **Double justification du petit `[32,64,128]`** : il égale les gros à pleines données **et** il est **plus robuste au régime peu-de-données** (N=20). Pour le bras réel (peu de démos), petit U-Net = bon réflexe.

NB : le « 100 % » des sections précédentes (mesuré sur ≤50 ép.) était dans le bruit de ce **~98.6–99 %** réel — la conclusion (compression sans perte de perf) tient, mais le chiffre honnête à 500 rollouts est ~99 %, pas un 100 % strict.

### Pas de diffusion × données — 500 rollouts (`[32,64,128]` mini-CNN)

Succès % [IC95 Wilson], lignes = pas (`num_inference_steps`), colonnes = nb démos. Voir `results/runs/lift/grid_steps_x_data_500.png`.

| pas \ N | 150 | 100 | 50 | 20 | 10 |
|---|---|---|---|---|---|
| **2** | 31.0 | 6.4 | 8.0 | 5.8 | 3.0 |
| **4** | 98.6 | 95.4 | 98.2 | 97.4 | 85.0 |
| **10** | 98.4 | 96.2 | 98.8 | 98.8 | **95.6** |
| **20** | 98.0 | 96.8 | 99.4 | 99.0 | 94.0 |
| **50** | 96.0 | 95.6 | 98.2 | 99.0 | 93.6 |

Trois constats (remplacent l'ancien « pont pas↔données » mesuré à 50 ép.) :
1. **2 pas s'effondre partout** (3–31 %) — le plancher est bien au-dessus de 2.
2. **4 pas suffit à pleines données** (N ≥ 50 : ~98 %), et **monter les pas n'aide pas** (10/20/50 ≈ identiques ; 50 pas *baisse* même un peu à N=150 → 96 %). Le point de fonctionnement @ 4 pas est validé.
3. **MAIS à données rares (N=10), 4 pas ne suffit plus (85 %)** : 85 → **95.6 % en passant à 10 pas** (+10 pts, sans réentraîner), puis plateau (~94–95 %, sans atteindre le ~98 % des données abondantes).

→ Le « plancher » de pas est **dépendant des données** : **4 pas à pleines données, ~10 pas si peu de démos** ; au-delà de 10, rien à gagner. Les pas **récupèrent une partie** du déficit de données mais **plafonnent sous la perf pleines-données** — ils ne *remplacent* pas les démos.

## Leçons clés

1. **U-Net surdimensionné ×160** : 252 M → 1.6 M sans perte. **Plancher de capacité = `[32,64,128]`** ; en dessous, falaise nette (`[16,32,64]` → 2 %).
2. **Pas de diffusion : plancher dépendant des données.** 2 pas casse partout (3–31 %). **4 pas suffit à pleines données** (~98 %, et plus n'aide pas — 50 pas baisse même un peu), **mais ~10 pas sont nécessaires à données rares** (N=10 : 85 % @4 → 95.6 % @10, puis plateau). Les pas récupèrent une partie du déficit de données sans le combler.
3. **La vision était le vrai mur** : une fois le U-Net minimal, ResNet18 (11.2 M) domine. Un **mini-CNN 0.03 M** from scratch suffit (Lift visuellement simple — cohérent avec « DINOv2 ≈ ResNet gelé » en phase 3). Gain params + vitesse d'entraînement ÷4 ; **latence ~inchangée** (le U-Net domine à l'inférence).
4. **Lift est peu gourmand en données… jusqu'à un seuil** : ~20 démos ≈ 97 % (`[32,64,128]`), mais **falaise à N=10 (85 %)**. Le coût de la rareté se voit aussi dans `t_success` (succès plus lent). Encourageant pour le bras réel — viser **≥ 20 démos**.
5. **Interaction capacité × données** (le résultat clé de la grille) : à pleines données la taille du U-Net ne change rien (~98–99 %), mais **à données rares (N=20) le gros U-Net sur-apprend** (88 % vs 97.4 % pour le petit). → le petit `[32,64,128]` est **doublement justifié** : aussi bon à pleines données, plus robuste à données rares.
6. **Le succès binaire est bruité — il faut assez de rollouts.** À 50 ép., IC95 ≈ ±7 pts noyait tous les écarts (les conclusions « ~20 ≈ 98 %, falaise » de la v1 étaient du bruit). **500 rollouts appariés** (IC95 ≈ ±2 pts) révèlent les vrais régimes. Et la **val-loss ne mesure pas la fonctionnalité** (juste l'ajustement aux démos) : c'est le succès qui juge. La val-loss reste utile pour **prédire les falaises de capacité** (0.083 → 0.243 → 0.663) et choisir le checkpoint.

## Détails techniques

**Protocole d'éval (Phase 0)** — figé pour tous les modèles :
- **Split train / val 50, contigu** (val = démos 150–199 figées, jamais entraînées ; train = préfixe `0..N-1`). Contigu *obligé* : l'EpisodeAwareSampler de `lerobot-train` indexe dans l'espace original → train non préfixe-0 déborde (IndexError).
- **Rollout sur les 50 init states val**, métriques continues ; **early-stop par modèle** via val-loss (croisée avec rollout). Split versionné : `results/runs/lift/phase4_split.json`.

**Entraînement** : Mac MPS lent (~16 h/12K steps avec ResNet ; ~26 min avec le mini-CNN) ou box GPU (`ssh gpu`, souvent offline → Mac). Le **mini-CNN** a un save/load custom (`src/mini_cnn.py` : la config dit `resnet18` → swap obligatoire avant load).

**Plafond démos** : nos modèles sont **au niveau expert** sur succès + temps-au-succès (`54_demo_ceiling.py`).

**Protocole rigoureux 500 rollouts** (grille données × U-Net) : **500 départs figés et appariés** (mêmes pour tous les modèles), générés par `env.reset()` (même distribution que les démos, jamais entraînés), sauvés dans `phase4_eval500.npy`. **Arrêt au 1er succès** (le succès = « réussi à un moment » → ne change pas le taux, ~2.5× plus vite). **IC95 de Wilson** (robuste près de 100 %). À 50 ép. l'IC95 vaut ±~7 pts (les évals historiques ≤50 ép. restent fiables car comparées entre elles dans la même passe, mais ne séparent pas des modèles à ~95–100 %) ; à 500 il tombe à ±~2 pts.

⚠️ **Bug d'éval longue trouvé + corrigé** : réutiliser un même env mujoco pour **des centaines de rollouts** (ou des centaines de `reset()` de génération) dégrade le **renderer offscreen** → les épisodes tardifs reçoivent des images pourries → faux échecs (symptôme : un modèle 100 % tombe à 2 %). Correctif `lift_eval.rollout_eval_chunked` : **recrée l'env tous les 50 épisodes** (sous le seuil sûr mesuré à 100) + `env.env.close()` ; génération des états avec un env **jetable**. Les évals ≤50 ép. (sweep, plafond) étaient sous le seuil → non affectées.

**Outils** : `47` (rollout), `48` (val-loss), `52` (sweep eval), `54/55` (démos), `56` (latence), `57` (sweep pas), `58` (grille latence×succès), `61` (train mini-CNN), `62` (eval mini-CNN), `63` (efficacité données 50 ép. — superseded), `64/66` (pont pas↔données 50 ép.), `65` (vidéos), `50` (train + val-loss continue), `67` (données 500 rollouts), `run_69_grid.sh` + `69_grid_eval.py` (grille données×U-Net 500), `70` (assemblage grille données×U-Net), `71` (sweep pas×données 500) + `72` (assemblage pas×données), `src/lift_eval.py` (dont `rollout_eval_chunked`), `src/mini_cnn.py`. Grilles : [`../results/runs/lift/grid_data_x_unet_500.json`](../results/runs/lift/grid_data_x_unet_500.json) · [`../results/runs/lift/grid_steps_x_data_500.json`](../results/runs/lift/grid_steps_x_data_500.json) · Tableau maître sweep : [`../results/runs/lift/51_unet_sweep_eval/SUMMARY.md`](../results/runs/lift/51_unet_sweep_eval/SUMMARY.md).

## Suite

Limites du modèle bien cartographiées (capacité, pas, vision, données). Frontières suivantes, ailleurs : **sim-to-real** (markers Mujoco, domain randomization), **tâche plus dure** (Can/Square — où les planchers seraient sûrement plus hauts).
