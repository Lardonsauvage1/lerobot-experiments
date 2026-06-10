# Phase 4 — Compression & limites du modèle

> Le run 46 résout Lift à 100 % mais pèse **263.7 M params** (938 ms/décision). Question : jusqu'où **rétrécir le modèle, réduire les pas de diffusion et réduire les données** en gardant les perfs ?
> - **Avec coords objet en entrée** (béquille sim, état 19D) : on descend à **1.65 M / ~44 ms** à **~99 %** (÷160 params, ÷21 latence) et **~20 démos suffisent**. → **Partie 1.**
> - **En vision pure** (sans coords, transférable au réel) : la compression maximale est plus modeste — **~29 M** pour le même **~99 %** (÷9 params). → **Partie 2.**
>
> Liens : [◀ Phase 3 Lift](LIFT.md) · **Phase 4 (ici)** · [index](README.md)

> ⚠️ **Caveat à lire avant les chiffres de la Partie 1.**
> Tous les résultats de la Partie 1 utilisent un état 19D qui **inclut la pose complète du cube** (`object` : position + quaternion + relatif). C'est ce que donne le simulateur, mais **un vrai bras n'a jamais cette info** — seulement sa caméra et ses encodeurs articulaires. Les chiffres « ~99 % » et « ~20 démos » de la Partie 1 **dépendent donc d'une béquille sim non transférable**.
> La **Partie 2** refait l'étude **sans cette béquille** (vision pure) : les conclusions *relatives* (architecture, taille U-Net, nb démos) tiennent, mais les chiffres *absolus* changent. Le récit phase 5 v1 (transférabilité + tâche Can, méthodo ré-évaluée depuis) est archivé dans [`CAN_archive.md`](CAN_archive.md).

**Convention.** Dans les tableaux, **Params = total du modèle** (vision + U-Net) sauf si la colonne dit explicitement « U-Net » (U-Net seul). Métriques continues (critère dé-saturé, le succès binaire saturant à 100 %) :
- **Succès** — % de rollouts réussis (le juge final, avec assez de mesures).
- **t_succ** — pas-environnement médians jusqu'au 1ᵉʳ succès (plus bas = plus efficace).
- **max_z** — hauteur max atteinte par le cube (marge : le succès = dépasser un seuil ; >~1 = bien soulevé).

---

# Partie 1 — Compression *avec* coords objet (béquille sim)

## Contexte

Point de départ (run 46) — **presque tout le poids est dans le U-Net** :

| Composant | Params | Part |
|---|---|---|
| **U-Net débruiteur** (`down_dims [512,1024,2048]`) | 252.5 M | **95.8 %** |
| Vision (ResNet18 + spatial-softmax) | 11.2 M | 4.2 % |
| **Total** | **263.7 M** | ~1.05 Go fp32 |

Choix de méthode : **métrique = latence** (en diffusion, une décision = *N passes U-Net* → 2 leviers **multiplicatifs** : pas × taille U-Net) ; on garde l'image ; métriques **continues** (cf. convention).

## Les 4 leviers

1. **Largeur du U-Net** (`down_dims`) — de `[512,1024,2048]` à `[8,16,32]`.
2. **Pas de diffusion** (`num_inference_steps`) — gratuit, sans réentraînement, 10 → 1.
3. **Vision** — ResNet18 remplacé par un mini-CNN maison (0.03 M).
4. **Données** — combien de démos suffisent (150 → 10).

## Résultats

> **Point de fonctionnement retenu : U-Net `[32,64,128]` + vision mini-CNN → 1.65 M params, ~99 % à 4 pas, ~44 ms.** vs baseline 263.7 M @ 10 pas (938 ms) → **÷160 params, ÷21 latence**, perf équivalente (98.6 % sur 500 rollouts, à égalité statistique avec les U-Nets 3–10× plus gros).

### Levier 1+2 — Sweep U-Net & grille latence

> ⚠️ Mesures sur **50 ép. (indicatif)** : l'IC95 vaut ±~7 pts à ce N, donc les `100 %`/`98 %` ne se distinguent **pas** entre eux. Les chiffres fiables (500 rollouts) sont plus bas.

**Sweep U-Net** — on fait varier la largeur du U-Net. **Fixe :** vision **ResNet18**, **10 pas**, **N=150** démos, état 19D.

| Modèle (`down_dims`) | Params (total) | U-Net seul | Succès | t_succ | max_z |
|---|---|---|---|---|---|
| baseline `[512,1024,2048]` | 263.7 M | 252 M | 100 % | 43.0 | 1.033 |
| `[256,512,1024]` | 76.6 M | 65 M | 98 % | 44.0 | 0.999 |
| `[128,256,512]` | 28.7 M | 17 M | 98 % | 48.0 | 0.951 |
| `[64,128,256]` | 16.2 M | 5 M | 100 % | 47.0 | 0.969 |
| **`[32,64,128]`** ⭐ | 12.8 M | 1.6 M | 100 % | 46.5 | 0.990 |
| `[16,32,64]` | 11.8 M | 0.4 M | **2 %** 💥 | 65.0 | 0.829 |
| `[8,16,32]` | 11.5 M | 0.3 M | **0 %** 💥 | — | 0.827 |

→ **Plancher de capacité net : `[32,64,128]`** (1.6 M de U-Net). En dessous, falaise (`[16,32,64]` → 2 %).

**Grille latence × succès** — on fait varier pas × largeur U-Net. **Fixe :** vision **ResNet18**, **N=150**, état 19D. Lignes = pas, colonnes = U-Net, cases = **latence ms · succès** :

| pas \ U-Net | `[512,1024,2048]`| `[256,512,1024]` | `[128,256,512]` | `[64,128,256]` |`[32,64,128]`⭐ | `[16,32,64]` | `[8,16,32]` |
|---|---|---|---|---|---|---|---|
| **10** | 938·100% | 297·100% | 114·98% | 111·100% | 110·100% | 108·2% | 109·0% |
| **5** | 457·100% | 145·98% | 60·100% | 58·100% | 58·100% | 57·0% | 57·0% |
| **4**⭐ | 396·100% | 109·100% | 51·100% | 48·100% | **49·100%** | 48·0% | 48·0% |
| **2** | 199·56% | 67·68% | 29·76% | 29·60% | 28·12% | 28·0% | 28·0% |
| **1** | 103·0% | 28·0% | 18·0% | 17·0% | 17·0% | 17·0% | 17·0% |

→ **4 pas suffisent** (au-dessus, on paie de la latence pour rien) ; **2 pas casse**. Combiné au petit U-Net, la latence chute de 938 à ~49 ms.

### Levier 3 — Vision mini-CNN

On fait varier l'encodeur visuel. **Fixe :** U-Net `[32,64,128]`, **4 pas**, **N=150**, état 19D. *(Mesures sur 50 ép., indicatif : les deux 100 % sont à égalité dans le bruit.)*

| Vision | Params total | Succès @4 pas | Latence @4 pas | val-loss |
|---|---|---|---|---|
| **ResNet18** | 12.8 M | 100 % | 49 ms | 0.083 |
| **mini-CNN** (0.03 M) | **1.65 M** | **100 %** | **43.7 ms** | **0.071** |

→ Sur Lift (visuellement simple, image quasi redondante avec les coords cube), un **mini-CNN 0.03 M from scratch** égale ResNet18. Gain en params et en **vitesse d'entraînement (÷4)** ; **latence ~inchangée** (le U-Net domine à l'inférence).

### Levier 4 — Données × capacité (500 rollouts, mesure fiable)

> ✅ **C'est ici que les chiffres comptent.** 500 départs **figés et appariés** (mêmes pour tous les modèles), **IC95 de Wilson ≈ ±2 pts**. La 1ʳᵉ version (50 ép., ±7 pts) noyait tous les écarts dans le bruit — voir Détails techniques.

On fait varier données × largeur U-Net. **Fixe :** vision **mini-CNN** (+~1.65 M de base), **4 pas**, état 19D. Succès % [IC95], lignes = nb démos, colonnes = **U-Net (params U-Net seul)** :

| N \ U-Net | `[32,64,128]` (1.6 M) | `[64,128,256]` (5 M) | `[128,256,512]` (17 M) |
|---|---|---|---|
| **150** | 98.6 [97.1–99.3] | 99.0 [97.7–99.6] | **99.8** [98.9–100] |
| **100** | 95.4 [93.2–96.9] | 98.4 [96.9–99.2] | **99.8** [98.9–100] |
| **50** | 98.2 [96.6–99.1] | **99.6** [98.6–99.9] | 99.4 [98.3–99.8] |
| **20** | **97.4** [95.6–98.5] | 92.2 [89.5–94.2] | 88.0 [84.9–90.6] |
| **10** | 85.0 [81.6–87.9] | 78.2 [74.4–81.6] | 93.8 [91.3–95.6] |

Trois régimes (`results/runs/lift/grid_data_x_unet_500.png`) :
1. **N ≥ 50 — la capacité ne compte quasi pas** : toutes les tailles à ~98–99.8 %. Le sweet spot `[32,64,128]` ne sacrifie rien.
2. **N = 20 — le gros U-Net sur-apprend** : inversion franche, IC95 disjoints (`[32,64,128]` 97.4 % > `[64,128,256]` 92.2 % > `[128,256,512]` 88.0 %). À données rares, **moins de capacité généralise mieux**.
3. **N = 10 — la variance d'*entraînement* domine** : non-monotone (`[128,256,512]` 93.8 % « gagne » par tirage chanceux alors qu'il ne fait que 88 % à N=20). Le résultat dépend surtout du modèle tiré, pas de la capacité.

→ **Le petit `[32,64,128]` est doublement justifié** : aussi bon à pleines données **et** plus robuste à données rares (N=20). Pour le bras réel (peu de démos), petit U-Net = bon réflexe.

### Levier 2×4 — Pas de diffusion × données (500 rollouts)

On fait varier pas × données. **Fixe :** U-Net `[32,64,128]`, vision **mini-CNN**, état 19D. Succès % [IC95], lignes = pas, colonnes = nb démos (`grid_steps_x_data_500.png`) :

| pas \ N | 150 | 100 | 50 | 20 | 10 |
|---|---|---|---|---|---|
| **2** | 31.0 | 6.4 | 8.0 | 5.8 | 3.0 |
| **4** | 98.6 | 95.4 | 98.2 | 97.4 | 85.0 |
| **10** | 98.4 | 96.2 | 98.8 | 98.8 | **95.6** |
| **20** | 98.0 | 96.8 | 99.4 | 99.0 | 94.0 |
| **50** | 96.0 | 95.6 | 98.2 | 99.0 | 93.6 |

1. **2 pas s'effondre partout** (3–31 %) — le plancher est bien au-dessus.
2. **4 pas suffisent à pleines données** (N ≥ 50 : ~98 %) ; monter les pas n'aide pas (50 pas *baisse* même un peu à N=150).
3. **MAIS à données rares (N=10), 4 pas ne suffisent plus (85 %)** : passer à 10 pas récupère **+10 pts (95.6 %)**, puis plateau — sans atteindre le ~98 % des données abondantes.

→ Le **plancher de pas dépend des données** : 4 pas à pleines données, ~10 pas si peu de démos. Les pas **récupèrent une partie** du déficit de données mais ne **remplacent pas** les démos.

## Leçons clés (Partie 1)

1. **U-Net surdimensionné ×160** : 252 M → 1.6 M sans perte. Plancher de capacité = `[32,64,128]` ; en dessous, falaise (`[16,32,64]` → 2 %).
2. **Pas de diffusion : plancher dépendant des données.** 2 pas casse partout. 4 pas suffisent à pleines données ; ~10 pas nécessaires à données rares ; au-delà, rien à gagner.
3. **La vision était le vrai mur — mais facile *avec* la béquille.** Une fois le U-Net minimal, un mini-CNN 0.03 M suffit, car l'image est quasi redondante avec les coords cube. (La Partie 2 montre que sans coords, ce n'est plus vrai.)
4. **Lift est peu gourmand en données… jusqu'à un seuil** : ~20 démos ≈ 97 % (`[32,64,128]`), mais falaise à N=10 (85 %). Pour le bras réel, viser **≥ 20 démos**.
5. **Interaction capacité × données** (résultat clé) : à pleines données la taille du U-Net ne change rien ; à données rares (N=20) le gros U-Net sur-apprend (88 % vs 97.4 %). → petit U-Net = aussi bon et plus robuste.
6. **Le succès binaire est bruité — il faut assez de rollouts.** À 50 ép. (IC95 ±7 pts) les écarts disparaissent ; 500 rollouts appariés (±2 pts) révèlent les régimes. La **val-loss ne mesure pas la fonctionnalité** (juste l'ajustement aux démos) mais reste utile pour **prédire les falaises de capacité** et choisir le checkpoint.

## Détails techniques

**Protocole d'éval (Phase 0, figé pour tous les modèles)** : split train / val 50 **contigu** (val = démos 150–199, jamais entraînées ; train = préfixe `0..N-1` — contigu *obligé*, l'EpisodeAwareSampler indexe dans l'espace original). Rollout sur les 50 init states val + early-stop par modèle via val-loss. Split versionné : `results/runs/lift/phase4_split.json`.

**Protocole 500 rollouts** : 500 départs **figés et appariés** (générés par `env.reset()`, même distribution que les démos, jamais entraînés, sauvés dans `phase4_eval500.npy`), **arrêt au 1ᵉʳ succès** (~2.5× plus rapide, ne change pas le taux), **IC95 de Wilson** (robuste près de 100 %). IC95 ≈ ±7 pts à 50 ép., ±2 pts à 500.

⚠️ **Bug d'éval longue (corrigé)** : réutiliser un même env mujoco sur des centaines de rollouts dégrade le **renderer offscreen** → images pourries → faux échecs (un modèle 100 % tombe à 2 %). Correctif `lift_eval.rollout_eval_chunked` : recrée l'env tous les 50 épisodes + `env.env.close()`. Les évals ≤50 ép. étaient sous le seuil → non affectées.

**Entraînement** : Mac MPS (~16 h/12K steps avec ResNet, ~26 min avec le mini-CNN). Le mini-CNN a un save/load custom (`src/mini_cnn.py`).

<details><summary><b>Scripts & artefacts</b></summary>

`47` rollout · `48` val-loss · `52` sweep eval · `54/55` démos · `56` latence · `57` sweep pas · `58` grille latence×succès · `61/62` train/eval mini-CNN · `50` train+val-loss continue · `67` données 500 · `run_69_grid.sh`+`69`/`70` grille données×U-Net · `71`/`72` pas×données · `src/lift_eval.py`, `src/mini_cnn.py`.
Grilles : [`grid_data_x_unet_500.json`](../results/runs/lift/grid_data_x_unet_500.json) · [`grid_steps_x_data_500.json`](../results/runs/lift/grid_steps_x_data_500.json) · [sweep SUMMARY](../results/runs/lift/51_unet_sweep_eval/SUMMARY.md).
</details>

---

# Partie 2 — Vision pure (sans la béquille cube)

> **La même question, sans tricher.** On refait la grille **taille U-Net × données** mais en **vision pure 9D** (proprio seule + image agentview, **aucune coordonnée d'objet**), avec un encodeur **ResNet18**. Question : jusqu'où la vision pure va-t-elle, et **que coûte le retrait de la béquille** ?
>
> **Réponse : la vision pure atteint 99–100 % sur Lift.** La béquille n'était **pas** nécessaire pour résoudre la tâche — elle masquait un **besoin de capacité** (vision + U-Net). Sweet spot **`[128,256,512]` ResNet18 (28.6 M)** : **99.4 %** à pleines données.

## Grille taille U-Net × données (ResNet18, vision pure 9D, @ 10 pas, 500 rollouts)

On fait varier données × largeur U-Net. **Fixe :** vision **ResNet18**, **10 pas**, **état 9D** (proprio + image agentview, sans coords objet), early-stop par val-loss.
Succès % [IC95], lignes = U-Net (**params totaux**), colonnes = nb démos :

| U-Net × N | 150 | 100 | 50 | 20 | 10 |
|---|---|---|---|---|---|
| `[64,128,256]` (16 M) | 81.2 | 95.2 | 78.8 | 90.8 | 85.6 |
| **`[128,256,512]`** ⭐ (28.6 M) | **99.4** [98.3–99.8] | 92.6 [90.0–94.6] | **99.8** [98.9–100] | 94.4 [92.0–96.1] | 88.6 [85.5–91.1] |
| `[256,512,1024]` (76 M) | **98.8** [97.4–99.4]† | 94.6 [92.3–96.3] | 83.4 [79.9–86.4] | 94.6 [92.3–96.3] | 79.4 [75.6–82.7] |

**Sources** : cellules `[128,256,512]` et `[256,512,1024]` vérifiées (`results/runs/lift_visionpure_resnet/`) ; † `[256,512,1024]×150` = run 75 ; ligne `[64,128,256]` = run 73 + évals data-efficiency antérieures (non re-vérifiées). `best_step` des cellules grille = 3000–6000 (early-stop ; au-delà la val-loss remonte, ex. `[128,256,512]×150` : 0.072@6000 → 0.117@15000).

## Vision : mini-CNN vs ResNet18 — le mur que la béquille cachait

Même grille données × U-Net, mais on remplace l'encodeur par le **mini-CNN** (0.03 M) de la Partie 1 — test direct de « peut-on garder la vision triviale sans la béquille ? ». **Fixe :** vision **mini-CNN**, **10 pas**, **état 9D**. Succès % à 500 rollouts, colonnes = nb démos, **params U-Net seul** (`results/runs/lift_visionpure/`) :

| U-Net × N | 150 | 100 | 50 | 20 | 10 |
|---|---|---|---|---|---|
| `[32,64,128]` (1.6 M) | 47.2 | 42.2 | 46.6 | 59.2 | 65.0 |
| `[64,128,256]` (5 M) | 58.0 | 56.0 | 57.4 | 45.2 | 56.6 |
| `[128,256,512]` (17 M) | 62.6 | 81.2 | 32.8 | 51.0 | 16.2 💥 |

**Réponse : non.** Le mini-CNN **plafonne à 45–65 %** et part en chaos (la ligne `[128,256,512]` fait 62→81→33→51→16 % ; le 81.2 % @N=100 est un tirage chanceux, IC95 disjoints, pas une propriété). Agrandir le U-Net n'aide pas : sans bon encodeur visuel, le débruiteur n'a rien d'exploitable.

→ **C'est la preuve que la béquille cachait un problème de *vision*, pas de débruitage.** Avec coords cube, le mini-CNN suffisait car l'image était quasi inutile (l'état contenait la réponse). En vision pure, **toute l'info de la tâche passe par l'image** → il faut un encodeur capable (ResNet18). Le mini-CNN n'était pas « assez bon pour Lift », mais « assez bon pour Lift *quand on trichait* ».

## Leçons clés (Partie 2)

1. **La vision pure résout Lift (~99 %)** — le `~81 %` de run 73 (`[64,128,256]`) n'était pas un plafond mais une **limite de capacité** : doubler le U-Net (`[64,128,256]`→`[128,256,512]`) fait passer de **81.2 % à 99.4 %** à pleines données. L'objectif transférable est **atteignable**.
2. **Sweet spot `[128,256,512]` (28.6 M)** — 99.4 % @150, 99.8 % @50 ; monter à `[256,512,1024]` (76 M) **ne gagne rien** (98.8 %) et déstabilise (83.4 % @50) → plateau de capacité.
3. **Coût du retrait de la béquille** — l'optimum vision pure (ResNet18 + `[128,256,512]`, **28.6 M**) pèse **~17×** l'optimum « avec béquille » (1.65 M). Ce facteur combine **deux causes** : encodeur visuel plus gros (mini-CNN → ResNet18) **et** U-Net plus large (`[32,64,128]` → `[128,256,512]`) — sans la béquille, les deux deviennent nécessaires.
4. **Variance d'entraînement plus forte** — `[256,512,1024]` non-monotone (83.4 @50, 94.6 @20), `[128,256,512]` chute à 92.6 @100 entre deux 99 %. Sans l'ancrage des coords, chaque seed diverge plus → à données rares, **plusieurs seeds** sont nécessaires pour conclure.

→ **Bilan transférabilité** : sur Lift, viser **ResNet18 + `[128,256,512]` (~29 M)** donne ~99 % en vision pure — toujours **÷9 vs baseline 263.7 M**, mais loin du ÷160 « optimiste » qui dépendait de la béquille. Le vrai coût de compression transférable est **modéré, pas extrême**.

## Périmètre — ce qui n'a PAS été refait en vision pure

Toute la grille vision pure est à **10 pas de diffusion figés**. Deux axes de la Partie 1 n'ont **pas** d'équivalent vision pure (nécessiteraient de nouveaux entraînements) :
- **Pas de diffusion × données** — le plancher de pas pourrait différer quand toute l'info passe par l'image.
- **Latence × succès** (pas × U-Net) — établie seulement avec coords objet.

→ Les conclusions **pas / latence** de la Partie 1 sont à considérer comme **non confirmées en vision pure**.

---

## Suite

Limites cartographiées **en sim avec coords objet** (capacité, pas, vision, données) **et en vision pure** pour l'axe capacité × données + vision.

Frontières suivantes :
- **Compléter la vision pure** — refaire les grilles **pas × données** et **latence × succès** sans béquille (cf. *Périmètre*).
- **Transférabilité réelle** — phase 5 (vision pure, sans `cube_pos`/`can_pos`). Récit v1 archivé : [`CAN_archive.md`](CAN_archive.md).
- **Tâche plus dure** — Can travaillé en phase 5 v1, à reprendre proprement (convergence + sélection ckpt) en v2. Square / Tool Hang en réserve.
