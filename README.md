# lerobot-experiments

Espace d'apprentissage et d'expérimentation autour de l'**imitation learning** pour le contrôle robotique. Le but : comprendre les différentes approches (MLP, RNN, Transformer, Diffusion Policy, etc.) en les testant sur des datasets publics, avant de les appliquer à un bras 5 axes + pince custom.

> ⚠️ Projet pédagogique — pas de production, pas de robot cible immédiat. L'accent est mis sur la **compréhension** (essayer plusieurs techniques, comparer, mesurer) plutôt que sur la performance maximale.

## État actuel

- **Phase 1-2 — PushT (cube 2D à pousser)** : best 46.5% coverage (run 24, image + agent_pos). Lecture clé : CE+résidu sur bins discrets rattrape la perf "image+pos" sans agent_pos (run 31, 44.9%).
- **Phase 3 — Robomimic Lift (bras Panda 7-DoF + cube)** : best **70% success** (run 44, ResNet18 trainable + anti-overfit), confirmé **66% sur 200 épisodes**.

### Tableau Phase 3 (Lift)

| Run | Setup principal | Best success | Final 200 eps |
|---|---|---|---|
| 33 | MLP state seul, MSE | 0% | 0% |
| 34 | MLP state seul, CE+résidu | 0% | 0% |
| 35 | image+state, MSE | 28% | 18% |
| 36 | image+state, CE+résidu | 34% | 14% |
| 37 | "Robomimic-clone" (MLP[1024]+GMM K=5, chunk=1) | 28% | 26% |
| 38-39 | + image augmentation (échec) | 16-20% | 6-10% |
| 40 | + anti-overfit (Dropout+LN+AdamW+LS) | 48% | 35% |
| 41 | run 40 prolongé 500 epochs | 50% | 27% |
| 42 | + **ResNet trainable** sur MPS | 66% (crash NaN après) | — |
| 43 | + FrozenBatchNorm2d (tue le 66%) | 30% | 10% |
| **44** | **Run 42 + NaN guard + save best immédiat** | **70%** ⭐ | **66%** ⭐ |
| 45 | DINOv2-small **frozen** au lieu de ResNet | 44% | 32% — comparable à frozen ResNet (run 40) |

**Leçons clés Phase 3** :
1. **Sans image → 0%**. Avec image features (ResNet18) → jump à 28%+.
2. **Dégeler le ResNet (LR 1e-5) est THE move** : +30pts (34% → 66%). Frozen ResNet = ImageNet stats → mismatch avec scène Lift.
3. **Anti-overfit obligatoire sur petit dataset** : Dropout 0.4 + LayerNorm + AdamW(wd=5e-4) + label_smoothing 0.1.
4. **MPS Apple Silicon** : `.contiguous()` après fancy indexing crucial pour éviter view-errors backward.
5. **BN trainable = breakthrough**, mais NaN possible → NaN guard + save best à disque immédiatement.
6. **DINOv2 frozen ≈ ResNet frozen** sur ce setup : la scène Lift étant visuellement simple, la richesse du pré-entraînement self-sup (DINOv2 sur 142M images) n'apporte pas vs ImageNet classique. C'est l'**adaptabilité** (backbone trainable) qui compte, pas la qualité du pré-entraînement.

## Pistes non explorées (pour une session future)

- **Foundation models robotiques** (SmolVLA, Octo, OpenVLA) : exploration tentée fin de phase 3, abandonnée au profit de finir proprement run 44. Demande conversion data Robomimic → LeRobot format + GPU plus puissant pour fine-tuner 450M+ params.
- **ACT (Action Chunking Transformer)** via lerobot-train : recette officielle LeRobot, pas testée.
- **Diffusion Policy** sur Colab GPU : prévu dans notebook, pas exécuté.
- **Receding horizon** sur run 44 : exécuter k=8 actions puis re-prédire (gratuit, juste inférence).

## Setup machine

Deux environnements coexistent :
- **Mac** : analyse, planification, design d'expériences (ce repo)
- **Linux + GPU** : exécution des trainings lourds, futur contrôle du robot

Workflow : édit/plan sur Mac → `git push` → `git pull` sur Linux → exécution → résultats commit + push → `git pull` sur Mac → analyse.

## Installation

```bash
python -m venv venv312
source venv312/bin/activate
pip install -r requirements.txt
```

Pour LeRobot avec PushT inclus :
```bash
pip install 'lerobot[pusht]'
```

Lancer un run en mode unbuffered (pour suivre la progression en live via `tail -f`) :
```bash
# Phase PushT
venv312/bin/python -u experiments/pusht/31_discrete_ce.py 2>&1 | tee results/logs/pusht/run_31.log

# Phase Lift (Robomimic)
venv312/bin/python -u experiments/lift/33_lift_mlp_baseline.py 2>&1 | tee results/logs/lift/run_33.log
```

## Structure du projet

```
.
├── src/                       # Bibliothèque commune utilisée par les expériences
│   ├── tracker.py             # Sauvegarde/lecture standardisée des runs (results/all_runs.jsonl)
│   ├── benchmark.py           # Comptage de paramètres, FLOPs, taille modèle
│   ├── data_cache.py          # Chargement + cache des données PushT (évite de re-télécharger)
│   └── visualize.py           # Génération des graphes loss/coverage
│
├── experiments/               # Scripts numérotés — un fichier par expérience indépendante
│   ├── pusht/                 # Phase 1 + 2 : tâche PushT (cube 2D à pousser)
│   │   ├── 08_pretrained_cnn.py        # CNN ResNet pré-entraîné vs from scratch
│   │   ├── 09_action_chunking.py       # Action chunking : prédire N actions futures
│   │   ├── 10_transformer.py           # Premier transformer pour PushT
│   │   ├── 17_rnn_chunk.py             # RNN avec chunk d'actions
│   │   ├── 24_precomputed_features.py  # Transformer + features ResNet pré-calculées (image+pos, 46.5%)
│   │   ├── 25_resnet_mlp.py            # MLP simple sur features ResNet
│   │   ├── 26_long_training.py         # 10k epochs avec dropout — voir si l'overfit aide (réponse: non)
│   │   ├── 27_position_history.py      # Historique de positions en entrée (hist=5/10)
│   │   ├── 28_baseline_no_history.py   # Baseline propre : MLP sans history
│   │   ├── 29_image_only.py            # ┐ Série "enquête loss vs coverage"
│   │   ├── 30_mdn.py                   # │  29 = MSE image seule
│   │   ├── 31_discrete_ce.py           # │  30 = MDN (mixture density network)
│   │   ├── 32_delta_grid.py            # │  31 = CE + résidu, bins absolus (best image-only : 44.9%)
│   │   │                               # ┘  32 = CE + résidu, bins delta (contrôle en vitesse)
│   │   ├── check_eval_variance.py      # Mesure la variance des évals
│   │   ├── check_eval_variance_big.py  # Même chose, 10×200 épisodes
│   │   └── archive/                    # Anciennes expériences (00-07)
│   │
│   └── lift/                  # Phase 3 : bras Panda 7-DoF, Robomimic Lift
│       └── 33_lift_mlp_baseline.py     # Baseline MLP, state→chunk20×7 (0% success, confirme mode collapse MSE)
│
├── notebooks/                 # Notebooks Jupyter pour Colab/Kaggle
│   ├── pusht_diffusion_colab.ipynb  # Diffusion Policy sur Colab GPU (T4)
│   └── pusht_diffusion_kaggle.ipynb # Idem sur Kaggle (sessions 9h)
│
├── results/                   # Résultats des runs — partiellement versionnés (voir .gitignore)
│   ├── all_runs.jsonl         # ✅ Index versionné de TOUS les runs (loss, coverage, params...)
│   ├── runs/
│   │   ├── pusht/<exp_id>/    # Runs PushT (phases 1 & 2)
│   │   └── lift/<exp_id>/     # Runs Lift (phase 3+)
│   │       ├── run_info.md        # ✅ Versionné — résumé lisible
│   │       ├── losses.png         # ✅ Versionné — courbes loss/coverage
│   │       ├── model_config.json  # ✅ Versionné — hyperparamètres
│   │       ├── model.pt           # ❌ Non versionné — poids du modèle (lourd)
│   │       └── ep*.mp4            # ❌ Non versionné — vidéos d'éval
│   ├── logs/
│   │   ├── pusht/             # Logs textuels PushT
│   │   └── lift/              # Logs textuels Lift
│   └── comparisons/           # Graphes croisés entre plusieurs runs
│
├── data_cache/                # ❌ Non versionné — features ResNet pré-calculées + données PushT cachées
│
├── docs/
│   └── JOURNEY.md             # Récit narratif et leçons techniques de la série 29-32
│
├── tests/                     # Tests unitaires (à étoffer)
└── requirements.txt
```

## Format des expériences

Chaque script dans `experiments/` :
1. Définit ses hyperparamètres en haut du fichier (constants en majuscules)
2. Charge les données (souvent via `data_cache.py` pour éviter de re-télécharger)
3. Construit + entraîne le modèle
4. Évalue en simulation PushT
5. Appelle `src.tracker.save_run(...)` qui ajoute une ligne à `results/all_runs.jsonl` et écrit `run_info.md` + `losses.png` dans `results/runs/<exp_id>/`

Cela permet de comparer toutes les expériences entre elles avec un format unifié.

## Vue d'ensemble des découvertes

### Phase 1 — exploration (runs 08 à 28)

- **L'historique de positions n'aide pas beaucoup** (hist=5 marginalement mieux que hist=0, hist=10 régresse) — voir runs `27_*` vs `28_*`
- **L'entraînement long n'apporte rien** au-delà de ~1000 epochs sur cette archi — la loss continue de baisser mais le coverage stagne / dérive (cf. run `26_long_training`, 10000 epochs)
- **Le bruit d'évaluation est élevé** : à 50 épisodes, écart-type ±3.7pt ; à 200 épisodes, ±1.7pt. Voir `check_eval_variance_big.py`. Conséquence : ne pas sur-interpréter les fluctuations de coverage entre runs/epochs proches.
- **Le Transformer (run 24) bat largement les MLP** (46% vs 27% coverage) — l'archi compte plus que la quantité d'historique.

### Phase 2 — enquête "loss vs coverage" (runs 29 à 32)

Détails dans [`docs/JOURNEY.md`](docs/JOURNEY.md). Conclusions principales :

- **La loss MSE est intrinsèquement décorrélée du coverage sur tâche multimodale** : sur PushT, optimiser MSE pousse le modèle à prédire la **moyenne** entre les différentes actions humaines valides → action morte. Les runs les plus bas en loss (`07_data_filtering` : MSE 0.0012) ont parfois le **pire** coverage (4.6%).
- **`agent_pos` apporte ~15pts mais n'est pas indispensable** : passer de "image + agent_pos" (46.5%) à "image seule" (31.8% avec MSE) coûte cher, **mais** la perte est presque entièrement récupérée en changeant la **formulation de la sortie** (44.9% avec bins discrets, sans agent_pos).
- **La classification discrète bat la régression sur tâche multimodale** : transformer "prédire (x, y) continus" en "choisir parmi 64 bins + résidu" évite le mode collapse → premier succès du projet (3/200 à epoch 75 de run 31).
- **La loss CE+résidu est mieux corrélée au coverage que MSE ou NLL/MDN** : sur run 31, baisser la loss baisse aussi le coverage. Sur runs MSE (24) et MDN (30), la loss continue de baisser après le pic de coverage.
- **Limite des bins absolus** : prédictions de chunk parallèle → téléportations possibles entre timesteps consécutifs (saccadé visible). Solution testée en run 32 : bins de **deltas** (contrôle en vitesse au lieu de position).
- **Run 32 — leçon inattendue** : passer aux bins de delta **dégrade** le coverage (44.9% → 21.3%). Hypothèse principale : prédire un delta nécessite de connaître l'état actuel (« où je suis » + « ce que je viens de commander »), info que le ResNet18 gelé n'extrait pas bien depuis l'image. Le delta est plus puissant en théorie mais plus exigeant en pratique — il faudrait soit donner agent_pos en input, soit un décodeur autoregressif, soit dégeler le ResNet.

## Prochaines pistes

1. **Décodeur autoregressif** (= BeT propre) : chaque action prédite voit les précédentes → résout structurellement le saccadé
2. **Diffusion Policy** via notebook Colab/Kaggle (référence SOTA, ~84-91% success en littérature)
3. **VQ-BeT** : VQ-VAE pour apprendre les tokens d'action au lieu d'un k-means basique
4. **Receding horizon** : exécuter k=8 actions puis re-prédire (gratuit, juste de l'inférence)
5. **Améliorations transverses** : unfreeze ResNet + GroupNorm, EMA weights, smoothness penalty

## Liens utiles

- LeRobot docs : https://huggingface.co/docs/lerobot
- Diffusion Policy paper : https://arxiv.org/abs/2303.04137
- BeT paper : https://arxiv.org/abs/2206.11251
- VQ-BeT paper : https://arxiv.org/abs/2403.03181
- Dataset PushT : https://huggingface.co/datasets/lerobot/pusht
