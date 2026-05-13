# lerobot-experiments

Espace d'apprentissage et d'expérimentation autour de l'**imitation learning** pour le contrôle robotique. Le but : comprendre les différentes approches (MLP, RNN, Transformer, Diffusion Policy, etc.) en les testant sur des datasets publics, avant de les appliquer à un bras 5 axes + pince custom.

> ⚠️ Projet pédagogique — pas de production, pas de robot cible immédiat. L'accent est mis sur la **compréhension** (essayer plusieurs techniques, comparer, mesurer) plutôt que sur la performance maximale.

## État actuel

- **Tâche** : [PushT](https://huggingface.co/datasets/lerobot/pusht) (LeRobot) — pousser un bloc en T sur une zone cible
- **Best coverage** : **44.9%** (Transformer 2L + bins discrets sur image seule, run `31_discrete_ce`)
- **Best run "absolu"** : 46.5% (`24_precomputed_features`, image + agent_pos)
- **Premier succès du projet** : run 31, 3/200 épisodes (1.5%) au checkpoint epoch 75 avec décodage argmax
- **Lecture clé** : passer de MSE-régression à classification (CE) sur bins discrets a quasi-rattrapé la perf "image+pos" en travaillant uniquement sur la formulation de la sortie

### Série la plus récente — enquête loss ↔ coverage

Les runs `29` à `32` répondent à la question : **pourquoi la loss ne reflète-t-elle pas la performance en simulation ?** Voir [`docs/JOURNEY.md`](docs/JOURNEY.md) pour le récit complet et les conclusions structurelles.

| Run | Loss | Input | Coverage |
|---|---|---|---|
| 24 | MSE | image + pos | 46.5% |
| 29 | MSE | **image seule** | 31.8% |
| 30 | NLL (MDN K=5) | image seule | 35.4% (best 40.9% @ epoch 75) |
| 31 | CE + résidu (bins absolus) | image seule | **44.9% argmax** (3/200 succès !) |
| 32 | CE + résidu (bins delta) | image seule | 21.3% argmax — moins bon, voir leçon |

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
venv312/bin/python -u experiments/31_discrete_ce.py 2>&1 | tee results/logs/run_31.log
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
│   ├── 08_pretrained_cnn.py        # CNN ResNet pré-entraîné vs from scratch
│   ├── 09_action_chunking.py       # Action chunking : prédire N actions futures
│   ├── 10_transformer.py           # Premier transformer pour PushT
│   ├── 17_rnn_chunk.py             # RNN avec chunk d'actions
│   ├── 24_precomputed_features.py  # Transformer + features ResNet pré-calculées (image+pos, 46.5%)
│   ├── 25_resnet_mlp.py            # MLP simple sur features ResNet
│   ├── 26_long_training.py         # 10k epochs avec dropout — voir si l'overfit aide (réponse: non)
│   ├── 27_position_history.py      # Historique de positions en entrée (hist=5/10)
│   ├── 28_baseline_no_history.py   # Baseline propre : MLP sans history, eval coverage tous les 200 epochs
│   │
│   ├── 29_image_only.py            # ┐ Série "enquête loss vs coverage"
│   ├── 30_mdn.py                   # │  29 = MSE image seule
│   ├── 31_discrete_ce.py           # │  30 = MDN (mixture density network)
│   ├── 32_delta_grid.py            # │  31 = CE + résidu, bins absolus (best image-only : 44.9%)
│   │                               # ┘  32 = CE + résidu, bins delta (contrôle en vitesse)
│   ├── check_eval_variance.py      # Mesure la variance des évals (combien d'épisodes nécessaires ?)
│   ├── check_eval_variance_big.py  # Même chose, 10×200 épisodes
│   └── archive/                    # Anciennes expériences (00-07)
│
├── notebooks/                 # Notebooks Jupyter pour Colab/Kaggle
│   ├── pusht_diffusion_colab.ipynb  # Diffusion Policy sur Colab GPU (T4)
│   └── pusht_diffusion_kaggle.ipynb # Idem sur Kaggle (sessions 9h)
│
├── results/                   # Résultats des runs — partiellement versionnés (voir .gitignore)
│   ├── all_runs.jsonl         # ✅ Index versionné de TOUS les runs (loss, coverage, params...)
│   ├── runs/<exp_id>/         # Un dossier par run
│   │   ├── run_info.md        # ✅ Versionné — résumé lisible
│   │   ├── losses.png         # ✅ Versionné — courbes loss/coverage
│   │   ├── model_config.json  # ✅ Versionné — hyperparamètres
│   │   ├── model.pt           # ❌ Non versionné — poids du modèle (lourd)
│   │   └── ep*.mp4            # ❌ Non versionné — vidéos d'éval
│   ├── logs/                  # ✅ Versionné — logs textuels des runs (utile en relecture)
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
