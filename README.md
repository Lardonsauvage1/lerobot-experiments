# lerobot-experiments

Espace d'apprentissage et d'expérimentation autour de l'**imitation learning** pour le contrôle robotique. Le but : comprendre les différentes approches (MLP, RNN, Transformer, Diffusion Policy, etc.) en les testant sur des datasets publics, avant de les appliquer à un bras 5 axes + pince custom.

> ⚠️ Projet pédagogique — pas de production, pas de robot cible immédiat. L'accent est mis sur la **compréhension** (essayer plusieurs techniques, comparer, mesurer) plutôt que sur la performance maximale.

## État actuel

- **Tâche** : [PushT](https://huggingface.co/datasets/lerobot/pusht) (LeRobot) — pousser un bloc en T sur une zone cible
- **Best coverage** atteint : ~46% (Transformer 2L, run `24_precomputed_features`)
- **Best success rate** : 0% — on n'a pas encore réussi à finaliser la tâche, en partie à cause d'archis non-multimodales (MLP collapse to mean)
- **Prochain step** : Diffusion Policy via `lerobot-train` (référence SOTA sur PushT, ~84-91% success en littérature)

## Depuis août 2026 : le bras réel (Roby)

L'état ci-dessus (PushT) date d'avril 2026. Le travail s'est ensuite porté sur le vrai bras
(Diffusion Policy LeRobot, tâche « pomme », 2 caméras). Ce dépôt en garde la partie
**entraînement et recherche** ; le code qui pilote le robot (inférence, garde, panneaux,
entraînement XPU `roby_train_xpu.py`) est dans
[`roby-le-gentil-robot`](https://github.com/Lardonsauvage1/roby-le-gentil-robot), `tools/pc/`.

- `experiments/real/entrainements_2026-09/` : lanceurs et configs des modèles essayés sur le
  bras en septembre 2026, avec la correspondance config → run.
- `src/can_occlusion.py` : banc « Can occluded » (simulation) qui reproduit le mode d'échec
  dominant du robot réel (le bras masque la cible).
- `src/camp.py` : CAMP-lite, mémoire compressée de l'historique d'actions (d'après
  arXiv 2606.21188).
- `experiments/lift/50_train_valloss.py` : script d'entraînement LeRobot modifié (perte de
  validation).

Non versionnés : `outputs/`, `results/`, `data_cache/`, poids (`*.safetensors`).

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
│   ├── 08_pretrained_cnn.py   # CNN ResNet pré-entraîné vs from scratch
│   ├── 09_action_chunking.py  # Action chunking : prédire N actions futures
│   ├── 10_transformer.py      # Premier transformer pour PushT
│   ├── 17_rnn_chunk.py        # RNN avec chunk d'actions
│   ├── 24_precomputed_features.py  # Transformer + features ResNet pré-calculées (best run actuel)
│   ├── 25_resnet_mlp.py       # MLP simple sur features ResNet
│   ├── 26_long_training.py    # 10k epochs avec dropout — voir si l'overfit aide
│   ├── 27_position_history.py # Historique de positions en entrée (hist=5/10)
│   ├── 28_baseline_no_history.py  # Baseline propre : MLP sans history, eval coverage tous les 200 epochs
│   ├── check_eval_variance.py # Mesure la variance des évals (combien d'épisodes nécessaires ?)
│   ├── check_eval_variance_big.py  # Même chose, 10×200 épisodes
│   └── archive/               # Anciennes expériences (00-07)
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
│   └── comparisons/           # Graphes croisés entre plusieurs runs
│
├── data_cache/                # ❌ Non versionné — features ResNet pré-calculées + données PushT cachées
│
├── docs/                      # Documentation libre (notes pédagogiques, etc.)
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

- **L'historique de positions n'aide pas beaucoup** (hist=5 marginalement mieux que hist=0, hist=10 régresse) — voir runs `27_*` vs `28_*`
- **L'entraînement long n'apporte rien** au-delà de ~1000 epochs sur cette archi — la loss continue de baisser mais le coverage stagne / dérive (cf. run `26_long_training`)
- **Le bruit d'évaluation est élevé** : à 50 épisodes, écart-type ±3.7pt ; à 200 épisodes, ±1.7pt. Voir `check_eval_variance_big.py`. Conséquence : ne pas sur-interpréter les fluctuations de coverage entre runs/epochs proches.
- **Le Transformer (run 24) bat largement les MLP** (46% vs 27% coverage) — l'archi compte plus que la quantité d'historique.
- **Les MLP/Transformer regression collapse vers la moyenne** sur PushT qui est multimodal → pour viser le SOTA il faut une archi générative (Diffusion Policy, VQ-BeT, etc.).

## Prochaines pistes

Voir aussi `.claude/projects/.../memory/ideas_future.md` pour la liste complète.

1. **Diffusion Policy** via `lerobot-train` (commande prête, perfs SOTA attendues)
2. **VQ-BeT** (alternative ~10x plus rapide à l'inférence que Diffusion)
3. **ACT** (encoder-decoder + chunking)
4. **Améliorations transverses** : unfreeze ResNet + GroupNorm, receding horizon, EMA weights, checkpoint le best coverage

## Liens utiles

- LeRobot docs : https://huggingface.co/docs/lerobot
- Diffusion Policy paper : https://arxiv.org/abs/2303.04137
- Dataset PushT : https://huggingface.co/datasets/lerobot/pusht
