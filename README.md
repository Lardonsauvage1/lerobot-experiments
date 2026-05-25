# lerobot-experiments

Projet d'apprentissage de l'**imitation learning** pour le contrôle robotique : on teste plusieurs approches (MLP, RNN, Transformer, **Diffusion Policy**) sur des tâches publiques, en mesurant et comparant, avant de viser un bras 5 axes + pince custom.

> ⚠️ **Projet pédagogique.** L'objectif est de **comprendre** (essayer, comparer, mesurer), pas la performance brute. La doc raconte donc le *raisonnement*, pas juste les résultats.

## Le parcours en un coup d'œil

| Phase | Tâche | Meilleur résultat | Leçon clé |
|---|---|---|---|
| **1-2** | PushT (cube 2D à pousser) | 46.5 % coverage | La **formulation de la sortie** compte plus que les features (la classification discrète rattrape « image+position ») ; **loss basse ≠ bonne perf**. → [`docs/PUSHT.md`](docs/PUSHT.md) |
| **3** | Robomimic Lift (bras Panda 7-DoF) | **100 % succès** (Diffusion Policy) | **Diffusion Policy ≫ behavior cloning** sur le multimodal (100 % vs 70 %) ; sans image → 0 %. → [`docs/LIFT.md`](docs/LIFT.md) |
| **4** | Compression & limites du modèle | **~99 % à ÷160 params / ÷21 latence**, et **~20 démos suffisent** | U-Net surdimensionné ×160 ; ResNet18 → mini-CNN 0.03 M ; Lift peu gourmand (≥20 démos) ; à données rares un **gros U-Net sur-apprend** (grille 500 rollouts, IC95 Wilson). → [`docs/COMPRESSION.md`](docs/COMPRESSION.md) |

## Résultats phares

- **Lift résolu à 100 %** avec une Diffusion Policy entraînée sur 150 démos (run 46/47).
- **Compression** : ÷160 params et ÷21 latence **sans perte de succès** —

  | | Modèle | Params | Latence/décision | Succès |
  |---|---|---|---|---|
  | Départ | baseline + ResNet18 @ 10 pas | 263.7 M | 938 ms | 100 % (50 ép.) |
  | **Final** | `[32,64,128]` + mini-CNN @ 4 pas | **1.65 M** | **43.7 ms** | **98.6 %** (500 rollouts) |

  Le modèle final est **à égalité statistique** avec des U-Nets 3–10× plus gros (grille 500 rollouts, IC95 Wilson). Tableaux complets : [`docs/COMPRESSION.md`](docs/COMPRESSION.md) (grille données × U-Net) + [`results/runs/lift/51_unet_sweep_eval/SUMMARY.md`](results/runs/lift/51_unet_sweep_eval/SUMMARY.md) (sweep + latence).
- Méthode : protocole d'éval propre (train 150 / val 50 figé, métriques continues), **éval finale à 500 rollouts appariés + IC95 de Wilson** (le succès sur 50 ép. était dans le bruit à ±7 pts), env recréé par tranche (anti-dégradation renderer).

## Naviguer dans le repo

**📖 Pour comprendre la logique** (récits, à lire dans l'ordre) :
1. [`docs/PUSHT.md`](docs/PUSHT.md) — PushT : enquête « loss vs performance », la formulation de sortie.
2. [`docs/LIFT.md`](docs/LIFT.md) — Lift : du behavior cloning raté (0 %) à 100 % en Diffusion Policy (+ les 5 bugs d'eval, le sim-to-real).
3. [`docs/COMPRESSION.md`](docs/COMPRESSION.md) — compression & limites : taille U-Net, pas de diffusion, vision, et efficacité données.

Index complet de la doc : [`docs/README.md`](docs/README.md).

**📊 Résultats détaillés** : `results/runs/<tâche>/<run>/` → `run_info.md` (fiche lisible) + courbes.
- **Tableau maître de la compression** : [`results/runs/lift/51_unet_sweep_eval/SUMMARY.md`](results/runs/lift/51_unet_sweep_eval/SUMMARY.md).

**🧪 Code** : `experiments/<tâche>/` = scripts numérotés (1 expérience = 1 fichier) · `src/` = bibliothèque commune.

**📁 Le reste** : `results/logs/` (logs des runs) · `notebooks/` (Colab/Kaggle) · `data_cache/` (données, non versionné).

## Setup & lancer

```bash
python -m venv venv312 && source venv312/bin/activate
pip install -r requirements.txt          # + 'lerobot[pusht]' pour PushT

# lancer une expérience (mode unbuffered pour suivre en live)
venv312/bin/python -u experiments/lift/47_phase0_eval.py --checkpoint <ckpt>
```

> **Setup machine** : Mac (M1, analyse/dev) + une box Linux+GPU pour les trainings lourds. Workflow : édit/plan sur Mac → `git push` → `pull` sur la box → exécution → résultats → `pull` sur Mac.

## Structure (compacte)

```
experiments/<tâche>/   scripts numérotés (pusht/, lift/) + archive/
src/                   lib commune (lift_data, lift_eval, lift_to_lerobot, tracker, benchmark…)
results/runs/<tâche>/  fiches + courbes par run (poids/vidéos = locaux, non versionnés)
results/logs/          logs textuels
docs/                  récits par phase (PUSHT, LIFT, COMPRESSION) + index
notebooks/             Colab/Kaggle
```

## Liens utiles

- LeRobot : https://huggingface.co/docs/lerobot · Diffusion Policy (paper) : https://arxiv.org/abs/2303.04137
- Dataset PushT : https://huggingface.co/datasets/lerobot/pusht · Robomimic : https://robomimic.github.io
