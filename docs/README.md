# Documentation — index & ordre de lecture

Les récits ci-dessous racontent le **raisonnement** du projet, phase par phase. À lire dans l'ordre pour comprendre la logique ; chacun est aussi autonome. **Même structure pour chacun** : Contexte · Le parcours · Résultats · Leçons clés · Détails techniques · Suite.

| # | Doc | Phase | En une phrase |
|---|---|---|---|
| 1 | [`PUSHT.md`](PUSHT.md) | PushT (1-2) | Enquête « loss vs performance » : pourquoi une loss basse ne donne pas un bon robot, et comment la **formulation de la sortie** (régression → classification discrète) change tout. |
| 2 | [`LIFT.md`](LIFT.md) | Lift (3) | Du **behavior cloning raté** (0 %) à **100 %** avec une **Diffusion Policy**. |
| 3 | [`COMPRESSION.md`](COMPRESSION.md) | Compression & limites (4) | **Rétrécir** (÷160 params, ÷21 latence) sans perdre la perf + **grilles 500 rollouts** (données × U-Net, pas × données) : à données rares un gros U-Net sur-apprend, le plancher de pas dépend des données. |
| 4 | [`CAN.md`](CAN.md) | Can — capacité (5) | **Le plafond venait de la capacité, pas de la vision pure.** Vision pure (sans coords objet) bloquée ~75 % → en cumulant ResNet34 + gros U-Net : **94,8 %@500** (61 M). Vision et décodeur paient et se cumulent. |
| 5 | [`METHODOLOGIE.md`](METHODOLOGIE.md) | Méthodologie (5) | **Peut-on faire confiance à nos mesures ?** 500 rollouts (pas 50) ; le succès **ne se lisse à aucune échelle** (interpoler est invalide) ; **loss au plancher ≠ convergé** (mini-CNN 2 %→74 %) ; aucun proxy offline ne remplace le rollout (`grad_norm` = *quand*, pas *combien*). Protocole détaillé : [`PHASE5_PROTOCOLE.md`](PHASE5_PROTOCOLE.md). |
| — | [`CAN_archive.md`](CAN_archive.md) | Can (5 v1) 🗄️ | **v1 archivée.** Démarrage avec coords objet → pivot vision pure ; runs sous-entraînés (10k ≠ converger), val_loss bruité. Bornes inférieures + observations (effondrement wrist OOD). |
| 6 | [`CAMERA.md`](CAMERA.md) | Caméra / sim-to-real (6) | **Robustesse au déplacement caméra.** Le modèle 74 % s'effondre à 13 % dès 2 cm/2°. L'augmentation caméra (jitter 10 cm/10°) ne marche qu'**avec de la capacité** : mini-CNN 0.03 M → 3 %, ResNet18 11 M → 18 %. La loss est aveugle à cet écart. |

## Notes transverses

| Doc | En une phrase |
|---|---|
| [`PERF_TEMPS_ENTRAINEMENT.md`](PERF_TEMPS_ENTRAINEMENT.md) | **Modèle calibré du temps d'entraînement (M1).** Le temps suit les FLOPs de la **vision** (caméras × backbone × résolution²), pas le compte de params ; U-Net quasi gratuit. + la **falaise mémoire** : la résolution 160² fait swapper les 16 Go (step ×440). |
| [`CONVERGENCE.md`](CONVERGENCE.md) | **Convergence = succès (rollouts) vs steps.** La capacité accélère *et* relève la convergence (ResNet34 : 94 % à 20k ; mini-CNN : 76-81 % à 46-73k). Courbe en **sigmoïde**, décollage bien après la loss. Étude **auto-extensible** : tout nouveau `rollouts_500.csv` rejoint le graphe. |
| [`SCHEDULE.md`](SCHEDULE.md) | **LR constant vs cosine.** Un cosine annealé à ~0 **gèle** le modèle (faux plateau : « loss parfaite, 2 % » cachait +72 pts). Constant = décolle plus tôt + **stoppable au plateau** ; cosine = pic un peu plus haut mais budget à deviner. → on entraîne en **constant**. |

## Où sont les résultats chiffrés

- **Tableau maître de la compression** (le plus complet) : [`../results/runs/lift/51_unet_sweep_eval/SUMMARY.md`](../results/runs/lift/51_unet_sweep_eval/SUMMARY.md) — sweep U-Net, plafond démos, latence × succès.
- **Par run** : `../results/runs/<tâche>/<run>/run_info.md` (fiche lisible) + courbes PNG + JSON.
- **Logs bruts** : `../results/logs/<tâche>/`.

## Conventions

- `experiments/<tâche>/NN_nom.py` — une expérience = un fichier numéroté (l'ordre = la chronologie).
- Les **poids de modèles et vidéos restent locaux** (gitignorés) ; seuls les résumés, courbes et JSON sont versionnés.
- Voir aussi `../README.md` (porte d'entrée + vue d'ensemble).
