# Documentation — index & ordre de lecture

Les récits ci-dessous racontent le **raisonnement** du projet, phase par phase. À lire dans l'ordre pour comprendre la logique ; chacun est aussi autonome. **Même structure pour chacun** : Contexte · Le parcours · Résultats · Leçons clés · Détails techniques · Suite.

| # | Doc | Phase | En une phrase |
|---|---|---|---|
| 1 | [`PUSHT.md`](PUSHT.md) | PushT (1-2) | Enquête « loss vs performance » : pourquoi une loss basse ne donne pas un bon robot, et comment la **formulation de la sortie** (régression → classification discrète) change tout. |
| 2 | [`LIFT.md`](LIFT.md) | Lift (3) | Du **behavior cloning raté** (0 %) à **100 %** avec une **Diffusion Policy**. |
| 3 | [`COMPRESSION.md`](COMPRESSION.md) | Compression & limites (4) | **Rétrécir** (÷160 params, ÷21 latence) sans perdre la perf + **grilles 500 rollouts** (données × U-Net, pas × données) : à données rares un gros U-Net sur-apprend, le plancher de pas dépend des données. |
| 4 | [`CAN.md`](CAN.md) | Can (5) ⏳ | **Tâche plus dure** (pick-and-place : saisir + transporter + déposer) : les conclusions de compression généralisent-elles ? *En cours.* |

## Où sont les résultats chiffrés

- **Tableau maître de la compression** (le plus complet) : [`../results/runs/lift/51_unet_sweep_eval/SUMMARY.md`](../results/runs/lift/51_unet_sweep_eval/SUMMARY.md) — sweep U-Net, plafond démos, latence × succès.
- **Par run** : `../results/runs/<tâche>/<run>/run_info.md` (fiche lisible) + courbes PNG + JSON.
- **Logs bruts** : `../results/logs/<tâche>/`.

## Conventions

- `experiments/<tâche>/NN_nom.py` — une expérience = un fichier numéroté (l'ordre = la chronologie).
- Les **poids de modèles et vidéos restent locaux** (gitignorés) ; seuls les résumés, courbes et JSON sont versionnés.
- Voir aussi `../README.md` (porte d'entrée + vue d'ensemble).
