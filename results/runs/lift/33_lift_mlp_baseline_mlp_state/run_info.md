# lift/33_lift_mlp_baseline — MLP 19 → 256 → 256 → 140 (chunk=20 × action=7)

## Expérience
Phase 3 — baseline MLP imitation sur Robomimic Lift. Premier transfert depuis PushT.

## Modèle
Type : MLP 19 → 256 → 256 → 140 (chunk=20 × action=7)
Paramètres : 106,892
Taille : 0.41 Mo

### Architecture détaillée
```
  net.0: Linear(in_features=19, out_features=256, bias=True)
  net.2: Linear(in_features=256, out_features=256, bias=True)
  net.4: Linear(in_features=256, out_features=140, bias=True)
```

## Dataset
robomimic/lift_ph — 200 épisodes
Samples train : 4701 / test : 1165
Seed : 42

## Résultats
Loss train finale : 0.119629
Loss test finale  : 0.788576
Meilleure test    : 0.474956
Temps entraînement: 872.8s
Temps/epoch       : 0.11s
Inférence         : 0.040760040283203125ms (0.002ms/frame)
FLOPs/inférence   : 213,784
FLOPs/frame       : 10,689
Success rate      : 0%
Coverage moyen    : 82.1%
Reward moyen      : 0.00

## Paliers de performance
Aucun palier atteint

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
