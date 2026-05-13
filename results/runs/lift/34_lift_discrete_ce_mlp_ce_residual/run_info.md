# lift/34_lift_discrete_ce — MLP 19 → 256 → 256 → CE+résidu (N=21 bins × 7 dims × chunk=20)

## Expérience
Lift CE + résidu (formule run 31 PushT appliquée au 7D Lift).

## Modèle
Type : MLP 19 → 256 → 256 → CE+résidu (N=21 bins × 7 dims × chunk=20)
Paramètres : 1,582,072
Taille : 6.04 Mo

### Architecture détaillée
```
  body.0: Linear(in_features=19, out_features=256, bias=True)
  body.2: Linear(in_features=256, out_features=256, bias=True)
  bin_head: Linear(in_features=256, out_features=2940, bias=True)
  residual_head: Linear(in_features=256, out_features=2940, bias=True)
```

## Dataset
robomimic/lift_ph — 200 épisodes
Samples train : 4701 / test : 1165
Seed : 42

## Résultats
Loss train finale : 0.504578
Loss test finale  : 1.843580
Meilleure test    : 0.971957
Temps entraînement: 359.7s
Temps/epoch       : 0.64s
Inférence         : 0.12997984886169434ms (0.006ms/frame)
FLOPs/inférence   : 3,164,144
FLOPs/frame       : 158,207
Success rate      : 0%
Coverage moyen    : 82.1%
Reward moyen      : 0.00

## Paliers de performance
Aucun palier atteint

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
