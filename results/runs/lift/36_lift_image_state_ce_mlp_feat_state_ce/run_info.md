# lift/36_lift_image_state_ce — ResNet18(gelé) + MLP 531 → 256 → 256 → CE+résidu (N=21×7×20)

## Expérience
Lift : features ResNet + state → CE+résidu (run 31 PushT formula appliquée à Lift).

## Modèle
Type : ResNet18(gelé) + MLP 531 → 256 → 256 → CE+résidu (N=21×7×20)
Paramètres : 1,713,144
Taille : 6.54 Mo

### Architecture détaillée
```
  body.0: Linear(in_features=531, out_features=256, bias=True)
  body.2: Linear(in_features=256, out_features=256, bias=True)
  bin_head: Linear(in_features=256, out_features=2940, bias=True)
  residual_head: Linear(in_features=256, out_features=2940, bias=True)
```

## Dataset
robomimic/lift_ph — 200 épisodes
Samples train : 4701 / test : 1165
Seed : 42

## Résultats
Loss train finale : 0.479387
Loss test finale  : 1.515628
Meilleure test    : 0.945863
Temps entraînement: 377.5s
Temps/epoch       : 0.57s
Inférence         : 0.16021013259887695ms (0.008ms/frame)
FLOPs/inférence   : 3,426,288
FLOPs/frame       : 171,314
Success rate      : 14%
Coverage moyen    : 83.4%
Reward moyen      : 1.28

## Paliers de performance
Aucun palier atteint

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
