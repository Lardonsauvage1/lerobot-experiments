# lift/38_lift_image_aug_ce — ResNet18(gelé,aug) + MLP 531 → 256 → 256 → CE+résidu (n_aug=8)

## Expérience
Run 36 + image augmentation 8x pour fight overfit.

## Modèle
Type : ResNet18(gelé,aug) + MLP 531 → 256 → 256 → CE+résidu (n_aug=8)
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
Loss train finale : 0.585587
Loss test finale  : 1.321572
Meilleure test    : 0.944135
Temps entraînement: 380.4s
Temps/epoch       : 0.58s
Inférence         : 0.17897486686706543ms (0.009ms/frame)
FLOPs/inférence   : 3,426,288
FLOPs/frame       : 171,314
Success rate      : 10%
Coverage moyen    : 83.0%
Reward moyen      : 0.98

## Paliers de performance
Aucun palier atteint

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
