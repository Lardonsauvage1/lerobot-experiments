# lift/39_lift_image_aug_50pct — ResNet18(gelé,aug50%) + MLP 531 → 256 → 256 → CE+résidu (n_aug=8)

## Expérience
Run 38 fix : 50% v0 + 50% aug. Cherche à matcher distribution eval clean.

## Modèle
Type : ResNet18(gelé,aug50%) + MLP 531 → 256 → 256 → CE+résidu (n_aug=8)
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
Loss train finale : 0.566341
Loss test finale  : 1.363736
Meilleure test    : 0.943512
Temps entraînement: 381.0s
Temps/epoch       : 0.57s
Inférence         : 0.15415549278259277ms (0.008ms/frame)
FLOPs/inférence   : 3,426,288
FLOPs/frame       : 171,314
Success rate      : 6%
Coverage moyen    : 82.7%
Reward moyen      : 0.64

## Paliers de performance
Aucun palier atteint

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
