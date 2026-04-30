# 28_baseline_no_hist — ResNet18 → MLP [256, 128] chunk=20 no-hist

## Expérience
Baseline sans history, 10000 epochs, eval tous les 200

## Modèle
Type : ResNet18 → MLP [256, 128] chunk=20 no-hist
Paramètres : 169,896
Taille : 0.65 Mo

### Architecture détaillée
```
  net.0: Linear(in_features=514, out_features=256, bias=True)
  net.2: Linear(in_features=256, out_features=128, bias=True)
  net.4: Linear(in_features=128, out_features=40, bias=True)
```

## Dataset
lerobot/pusht — 206 épisodes
Samples train : 17388 / test : 4348
Seed : 42

## Résultats
Loss train finale : 0.004927
Loss test finale  : 0.058403
Meilleure test    : 0.048640
Temps entraînement: 38690.2s
Temps/epoch       : 1.83s
Inférence         : 0ms (0.0ms/frame)
FLOPs/inférence   : 0
FLOPs/frame       : 0
Success rate      : 0%
Coverage moyen    : 19.8%
Reward moyen      : 13.68

## Paliers de performance
  loss_le_0.1 : epoch 54 (19.4s)
  loss_le_0.05 : epoch 234 (84.2s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
