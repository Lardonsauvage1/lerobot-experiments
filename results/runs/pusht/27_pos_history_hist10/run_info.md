# 27_pos_history — ResNet18 → MLP [256, 128] hist=10 chunk=20

## Expérience
MLP avec 10 positions passées en entrée

## Modèle
Type : ResNet18 → MLP [256, 128] hist=10 chunk=20
Paramètres : 174,504
Taille : 0.67 Mo

### Architecture détaillée
```
  net.0: Linear(in_features=532, out_features=256, bias=True)
  net.2: Linear(in_features=256, out_features=128, bias=True)
  net.4: Linear(in_features=128, out_features=40, bias=True)
```

## Dataset
lerobot/pusht — 206 épisodes
Samples train : 15905 / test : 3977
Seed : 42

## Résultats
Loss train finale : 0.006609
Loss test finale  : 0.039557
Meilleure test    : 0.037760
Temps entraînement: 464.1s
Temps/epoch       : 0.46s
Inférence         : 0ms (0.0ms/frame)
FLOPs/inférence   : 0
FLOPs/frame       : 0
Success rate      : 0%
Coverage moyen    : 17.2%
Reward moyen      : 8.75

## Paliers de performance
  loss_le_0.1 : epoch 18 (8.0s)
  loss_le_0.05 : epoch 71 (32.7s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
