# 27_pos_history — ResNet18 → MLP [256, 128] hist=5 chunk=20

## Expérience
MLP avec 5 positions passées en entrée

## Modèle
Type : ResNet18 → MLP [256, 128] hist=5 chunk=20
Paramètres : 171,944
Taille : 0.66 Mo

### Architecture détaillée
```
  net.0: Linear(in_features=522, out_features=256, bias=True)
  net.2: Linear(in_features=256, out_features=128, bias=True)
  net.4: Linear(in_features=128, out_features=40, bias=True)
```

## Dataset
lerobot/pusht — 206 épisodes
Samples train : 16729 / test : 4183
Seed : 42

## Résultats
Loss train finale : 0.007627
Loss test finale  : 0.044084
Meilleure test    : 0.040907
Temps entraînement: 447.1s
Temps/epoch       : 0.45s
Inférence         : 0ms (0.0ms/frame)
FLOPs/inférence   : 0
FLOPs/frame       : 0
Success rate      : 0%
Coverage moyen    : 24.9%
Reward moyen      : 15.70

## Paliers de performance
  loss_le_0.1 : epoch 22 (11.1s)
  loss_le_0.05 : epoch 86 (42.9s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
