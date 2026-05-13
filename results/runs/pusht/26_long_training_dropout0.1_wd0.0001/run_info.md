# 26_long_training — ResNet18 → MLP [256, 128] drop=0.1 chunk=20

## Expérience
10k epochs, dropout=0.1, wd=0.0001

## Modèle
Type : ResNet18 → MLP [256, 128] drop=0.1 chunk=20
Paramètres : 169,896
Taille : 0.65 Mo

### Architecture détaillée
```
  net.0: Linear(in_features=514, out_features=256, bias=True)
  net.3: Linear(in_features=256, out_features=128, bias=True)
  net.6: Linear(in_features=128, out_features=40, bias=True)
```

## Dataset
lerobot/pusht — 206 épisodes
Samples train : 17388 / test : 4348
Seed : 42

## Résultats
Loss train finale : 0.028939
Loss test finale  : 0.058320
Meilleure test    : 0.050492
Temps entraînement: 5770.3s
Temps/epoch       : 0.49s
Inférence         : 0ms (0.0ms/frame)
FLOPs/inférence   : 0
FLOPs/frame       : 0
Success rate      : 0%
Coverage moyen    : 16.8%
Reward moyen      : 10.52

## Paliers de performance
  loss_le_0.1 : epoch 47 (30.0s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
