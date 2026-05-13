# 26_long_training — ResNet18 → MLP [256, 128] drop=0.05 chunk=20

## Expérience
10k epochs, dropout=0.05, wd=1e-05

## Modèle
Type : ResNet18 → MLP [256, 128] drop=0.05 chunk=20
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
Loss train finale : 0.018041
Loss test finale  : 0.056393
Meilleure test    : 0.046933
Temps entraînement: 5224.3s
Temps/epoch       : 0.48s
Inférence         : 0ms (0.0ms/frame)
FLOPs/inférence   : 0
FLOPs/frame       : 0
Success rate      : 0%
Coverage moyen    : 19.7%
Reward moyen      : 12.08

## Paliers de performance
  loss_le_0.1 : epoch 47 (22.1s)
  loss_le_0.05 : epoch 296 (132.3s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
