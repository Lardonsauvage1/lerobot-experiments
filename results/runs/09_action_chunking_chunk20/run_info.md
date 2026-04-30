# 09_action_chunking — CNN+MLP chunk=20

## Expérience
Action chunking avec chunk=20 sur PushT

## Modèle
Type : CNN+MLP chunk=20
Paramètres : 548,872
Taille : 2.09 Mo

### Architecture détaillée
```
  cnn.conv.0: Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  cnn.conv.3: Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  cnn.fc: Linear(in_features=8192, out_features=64, bias=True)
  mlp.0: Linear(in_features=66, out_features=128, bias=True)
  mlp.2: Linear(in_features=128, out_features=64, bias=True)
  mlp.4: Linear(in_features=64, out_features=40, bias=True)
```

## Dataset
lerobot/pusht — 50 épisodes
Samples train : 4228 / test : 1057
Seed : 42

## Résultats
Loss train finale : 0.025838
Loss test finale  : 0.038856
Meilleure test    : 0.038856
Temps entraînement: 248.3s
Temps/epoch       : 4.97s
Inférence         : 0.895ms (0.045ms/frame)
FLOPs/inférence   : 1,097,056
FLOPs/frame       : 54,852
Success rate      : 0%
Coverage moyen    : 16.7%
Reward moyen      : 23.83

## Paliers de performance
  loss_le_0.1 : epoch 16 (81.0s)
  loss_le_0.05 : epoch 33 (164.3s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
