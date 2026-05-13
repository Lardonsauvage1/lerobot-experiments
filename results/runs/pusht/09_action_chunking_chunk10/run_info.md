# 09_action_chunking — CNN+MLP chunk=10

## Expérience
Action chunking avec chunk=10 sur PushT

## Modèle
Type : CNN+MLP chunk=10
Paramètres : 547,572
Taille : 2.09 Mo

### Architecture détaillée
```
  cnn.conv.0: Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  cnn.conv.3: Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  cnn.fc: Linear(in_features=8192, out_features=64, bias=True)
  mlp.0: Linear(in_features=66, out_features=128, bias=True)
  mlp.2: Linear(in_features=128, out_features=64, bias=True)
  mlp.4: Linear(in_features=64, out_features=20, bias=True)
```

## Dataset
lerobot/pusht — 50 épisodes
Samples train : 4628 / test : 1157
Seed : 42

## Résultats
Loss train finale : 0.014274
Loss test finale  : 0.025232
Meilleure test    : 0.025232
Temps entraînement: 271.4s
Temps/epoch       : 5.43s
Inférence         : 0.554ms (0.055ms/frame)
FLOPs/inférence   : 1,094,496
FLOPs/frame       : 109,449
Success rate      : 0%
Coverage moyen    : 12.6%
Reward moyen      : 11.77

## Paliers de performance
  loss_le_0.1 : epoch 8 (45.8s)
  loss_le_0.05 : epoch 17 (94.4s)
  loss_le_0.03 : epoch 36 (196.4s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
