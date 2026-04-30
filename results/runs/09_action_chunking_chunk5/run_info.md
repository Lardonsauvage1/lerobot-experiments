# 09_action_chunking — CNN+MLP chunk=5

## Expérience
Action chunking avec chunk=5 sur PushT

## Modèle
Type : CNN+MLP chunk=5
Paramètres : 546,922
Taille : 2.09 Mo

### Architecture détaillée
```
  cnn.conv.0: Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  cnn.conv.3: Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  cnn.fc: Linear(in_features=8192, out_features=64, bias=True)
  mlp.0: Linear(in_features=66, out_features=128, bias=True)
  mlp.2: Linear(in_features=128, out_features=64, bias=True)
  mlp.4: Linear(in_features=64, out_features=10, bias=True)
```

## Dataset
lerobot/pusht — 50 épisodes
Samples train : 4828 / test : 1207
Seed : 42

## Résultats
Loss train finale : 0.009302
Loss test finale  : 0.022960
Meilleure test    : 0.021655
Temps entraînement: 284.7s
Temps/epoch       : 5.69s
Inférence         : 0.859ms (0.172ms/frame)
FLOPs/inférence   : 1,093,216
FLOPs/frame       : 218,643
Success rate      : 0%
Coverage moyen    : 5.5%
Reward moyen      : 17.07

## Paliers de performance
  loss_le_0.1 : epoch 2 (11.0s)
  loss_le_0.05 : epoch 14 (80.0s)
  loss_le_0.03 : epoch 28 (161.0s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
