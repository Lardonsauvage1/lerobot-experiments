# 08_pretrained_cnn — CNN from scratch [16,32] + MLP [128, 64]

## Expérience
CNN from scratch + MLP sur PushT

## Modèle
Type : CNN from scratch [16,32] + MLP [128, 64]
Paramètres : 546,402
Taille : 2.08 Mo

### Architecture détaillée
```
  cnn.conv.0: Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  cnn.conv.3: Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  cnn.fc: Linear(in_features=8192, out_features=64, bias=True)
  mlp.0: Linear(in_features=66, out_features=128, bias=True)
  mlp.2: Linear(in_features=128, out_features=64, bias=True)
  mlp.4: Linear(in_features=64, out_features=2, bias=True)
```

## Dataset
lerobot/pusht — 50 épisodes
Samples train : 4988 / test : 1247
Seed : 42

## Résultats
Loss train finale : 0.004005
Loss test finale  : 0.012801
Meilleure test    : 0.012268
Temps entraînement: 325.4s
Temps/epoch       : 6.51s
Inférence         : 0.707ms (0.707ms/frame)
FLOPs/inférence   : 1,092,192
FLOPs/frame       : 1,092,192
Success rate      : 0%
Coverage moyen    : 5.5%
Reward moyen      : 17.25

## Paliers de performance
  loss_le_0.1 : epoch 1 (7.1s)
  loss_le_0.05 : epoch 1 (7.1s)
  loss_le_0.03 : epoch 5 (34.1s)
  loss_le_0.02 : epoch 18 (117.8s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
