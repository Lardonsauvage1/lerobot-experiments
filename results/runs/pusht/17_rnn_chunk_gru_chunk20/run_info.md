# 17_rnn_chunk — CNN+GRU(h=128,L=2) chunk=20

## Expérience
CNN + GRU (hidden=128, 2 couches) + action chunking=20

## Modèle
Type : CNN+GRU(h=128,L=2) chunk=20
Paramètres : 708,936
Taille : 2.7 Mo

### Architecture détaillée
```
  cnn.conv.0: Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  cnn.conv.3: Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  cnn.fc: Linear(in_features=8192, out_features=64, bias=True)
  rnn: GRU(66, 128, num_layers=2, batch_first=True)
  fc_out: Linear(in_features=128, out_features=40, bias=True)
```

## Dataset
lerobot/pusht — 206 épisodes
Samples train : 17388 / test : 4348
Seed : 42

## Résultats
Loss train finale : 0.006354
Loss test finale  : 0.038492
Meilleure test    : 0.038171
Temps entraînement: 4535.7s
Temps/epoch       : 15.12s
Inférence         : 0.587ms (0.029ms/frame)
FLOPs/inférence   : 1,068,896
FLOPs/frame       : 53,444
Success rate      : 0%
Coverage moyen    : 9.6%
Reward moyen      : 18.77

## Paliers de performance
  loss_le_0.1 : epoch 15 (238.0s)
  loss_le_0.05 : epoch 57 (894.9s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
