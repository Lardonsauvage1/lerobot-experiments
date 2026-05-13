# 18_rnn_window — CNN+GRU(h=128) window=5 chunk=20

## Expérience
RNN fenêtre glissante=5 + chunk=20

## Modèle
Type : CNN+GRU(h=128) window=5 chunk=20
Paramètres : 609,864
Taille : 2.33 Mo

### Architecture détaillée
```
  cnn.conv.0: Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  cnn.conv.3: Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  cnn.fc: Linear(in_features=8192, out_features=64, bias=True)
  rnn: GRU(66, 128, batch_first=True)
  fc_out: Linear(in_features=128, out_features=40, bias=True)
```

## Dataset
lerobot/pusht — 206 épisodes
Samples train : 16729 / test : 4183
Seed : 42

## Résultats
Loss train finale : 0.003564
Loss test finale  : 0.009515
Meilleure test    : 0.009154
Temps entraînement: 22905.1s
Temps/epoch       : 76.04s
Inférence         : 2.292ms (0.115ms/frame)
FLOPs/inférence   : 1,068,896
FLOPs/frame       : 53,444
Success rate      : 0%
Coverage moyen    : 19.6%
Reward moyen      : 14.97

## Paliers de performance
  loss_le_0.1 : epoch 6 (438.2s)
  loss_le_0.05 : epoch 12 (892.6s)
  loss_le_0.03 : epoch 19 (1430.2s)
  loss_le_0.02 : epoch 31 (2357.4s)
  loss_le_0.01 : epoch 126 (9410.1s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
