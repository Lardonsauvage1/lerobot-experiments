# 15_grayscale — CNN[16,32] gray + Transformer 2L chunk=20

## Expérience
Images NB + CNN [16, 32] + Transformer chunk=20

## Modèle
Type : CNN[16,32] gray + Transformer 2L chunk=20
Paramètres : 669,634
Taille : 2.55 Mo

### Architecture détaillée
```
  cnn.conv.0: Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  cnn.conv.3: Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  cnn.fc: Linear(in_features=8192, out_features=64, bias=True)
  obs_proj: Linear(in_features=66, out_features=64, bias=True)
  transformer_decoder.layers.0.self_attn.out_proj: NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
  transformer_decoder.layers.0.multihead_attn.out_proj: NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
  transformer_decoder.layers.0.linear1: Linear(in_features=64, out_features=256, bias=True)
  transformer_decoder.layers.0.linear2: Linear(in_features=256, out_features=64, bias=True)
  transformer_decoder.layers.1.self_attn.out_proj: NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
  transformer_decoder.layers.1.multihead_attn.out_proj: NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
  transformer_decoder.layers.1.linear1: Linear(in_features=64, out_features=256, bias=True)
  transformer_decoder.layers.1.linear2: Linear(in_features=256, out_features=64, bias=True)
  action_head: Linear(in_features=64, out_features=2, bias=True)
```

## Dataset
lerobot/pusht — 206 épisodes
Samples train : 17388 / test : 4348
Seed : 42

## Résultats
Loss train finale : 0.037344
Loss test finale  : 0.051530
Meilleure test    : 0.048250
Temps entraînement: 4117.5s
Temps/epoch       : 20.59s
Inférence         : 0.854ms (0.043ms/frame)
FLOPs/inférence   : 1,230,624
FLOPs/frame       : 61,531
Success rate      : 0%
Coverage moyen    : 23.1%
Reward moyen      : 33.01

## Paliers de performance
  loss_le_0.1 : epoch 23 (474.8s)
  loss_le_0.05 : epoch 164 (3364.0s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
