# 15_grayscale — CNN[4,8] gray + Transformer 2L chunk=20

## Expérience
Images NB + CNN [4, 8] + Transformer chunk=20

## Modèle
Type : CNN[4,8] gray + Transformer 2L chunk=20
Paramètres : 271,954
Taille : 1.04 Mo

### Architecture détaillée
```
  cnn.conv.0: Conv2d(1, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  cnn.conv.3: Conv2d(4, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  cnn.fc: Linear(in_features=2048, out_features=64, bias=True)
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
Loss train finale : 0.050707
Loss test finale  : 0.063975
Meilleure test    : 0.057854
Temps entraînement: 2192.9s
Temps/epoch       : 10.96s
Inférence         : 0.793ms (0.04ms/frame)
FLOPs/inférence   : 435,336
FLOPs/frame       : 21,766
Success rate      : 0%
Coverage moyen    : 41.5%
Reward moyen      : 50.82

## Paliers de performance
  loss_le_0.1 : epoch 52 (568.3s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
