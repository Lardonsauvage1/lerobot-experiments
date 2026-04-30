# 15_grayscale — CNN[8,16] gray + Transformer 2L chunk=20

## Expérience
Images NB + CNN [8, 16] + Transformer chunk=20

## Modèle
Type : CNN[8,16] gray + Transformer 2L chunk=20
Paramètres : 403,938
Taille : 1.54 Mo

### Architecture détaillée
```
  cnn.conv.0: Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  cnn.conv.3: Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  cnn.fc: Linear(in_features=4096, out_features=64, bias=True)
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
Loss train finale : 0.047615
Loss test finale  : 0.058574
Meilleure test    : 0.053794
Temps entraînement: 2932.9s
Temps/epoch       : 14.66s
Inférence         : 0.755ms (0.038ms/frame)
FLOPs/inférence   : 699,280
FLOPs/frame       : 34,964
Success rate      : 0%
Coverage moyen    : 33.2%
Reward moyen      : 45.24

## Paliers de performance
  loss_le_0.1 : epoch 45 (665.4s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
