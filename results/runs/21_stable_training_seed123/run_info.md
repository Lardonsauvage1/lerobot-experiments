# 21_stable_training — Transformer 2L chunk=20 warmup+clip

## Expérience
Transformer 2L chunk=20, warmup+clip, seed=123

## Modèle
Type : Transformer 2L chunk=20 warmup+clip
Paramètres : 669,922
Taille : 2.56 Mo

### Architecture détaillée
```
  cnn.conv.0: Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
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
Seed : 123

## Résultats
Loss train finale : 0.011875
Loss test finale  : 0.039875
Meilleure test    : 0.037619
Temps entraînement: 8768.6s
Temps/epoch       : 29.23s
Inférence         : 1.418ms (0.071ms/frame)
FLOPs/inférence   : 1,231,200
FLOPs/frame       : 61,560
Success rate      : 0%
Coverage moyen    : 32.4%
Reward moyen      : 33.92

## Paliers de performance
  loss_le_0.1 : epoch 13 (379.5s)
  loss_le_0.05 : epoch 38 (1109.9s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
