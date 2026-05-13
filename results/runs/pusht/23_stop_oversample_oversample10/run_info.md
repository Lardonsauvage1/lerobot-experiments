# 23_stop_oversample — Transformer 2L chunk=20 +stop ×10

## Expérience
Stop signal avec surenchantillonnage ×10

## Modèle
Type : Transformer 2L chunk=20 +stop ×10
Paramètres : 669,987
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
  stop_head: Linear(in_features=64, out_features=1, bias=True)
```

## Dataset
lerobot/pusht — 206 épisodes
Samples train : 23429 / test : 5858
Seed : 42

## Résultats
Loss train finale : 0.027319
Loss test finale  : 0.051795
Meilleure test    : 0.049536
Temps entraînement: 3944.0s
Temps/epoch       : 39.44s
Inférence         : 1.308ms (0.065ms/frame)
FLOPs/inférence   : 1,231,328
FLOPs/frame       : 61,566
Success rate      : 0%
Coverage moyen    : 38.2%
Reward moyen      : 49.25

## Paliers de performance
  loss_le_0.1 : epoch 18 (709.3s)
  loss_le_0.05 : epoch 86 (3384.9s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
