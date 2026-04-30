# 10_transformer — CNN+Transformer 1L 4H d=64 chunk=20

## Expérience
Transformer decoder 1 couches + action chunking=20

## Modèle
Type : CNN+Transformer 1L 4H d=64 chunk=20
Paramètres : 603,170
Taille : 2.3 Mo

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
  action_head: Linear(in_features=64, out_features=2, bias=True)
```

## Dataset
lerobot/pusht — 50 épisodes
Samples train : 4228 / test : 1057
Seed : 42

## Résultats
Loss train finale : 0.035045
Loss test finale  : 0.045994
Meilleure test    : 0.045501
Temps entraînement: 220.2s
Temps/epoch       : 4.4s
Inférence         : 0.697ms (0.035ms/frame)
FLOPs/inférence   : 1,149,280
FLOPs/frame       : 57,464
Success rate      : 0%
Coverage moyen    : 19.8%
Reward moyen      : 14.27

## Paliers de performance
  loss_le_0.1 : epoch 13 (59.2s)
  loss_le_0.05 : epoch 39 (172.2s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
