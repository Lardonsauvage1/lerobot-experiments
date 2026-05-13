# 30_mdn — ResNet18(gelé) → Transformer 2L → MDN(K=5) chunk=20

## Expérience
MDN sur image seule — modélise la multimodalité par mélange gaussien K=5.

## Modèle
Type : ResNet18(gelé) → Transformer 2L → MDN(K=5) chunk=20
Paramètres : 170,521
Taille : 0.65 Mo

### Architecture détaillée
```
  obs_proj: Linear(in_features=512, out_features=64, bias=True)
  transformer_decoder.layers.0.self_attn.out_proj: NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
  transformer_decoder.layers.0.multihead_attn.out_proj: NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
  transformer_decoder.layers.0.linear1: Linear(in_features=64, out_features=256, bias=True)
  transformer_decoder.layers.0.linear2: Linear(in_features=256, out_features=64, bias=True)
  transformer_decoder.layers.1.self_attn.out_proj: NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
  transformer_decoder.layers.1.multihead_attn.out_proj: NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
  transformer_decoder.layers.1.linear1: Linear(in_features=64, out_features=256, bias=True)
  transformer_decoder.layers.1.linear2: Linear(in_features=256, out_features=64, bias=True)
  mdn_head: Linear(in_features=64, out_features=25, bias=True)
```

## Dataset
lerobot/pusht — 206 épisodes
Samples train : 17388 / test : 4348
Seed : 42

## Résultats
Loss train finale : -0.502673
Loss test finale  : -0.400897
Meilleure test    : -0.408001
Temps entraînement: 1868.7s
Temps/epoch       : 11.68s
Inférence         : 0.9652698040008545ms (0.048ms/frame)
FLOPs/inférence   : 341,042
FLOPs/frame       : 17,052
Success rate      : 0%
Coverage moyen    : 35.4%
Reward moyen      : 32.75

## Paliers de performance
  loss_le_0.1 : epoch 31 (290.5s)
  loss_le_0.05 : epoch 35 (328.0s)
  loss_le_0.03 : epoch 36 (337.3s)
  loss_le_0.02 : epoch 38 (355.9s)
  loss_le_0.01 : epoch 38 (355.9s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
