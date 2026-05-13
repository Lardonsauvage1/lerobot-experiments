# 24_precomputed_features — ResNet18(gelé) → Transformer 2L chunk=20

## Expérience
Features ResNet18 pré-calculées → Transformer pour action chunking

## Modèle
Type : ResNet18(gelé) → Transformer 2L chunk=20
Paramètres : 169,154
Taille : 0.65 Mo

### Architecture détaillée
```
  obs_proj: Linear(in_features=514, out_features=64, bias=True)
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
Loss train finale : 0.061224
Loss test finale  : 0.072194
Meilleure test    : 0.066231
Temps entraînement: 1453.8s
Temps/epoch       : 14.54s
Inférence         : 1.314154863357544ms (0.066ms/frame)
FLOPs/inférence   : 338,308
FLOPs/frame       : 16,915
Success rate      : 0%
Coverage moyen    : 46.5%
Reward moyen      : 59.01

## Paliers de performance
  loss_le_0.1 : epoch 41 (591.4s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
