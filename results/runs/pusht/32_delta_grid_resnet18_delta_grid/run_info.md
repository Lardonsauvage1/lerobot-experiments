# 32_delta_grid — ResNet18(gelé) → Transformer 2L → Grille delta 8×8 + résidu chunk=20

## Expérience
Bins en grille uniforme 8×8 sur deltas humains action→action. Image seule, reconstruction cumulative.

## Modèle
Type : ResNet18(gelé) → Transformer 2L → Grille delta 8×8 + résidu chunk=20
Paramètres : 181,376
Taille : 0.69 Mo

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
  bin_head: Linear(in_features=64, out_features=64, bias=True)
  residual_head: Linear(in_features=64, out_features=128, bias=True)
```

## Dataset
lerobot/pusht — 206 épisodes
Samples train : 17224 / test : 4306
Seed : 42

## Résultats
Loss train finale : 1.301488
Loss test finale  : 1.391431
Meilleure test    : 1.381302
Temps entraînement: 1634.8s
Temps/epoch       : 9.31s
Inférence         : 0.9104049205780029ms (0.046ms/frame)
FLOPs/inférence   : 362,752
FLOPs/frame       : 18,137
Success rate      : 0%
Coverage moyen    : 21.1%
Reward moyen      : 23.59

## Paliers de performance
Aucun palier atteint

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
