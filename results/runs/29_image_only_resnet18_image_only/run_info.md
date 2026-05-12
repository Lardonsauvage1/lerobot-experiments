# 29_image_only — ResNet18(gelé) → Transformer 2L (image seule, sans agent_pos) chunk=20

## Expérience
Image seule (sans agent_pos) → Transformer pour action chunking. Comparaison au run 24 pour mesurer l'apport de agent_pos.

## Modèle
Type : ResNet18(gelé) → Transformer 2L (image seule, sans agent_pos) chunk=20
Paramètres : 169,026
Taille : 0.64 Mo

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
  action_head: Linear(in_features=64, out_features=2, bias=True)
```

## Dataset
lerobot/pusht — 206 épisodes
Samples train : 17388 / test : 4348
Seed : 42

## Résultats
Loss train finale : 0.066702
Loss test finale  : 0.089571
Meilleure test    : 0.081010
Temps entraînement: 899.4s
Temps/epoch       : 8.99s
Inférence         : 0.784534215927124ms (0.039ms/frame)
FLOPs/inférence   : 338,052
FLOPs/frame       : 16,902
Success rate      : 0%
Coverage moyen    : 31.8%
Reward moyen      : 36.25

## Paliers de performance
  loss_le_0.1 : epoch 61 (551.6s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
