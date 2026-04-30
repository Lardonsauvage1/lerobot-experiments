# 14_vae_transformer — CNN+VAE(z=16)+Transformer 2L chunk=20

## Expérience
Transformer + VAE (z=16) + action chunking=20

## Modèle
Type : CNN+VAE(z=16)+Transformer 2L chunk=20
Paramètres : 686,530
Taille : 2.62 Mo

### Architecture détaillée
```
  cnn.conv.0: Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  cnn.conv.3: Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  cnn.fc: Linear(in_features=8192, out_features=64, bias=True)
  vae_encoder.net.0: Linear(in_features=40, out_features=128, bias=True)
  vae_encoder.net.2: Linear(in_features=128, out_features=64, bias=True)
  vae_encoder.fc_mu: Linear(in_features=64, out_features=16, bias=True)
  vae_encoder.fc_logvar: Linear(in_features=64, out_features=16, bias=True)
  obs_proj: Linear(in_features=82, out_features=64, bias=True)
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
Loss train finale : 0.032667
Loss test finale  : 0.049919
Meilleure test    : 0.049812
Temps entraînement: 4793.2s
Temps/epoch       : 23.97s
Inférence         : 0.903ms (0.045ms/frame)
FLOPs/inférence   : 1,263,968
FLOPs/frame       : 63,198
Success rate      : 0%
Coverage moyen    : 27.8%
Reward moyen      : 24.60

## Paliers de performance
  loss_le_0.1 : epoch 16 (363.1s)
  loss_le_0.05 : epoch 176 (4219.0s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
