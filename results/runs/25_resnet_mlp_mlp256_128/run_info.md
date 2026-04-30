# 25_resnet_mlp — ResNet18(gelé) → MLP [256, 128] chunk=60

## Expérience
Features ResNet18 → MLP [256, 128] chunk=60

## Modèle
Type : ResNet18(gelé) → MLP [256, 128] chunk=60
Paramètres : 180,216
Taille : 0.69 Mo

### Architecture détaillée
```
  net.0: Linear(in_features=514, out_features=256, bias=True)
  net.2: Linear(in_features=256, out_features=128, bias=True)
  net.4: Linear(in_features=128, out_features=120, bias=True)
```

## Dataset
lerobot/pusht — 206 épisodes
Samples train : 10807 / test : 2702
Seed : 42

## Résultats
Loss train finale : 0.016484
Loss test finale  : 0.096174
Meilleure test    : 0.090287
Temps entraînement: 325.6s
Temps/epoch       : 0.33s
Inférence         : 0.07273554801940918ms (0.001ms/frame)
FLOPs/inférence   : 360,432
FLOPs/frame       : 6,007
Success rate      : 0%
Coverage moyen    : 19.8%
Reward moyen      : 12.33

## Paliers de performance
  loss_le_0.1 : epoch 102 (35.4s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
