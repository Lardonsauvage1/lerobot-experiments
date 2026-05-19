# lift/37_lift_robomimic_bc — ResNet18(gelé) + MLP 531 → 1024 → 1024 → GMM(K=5)

## Expérience
Clone de la recette BC du papier Robomimic (chunk=1, GMM, MLP[1024,1024]+LN).

## Modèle
Type : ResNet18(gelé) + MLP 531 → 1024 → 1024 → GMM(K=5)
Paramètres : 1,675,339
Taille : 6.39 Mo

### Architecture détaillée
```
  body.0: Linear(in_features=531, out_features=1024, bias=True)
  body.3: Linear(in_features=1024, out_features=1024, bias=True)
  gmm_head: Linear(in_features=1024, out_features=75, bias=True)
```

## Dataset
robomimic/lift_ph — 200 épisodes
Samples train : 7741 / test : 1925
Seed : 42

## Résultats
Loss train finale : -22.749010
Loss test finale  : 64.967453
Meilleure test    : -9.914659
Temps entraînement: 1460.9s
Temps/epoch       : 0.65s
Inférence         : 0.23116469383239746ms (0.231ms/frame)
FLOPs/inférence   : 3,350,678
FLOPs/frame       : 3,350,678
Success rate      : 26%
Coverage moyen    : 115.1%
Reward moyen      : 32.60

## Paliers de performance
  loss_le_0.1 : epoch 1 (0.8s)
  loss_le_0.05 : epoch 1 (0.8s)
  loss_le_0.03 : epoch 1 (0.8s)
  loss_le_0.02 : epoch 1 (0.8s)
  loss_le_0.01 : epoch 1 (0.8s)

## Fichiers
- `model.pt` — poids du modèle
- `losses.png` — graphes loss vs epoch / temps / FLOPs
- `ep0.mp4`, `ep1.mp4`, `ep2.mp4` — vidéos des épisodes d'évaluation
