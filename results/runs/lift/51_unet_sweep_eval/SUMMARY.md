# Sweep U-Net — résultats (Phase 4, compression pour la latence)

Tâche : Robomimic **Lift**. Protocole Phase 0 : train 150 / val 50 (split contigu figé), éval rollout
sur les **50 init states val held-out**. Chaque modèle entraîné 7500 steps avec `50_train_valloss.py`
(val-loss continue) ; **meilleur checkpoint = minimum de val-loss** (set val complet), puis rollout 50 ép.

## Tableau comparatif

| Modèle (`down_dims`) | Params | Checkpoint retenu | Succès | t_success | max_z (marge) | hold |
|---|---|---|---|---|---|---|
| **Baseline** `[512,1024,2048]` | **263.7 M** | 6000 | **100 %** | 43.0 | 1.033 | 0.48 |
| `[256,512,1024]` | 76.6 M | 4500 | 98 % | 44.0 | 0.999 | 0.55 |
| `[128,256,512]` | 28.7 M | 4500 | 98 % | 48.0 | 0.951 | 0.49 |
| **`[64,128,256]`** | **16.2 M** | 6000 | **100 %** | 47.0 | 0.969 | 0.48 |

*(succès sur 50 ép. : 98 % = 49/50, dans le bruit d'échantillonnage. t_success = nb de steps avant levage, ↓ = plus décisif. max_z = hauteur max du cube, ↑ = meilleure marge. hold = fraction de steps maintenus après le 1er succès.)*

## Val-loss propre par checkpoint (set val complet)

| step | `[256,512,1024]` | `[128,256,512]` | `[64,128,256]` |
|---|---|---|---|
| 1500 | 0.0824 | 0.0798 | 0.1054 |
| 3000 | 0.0686 | 0.0708 | 0.0832 |
| 4500 | **0.0667** | **0.0703** | 0.0736 |
| 6000 | 0.0718 | 0.0735 | **0.0722** |
| 7500 | 0.0742 | 0.0732 | 0.0730 |

→ overfit léger après ~4500-6000 selon la taille (val-loss remonte), cohérent avec le baseline (min ~6000).

## Conclusion

- **Le U-Net se rétrécit ÷16 (263.7 M → 16.2 M) sans perte de succès** : les 4 tailles tiennent 98-100 % sur les départs val jamais vus. Le gros U-Net du run 46 était massivement surdimensionné pour Lift.
- Coût marginal : t_success un peu plus lent (43 → 47) et marge max_z légèrement plus basse (1.03 → 0.97), mais toujours largement au-dessus du seuil de levage.
- **Sur le modèle 16.2 M, la vision (ResNet18, 11 M) pèse 68 %** → le U-Net n'est plus que ~5 M. Pour descendre encore, il faudrait attaquer la vision.
- ⚠️ Reste à mesurer la **latence d'inférence** (objectif réel) — non couverte par ce tableau (params seulement). Le levier des pas de diffusion (10 → moins) est encore intact.

## Fichiers
- `sweep_eval.json` — données brutes (params, val-loss, métriques rollout).
- `sweep_pareto.png` — Pareto params vs (succès, t_success, max_z).
- `videos/<label>_ep<idx>_<OK|FAIL>.mp4` — vidéos d'épisodes (3 scènes val, mêmes pour tous les modèles).
