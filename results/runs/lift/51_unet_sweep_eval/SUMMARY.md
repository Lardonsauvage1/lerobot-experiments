# Sweep U-Net — résultats (Phase 4, compression pour la latence)

Tâche : Robomimic **Lift**. Protocole Phase 0 : train 150 / val 50 (split contigu figé), éval rollout
sur les **50 init states val held-out**. Chaque modèle entraîné 7500 steps avec `50_train_valloss.py`
(val-loss continue) ; **meilleur checkpoint = minimum de val-loss** (set val complet), puis rollout 50 ép.

## Tableau comparatif

| Modèle (`down_dims`) | Params | U-Net | Checkpoint | Succès | t_success | max_z | val_loss |
|---|---|---|---|---|---|---|---|
| **Baseline** `[512,1024,2048]` | **263.7 M** | 252 M | 6000 | **100 %** | 43.0 | 1.033 | — |
| `[256,512,1024]` | 76.6 M | 65 M | 4500 | 98 % | 44.0 | 0.999 | 0.067 |
| `[128,256,512]` | 28.7 M | 17 M | 4500 | 98 % | 48.0 | 0.951 | 0.070 |
| `[64,128,256]` | 16.2 M | 5 M | 6000 | **100 %** | 47.0 | 0.969 | 0.072 |
| **`[32,64,128]`** ⭐ | **12.8 M** | **1.6 M** | 7500 | **100 %** | 46.5 | 0.990 | 0.083 |
| `[16,32,64]` | 11.8 M | 0.4 M | 10500 | **2 %** 💥 | 65.0 | 0.829 | 0.243 |
| `[8,16,32]` | 11.5 M | 0.3 M | 9000 | **0 %** 💥 | — | 0.827 | 0.663 |

*(succès sur 50 ép. ; 98 % = 49/50, dans le bruit. t_success = steps avant levage (↓ mieux) ; max_z = hauteur max cube (↑ mieux ; ~0.83 = jamais levé).)*

**Plancher U-Net trouvé** : `[32,64,128]` (U-Net **1.6 M**, total 12.8 M) est le **plus petit qui tient 100 %** ; en dessous, falaise nette → `[16,32,64]` (0.4 M) s'effondre à 2 %, `[8,16,32]` (0.3 M) à 0 %. Le U-Net passe de 252 M → 1.6 M (**÷160**) sans perte. La **val-loss prédit la falaise** (0.083 → 0.243 → 0.663). Mais sous `[32,64,128]` le total ne bouge plus (vision ResNet18 = 11.2 M domine) → descendre plus = perdre 100 % pour ~1 M. ⭐ `[32,64,128]` = sweet spot ; pour aller plus loin, attaquer la **vision**. Voir `sweep_pareto.png` (4 panneaux).

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

## Plafond théorique — démos expertes (50 scènes val)

Mesuré en remettant l'env sur chaque état enregistré des démos (`54_demo_ceiling.py`, même critère `is_success`) :

| | Succès | t_success | max_z | hold |
|---|---|---|---|---|
| **Démos expertes (plafond)** | 100 % | 43.0 | 0.880 | 1.00 |
| Baseline 263.7 M | 100 % | 43.0 | 1.033 | 0.48 |
| Gagnant 16.2 M | 100 % | 47.0 | 0.969 | 0.48 |

- **Sur les métriques comparables (succès + t_success), nos modèles sont AU plafond expert** : 100 % de succès, et levage aussi rapide que l'humain (43-47 ≈ 43). Pas de marge théorique à récupérer.
- **max_z et hold ne sont pas comparables directement** : les démos sont courtes (longueur ~36-54 steps, elles s'arrêtent au levage) → `hold=1.00` et `max_z=0.88` par construction. Nos modèles tournent 200 steps → ils montent plus haut (max_z>0.88) et le `hold=0.48` mesure du comportement post-succès hors-distribution que les démos ne couvrent pas. Ce n'est donc pas un déficit vs expert.

## Fichiers
- `sweep_eval.json` — données brutes (params, val-loss, métriques rollout).
- `demo_ceiling.json` — plafond démos expertes.
- `sweep_pareto.png` — Pareto params vs (succès, t_success, max_z).
- `videos/<label>_ep<idx>_<OK|FAIL>.mp4` — vidéos d'épisodes (3 scènes val, mêmes pour tous les modèles).
