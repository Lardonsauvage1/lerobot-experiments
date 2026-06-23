# Convergence — succès (rollouts) vs steps d'entraînement

**Question :** à quelle vitesse un modèle atteint-il sa performance réelle, et comment
la **capacité** change-t-elle cette vitesse ? On mesure la convergence sur le **vrai
signal** (taux de succès en rollouts), pas sur la val_loss.

## Règle de l'étude

Référence = **500 rollouts** par checkpoint (IC95 ≈ ±4 pts). Le **50 rollouts**
(IC95 ≈ ±13 pts) est accepté comme **provisoire** (tracé en pointillés) — utile pour
voir vite la forme, à repasser en 500 plus tard sur les checkpoints clés. Pour un même
run, le 500r prime sur le 50r. Sont exclus :
- les évals **camshift / camaug** (succès sous caméra décalée = autre mesure) ;
- les évals à un seul checkpoint (pas une trajectoire).

## Auto-extensible — comment ajouter un modèle

Le script `experiments/phase5_methodology/33_plot_convergence_rollouts.py`
**découvre seul** tous les `**/*rollouts_500.csv` sous `results/runs/`, recolle les
fragments d'un même run (suffixes `_continue`/`_resume`/`_150k`/`_200k`/…) et exclut
le camshift.

> **Pour ajouter un entraînement à l'étude :** produire un
> `results/runs/<…>/<run>/rollouts_500.csv` (colonnes `step,success_rate,…`) à
> checkpoints réguliers, puis **relancer le script**. La courbe et le tableau
> (`convergence_metrics.md`) se régénèrent. Un nouveau run inconnu apparaît
> automatiquement sous son nom (ajouter un libellé lisible dans `LABELS` si besoin).
>
> ⚠️ Une éval **mono-checkpoint** (type `vision_500.json`, un seul step) **ne nourrit
> pas** cette étude : il faut des rollouts à *plusieurs* steps réguliers.

Sorties : `results/runs/phase5_methodology/convergence_rollouts.png` et
`…/convergence_metrics.md`.

## Résultats actuels

| run | capacité | plafond | décollage (>5 %) | 50 % du max | 90 % du max |
|---|---|---|---|---|---|
| ResNet34 dense | gros | **94 %** | 6 000 | 11 000 | **20 000** |
| mini-CNN cosine | minuscule | 81 % | 34 000 | 45 000 | **73 000** |
| mini-CNN constant | minuscule | 76 % | 24 000 | 30 000 | **46 000** |

Trajectoire ResNet34 : `1k:0 · 5k:3 · 10k:30 · 15k:80 · 20k:91 · 30k:93 %`.

## Leçons

1. **La capacité accélère ET relève la convergence du succès.** Le gros modèle atteint
   90 % de son max à **20k steps / plafond 94 %** ; le mini-CNN met **46–73k steps**
   pour plafonner **15–18 pts plus bas**. Plus de capacité = converge plus vite *et*
   plus haut (régime non sur-appris).
2. **La courbe de succès est une sigmoïde** : phase latente quasi-nulle → décollage
   brutal → plateau. Rien à voir avec la décroissance lisse de la loss.
3. **Le décollage du succès arrive bien après la convergence de la loss** (loss 90 %-
   convergée vers 750–6000 steps, succès qui décolle à 6–34k). → La val_loss est un
   mauvais chronomètre de convergence ; **seul le rollout** dit quand le modèle marche.
   (Cohérent avec [`PHASE5_PROTOCOLE.md`](PHASE5_PROTOCOLE.md) et [`CAMERA.md`](CAMERA.md).)
4. **Schedule** : `cosine` plafonne plus haut (81 % vs 76 %) mais converge plus lentement
   (90 %-max à 73k vs 46k) ; `constant` est plus rapide mais plafonne plus bas.

---
*Généré le 2026-06-22 ; régénérer après chaque nouveau run à rollouts réguliers.*
