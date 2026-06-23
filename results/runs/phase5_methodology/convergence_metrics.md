# Convergence — métriques (succès en rollouts vs steps)

*Généré par `experiments/phase5_methodology/33_plot_convergence_rollouts.py`.*

| run | rollouts | n pts | plage steps | plafond | décollage (>5%) | 50% du max | 90% du max |
|---|---|---|---|---|---|---|---|
| ResNet34 + gros U-Net (61M) | 50 (prov.) | 20 | 2000–40000 | 100% | 8000 | 20000 | 28000 |
| ResNet34 dense (gros modèle) | 500 | 30 | 1000–30000 | 94% | 6000 | 11000 | 20000 |
| mini-CNN cosine (minuscule) | 500 | 198 | 1000–250000 | 81% | 34000 | 45000 | 73000 |
| mini-CNN constant (minuscule) | 500 | 182 | 1000–248000 | 76% | 24000 | 30000 | 46000 |
