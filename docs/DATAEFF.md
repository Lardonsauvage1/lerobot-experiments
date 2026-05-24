# Phase 5 — Efficacité données : combien de démos suffisent ?

> Question clé pour le bras réel (où l'on aura **peu de données**) : jusqu'où descendre en nombre de démos en gardant les perfs ? Réponse sur Lift : **~20 démos ≈ 98 %**, et même **10 démos → 84 %**. Pas de falaise, une pente douce.
> Liens : [◀ Phase 4 Compression](COMPRESSION.md) · **Phase 5 (ici)** · [index](README.md)

## Contexte

Le modèle final (mini-CNN + U-Net `[32,64,128]`, 1.65 M) a été entraîné sur **150 démos**. Mais sur un vrai bras, collecter 150 démos est coûteux. On mesure donc la **dégradation des perfs quand on réduit le nombre de démos d'entraînement**.

## Le parcours

Même modèle, même protocole, on fait varier **N = 150 / 100 / 50 / 20 / 10 démos** :
- **Train** = les N premières démos (préfixe `0..N-1`).
- **Val = 50 démos figées (150-199), held-out et identiques pour tous les N** → comparaison juste (seul le train change).
- Entraînement 6000 steps (save tous les 750), meilleur checkpoint = min val-loss, **rollout @ 4 pas** sur les 50 val.

## Résultats

| N démos | succès @4 pas | t_success | max_z | val-loss | best ckpt |
|---|---|---|---|---|---|
| **150** | **100 %** | 46.0 | 1.016 | 0.069 | 10500 |
| 100 | 96 % | 55.5 | 0.973 | 0.084 | 5250 |
| 50 | 94 % | 55.0 | 0.974 | 0.087 | 6000 |
| 20 | **98 %** | 68.0 | 0.977 | 0.096 | 5250 |
| 10 | **84 %** | 57.5 | 0.920 | 0.119 | 3000 |

Courbe : `../results/runs/lift/63_data_efficiency.png` · données : `63_data_efficiency.json`.

## Leçons clés

1. **Pas de falaise — une pente douce.** Contrairement au U-Net (capacité) et aux pas de diffusion, le succès ne s'effondre pas brutalement.
2. **Plateau ~94-100 % de 20 à 150 démos.** Les écarts 20/50/100 (94-98 %) sont **dans le bruit** (±3-4 % à 50 ép.) → essentiellement équivalents. **~20 démos suffisent** pour ~98 % sur Lift.
3. **Même 10 démos → 84 %** : utilisable, seul cran clairement en dessous.
4. **Le coût des données se voit surtout dans `t_success`** (46 → 55-68 steps) : moins de démos = levage plus lent/hésitant, plus que dans le taux de succès brut.
5. **La val-loss suit la dégradation douce** (0.069 → 0.119) — ici elle est un bon indicateur (pas de cliff).

## Détails techniques

- **Préfixes contigus** (`0..N-1`) imposés par l'EpisodeAwareSampler de lerobot-train (cf. `COMPRESSION.md`).
- ⚠️ Le « step du min de val-loss live » sort identique (4650) pour les runs à 6000 steps → **artefact de seed** (mêmes batchs val tirés dans le même ordre, seed=42), pas un vrai signal. C'est la **val-loss propre par checkpoint** (set complet) qui choisit le `best ckpt`.
- **Spécifique à Lift** (tâche visuellement et dynamiquement simple). Une tâche plus dure demanderait plus de démos.

## Suite

**Pour le bras réel** : sur une tâche type Lift, viser **~20-50 démos** donne déjà de hautes perfs ; 10 reste utilisable. Encourageant pour la contrainte « peu de données ».
Pistes pour pousser le bas (N=10-20) si besoin : **plus de démos**, **data augmentation**, régularisation — pas un plus gros modèle (qui sur-apprendrait davantage à données égales). Voir aussi : tâche plus dure, sim-to-real.
