# Le tableau des leviers — ce qui marche, en échecs supprimés

*Généré par `experiments/can/160_tableau_leviers.py`. Ne pas éditer à la main.*

Une ligne n'entre ici que si les deux bras diffèrent par **une seule chose**, sur les
**mêmes 500 états**, et si possible à graine égale. Les gains sont donnés en **réduction
du nombre d'échecs** et pas seulement en points : passer de 60 à 61 % est bien plus facile
que de 90 à 91 % (cf. `METHODOLOGIE.md` §6).

⚠️ McNemar compare deux **modèles**, pas deux **configurations** : il ne voit pas le hasard
d'entraînement. Une ligne à p < 0,05 sur une seule graine reste à confirmer.

## Ce qui améliore

| levier | réussite | échecs /500 | échecs en moins | McNemar | appariement |
|---|---|---|---|---|---|
| ⭐ Décodeur ×2 · graine 42 APPARIÉE | 84.2 → 93.8 % | 79 → 31 | **+60.8 %** | <1e-4 | graine 42, poignet NEUF des 2 côtés |
| Décodeur ×2 (mono, graine 42) | 80.4 → 91.2 % | 98 → 44 | **+55.1 %** | <1e-4 | graine 42 |
| Décodeur ×2 (bi-caméra gelé) | 86.2 → 93.8 % | 69 → 31 | **+55.1 %** | 0.0001 | graine 42 |
| Décodeur ×2 (mono, graine 43) | 76.6 → 89.0 % | 117 → 55 | **+53.0 %** | <1e-4 | 2e réplique de chaque côté |
| ⭐ Décodeur ×2 · graine 43 APPARIÉE | 85.4 → 91.4 % | 73 → 43 | **+41.1 %** | 0.0029 | graine 43, poignet NEUF des 2 côtés |
| Poignet REPRIS vs NEUF (graine 43) | 85.4 → 91.2 % | 73 → 44 | **+39.7 %** | 0.0015 | graine 43, décodeur identique |
| Encodeur GELÉ pour intégrer le poignet | 80.4 → 86.2 % | 98 → 69 | **+29.6 %** | 0.0134 | graine 42 |
| Alterner plutôt que figer (porte) | 78.0 → 82.2 % | 110 → 89 | **+19.1 %** | 0.0504 | MÊME modèle |
| ⭐ 5 pas de débruitage au lieu de 10 (gros décodeur) | 93.8 → 94.8 % | 31 → 26 | **+16.1 %** | 0.4869 | MÊME modèle, seul le réglage d'exécution change |
| ⭐ 5 pas de débruitage au lieu de 10 (petit décodeur) | 87.6 → 89.6 % | 62 → 52 | **+16.1 %** | 0.2451 | MÊME modèle |
| Rattrapages COUPÉS vs complets | 66.8 → 72.0 % | 166 → 140 | **+15.7 %** | 0.0563 | graine 42 |
| Poignet REPRIS vs NEUF (graine 42) | 84.2 → 86.2 % | 79 → 69 | **+12.7 %** | 0.3729 | graine 42, décodeur identique |
| Encodeur poignet NEUF vs repris | 84.2 → 86.2 % | 79 → 69 | **+12.7 %** | 0.3729 | graine 42 |
| 3 pas de débruitage au lieu de 10 (petit décodeur) | 87.6 → 89.0 % | 62 → 55 | **+11.3 %** | 0.4188 | MÊME modèle |
| Porte DURE vs porte souple | 81.6 → 83.6 % | 92 → 82 | **+10.9 %** | 0.3481 | MÊME modèle K |
| 2e caméra EXTÉRIEURE (dessus) | 80.4 → 82.0 % | 98 → 90 | **+8.2 %** | 0.5159 | graine 42 |
| Dropout de caméra (poignet naïf) | 61.6 → 64.2 % | 192 → 179 | **+6.8 %** | 0.2181 | graine 42 |
| Décodeur ×2 (bi-caméra, graine 43) | 91.2 → 91.4 % | 44 → 43 | **+2.3 %** | 1.0000 | graine 43 |

## Ce qui dégrade

| levier | réussite | échecs /500 | échecs en moins | McNemar | appariement |
|---|---|---|---|---|---|
| Porte apprise vs TIRAGE AU SORT | 82.2 → 82.2 % | 89 → 89 | **+0.0 %** | 1.0000 | MÊME modèle L42 |
| +200 épisodes experts | 80.4 → 79.4 % | 98 → 103 | **-5.1 %** | 0.7228 | graine 42 |
| Routeur souple sur modèle gelé | 86.2 → 83.2 % | 69 → 84 | **-21.7 %** | 0.1837 | graine 42 |
| 20 pas de débruitage au lieu de 10 | 93.8 → 92.4 % | 31 → 38 | **-22.6 %** | 0.3713 | MÊME modèle — au-delà du point de rendement |
| 20 pas de débruitage au lieu de 10 (petit décodeur) | 87.6 → 83.0 % | 62 → 85 | **-37.1 %** | 0.0128 | MÊME modèle — l'excès de pas COÛTE |
| Décodeur ×4 vs ×2 (bi-caméra) | 93.8 → 91.0 % | 31 → 45 | **-45.2 %** | 0.0980 | graine 42 — le 2e doublement |
| Routeur souple sur poignet naïf | 61.6 → 42.2 % | 192 → 289 | **-50.5 %** | <1e-4 | graine 42 |
| Décodeur ×4 vs ×2 (mono) | 91.2 → 86.0 % | 44 → 70 | **-59.1 %** | 0.0054 | graine 42 — le 2e doublement |
| Rattrapages COMPLETS | 79.4 → 66.8 % | 103 → 166 | **-61.2 %** | <1e-4 | graine 42, données égales |
| Recadrage aléatoire (poignet naïf) | 61.6 → 34.8 % | 192 → 326 | **-69.8 %** | <1e-4 | graine 42 |
| Poignet en concaténation NAÏVE | 80.4 → 61.6 % | 98 → 192 | **-95.9 %** | <1e-4 | graine 42 |
| Recadrage aléatoire (mono, 96 px) | 80.4 → 21.6 % | 98 → 392 | **-300.0 %** | <1e-4 | graine 42 |
| 2 pas de débruitage au lieu de 10 | 93.8 → 54.6 % | 31 → 227 | **-632.3 %** | <1e-4 | MÊME modèle — la falaise |
