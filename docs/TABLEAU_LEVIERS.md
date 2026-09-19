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
| Décodeur ×2 (mono, graine 42) | 80.4 → 91.2 % | 98 → 44 | **+55.1 %** | <1e-4 | graine 42 |
| Décodeur ×2 (bi-caméra gelé) | 86.2 → 93.8 % | 69 → 31 | **+55.1 %** | 0.0001 | graine 42 |
| Décodeur ×2 (mono, graine 43) | 76.6 → 89.0 % | 117 → 55 | **+53.0 %** | <1e-4 | 2e réplique de chaque côté |
| Encodeur GELÉ pour intégrer le poignet | 80.4 → 86.2 % | 98 → 69 | **+29.6 %** | 0.0134 | graine 42 |
| Alterner plutôt que figer (porte) | 78.0 → 82.2 % | 110 → 89 | **+19.1 %** | 0.0504 | MÊME modèle |
| Rattrapages COUPÉS vs complets | 66.8 → 72.0 % | 166 → 140 | **+15.7 %** | 0.0563 | graine 42 |
| Encodeur poignet NEUF vs repris | 84.2 → 86.2 % | 79 → 69 | **+12.7 %** | 0.3729 | graine 42 |
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
| Décodeur ×4 vs ×2 (bi-caméra) | 93.8 → 91.0 % | 31 → 45 | **-45.2 %** | 0.0980 | graine 42 — le 2e doublement |
| Routeur souple sur poignet naïf | 61.6 → 42.2 % | 192 → 289 | **-50.5 %** | <1e-4 | graine 42 |
| Décodeur ×4 vs ×2 (mono) | 91.2 → 86.0 % | 44 → 70 | **-59.1 %** | 0.0054 | graine 42 — le 2e doublement |
| Rattrapages COMPLETS | 79.4 → 66.8 % | 103 → 166 | **-61.2 %** | <1e-4 | graine 42, données égales |
| Recadrage aléatoire (poignet naïf) | 61.6 → 34.8 % | 192 → 326 | **-69.8 %** | <1e-4 | graine 42 |
| Poignet en concaténation NAÏVE | 80.4 → 61.6 % | 98 → 192 | **-95.9 %** | <1e-4 | graine 42 |
| Recadrage aléatoire (mono, 96 px) | 80.4 → 21.6 % | 98 → 392 | **-300.0 %** | <1e-4 | graine 42 |

## ⭐ Verdict au niveau du GROUPE (le seul qui voie le hasard d'entraînement)

McNemar compare deux **modèles**. Pour trancher entre deux **configurations**, il faut plusieurs
entraînements par condition et un test de groupe (Welch, une observation = un entraînement).

| levier | échecs (moyenne) | en moins | Welch |
|---|---|---|---|
| **décodeur ×2, bi-caméra gelé** | 69,2 → 37,0 | **−46,6 %** | **p = 0,042** ✅ |
| décodeur ×2, mono | 107,5 → 49,5 | −54,0 % | p = 0,052 |

*Bi-caméra : 4 graines `[64,128,256]` (69·44·85·79) contre 2 graines `[128,256,512]` (31·43).
Mono : 2 répliques de chaque côté (98·117 contre 44·55).*

⚠️ L'estimation a **baissé** en passant d'une à deux graines — de 55,1 % à 46,6 %. Retour à la
moyenne classique : c'est exactement pourquoi on ne conclut pas sur une graine.

## La taille du décodeur a un OPTIMUM

| décodeur | mono | bi-caméra gelé |
|---|---|---|
| `[64,128,256]` | 107,5 échecs | 69,2 échecs |
| **`[128,256,512]`** | **49,5** | **37,0** |
| `[256,512,1024]` | 70,0 | 45,0 |

Le **second** doublement rajoute des échecs : +41 % en mono, +22 % en bi-caméra. Deux
configurations indépendantes, même sommet. Explication la plus simple : les trois tailles voient
le **même budget** (320k échantillons) pour ~28, 41 et 90 M de paramètres. La recette standard à
96 % emploie un décodeur encore plus large — mais avec 2,56 M d'échantillons, 8× plus.

**→ À budget fixé, la taille du décodeur s'accorde, elle ne se maximise pas.**
