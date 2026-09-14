# Protocole séquentiel — tête auxiliaire (décidé AVANT de voir le résultat)

Le plancher de bruit du banc est de **4 à 8 points** sur 500 rollouts (mesuré sur deux
répliques involontaires : 54,2/61,6 pour le poignet, 76,6/80,4 pour la caméra seule).

À 3 graines par groupe, un test de Welch ne détecte qu'un effet d'environ **12 points**.
Payer 8 h de machine pour apprendre « non concluant » n'a pas de sens quand un seul run
suffit à trancher dans la majorité des cas.

## La règle de décision, fixée d'avance

Référence poignet observée : **54,2 % et 61,6 %** (moyenne 57,9, écart-type 5,2).
Bande de bruit à ±2 écarts-types : environ **[48 ; 68]**.

| résultat de `aux_s42` | lecture | suite |
|---|---|---|
| **≥ 72 %** | hors bande, effet large | concluant — 1 graine de confirmation |
| **≤ 64 %** | dans la bande | pas d'effet exploitable — on arrête |
| **65 – 71 %** | ambigu | on lance les graines 43 et 44 |

Cette règle est écrite **avant** la mesure. Ajouter des graines après avoir vu un chiffre
prometteur, puis annoncer le résultat, serait du bricolage statistique.

## Ce qui tourne quand même

`ref_s43` — troisième réplique de la référence. Elle ne teste rien, mais elle resserre
l'estimation du bruit, qui sert d'étalon à **toutes** les comparaisons du dépôt. C'est
l'actif le plus réutilisable de la campagne.
