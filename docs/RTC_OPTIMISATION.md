# Optimisation du RTC — journal des essais (Sam, 27/10 après-midi)

**But** : maximiser la réussite de E0 (80 démos humaines, sim durcie complète : 10 Hz, jeu 0,06, zone morte + bruit,
couleur) sous latence d'inférence, en ne changeant QUE l'exécution (aucun réentraînement).

**Contraintes du réel (Sam)** : inférence jusqu'à 300 ms = **3 décisions** ; **une seule inférence à la fois** (la
suivante ne démarre qu'à la livraison de la précédente, ou plus tard) → jamais « inférer à chaque mouvement ».
Criblage au pire cas (latence fixe 3), le meilleur réglage confirmé en latence variable 1-3 (100-300 ms).

**Protocole** : 200 épisodes de 40 s, critère strict, gb10, mêmes états ; p = McNemar apparié contre la référence
indiquée. Écarts < ~15 échecs = bruit probable → réplique sur E0_s43 (2e graine) avant de conclure.
Évaluation : `experiments/can/367_eval_seule_gb10.sh` → `results/runs/can/strict40_banc_NOM/bilan.json`.
Réglages : variables `SIMR_RTC_*` de `src/sim_reelle.py` (défauts = comportement d'avant ; sauvegarde
`results/sauvegardes/sim_reelle_avant_rtc_opt_27oct.py`).

| Réglage | Sens |
|---|---|
| `GEL` | actions du nouveau paquet remplacées par l'ancien plan (depuis « maintenant ») ; `auto` = latence du calcul |
| `GUIDE` | actions suivantes mélangées à l'ancien plan, poids décroissant |
| `HIST` | 1 = deux images successives (comme à l'entraînement) ; 0 = image fraîche dupliquée (Roby) |
| `S0` | indice de l'action « maintenant » (Roby 0 ; LeRobot 1) |
| `PERIODE` | décisions entre deux lancements (0 = à la livraison, soit toutes les L) |
| `LATENCE_MIN` | latence variable tirée à chaque calcul entre MIN et 3 |

## ⭐ Meilleur réglage confirmé

**Bilan 3 graines (21:21)** — RTC **gel 3, guide 0, indice Roby (S0 = 0)** contre saut de 3 actions :
latence fixe 300 ms : 115 / 124 / 128 contre 146 / 154 / 144 (**−21 % d'échecs**, chaque graine p ≤ 0,04) ;
latence variable 100-300 ms : 104 / 129 / 106 contre 127 / 132 / 126 (**−11 %**, 2 graines sur 3 significatives).
Réussite réaliste (latence variable) : **48 / 35,5 / 47 %** contre 36,5 / 34 / 37 % pour le saut et ~12 % pour le RTC d'origine.

**RTC gel 3, guide 0, mode Roby (HIST 0, S0 0)**, latence fixe 3 (pire cas) — 🔁 **répliqué** :
E0 **115 échecs (42,5 %)** contre saut de 3 146 (p=0,001) ; E0_s43 **124 (38,0 %)** contre saut de 3 154 (p=1e-4)
et RTC Roby d'origine 180 (p=6e-12). Sur Roby : `--rtc` avec gel = latence (3), guidage 0.

## Essais

| # | Nom | Modèle | Latence | Réglage | Échecs | Réussite | p (contre) | Lecture |
|---|---|---|---|---|---|---|---|---|
| 0a | f3_na8 | E0 | 3 fixe | saut, 8 actions/paquet | 177 | 11,5 % | — | référence sans RTC |
| 0b | f3_na3 | E0 | 3 fixe | saut, 3 actions/paquet | 146 | 27,0 % | 9e-5 (0a) | redécider tous les 3 pas |
| 0c | rtc | E0 | 3 fixe | Roby : gel 4, guide 4 | 175 | 12,5 % | 0,86 (0a) | guidage = actions jouées ≈ ancien plan |
| 0d | rtccorr | E0 | 3 fixe | corrigé : gel 4, guide 4, HIST 1, S0 1 | 187 | 6,5 % | 0,09 (0a) | |
| 1 | rtc_g3u2 | E0 | 3 fixe | Roby, gel 3, guide 2 | 157 | 21,5 % | 0,011 (0c) | guidage court coûte encore |
| 2 | **rtc_g3u0** | E0 | 3 fixe | **Roby, gel 3, guide 0** | **115** | **42,5 %** | 0,001 (0b) | ⭐ meilleur à ce jour |
| 3 | rtccorr_g3u0 | E0 | 3 fixe | corrigé, gel 3, guide 0 | 159 | 20,5 % | 2e-7 (2) | corrigé < Roby : inexpliqué |
| 0e | s43_f3_na8 / s43_f3_na3 / s43_rtc / s43_rtccorr | E0_s43 | 3 fixe | références 2e graine | 175 / 154 / 180 / 191 | 12,5 / 23,0 / 10,0 / 4,5 % | — | |
| 4 | **s43_rtc_g3u0** | E0_s43 | 3 fixe | **Roby, gel 3, guide 0** | **124** | **38,0 %** | 1e-4 (s43 na3) · 6e-12 (s43 rtc) | 🔁 **réplique le gain** |
| 5 | s43_rtccorr_g3u0 | E0_s43 | 3 fixe | corrigé, gel 3, guide 0 | 151 | 24,5 % | 7e-4 (4) | 🔁 le mode Roby bat le « corrigé » sur les 2 graines → décomposé par h1s0 / h0s1 |
| 6 | rtcopt_p4 | E0 | 3 fixe | gel 3, guide 0, calcul toutes les 4 décisions | 128 | 36,0 % | 0,14 (2) | plus rare = pas mieux (tendance pire) → garder période = latence |
| 7 | **rtcopt_var13** | E0 | **1-3 variable** | gel = latence du calcul (auto), guide 0 | **96** | **52,0 %** | 2e-4 (saut na3 même latence : 127) | ⭐ réaliste ; ⚠️ « auto » suppose connaître la durée du calcul à l'avance → testé avec gel fixe (8-10) |
| 8 | rtcopt_g4u0 | E0 | 3 fixe | gel 4, guide 0 | 142 | 29,0 % | 0,002 (2) | geler au-delà de la latence coûte → gel = latence |
| 9 | rtcopt_h0s1 | E0 | 3 fixe | image dupliquée, S0 = 1 (indice « corrigé » seul) | 165 | 17,5 % | 1e-9 (2) | 🔑 c'est l'INDICE de Roby (S0 = 0, actions jouées avec 1 pas de retard sur le plan) qui fait le gain → test d'un décalage plus grand (SAUT_DELTA) |
| 10 | rtcopt_s43_var13 | E0_s43 | 1-3 variable | gel auto, guide 0 | 131 | 34,5 % | 0,49 (4) · 9e-4 (E0_s43 saut na8 variable : 159) | gain de la latence variable NON répliqué contre la latence fixe ; référence saut na3 variable lancée |
| 11 | rtcopt_g2u0 | E0 | 3 fixe | gel 2, guide 0 | 104 | 48,0 % | 0,16 (2) | tendance meilleure, non significative |
| 12 | rtcopt_h1s0 | E0 | 3 fixe | historique d'images, S0 = 0 | 118 | 41,0 % | 0,80 (2) | l'historique (image dupliquée ou non) ne compte pas |
| 13 | rtcopt_var13_g1 | E0 | 1-3 variable | gel fixe 1, guide 0 | 98 | 51,0 % | 0,88 (7) | |
| 14 | **rtcopt_var13_g3** | E0 | 1-3 variable | **gel fixe 3** (réalisable : pas besoin de connaître la latence d'avance) | **104** | **48,0 %** | 0,37 (7) | ✅ le gel « auto » n'est pas nécessaire : gel 1, 3 ou auto ≈ pareil |
| 15 | rtcopt_var13_g2 | E0 | 1-3 variable | gel fixe 2 | 109 | 45,5 % | 0,10 (7) | ≈ |
| 16 | rtcopt_d1 | E0 | 3 fixe | gel 3, saut L-1 (décalage de 2 pas) | 145 | 27,5 % | 8e-4 (2) | ❌ décaler davantage nuit : le décalage d'UN pas est l'optimum |
| 17 | rtcopt_d1_g2 | E0 | 3 fixe | gel 2, saut L-1 | 127 | 36,5 % | 0,21 (2) | ❌ idem |
| 18 | rtcopt_g0s1 | E0 | 3 fixe | aucun gel, indice corrigé (= saut pur, image dupliquée) | 160 | 20,0 % | 4e-7 (2) | témoin ≈ saut de 3 (146) : l'image dupliquée ne coûte presque rien |
| 19 | rtcopt_s43_var_na3 | E0_s43 | 1-3 variable | saut, 3 actions/paquet (sans RTC) | 132 | 34,0 % | 1,0 (10) | ⚠️ sur E0_s43, le RTC en latence variable (131) ne bat PAS le saut de 3 (132) ; sur E0 si (96 contre 127) |
| 20 | **rtcopt_g0s0** | E0 | 3 fixe | **aucun gel, aucun guidage, indice Roby (S0 = 0)** = saut avec décalage d'1 pas | **123** | **38,5 %** | 0,35 (2) · 0,005 (saut na3 : 146) | 🔑 le gain vient presque entièrement du DÉCALAGE D'UN PAS, pas du raccord RTC |
| 21 | rtcopt_s43_g2u0 | E0_s43 | 3 fixe | gel 2, guide 0 | 136 | 32,0 % | 0,10 (4) | la tendance gel 2 > gel 3 de E0 ne se réplique pas → garder gel 3 |
| 22 | rtcopt_s43_g0s0 | E0_s43 | 3 fixe | aucun gel, indice Roby (décalage seul) | 148 | 26,0 % | 0,001 (4 : 124) · 0,5 (saut na3 : 154) | ❌ sur E0_s43 le décalage SEUL n'aide pas : il faut aussi le gel 3 → la ligne 20 est corrigée : le réglage robuste = gel 3 + guide 0 + indice Roby |
| 23 | rtcopt_var13_g0s0 | E0 | 1-3 variable | aucun gel, indice Roby | 106 | 47,0 % | 0,89 (14) · 0,02 (saut na3 var : 127) | |
| 24 | rtcopt_s43_var13_g3 | E0_s43 | 1-3 variable | gel 3, guide 0 | 129 | 35,5 % | 0,81 (19 : saut na3 var 132) | ⚠️ en latence variable, E0_s43 ne gagne rien sur le saut de 3 |
| 25 | rtcopt_s43_var13_g0s0 | E0_s43 | 1-3 variable | aucun gel, indice Roby | 128 | 36,0 % | 0,70 (19) | idem |
| — | (lancés 19:51) | E0_s44 | — | **3e graine** entraînée (`372`) + f3_na3, rtc_g3u0, var_na3, var13_g3 ; et période fixe 3 en latence variable (E0, E0_s43) | | | | pour trancher les désaccords entre graines |
| 26 | rtcopt_var13_g3_p3 | E0 | 1-3 variable | gel 3, guide 0, calcul toutes les 3 décisions même si livré plus tôt | 103 | 48,5 % | 1,0 (14) · 0,003 (saut na3 var : 127) | cadence fixe = cadence « dès livraison » : au choix, selon ce qui est simple sur Roby |
| 27 | rtcopt_s43_var13_g3_p3 | E0_s43 | 1-3 variable | idem | 129 | 35,5 % | 1,0 (24) · 0,8 (saut na3 var : 132) | idem ; toujours aucun gain sur E0_s43 en latence variable |
| 28 | E0_s44 (réf.) | E0_s44 | 1-3 variable | saut 8 actions (référence du banc) | 165 | 17,5 % | — | 3e graine cohérente avec E0 (159) et E0_s43 (159) |
| 29 | rtcopt_s44_f3_na3 | E0_s44 | 3 fixe | saut 3 actions | 144 | 28,0 % | — | (E0 146, E0_s43 154) |
| — | (lancés 20:51) | E0, E0_s43 | 3 fixe / 1-3 | **saut 3 actions + décalage d'1 pas** (`SIMR_DECALAGE=1`, préfixe des actions engagées décalé aussi) | | | | le décalage marche-t-il hors RTC ? |
| 30 | **rtcopt_s44_rtc_g3u0** | E0_s44 | 3 fixe | gel 3, guide 0, indice Roby | **128** | **36,0 %** | 0,036 (29 : saut na3 144) | 🔁 3e graine : le gain en latence fixe se réplique |
| 31 | rtcopt_s44_var_na3 | E0_s44 | 1-3 variable | saut 3 actions | 126 | 37,0 % | 2e-6 (28) | |
| 32 | **rtcopt_s44_var13_g3** | E0_s44 | 1-3 variable | gel 3, guide 0, indice Roby | **106** | **47,0 %** | 0,022 (31) | 🔁 gain en latence variable sur 2 graines sur 3 (E0 104/127, s44 106/126 ; s43 129/132) |
| 33 | rtcopt_f3_na3_dec1 | E0 | 3 fixe | saut 3 actions + décalage 1 (sans RTC) | 147 | 26,5 % | 1,0 (0b : 146) | ❌ le décalage n'aide qu'avec le gel du RTC |
| 34 | rtcopt_var_na3_dec1 | E0 | 1-3 variable | idem, latence variable | 123 | 38,5 % | 0,70 (saut na3 var : 127) | ❌ idem |
| — | (lancés 21:21) | E0 | 3 fixe / 1-3 | **gel donné par le canal de MASQUE** des actions engagées (comme à l'entraînement) au lieu de l'inpainting, S0 = 0 ou 1 | | | | |
| 35 | rtcopt_s43_f3_na3_dec1 | E0_s43 | 3 fixe | saut 3 + décalage 1 | 164 | 18,0 % | 0,23 (154) | ❌ |
| 36 | rtcopt_s43_var_na3_dec1 | E0_s43 | 1-3 variable | saut 3 + décalage 1 | 154 | 23,0 % | 0,006 (132) | ❌ nuit même : le décalage ne vaut QU'avec le gel RTC |
| — | (lancés 21:51) | E0 | **2 fixe / 1 fixe** | meilleur RTC (gel = latence, guide 0) à 200 ms et 100 ms | | | | combien rapporterait d'accélérer l'inférence ? |
| 37 | rtcopt_masque_s0 | E0 | 3 fixe | gel 3 par CANAL DE MASQUE (actions propres imposées), indice Roby | 118 | 41,0 % | 0,77 (2) | = inpainting : la méthode de gel est indifférente |
| 38 | rtcopt_masque_s1 | E0 | 3 fixe | gel par masque, indice LeRobot (S0 = 1, « comme à l'entraînement ») | 157 | 21,5 % | 1e-6 (2) | ❌ l'indice « correct » perd encore → l'indice Roby n'est pas un artefact de l'inpainting |
| 39 | rtcopt_var13_masque_s1 | E0 | 1-3 variable | idem en latence variable | 154 | 23,0 % | 3e-9 (14) | ❌ idem |
| 40 | rtcopt_lat2_g2 | E0 | **2 fixe (200 ms)** | gel 2, guide 0 | 104 | 48,0 % | 0,19 (2 : 115 à 300 ms) | inférence à 200 ms : ~−10 % d'échecs (à répliquer, lancé sur s43/s44) |
| 41 | rtcopt_lat1_g1 | E0 | **1 fixe (100 ms)** | gel 1, guide 0 → un calcul à CHAQUE décision, 1 action jouée par paquet | 148 | 26,0 % | 2e-4 (2 : 115) · 4e-7 (40 : 104) | ⚠️ PLUS RAPIDE = PIRE : replanifier à chaque pas casse l'intérêt des paquets → même avec une inférence rapide, garder ≥ 2-3 actions par paquet (testé : période 2 et 3 à 100 ms, période 3 à 200 ms) |
| 42 | rtcopt_s43_lat2_g2 | E0_s43 | 2 fixe | gel 2, guide 0 | 126 | 37,0 % | 0,91 (4 : 124) | |
| 43 | rtcopt_s44_lat2_g2 | E0_s44 | 2 fixe | gel 2, guide 0 | 113 | 43,5 % | 0,06 (30 : 128) | 200 contre 300 ms : 104/126/113 contre 115/124/128 → **−5 % seulement**, non significatif |
| 44 | rtcopt_var12_g2 | E0 | 1-2 variable | gel 2, guide 0 | 106 | 47,0 % | 0,90 (14 : 104 en 1-3) | une inférence plus rapide n'apporte presque rien une fois le RTC bien réglé |
| 45 | rtcopt_lat1_g1_p3 | E0 | 1 fixe | gel 1, calcul toutes les 3 décisions | 121 | 39,5 % | 6e-4 (41 : 148) | ✅ avec une inférence rapide, ATTENDRE et jouer 3 actions répare tout (≈ 300 ms : 115) |
| 46 | rtcopt_lat2_g2_p3 | E0 | 2 fixe | gel 2, calcul toutes les 3 décisions | 109 | 45,5 % | 0,58 (40 : 104) | ≈ |
| 47 | rtcopt_lat1_g1_p2 | E0 | 1 fixe | gel 1, calcul toutes les 2 décisions | 122 | 39,0 % | — (41 : 148) | 2 actions par paquet suffisent aussi |
| — | (lancés 23:21) | E0, E0_s43, E0_s44 | 1-3 variable | **confirmation sur 200 états NEUFS (200-399)** : RTC gel 3 contre saut de 3 (`373`) | | | | resserrer l'estimation du gain réaliste |

