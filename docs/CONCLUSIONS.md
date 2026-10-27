# Toutes nos conclusions — une ligne chacune

> Récapitulatif du projet (mai → 27/10/2026). Chaque ligne : la conclusion, le chiffre qui la porte, son statut, et le
> lien vers l'explication / les preuves. Lecture : **A > B** = A meilleur que B ; « échecs » = sur 200 ou 500 épisodes.
>
> **Statuts** : ✅ établi · 🔁 répliqué (2 graines ou plus) · ❌ réfuté (on l'a cru, c'est faux) · ⚠️ contaminé
> (évaluation vision d'avant le 23/09, où un repère robosuite était dessiné au point de saisie — à refaire) ·
> 🔄 à rejuger (mesuré dans de mauvaises conditions) · 🧭 décision.
>
> Liens « mémoire » = notes de Claude dans `~/.claude/projects/…/memory/` (hors dépôt, lisibles sur le Mac).

[M]: ../../../../.claude/projects/-Users-nielsmurawka-Documents-VScodeProject-experience-Le/memory

## 1. Mesurer juste (méthodologie)

| Conclusion | Chiffre clé | Statut | Explication / preuves |
|---|---|---|---|
| Juger au **taux de succès**, jamais à la val-loss | Can : brut ~20 % contre EMA ~74 % à loss égale ; corr. loss↔succès +0,27 | ✅ | [METHODOLOGIE §4](METHODOLOGIE.md#4-quand-arrêter-sans-simulation-) · mémoire `feedback_val_loss_not_judge` |
| Une loss normale peut cacher une tâche détruite | actions absolues : perte 0,025 (normale), 454 échecs contre 207 | ✅ | [TABLEAU_LEVIERS](TABLEAU_LEVIERS.md#ce-qui-dégrade) |
| **500 rollouts > 50** (ou 200 appariés) | IC ±12,7 pts à 50 contre ±4 à 500 ; même modèle 20-36 % à n=50 | ✅ | [METHODOLOGIE §1](METHODOLOGIE.md#1-combien-de-rollouts---lintervalle-de-confiance) |
| Deux entraînements identiques diffèrent (bruit inter-run) | 7,4 pts (p=0,0007) ; à graines homogènes σ≈0,6 pt | ✅ | [BALAYAGE_UNET](BALAYAGE_UNET.md#-le-plancher-de-bruit-du-projet-était-surestimé) · [README](../README.md#le-plancher-de-bruit-et-pourquoi-il-change-tout) |
| Sim durcie : la moitié des états se jouent à pile ou face | 68/200 basculent entre deux graines | ✅ | [AUTOPSIE_C12 §1](recherche/AUTOPSIE_C12.md#1-le-bruit-dabord--la-moitié-des-états-se-jouent-à-pile-ou-face) |
| Compter en **échecs supprimés** ; les leviers se multiplient | 98 × 0,449 × 0,704 = 31 prédit, 31 observé | ✅ | [METHODOLOGIE §6](METHODOLOGIE.md#6-le-point-de-pourcentage-nest-pas-la-bonne-échelle) |
| Le seuil de significativité se resserre près de 0/100 % | 6,3 pts à 50 %, 3,0 pts à 94 % (n=500) | ✅ | [METHODOLOGIE §7](METHODOLOGIE.md#7-le-seuil-de-significativité-dépend-du-niveau) |
| Le succès n'est pas lisse : interpoler entre checkpoints est faux | ~33 pts entre checkpoints voisins | ✅ | [METHODOLOGIE §2](METHODOLOGIE.md#2-le-succès-nest-pas-lisse--interpoler-est-invalide) |
| Loss au plancher ≠ convergé | mini-CNN 2 % → 74 % en prolongeant | ✅ | [METHODOLOGIE §3](METHODOLOGIE.md#3-loss-au-plancher---convergé-) |
| Établir la **courbe de budget** avant de comparer des leviers | square 10k 290 → 30k 232 échecs : un banc entier jugé à 1/3 du budget | ✅ | mémoire `feedback_plateau_avant_leviers` |
| Une config n'est valide qu'avec son budget | R34 + LR constant : 2,7 % à 320k échantillons, 67 % à 960k | ✅ | mémoire `feedback_budget_vs_config` |
| Poser le contrôle sur données saines EN PREMIER | 4 fausses hypothèses, ~20 h perdues | ✅ | [MEMOIRE leçons](recherche/MEMOIRE.md#-leçons-de-méthode) |
| L'éval dessinait un **repère au point de saisie** absent de l'entraînement | poignet seul 274 → 27 échecs une fois retiré (p=4e-70) | ✅ (évals antérieures ⚠️) | mémoire `project_repere_visualisation` · `src/can_eval.py` |
| L'étiquette « agentview masquée » était fausse : c'est birdview | canette invisible 0,0 % (agentview) contre 57,7 % (birdview) | ✅ | mémoire `project_camera_occlusion_erreur` |
| Les épisodes ne sont pas indépendants (`reset_to` = physique seule) | correcteur déterministe : 172/163/163/169 pas selon l'épisode d'avant | ✅ | mémoire `project_episodes_non_independants` |
| Comparer les évals sur la **même machine** (Mac ≠ atomman) | même soupe 147 = 147, mais 64/200 épisodes basculent | ✅ | [HGDAGGER_CAMPAGNES C11](HGDAGGER_CAMPAGNES.md#campagne-11-hgd11-lancée-le-2410-à-23-h--sim-rapprochée-du-réel--actions-engagées) |
| Le tirage des états initiaux change le score à lui seul | même élève 43 % (collecte) contre 36 % (éval) | ✅ | mémoire `project_declencheur_hgdagger` |
| Critère strict (écrou enfilé) ≈ critère officiel robosuite | ≤ 3-4 échecs d'écart | ✅ | [AUTOPSIE_C12 §9](recherche/AUTOPSIE_C12.md#9-succès-strict-srcsucces_strictpy-348-349-2610) |
| Rollouts non reproductibles (débruitage sans graine) ; McNemar reste valide | ~100 000 tirages non grainés par éval | ✅ | mémoire `project_mesures_concurrentes` |
| « La concurrence GPU fausse les mesures » | 2 évals simultanées dans la plage normale | ❌ | mémoire `project_mesures_concurrentes` |
| « Les stats image corrigées valent +7,4 pts » | stats jamais lues (`use_imagenet_stats=True`) : c'était du bruit | ❌ | mémoire `project_stats_images_bug` |
| Juger un modèle linéaire en forme close, pas à la descente de gradient | MAE 61 (SGD non convergé) contre 13,8 | ✅ | `experiments/can/187_lineaire_forme_close.py` |

## 2. Entraînement

| Conclusion | Chiffre clé | Statut | Explication / preuves |
|---|---|---|---|
| **Recuit** (LR → 0 sur ~5k pas) > checkpoint brut | joint 15 % → 83 % ; mini-CNN ~60 → 80 % | ✅ | mémoire `feedback_cooldown_default` |
| **LR constant** > cosine pour expérimenter (le cosine confond budget et convergence) | cosine annealé = faux plateau ; constant 2 → 74 % | ✅ 🧭 | [SCHEDULE §1](SCHEDULE.md#1-le-piège--un-cosine-annealé-à-0--faux-plateau) |
| Cosine bien réglé > constant à budget égal (run31) | 94,8 % contre ~50-69 % | ✅ | [SCHEDULE](SCHEDULE.md#suite---run31-cosine-vs-run31-jumeau-constant-mesure-directe-même-archidonnées) |
| **EMA** > poids bruts, et plus stable | 80k : 74 % contre 20 % | ✅ | [ENSEMBLING_JOINT EMA](ENSEMBLING_JOINT.md#ema--settling--online--résultats) |
| En boucle (HG-DAgger) : jouer/évaluer l'EMA, entraîner le brut | brut 11,7 % contre EMA 36,7 % | ✅ | mémoire `project_poids_chauds_hgdagger` |
| **SWA** (moyenne de ~10 checkpoints tardifs) > meilleur checkpoint seul | joint 88 % contre 56 % | ⚠️ | [ENSEMBLING_JOINT SWA](ENSEMBLING_JOINT.md#swa--moyennage-de-poids-de-checkpoints-n50) |
| Un checkpoint cassé empoisonne une petite moyenne | best + nul → 8 % | ✅ | [ENSEMBLING_JOINT](ENSEMBLING_JOINT.md#swa--moyennage-de-poids-de-checkpoints-n50) |
| SWA inutile si les poids sont groupés, nuisible à travers les stades | cos late5 97 ≈ 94,8 ; late10 89 | ✅ | [ENSEMBLING_JOINT matrice](ENSEMBLING_JOINT.md#matrice-schedule--swa-quand-le-swa-aide-t-il-) |
| Prolonger loin ne dégrade pas (pas de vrai surapprentissage) | 80k → 300k stable ~76-81 % | ✅ | [SCHEDULE §3](SCHEDULE.md#3-extensions-80k300k--ça-sature-ça-ne-se-dégrade-pas) |
| Le succès décolle bien après la convergence de la loss | loss à 750-6000 pas, succès à 6-34k | ✅ | [CONVERGENCE](CONVERGENCE.md#leçons) |
| Sim durcie : entraîner plus longtemps n'aide pas | 15k 159 → 100k 171 échecs | ✅ | mémoire `project_banc_nuit_27oct` |
| 10 tours de 10k (Adam neuf à chaque fois) ≡ 100k d'un trait | 171 contre 171, p=1 | ✅ | mémoire `project_banc_nuit_27oct` · `experiments/can/368_long_gb10.sh` |
| Sim durcie : le budget plafonne tôt | ~150 échecs dès 7-10k | ✅ | mémoire `project_actions_engagees_c11` |

## 3. Architecture

| Conclusion | Chiffre clé | Statut | Explication / preuves |
|---|---|---|---|
| **CNN pré-entraîné ImageNet (gel strict) > CNN appris de zéro** | square : plafond 80-84 % contre ~60 % ; HG-DAgger soupe 90 % contre 77 % (p=9e-4) | ✅ | [HGDAGGER_CAMPAGNES](HGDAGGER_CAMPAGNES.md#résultats-mac-200-essais-jeu-006-appariés) · mémoire `project_c9_c10_encodeur` |
| ImageNet entièrement entraînable : chute au 1er tour HG-DAgger | 83 → 58 % à t01 | ✅ | mémoire `project_c9_c10_encodeur` |
| Le « gel » BatchNorm ne figeait que les poids (bug corrigé) | running_var jusqu'à 28 d'écart ; gel strict gratuit (78,5 contre 79,5 %) | ✅ | [HGDAGGER_CAMPAGNES gel](HGDAGGER_CAMPAGNES.md#-le-gel-ne-gelait-que-la-moitié--corrigé-avant-c10) · `experiments/can/318_verifier_gel.py` |
| Geler seulement conv1+bn1 : nul | 159 / 162 contre 159 | ✅ | mémoire `project_banc_nuit_27oct` |
| ResNet34 ImageNet à 96 px < de zéro (Can joint) | 24-28 % contre 83 % | ⚠️ | mémoire `project_backlog` |
| **Gros U-Net [128,256,512] > [64,128,256]** quand les données s'accumulent | HG-DAgger −23 % d'échecs (p=5e-6) ; nul au départ | ✅ | mémoire `project_gros_unet_hgdagger` |
| Décodeur ×2 supprime ~55 % des échecs (Can) | 76 → 34 échecs (3 graines) | ⚠️ | [BALAYAGE_UNET](BALAYAGE_UNET.md#le-résultat-apparié-proprement) |
| La taille du décodeur a un optimum (×4 rajoute des échecs) | +37 % d'échecs | ⚠️ | [BALAYAGE_UNET](BALAYAGE_UNET.md#la-taille-a-un-optimum-elle-ne-se-maximise-pas) |
| Le décodeur porte la perf et n'est pas compressible (vision pure) | ÷2 : −18 pts (p=0,002) | ⚠️ | [MEMOIRE](recherche/MEMOIRE.md#les-3-résultats-significatifs) |
| **5 pas de débruitage = 10** (−45 % de calcul) ; 2 pas = falaise | 26 contre 31 ; 2 pas 227 | ⚠️ (à re-mesurer) | [TABLEAU_LEVIERS](TABLEAU_LEVIERS.md#ce-qui-améliore) · mémoire `project_pas_debruitage` |
| DDIM déterministe < échantillonnage stochastique | 173 contre 159 (p=0,03) | ✅ | mémoire `project_banc_nuit_27oct` |
| Encodeurs séparés > encodeur partagé (2 caméras) | 55,2 contre 45,0 % | ⚠️ | [CAN_archive](CAN_archive.md#2-cams-encodeurs-séparés-08_proprio_wrist_sep--55--500) |
| Nombre de keypoints du spatial softmax : aucun effet | K=32-256 : 53-65 %, p≥0,10 | ⚠️ | [README](../README.md#plus-de-points-dintérêt-naide-pas) |
| Le calcul est dans le décodeur (déroulé à chaque pas) | 89 % des paramètres appliqués | ✅ | mémoire `project_pas_debruitage` |
| Diffusion Policy ≫ meilleur BC (Lift) | 100 % contre 70 % | ✅ | [LIFT](LIFT.md#résultats) |
| Lift : U-Net réductible ÷160, latence ÷21, mais grâce aux coordonnées objet | 98,6 % à 1,65 M / 44 ms | ✅ | [COMPRESSION](COMPRESSION.md#résultats) |
| Vision pure : mini-CNN ≪ ResNet18 (la béquille cachait le problème) | 45-65 % erratique contre ~99 % | ✅ | [COMPRESSION](COMPRESSION.md#vision--mini-cnn-vs-resnet18--le-mur-que-la-béquille-cachait) |

## 4. Caméras / vision

| Conclusion | Chiffre clé | Statut | Explication / preuves |
|---|---|---|---|
| **agentview + poignet > birdview + poignet** (square) | 207 contre 294 échecs (−30 %, p=5,5e-9) | ✅ | [TABLEAU_LEVIERS](TABLEAU_LEVIERS.md#ce-qui-améliore) · mémoire `project_square_leviers_nuls` |
| Voir l'objet ≠ utile : birdview le voit 100 % mais seule = 467 échecs | prédiction réfutée | ❌ | mémoire `project_square_leviers_nuls` |
| **Une vue inutile coûte à l'entraînement** ; la masquer après coup ne récupère rien | 3 cam 236, masquée 251, 2 cam 207 (p=0,0014) | ✅ | [TABLEAU_LEVIERS](TABLEAU_LEVIERS.md#ce-qui-dégrade) · mémoire `project_camera_dilution` |
| **Recadrage aléatoire dégrade** | 207 → 319 (p=9e-13) | ✅ | mémoire `project_recadrage_aleatoire` |
| 160 px < 96 px en sim durcie | 177 contre 159 (p=0,018) | ✅ | mémoire `project_banc_nuit_27oct` |
| 160 px = 96 px ; décodeur ×2 et 2e caméra nuls (square, birdview, 10k) | p=1,00 / 0,24 / 0,59 | 🔄 | mémoire `project_square_leviers_nuls` |
| Poignet greffé sur l'encodeur GELÉ de la vue globale > vue seule > conjoint | 86,2 contre 78,5 contre 54,8 % | ⚠️ | [POURQUOI_LE_POIGNET](POURQUOI_LE_POIGNET.md#le-résultat-1) |
| « Le poignet dégrade à toute capacité » | venait du repère dessiné au point de saisie | ❌ | mémoire `project_wrist_camera` |
| Vue extérieure = localisation grossière, poignet = alignement fin | sans poignet 18 % (4 cm), sans extérieure 0 % | ⚠️ | [PORTE_DURE](PORTE_DURE.md#-ce-que-fait-chaque-caméra) |
| Porte dure (une vue encodée par pas) : −25 % d'encodage sans coût démontré | 83,4 contre 86,2 % (p=0,28) | ⚠️ | [PORTE_DURE](PORTE_DURE.md#résultats) |
| Entraîner sur des vues périmées échoue | k=6 : 74,5 → 30,5 % | ⚠️ | [PORTE_DURE add. 3](PORTE_DURE.md#addendum-3--entraîner-sur-la-péremption--échec) |
| Décaler la caméra effondre une politique non augmentée ; l'augmentation exige de la capacité | 74 → 13 % dès 2 cm/2° | ✅ | [CAMERA](CAMERA.md#leçons-clés) |
| La politique est en boucle fermée (corrige un objet déplacé de 4 cm) | 88 % de corrections | ⚠️ | [PORTE_DURE](PORTE_DURE.md#la-policy-est-elle-en-boucle-fermée-) |

## 5. Actions et exécution

| Conclusion | Chiffre clé | Statut | Explication / preuves |
|---|---|---|---|
| **Déplacements (delta) ≫ coordonnées absolues** — le plus gros effet du projet | 207 contre 454 échecs (p=9e-59) | ✅ | [TABLEAU_LEVIERS](TABLEAU_LEVIERS.md#ce-qui-dégrade) · mémoire `project_actions_absolues` |
| Cartésien (OSC) ≫ joint absolu | 94,8 % contre ~15 % | ✅ 🧭 | [ENSEMBLING_JOINT](ENSEMBLING_JOINT.md#leçons-clés) |
| Delta joint : effondrement de magnitude (séquentiel) / affamé de résolution (par paquet) | 0/50 ; 6 % à 96 px contre 42 % à 224 | ⚠️ | mémoire `project_backlog` |
| Temporal ensembling = multiplicateur de marge, KO en multimodal | joint 56 → 90 ; plafond 97 → 97 ; PushT 0,67 → 0,40 | ⚠️ | [ENSEMBLING_JOINT](ENSEMBLING_JOINT.md#temporal-ensembling-sans--avec-m001-n5030) |
| **Redécider plus souvent** (paquets courts, n_action ≥ latence) | 8 → 177 ; 3 → 146 (p=9e-5) ; répliqué 175 → 154 | 🔁 | mémoire `project_durcissement_rtc_27oct` · [RTC_OPTIMISATION](RTC_OPTIMISATION.md) |
| Paquets de 1-2 actions sous latence 3 = gain | artefact : la sim utilisait des actions décidées dans le futur | ❌ | mémoire `project_banc_nuit_27oct` |
| RTC de Roby (gel 4, guide 4) n'aide pas | 175 contre 177 ; corrigé 187 | ✅ | [RTC_OPTIMISATION](RTC_OPTIMISATION.md) |
| **Le guidage était le coupable : RTC gel 3 sans guidage > saut de 3** | E0 115 contre 146 (p=0,001) ; E0_s43 124 contre 154 (p=1e-4) | 🔁 |
| RTC : garder l'indice « faux » de Roby (actions jouées un pas en retard sur le plan) | le corriger : 115 → 165 échecs (p=1e-9) ; décaler de 2 : 145 ; décalage sans gel ou hors RTC : aucun effet | 🔁 | [RTC_OPTIMISATION](RTC_OPTIMISATION.md) |
| RTC gel 3 / guide 0 / indice Roby, 3 graines : latence fixe −21 % d'échecs, latence variable 100-300 ms −11 % | ~47 % de réussite réaliste contre ~36 % (saut) et ~12 % (RTC d'origine) | 🔁 | [RTC_OPTIMISATION](RTC_OPTIMISATION.md) | [RTC_OPTIMISATION](RTC_OPTIMISATION.md) |
| RTC : la méthode de gel (inpainting ou canal de masque), l'historique d'images et la cadence (dès livraison ou fixe) ne comptent pas | 118 contre 115 ; 118 contre 115 ; 103 contre 104 | ✅ | [RTC_OPTIMISATION](RTC_OPTIMISATION.md) |
| Inférence plus rapide : 200 ms ne rapporte que ~5 % ; 100 ms en recalculant à chaque pas est PIRE | 300 ms 115/124/128 → 200 ms 104/126/113 ; 100 ms 148 | 🔁 (200 ms) / ✅ | [RTC_OPTIMISATION](RTC_OPTIMISATION.md) |
| Quelle que soit la vitesse d'inférence, jouer 2-3 actions par paquet | 100 ms : 1 action/paquet 148 → 2 actions 122, 3 actions 121 | ✅ | [RTC_OPTIMISATION](RTC_OPTIMISATION.md) |
| Conditionner sur les actions engagées : aucun gain | 156 contre 150 (p=0,50) | ✅ | [SIM_REELLE](recherche/SIM_REELLE.md#test-du-préfixe-2510-335_test_prefixe_macsh) |
| ~3/4 du coût de la latence vient du **raccord** entre paquets, pas de l'âge des images | âge seul 96 → 109 ; latence 300 ms ~152 | ✅ | [SIM_REELLE](recherche/SIM_REELLE.md#raccord-ou-âge-des-images--2410-328-modèle-7k-10-hz--jeu-006-200-rollouts) |
| Les allers-retours à l'approche viennent du raccord | inversions 15,6 % juste après un paquet contre 3-8 % | ✅ | [AUTOPSIE_C12](recherche/AUTOPSIE_C12.md#a-le-bras-avance-et-recule-à-lapproche--cest-le-raccord) |
| Un mauvais ordre des caméras ne plante pas : il fausse en silence | — | ✅ | [DEPLOIEMENT_CART160](DEPLOIEMENT_CART160_3CAM.md#-le-point-bloquant--la-correspondance-caméras) |

## 6. Sim → réel (ce qui rend la tâche dure)

| Conclusion | Chiffre clé | Statut | Explication / preuves |
|---|---|---|---|
| **Trois coupables : cadence 10 Hz, latence, peu de démos** | sim propre 44 → 10 Hz 83 → +latence 133 → E0 159-162 | ✅ | mémoire `project_durcissement_rtc_27oct` · `experiments/can/370_durcir_gb10.sh` |
| Décider à 10 Hz au lieu de 20 Hz | 44 → 83 (p=2e-5) | ✅ | idem |
| Latence 100-300 ms | 44 → 67 seule (p=0,006) ; +37 en cumul | ✅ | idem · [AUTOPSIE_C12](recherche/AUTOPSIE_C12.md#2-contrefactuels--cest-la-latence) |
| Même un oracle parfait s'effondre sous latence | 99 % → 55,5 % (1-3) → 0,5 % (300 ms fixes) | ✅ | [AUTOPSIE_C12](recherche/AUTOPSIE_C12.md#loracle-correcteur-scripté-qui-lit-la-vérité-sous-la-même-latence) |
| Dérive de couleur : coûte un peu | 44 → 59 (p=0,04) | ✅ | mémoire `project_durcissement_rtc_27oct` |
| Jeu articulaire, zone morte, bruit, épisodes 40 s : **nuls** | 41 / 35 / 38 contre 44 | ✅ | idem |
| Biais du zéro articulaire (3,4°) coûte au modèle | 58,6 → 36,0 % ; les anciens chiffres (25,6 / 17,2 %) étaient contaminés | ✅ (anciens ❌) | mémoire `project_biais_calibration` |
| Les échecs sont au contact fin (poteau 44 %, prise 15 %) | décalage non résorbé, sans oscillation | ✅ | [AUTOPSIE_C12](recherche/AUTOPSIE_C12.md#3-entonnoir--où-sarrêtent-les-132-échecs) |
| Sim durcie désactivée = identique au bit près | 12/12 épisodes | ✅ | [SIM_REELLE](recherche/SIM_REELLE.md#simulation-rapprochée-du-réel-square--23-24102026) |

## 7. Données

| Conclusion | Chiffre clé | Statut | Explication / preuves |
|---|---|---|---|
| **150 démos ≫ 80** | 44 contre 100 échecs (p=1e-8) | ✅ | mémoire `project_durcissement_rtc_27oct` |
| Les 80 et 150 démos couvrent uniformément l'espace des départs (pas de doublons) | voisin le plus proche 1,3 contre 1,1 au hasard | ✅ | analyse du 27/10 (positions y et orientation de l'écrou) |
| Les échecs d'E0 ne dépendent ni de la position ni de la distance à une démo | p=0,23 / 0,39 ; la zone « ratée » ne se réplique pas sur la 2e graine | ✅ | analyse du 27/10 |
| Démos du correcteur scripté inutilisables en sim durcie (toutes variantes) | 195-199 échecs (~1 %) contre 159 pour 80 démos humaines | 🔁 | mémoire `project_banc_nuit_27oct` · test en sim propre en cours (`371`) |
| **Sim propre : démos du correcteur + DART ≫ sans DART**, proches des démos humaines | DART 77 échecs (61,5 %) contre 146 sans (p=6e-11) ; humain 150 démos 44 (p=9e-4), humain 80 démos 100 | ✅ | `experiments/can/371_corr_propre_gb10.sh` · mémoire `project_correcteur_dart_propre` |
| Sim propre : oracle complet (lent, prise propre) = correcteur simple | 154 contre 146 (p=0,43), avec 132 démos | ✅ | idem |
| Ajouter +200 épisodes experts : neutre ; ajouter des rattrapages complets : dégrade | 98 → 103 ; 103 → 166 | ⚠️ | [POURQUOI_LE_POIGNET](POURQUOI_LE_POIGNET.md#bilan-final-de-la-campagne-17-septembre-2026) |
| Lift peu gourmand : ~20 démos ≈ 98 % ; à N=20 le gros U-Net sur-apprend | 97,4 > 92,2 > 88,0 % | ✅ | [COMPRESSION](COMPRESSION.md#levier-4--données--capacité-500-rollouts-mesure-fiable) |
| Donner les coordonnées de l'objet = béquille non transférable au réel | → passage à la vision pure | 🧭 | [CAN_archive](CAN_archive.md#pivot--pourquoi-pas-de-coords-objet) |

## 8. HG-DAgger

| Conclusion | Chiffre clé | Statut | Explication / preuves |
|---|---|---|---|
| Les corrections paient **au recuit** | t10 57,5 % contre départ 41,6 % (p=0,0005) | ✅ | [HGDAGGER_CAMPAGNES](HGDAGGER_CAMPAGNES.md#pourquoi-chaque-campagne-a-changé) |
| **Soupe de poids** des tours > chaque tour (C4) | 81,5 % contre 68,9 % (−40 % d'échecs) | 🔁 (C4) / ❌ ailleurs | [HGDAGGER_CAMPAGNES soupes](HGDAGGER_CAMPAGNES.md#soupes-de-poids-moyenne-uniforme-des-recuits-de-plusieurs-tours-270_soupe_poidspy) |
| Soupe intra-tour : nulle | 68,5 % = t11 | ✅ | idem |
| Boucle itérative > réentraîner de zéro sur les données agrégées | 52,5 % contre 67,5 / 76 % | ✅ | idem |
| Correction a posteriori (C4) > C2/C3, qui plafonnent ~60 % | C4 67-75 % | ✅ | [HGDAGGER_CAMPAGNES C4](HGDAGGER_CAMPAGNES.md#campagne-4-hgd4-lancée-le-1410-vers-18-h) |
| Encodeur ImageNet gelé (C10) ≫ encodeur de zéro (C8) | soupe 90 % contre 77 % | ✅ | [HGDAGGER_CAMPAGNES](HGDAGGER_CAMPAGNES.md#résultats-mac-200-essais-jeu-006-appariés) |
| Sim durcie : correcteur qui rend la main (C11) dégrade ; jusqu'au bout (C12) aide | C11 156 → 176 ; C12 → 131 (p≈0,01) | ✅ | [HGDAGGER_CAMPAGNES C11/C12](HGDAGGER_CAMPAGNES.md#campagne-11-hgd11-lancée-le-2410-à-23-h--sim-rapprochée-du-réel--actions-engagées) |
| Le déclencheur linéaire (274 coefficients) voit 98 % des échecs… | 9 % de fausses alarmes | ✅ | mémoire `project_declencheur_hgdagger` |
| … mais **se périme** si on ne le recalibre pas à chaque tour | proba médiane 0,25 → 0,02 | ✅ | mémoire `project_declencheur_derive` |
| Reprendre la main tôt > tard | correcteur 91 % contre 80 % | ✅ | [HGDAGGER_PROTOCOLE](HGDAGGER_PROTOCOLE.md#les-trois-réglages-mesurés-sur-les-mêmes-250-états-2026-10-11) |
| Correcteur scripté (lit la vérité de la sim) insensible au biais | 98 % à 0 / 0,03 / 0,06 rad | ✅ | mémoire `project_correcteur_scripte` |
| Mur de portée square : 100 % depuis un état quelconque impossible | rien au-delà de x≈0,22 | ✅ | mémoire `project_mur_portee_square` |
| Le collecteur ne corrige jamais après la faute (pas de ressaisie apprise) | 0/307 reprises après l'instant visible | ✅ | [AUTOPSIE_C12](recherche/AUTOPSIE_C12.md#cause-de-b-et-c--le-collecteur-ne-corrige-jamais-après-la-faute) |
| Les tours = marche aléatoire des poids | deltas quasi orthogonaux, ~13 %/tour | ✅ | `experiments/can/273_analyse_poids_tours.py` |

## 9. Superviseur et têtes annexes (sur politique gelée)

| Conclusion | Chiffre clé | Statut | Explication / preuves |
|---|---|---|---|
| Le temps restant se lit **linéairement** (275 paramètres = MLP 103 k) ; 20 démos suffisent | MAE 13,84 contre 13,89 | ✅ | mémoire `project_tete_temps_restant` |
| Prédire le temps DANS la politique lui coûte (dose-réponse) | 207 → 233 → 282 échecs | ✅ | [TABLEAU_LEVIERS](TABLEAU_LEVIERS.md#ce-qui-dégrade) |
| Tête « c'est fini » auto-étiquetée sur rollouts ≫ apprise sur démos | 97 % / 8 % contre 74 % / 24 % | ✅ | mémoire `project_tete_binaire_fini` |
| … mais arrêter le bras à l'annonce fige des tâches inachevées | validation ~48 % | ✅ | [REPRENDRE](../REPRENDRE.md) |
| Superviseur deux têtes FINI + PERDU | 94 % de décisions justes, 12,2 s gagnées par échec | ✅ | mémoire `project_superviseur_deux_tetes` |
| 10 familles de prédicteurs > 10 copies ; la pente de leur moyenne détecte 99 % des échecs | 7 % de fausses alarmes, 9,3 s d'avance | ✅ | mémoire `project_ensemble_tetes_temps` |
| Une minuterie suffit | n'avertit que 0,3 s avant (verdict du 25/09 = artefact n=40) | ❌ | idem |

## 10. Mémoire / occlusion (Can)

| Conclusion | Chiffre clé | Statut | Explication / preuves |
|---|---|---|---|
| Aucun mécanisme de mémoire ne bat la baseline | 12 comparaisons, aucune significative | 🔄 (module CAMP cassé) | [MEMOIRE](recherche/MEMOIRE.md#aucun-mécanisme-de-mémoire-ne-récupère-les-17-points) |
| La boucle descente/remontée n'est pas causée par l'occlusion | 3,6 contre 3,7 cycles | ✅ | [MEMOIRE](recherche/MEMOIRE.md#-correction-majeure--la-boucle-nest-pas-causée-par-locclusion) |
| Le vrai goulot était la préhension | une fois soulevé, 95 % de réussite | ⚠️ | [MEMOIRE](recherche/MEMOIRE.md#-le-vrai-goulot--la-préhension-pas-la-mémoire) |

## 11. Robot réel et infrastructure

| Conclusion | Chiffre clé | Statut | Explication / preuves |
|---|---|---|---|
| Premiers rollouts autonomes réussis sur le bras | 5/14 (08/09), 7/15 (13/09) | ✅ | [README](../README.md#quelques-chiffres) |
| Robot erratique = observation non normalisée (preprocessor oublié) | 1 rad d'erreur → 0,017 rad | ✅ | [PROBLEME_ROBOT_KO](real/PROBLEME_ROBOT_KO.md#2-la-cause-diagnostiquée) |
| Inférence : CPU pour les petits modèles, Arc (OpenVINO) pour les gros | 30 M : 46 contre 99 ms ; 263 M : 373 contre 153 ms | ✅ | [HANDOFF](HANDOFF_ATOMMAN_HGDAGGER.md#inférence--sur-ligpu-arc-via-openvino-déjà-en-prod) |
| Le temps d'entraînement suit les FLOPs de la vision, pas les paramètres | U-Net ×3 : +9 % seulement | ✅ | [PERF_TEMPS_ENTRAINEMENT](PERF_TEMPS_ENTRAINEMENT.md#coefficients-effet-de-chaque-levier-sur-t_step) |
| gb10 + images décodées une fois : ~7× plus rapide qu'atomman, identique au bit près | 0,048 contre 0,35 s/pas | ✅ | mémoire `reference_gb10_entrainement_rapide` · `src/cache_images.py` |
| « Le décodage vidéo freinait atomman » | 0,35 → 0,31 s/pas seulement | ❌ | idem |
| Évals découpées en 4 processus : 2× plus vite, identiques | 19 → 10 min | ✅ | mémoire `reference_evals_paralleles` |
| `scp`/`rsync` peuvent réussir sans rien copier : vérifier au md5 | 2 incidents | ✅ | mémoire `reference_lerobot_versions` |
