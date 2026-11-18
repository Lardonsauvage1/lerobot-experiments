# Guide d'enregistrement des démos réelles (Roby + bras guide)

> **Pour qui** : la session qui enregistre les démos du vrai bras Roby (5 axes + pince), téléopéré au bras guide, pour
> entraîner une Diffusion Policy LeRobot.
> **État au 18/11/2026.** Synthèse des journaux et des mémoires du projet. Aucune expérience n'a été relancée pour ce guide.
> Seules les conclusions encore valides sont gardées. Les résultats réfutés ou contaminés sont exclus, et les versions
> anciennes sont remplacées par les plus récentes.

**Lire les étiquettes de preuve**
- 🖥️ **sim** : mesuré en simulation. C'est presque tout le contenu. Les tâches sont square, Can, marteau et café, en sim
  « durcie » (10 Hz, latence, décalage articulaire, lumière) ou « propre ».
- 🤖 **réel** : mesuré sur le vrai bras ou sur des données réelles. Il y a très peu de lignes de ce type.
- 🧭 **décision** : décision de Niels, à appliquer telle quelle.
- **Unité des chiffres** : échecs sur N essais (« 207/500 ») ou taux de réussite.
- **Une graine** : un seul entraînement. Le bruit entre deux entraînements identiques est de 4 à 8 points, ou de 20 à 30
  échecs sur 200. Un écart plus petit que ça n'est pas une preuve.

---

## 0. La recette en une page

| # | Règle | Pourquoi (chiffre clé, preuve, source) |
|---|---|---|
| 1 | **80 démos** par geste ou modèle, chacune depuis un **départ différent** | 🧭 Niels (03/11), `feedback_pas_de_dec`. 🖥️ sim : 80 départs distincts font 69 échecs/200, contre 91 pour 40 départs × 2 démos (p=0,013, `DEMOS_OPTIMISATION.md` P24) |
| 2 | **Un seul geste par démo.** Si la prise rate, **jeter la démo et recommencer l'enregistrement** | 🖥️ sim Can : démos sans reprise 32/400 échecs (humain 24/400), contre 331/400 avec des reprises (`CHAINE_MARTEAU_DEMOS_HUMANISEES.md`, `project_demos_humanisees`) |
| 3 | **Finir chaque démo par 2 s d'immobilité ENREGISTRÉES** : bras tenu, pince dans son état de fin | 🧭 Niels (12/11). 🖥️ sim marteau : saisir 0 → 111/200 tenus ; déposer 78 → 93 % (`PASSAGE_AU_REEL.md` §12/11) |
| 4 | **Geste lent, continu, mené au bout** (environ 2× plus lent qu'un geste spontané), sans pause au milieu | 🖥️ sim marteau, 1 graine : démos 2× plus lentes, ouvrir 89 → 1 échec/200 (seul changement). Saisir 44 → 81/100 (p=9e-9), mais avec les zones de départ changées en même temps. **Pas plus lent** : ÷4 fait 68/100 contre 81 (p=0,03) (`CHAINE_MARTEAU_OPTIMISATION.md` 12-13/11) |
| 5 | **Varier les départs dans toute la zone**, en grille, coins compris, sans sur-représenter les bords | 🖥️ sim : déposer avec départs variés 99/100 contre 60/100 sans ; fermer 100 contre 80 (`CHAINE_MARTEAU_OPTIMISATION.md` 13/11). La moitié des échecs restants viennent de départs loin des démos (`ROBY_PASSAGE_AU_REEL.md`) |
| 6 | **Ne jamais réécrire les actions après coup.** Garder les mouvements du bras guide tels qu'enregistrés, même saccadés ou un moteur à la fois. Pas de script qui remplace les actions par une « ligne droite » ou les lisse : les images montreraient un chemin et les actions un autre. Un geste droit se fait **pendant** l'enregistrement. | 🖥️ sim Roby : bras guide brut 115/400 échecs, démos scriptées propres 137 ; les mêmes réétiquetées en ligne droite 393-395/400 (le modèle coupe en diagonale et ferme 4-7 cm à côté). Les contrôles hors ligne ne voyaient rien (`project_bras_guide_reetiquetage`) |
| 7 | **Pas de perturbation volontaire (DART)** pendant la téléop humaine | 🖥️ sim Roby : testé 3 fois, jamais utile (`ROBY_PASSAGE_AU_REEL.md` §2 bis) |
| 8 | **Stratégie** : plusieurs façons de faire sont permises, mais il faut **éviter celles que le modèle apprend mal** (au-delà de ~10 % du jeu, elles coûtent) | 🖥️ sim café, 18/11 : mélanges de bonnes stratégies 97-97,5 % (contre 95,5-100 % seules) ; 40 bonnes démos + 40 « mauvaises » = 11 → 29 échecs (`CAFE_STRATEGIES_RAPPORT.md`) |
| 9 | **Caméras fixées, identiques de la collecte au déploiement** (montage, pilote, résolution, compression, machine) | 🖥️ sim : un écart de retard image↔état fait passer un modèle de ~79 % à 38 % ; 🤖 réel : retard mesuré ~0,1 s (`ROBY_PASSAGE_AU_REEL.md` §5) |
| 10 | **Même lumière** à la collecte et à l'usage : pas de soleil direct, plafonnier constant | 🖥️ sim : un éclairage inconnu fait passer un modèle de 81 % à 58 % (`ROBY_PASSAGE_AU_REEL.md`) |
| 11 | **Garder le fichier brut de chaque séance (mcap)**, avec : angles du bras (`/joint_states`), angles du bras guide (`/leader/joint_states`), **commande** de pince (ouvrir/fermer demandé, pas la position mesurée des doigts), toutes les caméras. Le dataset d'entraînement est produit **ensuite** par conversion. | Si on change d'avis sur la fréquence, le format des actions ou les caméras, on reconvertit au lieu de refaire la collecte. Exemple : le dataset du 07/10, converti en coordonnées absolues (format qui détruit les résultats), doit être refait en déplacements (§4.3) |
| 12 | **Contrôle qualité automatique et bloquant** avant tout entraînement | 🧭 Niels (15/11) : des épisodes en mosaïque ont été découverts trop tard (marteau, `feedback_qualite_datasets`). Liste au §5 |

---

## 1. Structure d'une démo

| Règle | Détail pratique | Justification |
|---|---|---|
| **Un seul geste** | Approche → prise (ou action) → fin, d'un seul mouvement. **Aucune reprise enregistrée** : si la prise rate, couper, jeter la démo, replacer la scène et recommencer. | 🖥️ sim Can : 0/80 démos humaines contiennent une reprise, contre 61/80 démos scriptées ; politique à 32/400 contre 331/400 échecs (`CHAINE_MARTEAU_DEMOS_HUMANISEES.md` §1-3). Les reprises cassent aussi la tête « temps restant » utilisée pour relancer (seuil G5 +50 au lieu de −18). |
| **Queue de 2 s immobile** | En fin de geste, tenir la pose 2 s : bras immobile, pince **fermée** si l'objet est tenu, **ouverte** après un lâcher. Couper l'enregistrement seulement ensuite. Ces 2 s font partie de la démo. | 🧭 Niels (12/11, consigne au réel). 🖥️ sim marteau, 1 graine : saisir 0/200 → 111/200 tenus ; déposer 156 → 186/200 (`PASSAGE_AU_REEL.md`). Sans cette queue, le modèle n'a jamais vu « tenir » : il rouvre la pince ~1 s après la prise. La queue sert aussi de positifs aux observateurs « fini » (§6). |
| **Pas de pause au milieu** | Ne pas s'arrêter 0,5 s en bas avant de fermer, ni pince fermée avant de lever. | 🖥️ sim marteau saisir, 1 graine : avec pauses 145/200 échecs, contre 61 (`CHAINE_MARTEAU_OPTIMISATION.md` 12/11 20:40). |
| **Pas de « geste de retour » ajouté** | Ne pas enregistrer de démos « raté volontaire → retour pince ouverte → reprise ». Si un geste de reprise est un jour enseigné, il doit être **discernable** d'une approche (ex. pince FERMÉE pendant le retour) et passer la porte hors ligne « aliasing ». | 🖥️ sim Roby : 185 et 358/400 échecs (RAD, RAD2). 54 % des images du retour sont à < 2 cm d'une approche dont l'étiquette est opposée, et la politique fait des demi-tours en boucle (`feedback_aliasing_geste_reprise`). |
| **Tâches enchaînées** | Pour l'étape N, faire partir les démos de la zone où l'étape N−1 se termine réellement (dispersion mesurée sur ses fins), et non d'une pose unique. | 🖥️ sim marteau : déposer 60 → 99/100 et fermer 80 → 100/100 avec des départs pris dans ces zones (`CHAINE_MARTEAU_OPTIMISATION.md` 13/11). |

## 2. Façon de téléopérer

| Règle | Détail pratique | Justification |
|---|---|---|
| **Lent et posé, mené au bout** | Viser environ la moitié de la vitesse spontanée. Le geste doit **aboutir** (descendre vraiment avant de fermer). Ne pas commander en « tout ou rien ». | 🖥️ sim marteau (10 Hz, latence, décalage) : démos 2× plus lentes. Ouvrir 89 → 1 échec/200. Saisir 44 → 81/100 réussis (p=9e-9), mais les zones de départ ont changé en même temps (effet non séparé). Ralentir ÷4 dégrade : 68/100 contre 81 (p=0,03). Fermetures de pince par épisode : médiane 4 → 2 (`CHAINE_MARTEAU_OPTIMISATION.md`). Sur square, les démos scriptées en tout-ou-rien saturaient 24 % des actions, contre 4,5 % chez l'humain (`CONCLUSIONS.md` §7). ⚠️ Voir §8 : trop lent a nui sur square en sim propre. |
| **Bras guide brut** | Les saccades, le mouvement d'1-2 moteurs à la fois et la hauteur variable sont **acceptables**. Ne pas chercher à « nettoyer » après coup. | 🖥️ sim Roby : bras guide brut 115/400 contre démos scriptées propres 137/400 (la diversité aide) (`project_bras_guide_reetiquetage`). |
| **Pas de DART / dérive volontaire** | Ne pas perturber exprès la téléop humaine. | 🖥️ sim Roby : testé 3 fois, jamais utile (`ROBY_PASSAGE_AU_REEL.md`). Le DART ne sert qu'aux démos **scriptées** (27 → 61,5 % de réussite sur square, `project_correcteur_dart_propre`). |
| **Rester loin des butées articulaires** | Choisir un geste qui garde de la marge sur chaque axe, surtout le poignet (J5). Pas de geste qui « frôle » une butée. | 🖥️ sim café (couvercle) : un geste à 0-0,07 rad de la butée J5 se fige dès qu'un décalage de zéro existe (84/99 réussis sous décalage, jusqu'à 100 s). Le même geste avec ≥ 0,4 rad de marge fait 99/99 (`CAFE_CADRE.md` 17/11 00:12-00:44). |
| **Un style cohérent** | Une ou plusieurs stratégies, pourvu que chacune soit **apprise facilement**. Repérer celles qui ratent souvent ou exigent une précision extrême, et les limiter à ≤ 10 % du jeu. | 🖥️ sim café (18/11) : le taux global ≈ moyenne des taux de chaque stratégie, pondérée par sa part (±4 points sur 6 jeux sur 7). 10 % de mauvaise stratégie = rien de mesurable ; 25 % = 88,5 % contre 97,5 % (`CAFE_STRATEGIES_RAPPORT.md`). Même phénomène en sim durcie : 50 démos humaines + 30 démos d'un expert mal appris = 160 échecs contre 104 (`PASSAGE_AU_REEL.md` §5.8). |

## 3. Diversité, départs, séances

| Règle | Détail pratique | Justification |
|---|---|---|
| **80 départs différents** | Un départ = une position d'objet ET une pose de départ du bras. Jamais 2 démos du même départ. | 🖥️ sim square : 80 départs = 69 échecs/200, contre 91 pour 40 × 2 (p=0,013, `DEMOS_OPTIMISATION.md` P24). 🤖 réel (pomme, juge = val-loss) : le 1er modèle a surappris parce qu'il y avait trop peu d'épisodes distincts ; ce sont les épisodes qui manquaient, pas les images (`project_apple_real`). |
| **Couvrir toute la zone** | Grille régulière, coins compris, sans sur-représenter les bords. Varier aussi la pose de départ du bras dans une boîte (ex. marteau : ±4/±3/−10..+3 cm). | 🖥️ sim Roby : environ la moitié des échecs restants (24/49) partent de départs loin des démos, et aucune repondération ne les rattrape. Bords sur-représentés : 57/400, pire sous éclairage inconnu (`ROBY_PASSAGE_AU_REEL.md`). |
| **Plusieurs séances = OK** | Collecter sur plusieurs séances (redémarrages) est bon ou neutre. **Noter l'identifiant de séance** de chaque épisode. **Couvrir toute la zone dans CHAQUE séance.** | 🖥️ sim Roby : erreur de repère −24 % avec 4 séances (`ROBY_PASSAGE_AU_REEL.md` §2 bis). Le décalage du zéro change à chaque redémarrage (`project_biais_calibration`), et la banque d'augmentation se calcule par séance. |
| **Ne pas juger sur une seule collecte** | Trois jeux de 80 démos collectés selon le même protocole ne donnent pas le même modèle. | 🖥️ sim Roby : 49 / 78 / 53 échecs/400 (`ROBY_PASSAGE_AU_REEL.md`). |
| **Plus de démos reste le levier n°1** | Si le budget le permet, au-delà de la règle des 80 (choix de Niels), plus de démos aide nettement. | 🖥️ sim square : 150 démos 44 échecs contre 100 pour 80 (p=1e-8, `CONCLUSIONS.md` §7). |

## 4. Capteurs, format, calibration

### 4.1 Caméras

| Règle | Détail pratique | Justification |
|---|---|---|
| **Caméras fixes, ni ajoutées ni déplacées** | Garder le montage actuel. Choisir les vues **avant** d'entraîner. | 🧭 Niels (29/10, `feedback_contraintes_reel`). 🖥️ sim : une vue inutile coûte dans les poids. 3 caméras 236 échecs, contre 207 pour 2 ; la masquer après coup donne 251 (p=0,0014) (`project_camera_dilution`). |
| **Vue extérieure de trois-quarts + poignet** | Une vue extérieure en biais, pas à la verticale, plus le poignet. Ensemble, elles sont complémentaires. | 🖥️ sim square : agentview (3/4) + poignet = 207/500, contre birdview (dessus) + poignet 294 (p=5,5e-9). Seules : agentview 277, poignet 303, dessus 467 (`project_square_leviers_nuls`). La vue de dessus est masquée par le bras 57,7 % du temps en usage, la vue 3/4 jamais (`project_camera_occlusion_erreur`). |
| **Montage répétable** | Poignet : ≤ 0,5 cm / 0,5°, avec pions de centrage ou gabarit. Vues fixes : support rigide. | 🖥️ sim durcie : 0,5 cm sans perte détectée ; 2 cm = +42 échecs même avec augmentation (`PASSAGE_AU_REEL.md` §3.1, 1 graine). |
| **Image de référence** | À la collecte : bras en pose de départ, une image par caméra fixe, en résolution native, sauvegardée avec le dataset (`ref_<camera>.png`). Refaire la même image en début de chaque séance. | Sert au recalage SIFT et à l'alarme « caméra bougée » (`src/recalage.py`). 🖥️ sim : vue fixe déplacée de 2 cm, 162 → 137 échecs avec recalage (`PASSAGE_AU_REEL.md` §3.2). |
| **Correspondance topic ↔ clé écrite noir sur blanc** | Noter quel topic ROS correspond à quelle caméra physique (fixe / poignet / ext_a / ext_b) et la garder avec le dataset. **Ne jamais fusionner** deux jeux dont le montage caméra diffère, même si les topics portent le même nom. | Une inversion ne plante pas : le modèle sort simplement de fausses actions (`CONCLUSIONS.md` §5). 🤖 réel : le lot du 16/07 a les mêmes noms de topics que l'ancien, mais d'autres caméras physiques (`project_roby_batch2_wrist`). |
| **Pas de changement entre collecte et déploiement** | Garder la même caméra, le même pilote, la même résolution, la même compression, la même machine et le même nœud. Sinon, recollecter. | 🤖 réel : retard image↔état ~0,1 s. 🖥️ sim : entraîné avec image retardée puis déployé avec image fraîche = 38 % ; l'inverse = 72 % (au lieu de ~79 %) (`ROBY_PASSAGE_AU_REEL.md` §5). |

### 4.2 Lumière

| Règle | Détail pratique | Justification |
|---|---|---|
| **Lumière stable et identique** | Même éclairage à la collecte et à l'usage. Pas de soleil direct, pas de changement de plafonnier. | 🖥️ sim Roby : lumière inconnue, modèle 81 → 58 % ; sans augmentation d'éclairage, 82 → 33 % (`ROBY_PASSAGE_AU_REEL.md`). 🧭 Niels (05/11) : l'atelier varie peu, donc augmentation ECL (`project_roby_decisions_05nov`). |
| **Photo de calibration par séance** | Une image de la même scène à chaque séance (≥ 5 séances : matin, soir, lampes), pour régler l'amplitude de l'augmentation photométrique sur l'écart réel observé. | `PASSAGE_AU_REEL.md` §1.4. |

### 4.3 Proprioception, pince, actions

| Règle | Détail pratique | Justification |
|---|---|---|
| **Enregistrer la COMMANDE de pince** | L'état de pince du dataset est la commande (Bool : true = fermée), comme le servo réel. | 🖥️ sim Roby : un modèle entraîné sur la pince mesurée tombe à 56 % s'il ne reçoit que la commande. L'entraîner sur la commande donne ~81 % (`ROBY_PASSAGE_AU_REEL.md` §5). |
| **Pince binaire, sans battement** | Une fermeture par prise et une ouverture par lâcher. Pas de clics multiples. | Contrôle qualité « pince » du §5 (`CAFE_DEMOS.md` §4). |
| **Apprendre des DÉPLACEMENTS, pas des coordonnées** | À la conversion : action = TCP(t+1) − TCP(t), en cartésien. Le dataset du 07/10 (absolu) est à refaire en déplacements. | 🖥️ sim square : coordonnées absolues 454/500 échecs contre 207 (p=9e-59), le plus gros effet du projet. La perte ne voit rien (0,025 dans les deux cas) (`project_actions_absolues`). Cartésien gardé (Δq ne récupère rien, `ROBY_PASSAGE_AU_REEL.md`). |
| **Décalage du zéro par séance** | Mesurer à chaque séance l'écart pose commandée / pose lue à la pose de départ. S'il dépasse ~0,1 rad (~6°), **recalibrer le zéro** avant d'enregistrer. | 🖥️ sim square : 3,4° de décalage font passer le modèle relatif de 58,6 % à 36,0 %. Le modèle absolu tombe à 0 % (`project_biais_calibration`, chiffres corrigés). Sim durcie : 0 / 0,06 / 0,10 rad = 109 / 104 / 103 échecs, 0,30 rad = 144 (`PASSAGE_AU_REEL.md`). ⚠️ Ce seuil de 0,1 rad ne vaut que si l'entraînement utilise l'augmentation de biais articulaire (banque) ; sans elle, 3,4° (0,06 rad) suffisent déjà à faire chuter le modèle (square). |
| **Garder `/leader/joint_states`** | Enregistrer aussi les angles du bras guide. | `ROBY_PASSAGE_AU_REEL.md` §2 bis. |

### 4.4 Fréquence

| Règle | Détail pratique | Justification |
|---|---|---|
| **Enregistrer à la cadence native, rééchantillonner à la conversion** | Caméras (~60 Hz) et joints (~100 Hz) bruts dans le mcap. On convertit ensuite à la cadence du modèle (10 Hz aujourd'hui). | 🤖 réel : pipeline mcap → LeRobot avec rééchantillonnage au plus proche (résidu ~4 ms, `project_apple_real`). |
| **Fréquence du dataset = fréquence de pilotage** | Si le dataset est à 10 Hz, l'inférence doit tourner avec `--hz 10` (défaut 15 !). Sinon le robot joue 1,5× trop vite. | `PASSAGE_AU_REEL.md` §2, `project_deploiement_cart160`. |

## 5. Contrôle qualité OBLIGATOIRE et bloquant (avant tout entraînement)

🧭 Niels (15/11) : « beaucoup de problèmes venaient de datasets mal faits ». Modèle à suivre : `experiments/cafe/12_controle_jeu.py`
(`CAFE_DEMOS.md` §4). **Tout KO bloque l'entraînement.** Une seule exception est admise : un KO compris et vérifié à la main,
noté dans le journal.

| # | Point | Contrôle (seuils utilisés en sim, à garder au réel sauf mesure contraire) |
|---|---|---|
| 1 | **Épisodes** | Exactement 80 ; indices contigus ; longueurs cohérentes (parquet = méta) ; durée minimale raisonnable ; une seule tâche. |
| 2 | **Images** | Relues **dans les vidéos encodées**, pas dans les tableaux avant encodage, sur **toutes les caméras** et toutes les images. Mosaïque si la moyenne de \|diff horizontale\| dépasse 15/255. Noire si < 8, blanche si > 247. Figée si ≥ 3 images identiques alors que la pince bouge. Rejeter aussi une 1re image identique d'un épisode à l'autre. |
| 3 | **Réussite** | Chaque démo réussit son geste. Au réel, l'opérateur le note à chaque épisode. **0 reprise.** |
| 4 | **Queue immobile** | Les 1,5 dernières secondes strictement immobiles (< 1 mm par image, \|action xyz\| < 0,05). La phase de 2 s commence par ~0,3 s de décélération. |
| 5 | **Pince** | Binaire ; nombre de fermetures et d'ouvertures = nombre de gestes ; pas de battement. |
| 6 | **Bornes et cohérence** | Actions finies et dans les bornes. Sauts < 6 cm par image à 10 Hz et < 0,3 rad. Actions cohérentes avec l'état (le déplacement réalisé suit l'action enregistrée). |
| 7 | **Départs** | Tous différents. Chaque tiers de chaque axe contient ≥ 10 % des épisodes. Étendue ≥ 60 % de la zone. Zone couverte dans chaque séance. |
| 8 | **Planche** | 8 épisodes tirés au hasard : images début / milieu / fin de chaque caméra, à montrer à Niels. |
| 9 | **Caméras** | Correspondance topic ↔ clé conforme à la fiche du montage. Image de référence présente. |
| 10 | **Si des étiquettes ont été modifiées** | Passer la porte « imitateur markovien » (`experiments/roby/93b_porte_imitateur_markovien.py`). Elle doit faire au moins aussi bien qu'avec les démos brutes. Les contrôles hors ligne (cosinus 0,86, rapport des normes 1,00) n'avaient RIEN vu de l'effondrement à 393/400 (`project_bras_guide_reetiquetage`). |
| 11 | **Si des segments de reprise sont ajoutés** | Porte « aliasing » : part des images du geste proches (< 2 cm) d'un état d'une autre phase avec une étiquette opposée ≈ 0 % (`feedback_aliasing_geste_reprise`). |

Pièges connus de l'outillage :
- L'encodage vidéo (SVT-AV1) peut se figer ou corrompre des fichiers quand la machine est chargée (`CAFE_CADRE.md` 16/11). C'est
  une raison de plus de relire les vidéos encodées.
- `scp` et `rsync` peuvent rendre « succès » sans avoir rien copié : toujours vérifier au md5 (`reference_lerobot_versions`).

## 6. Enregistrements annexes utiles (observateurs, ordonnanceur)

| Quoi | Règle | Justification |
|---|---|---|
| **Budget** | Tout élément annexe (détecteur, tête, observateur) doit s'apprendre sur **les 80 démos + 80 épisodes joués par le modèle**, et pas plus. | 🧭 Niels (09/11, `feedback_elements_sur_demos`). Pas d'auto-amélioration à des centaines d'épisodes (`feedback_auto_amelioration_reel`). |
| **Négatifs « jumeaux »** | Si on enregistre des exemples d'échec pour un observateur, ils doivent ressembler le plus possible aux réussites et n'en différer **que** par l'échec. Ex. pince fermée à vide à la hauteur de levée, au-dessus de l'objet posé. | 🧭 Niels (13/11, `project_observateurs_demos_negatifs`). Sinon l'observateur apprend un indice (pince fermée, hauteur…) au lieu de la réussite. |
| **Aucune position d'objet** | Ne rien prévoir qui exige de mesurer la position d'un objet au réel (case du bac, cible fixe). | 🧭 Niels (09/11, `feedback_pas_position_objet`). |
| **Pas de compensation empirique** | Pas de décalage fixe (type DEC 10 mm) réglé en sim : corriger les défauts à la source. | 🧭 Niels (03/11, `feedback_pas_de_dec`). |
| **Corrections HG-DAgger** (si on en enregistre) | Reprendre la main **≤ ~1 s après une faute visible**, d'un **geste net**, jusqu'au bout. Ne garder que les corrections réussies. ≤ 30 par tour, 5 tours. | 🖥️ sim square durcie : −41 % d'échecs (4 graines). Des corrections bruitées font +37 % d'échecs. Un correcteur qui rend la main avant la fin dégrade (156 → 176) (`PASSAGE_AU_REEL.md` §31/10, `CONCLUSIONS.md` §8). ⚠️ Sur Roby/Can en sim, 4 campagnes n'ont pas amélioré le départ (§8). |

## 7. Ce qui a été testé et ne sert à rien ou nuit (établi)

| À ne pas faire | Chiffre clé | Source |
|---|---|---|
| Réétiqueter les démos (ligne droite, jalons, anticipation) | 🖥️ 393-395/400 échecs contre 115 brut | `project_bras_guide_reetiquetage` |
| Apprendre des coordonnées absolues | 🖥️ 454 contre 207/500 (p=9e-59) | `project_actions_absolues` |
| Garder des reprises dans les démos | 🖥️ 331/400 contre 32/400 sans reprise | `project_demos_humanisees` |
| Ajouter des démos « retour pince ouverte → reprise » | 🖥️ 358/400 (RAD2), 185/400 (RAD) | `feedback_aliasing_geste_reprise` |
| Pauses de 0,5 s en bas avant de fermer ou de lever | 🖥️ 145/200 contre 61 (1 graine) | `CHAINE_MARTEAU_OPTIMISATION.md` 12/11 |
| Perturbation DART pendant la téléop humaine | 🖥️ jamais utile en 3 essais sur Roby | `ROBY_PASSAGE_AU_REEL.md` |
| Mélanger une stratégie mal apprise (≥ 25 %) | 🖥️ 40 bonnes + 40 mauvaises : 11 → 29 échecs/200 ; 25 % : 88,5 % contre 97,5 % | `CAFE_STRATEGIES_RAPPORT.md` |
| Mélanger avec des démos d'une source mal apprise (expert scripté en sim durcie) | 🖥️ 50 humaines + 30 expert : 160 contre 104 | `PASSAGE_AU_REEL.md` §5.8 |
| 2 démos par départ au lieu de 80 départs | 🖥️ 91 contre 69/200 | `DEMOS_OPTIMISATION.md` P24 |
| Ajouter une caméra peu informative, puis la masquer | 🖥️ 236 (3 cam) et 251 (masquée) contre 207 (2 cam) | `project_camera_dilution` |
| Vue extérieure de dessus (verticale) | 🖥️ 294 contre 207 avec poignet ; 467 seule | `project_square_leviers_nuls` |
| Geste qui frôle une butée articulaire | 🖥️ 84/99 sous décalage contre 99/99 avec marge | `CAFE_CADRE.md` 17/11 |
| Sur-représenter les bords de la zone | 🖥️ 57/400, pire sous éclairage | `ROBY_PASSAGE_AU_REEL.md` |
| (entraînement) Recadrage aléatoire des images | 🖥️ 319 contre 207/500 (p=9e-13) | `project_recadrage_aleatoire` |
| (déploiement) Changer caméra, pilote ou machine après la collecte | 🖥️ 38 % ou 72 % au lieu de ~79 % | `ROBY_PASSAGE_AU_REEL.md` §5 |

## 8. Points ouverts ou non tranchés

| Question | Ce qu'on sait | Pourquoi ce n'est pas tranché |
|---|---|---|
| **Vitesse idéale** | Marteau (sim durcie, 12-13/11) : ralentir ÷2 aide beaucoup (ouvrir 89 → 1 ; saisir 44 → 81/100). Square (sim propre, démos scriptées, 28-29/10) : ralentir au-delà d'un optimum nuit (vitesse 0,7 → 84 échecs, 0,5 → 104-117), et « démos plus longues » a été réfuté. | Contextes différents : 10 Hz + latence d'un côté, 20 Hz sans latence de l'autre. Le réel ressemble au premier. Garder « lent et abouti » sans aller jusqu'au geste laborieux (démos à tâtonnements, cf. couvercle café). |
| **La queue d'immobilité aide-t-elle toutes les étapes ?** | Saisir et déposer : oui. Fermer : 91 → 78 % (1 graine, à surveiller). Sur Can, les démos humaines finissent en mouvement et la queue n'y était pas justifiée. | Décision de Niels (12/11) : l'appliquer partout au réel. Un seul cas contraire, sur 1 graine. |
| **Zone de départ localisée ou uniforme** | Départs humains / hasard / grille : aucun effet visible (124 / 118 / 118, 1 graine). Mais couvrir les coins compte (§3). | Seules des répartitions quasi uniformes ont été comparées. |
| **2 ou 3 caméras au réel** | Le modèle réel `cart160` utilise 3 caméras (ext_a, ext_b, poignet). En sim, une vue inutile coûte (dilution). | Jamais mesuré sur Roby. Les caméras sont fixées (Niels) ; il faut décider des vues **avant** d'entraîner. |
| **Démos sous décalage de zéro** | Sim café (17/11) : des démos enregistrées avec un décalage par épisode font égal ou mieux (A 465 / AB 470 ; G 442 / GB 479 sur 500 ; E 490 = EB). L'effet porte surtout sur la réussite au 1er essai. | Au réel, le décalage est naturellement présent dans les démos. Varier les séances (§3) en est l'équivalent probable : c'est une déduction, pas une mesure. Les résultats café du 16/11 (CB, BB…) sont invalides (rendu corrompu). ⚠️ A/AB/G/GB/E/EB ont été évalués le 17/11 avant 16:46 : une partie des épisodes (7-30 %) a été jouée avec un rendu recréé (couleurs du bras fausses) ; les taux sur les seuls blocs sains étaient proches des taux globaux (`CAFE_CADRE.md` 17/11 16:46), mais ces chiffres ne sont pas refaits avec le correctif. |
| **Pondération Y4 par couverture** | Adoptée en sim Roby. Sans elle : 98/400 contre 80. Elle demande la position de départ de chaque démo (clic sur la 1re image). | Son mécanisme n'est pas démontré. Repérer un départ à la main sur l'image n'est pas une position d'objet utilisée au robot, mais c'est à confirmer avec Niels. |
| **20 Hz au réel** | Sim : décider à 10 Hz au lieu de 20 Hz double les échecs (44 → 83). | Dépend de la latence réelle et d'atomman. Si on passe à 20 Hz, le dataset doit l'être aussi. |
| **HG-DAgger au réel** | Square durcie : −41 %. Roby/Can sim : aucun gain sur le départ. | Jamais testé au réel. |
| **Pince d'aplomb** | Sim square, démos SCRIPTÉES : brider le roulis/tangage (P7a) donne 81 échecs/200 sur 2 tirages, contre 76 et 109 sans bride. | `DEMOS_PISTES.md` attribue ce gain surtout à un expert « qui n'est plus saboté » (3 rejets au lieu de 47-66), pas à l'aplomb lui-même. Jamais mesuré avec des démos humaines : pas de règle. |
| **Échelle joint_3 du bras guide** | 0.972469 sur GitHub (#444, 02/11), 0.984931 sur le clone d'atomman (03/11). | À vérifier avant de collecter : `reference_teleop_bras_guide`, état au 03/11. |
| **Budget d'entraînement sur le dataset réel** | Sim : ~160 passes (30k pas pour 6 044 images). | Jamais établi sur le réel. Juger sur le robot, pas sur la val-loss. |

## 9. Ce qui a été mesuré AU RÉEL (rappel)

- Premiers rollouts autonomes : 5/14 (08/09), puis 7/15 (13/09) (`CONCLUSIONS.md` §11).
- Retard image ↔ état : ~0,1 s (`ROBY_PASSAGE_AU_REEL.md`).
- Données pomme : surapprentissage dû au manque d'épisodes **distincts**, jugé sur la val-loss (`project_apple_real`).
- Robot erratique quand l'observation n'est pas normalisée : 1 rad d'erreur, contre 0,017 rad une fois corrigé
  (`CONCLUSIONS.md` §11).
- Lot du 16/07 : `/head_camera/left` = FIXE, `/head_camera/right` = POIGNET. Il est incompatible avec le lot à 2 caméras fixes
  (`project_roby_batch2_wrist`).
- Tout le reste de ce guide est mesuré en simulation. Au robot, 20 épisodes par condition donnent un intervalle de ±20 points :
  seuls les grands écarts s'y voient (`PASSAGE_AU_REEL.md` §4).
