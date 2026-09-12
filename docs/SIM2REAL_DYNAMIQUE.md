# Sim-to-real : écarts de DYNAMIQUE et d'ACTIONNEMENT MÉCANIQUE qui font échouer une politique de manipulation sur bras réel

> **Cadre.** Architecture et entraînement identiques : la politique réussit en simulation mais échoue (ou se dégrade fortement) sur le bras physique. La cause n'est ni la perception ni le réseau, mais le **reality gap dynamique** — l'écart entre la physique idéalisée du simulateur et la mécanique réelle de l'actionnement et des contacts.
>
> **Thèse centrale (vérifiée).** Chaque commande émise par la politique transite par une chaîne motor → réducteur → bras → contact. Le simulateur modélise cette chaîne comme quasi idéale (couple instantané, engrenages parfaits, contacts rigides régularisés). Le réel y injecte des **discontinuités** (jeu, stiction), des **dynamiques d'ordre supérieur** (élasticité, bande passante, retard) et des **paramètres mal identifiés** (inertie, friction, charge). Résultat : *la même action produit un mouvement différent*, ce que la politique n'a jamais appris à gérer.
>
> Période couverte : 2017–2026 (cœur 2020–2026). Toutes les affirmations quantitatives ci-dessous ont été vérifiées contradictoirement sur la source primaire (voir § Note méthodologique).

---

## Vue d'ensemble — la chaîne d'actionnement et où elle casse

```
Politique → [action] → contrôleur bas niveau (PD/OSC/impédance) → moteur (courant→couple) →
réducteur (harmonic drive / courroie / câble) → bras (inertie, flexion) → contact (objet, surface)
            └ écart 9 ┘  └────── écarts 4, 5 ──────┘  └──── écarts 1, 2, 5 ────┘  └─ 3,6 ─┘  └── 7 ──┘
```

Deux familles de gaps, avec des remèdes différents :

- **Gap cinématique** (calibration, § 10) : *géométrique et corrigible par la mesure* (offsets, DH, TCP, hand-eye). Une petite erreur angulaire est amplifiée par les longueurs de liens en grande erreur cartésienne.
- **Gap dynamique/actionnement** (§ 1–9) : *stochastique et difficile à modéliser analytiquement* (friction, jeu, retard, élasticité, contact). Le champ converge sur trois familles de remèdes combinées : **(1) robustifier** (domain randomization dynamique), **(2) identifier/calibrer** (system ID, real2sim, actuator nets), **(3) adapter en ligne** (RMA, residual RL, fine-tuning réel).

**Constat transversal majeur : aucun remède isolé ne suffit.** L'ADR (Rubik's Cube OpenAI) ne transfère qu'à ~60 % global et ~20 % sur les scrambles les plus durs ; les actuator nets échouent seuls sur les escaliers ; la DR de friction/restitution plafonne à ~60–80 % du niveau « entraîné en réel » sur l'insertion. Les systèmes modernes empilent sysID + actuator net + DR ciblée + correction résiduelle en ligne.

---

## 1. Jeu des articulations / backlash (réducteurs, courroies, câbles)

**Mécanisme physique.** Le backlash crée une **zone morte** : une plage de mouvement de l'entrée qui produit *zéro* mouvement de sortie tant que les dents d'engrenage ne se re-engrènent pas. C'est une **discontinuité dure** (marche d'escalier) dans la relation commande→mouvement, avec **hystérésis** sur inversion de sens. Le harmonic drive (réducteur à onde de déformation) ajoute en plus une erreur de transmission. Les simulateurs intègrent une physique *lisse* : ils omettent ou mal-modélisent cette discontinuité. Le backlash **augmente avec l'usure** (jeu des dents sous charge, dégradation de la graisse) : une calibration sim figée dérive dans le temps, et le réducteur « est presque toujours la première pièce à lâcher ».

**Symptôme observable.** Les micro-mouvements commandés ne produisent aucun mouvement du bras jusqu'à rattrapage du jeu, puis l'articulation « saute ». Erreur de positionnement = largeur du backlash sur chaque inversion de sens. La politique « hallucine en heurtant la réalité crénelée du réducteur ». Variabilité d'un exemplaire à l'autre du même modèle (un servo « bouge à peine » ou « oscille »). **Piège silencieux en imitation learning** : une articulation dont le jeu/stiction avale les petites commandes a une variance quasi nulle dans les démonstrations → le réseau apprend à prédire ~0 pour cette articulation, *la loss converge quand même*, masquant une articulation fonctionnellement morte au déploiement.

**Mitigations.**
- Modélisation explicite du backlash/zone morte, ou **actuator net appris** capturant la discontinuité (les modèles analytiques n'y arrivent pas).
- Identification du backlash sans capteur de sortie, par détection du changement de couple en bord de zone morte ; compensation par modèle inverse (NARMAX, piecewise-linéaire) ou réseau de neurones.
- **DR de la dynamique d'actionnement** incluant backlash et saturation couple-vitesse ; ranges larges pour couvrir usure et variabilité d'exemplaire.
- **Diagnostic de variance par articulation** sur le dataset *avant* entraînement (contre le piège « articulation morte »).

**Sources.**
- https://www.origami-robotics.com/blog/dexterity-deadlocks.html (discontinuité, « hallucination »)
- https://www.geartechnology.com/gear-backlash-in-robotics-applications (usure, première pièce à lâcher)
- https://ieeexplore.ieee.org/document/475744/ ; https://www.semanticscholar.org/paper/9c054c2608dc2af266e84babcbe1e3fd062dc4eb (identification/compensation)
- https://medium.com/correll-lab/behind-a-failed-robot-learning-project-dataset-engineering-training-and-inference-efc1c2bd10e8 (silent failure imitation)
- https://www.patsnap.com/resources/blog/rd-blog/sim-to-real-gap-in-robotics-patsnap-eureka/ (backlash dans le gap MDP)

---

## 2. Frottements (stiction, Coulomb, visqueux ; dépendants position/charge/température)

**Mécanisme physique.** Le frottement articulaire combine **stiction** (frottement statique à briser avant tout mouvement, terme discontinu opposé à l'intention), **Coulomb** (constant en magnitude, opposé au sens de glissement) et **visqueux** (proportionnel à la vitesse). Modèle de base : `Iⱼθ̈ⱼ + Bⱼθ̇ⱼ = τⱼ + fⱼ`. Les principaux contributeurs réels sont la **charge axiale**, la **vitesse** et la **température** du mécanisme (système du 1er ordre piloté par la puissance dissipée). Le frottement varie sur les 4 quadrants (signe couple-charge × signe vitesse). Le simulateur l'idéalise (constante, ou omis).

**Point clé vérifié — l'omission de la stiction est un mode d'échec mesurable.** La domain randomization conventionnelle **exclut typiquement le frottement statique** de son espace de paramètres ; ajouter une DR consciente de la stiction est précisément ce qui rétablit le transfert. Sur petites articulations bas coût, la stiction est une **fraction disproportionnée du couple disponible** : ratio frottement/couple-max mesuré ≈ **0,13 % (quadrupède Go1) vs 0,98 % (hexapode Saturn Lite), soit ~7,5× plus élevé** (ratio dérivé des deux valeurs du papier).

**Symptôme observable.** Avec DR *sans* stiction, le robot pouvait marcher en arrière mais **« la marche avant le faisait tomber »** et il **ne pouvait pas monter les escaliers**. Dérive de l'erreur de suivi au cours d'une session à mesure que l'articulation chauffe ; erreur dépendante de la posture/charge qu'une friction constante en sim ne reproduit pas. Sur mains à câbles : **« dépassement d'impulsion significatif »** près des inversions de vitesse si la stiction est sous-estimée.

**Mitigations.**
- **DR consciente de la stiction** : randomiser `fⱼ` sur une plage large — le papier utilise **[0,0 ; 1,2]× nominal** (couvre aussi le cas quasi nul pour faciliter l'apprentissage).
- **Identification par moindres carrés** de l'inertie, de l'amortissement et de la stiction par excitation sinusoïdale ; identifier la friction à partir de données **dynamiques** (pas statiques).
- Modèles de friction avec termes **vitesse + température + charge** (réseau RBF pour la dépendance en position, température estimée sans capteur ; modèle KUKA KR10 corrigé en température).
- **Actuator nets** appris (les modèles servo analytiques de l'état de l'art échouent à capturer la friction complexe) ; **PACE** ajuste par articulation armature, amortissement visqueux, friction de Coulomb, biais et retard global → transfert zero-shot *sans* DR.

**Sources.**
- https://arxiv.org/abs/2503.01255 ; https://arxiv.org/html/2503.01255 (**source la plus directe** : exclusion stiction, ratios 0,13 %/0,98 %, chute en marche avant / escaliers, range [0 ; 1,2])
- https://www.researchgate.net/publication/321813628 ; https://onlinelibrary.wiley.com/doi/10.1155/2019/6931563 (friction vitesse+température+charge)
- https://arxiv.org/html/2509.23075v1 (identification dynamique, dépassement câbles)
- https://github.com/leggedrobotics/legged_gym ; https://github.com/leggedrobotics/pace-sim2real (actuator nets, PACE)

---

## 3. Inertie & dynamique des corps (Coriolis/centrifuge, couplage multi-DOF, charge utile)

**Mécanisme physique.** La dynamique d'un bras série suit `τ = M(q)q̈ + C(q,q̇)q̇ + g(q)` : matrice d'inertie `M`, termes de **Coriolis/centrifuges** `C` (couplage entre articulations dépendant de la vitesse), gravité `g`. Si les **inerties/masses/centres de masse** des liens sont mal identifiés, la même action produit une accélération différente — la correspondance action→résultat apprise est cassée. Les termes de Coriolis/centrifuges et la flexibilité articulaire sont **systématiquement sous-modélisés** sur les bras légers, rapides et à charge variable ; ils croissent avec la vitesse et la charge. Une **charge utile inconnue** déplace *simultanément* masse totale, centre de masse et distribution d'inertie — un décalage **structuré** (pas un bruit).

**Symptôme observable.** Quasi-perfection en sim, chute nette sur le matériel ; objet poussé au mauvais endroit. L'erreur de suivi empire à haute vitesse / sous charge ; vibrations absentes en sim. Les manœuvres dynamiques (lancer, fouet, non-préhensile) qui exploitent le couplage centrifuge échouent ou déstabilisent si le tenseur de couplage est erroné. **HALO** rapporte **0 % de succès** sur manœuvres agiles sous charge lourde avec DR large ; **50 %** avec compensation de masse seule.

**Mitigations.**
- **Dynamics randomization** : randomiser masse, dimensions, amortissement articulaire, friction, gains PID ; une **politique récurrente (LSTM)** réalise une **identification système implicite** en intégrant l'historique état-action (un réseau feedforward en est incapable). Peng et al. : LSTM 0,89 succès réel (0,91 sim) vs **0 essai réussi en réel** sans DR (feedforward).
- **System ID actif (ASID)** : une politique d'exploration excite délibérément le système (trajectoires à forte information de Fisher) pour identifier masse/inertie/friction/COM, puis ré-entraîne — meilleur que des données passives.
- **HALO** : system ID en simulation différentiable en deux temps (calibrer le modèle de base à vide, puis identifier masse/COM de la charge) → −41 à −44 % d'erreur de position, jusqu'à **100 %** de succès zero-shot.
- Modèles hybrides physique+données conservant la dynamique rigide en espace articulaire et ajoutant un terme résiduel pour flexibilité/dynamique non modélisée.

**Sources.**
- https://arxiv.org/abs/1710.06537 ; https://openai.com/index/sim-to-real-transfer-of-robotic-control-with-dynamics-randomization/ (Peng et al. + identification implicite)
- https://arxiv.org/abs/2603.15084 ; https://arxiv.org/html/2603.15084 (HALO : 0→100 %, −41/−44 %)
- https://arxiv.org/pdf/2404.12308 (ASID) ; https://arxiv.org/pdf/2505.14266 (excitation Fisher-optimale)
- https://arxiv.org/html/2405.04503v1 (Coriolis/flexibilité sous-modélisés) ; https://link.springer.com/article/10.1007/s11044-025-10094-w (couplage centrifuge)

---

## 4. Dynamique des actionneurs (bande passante moteur, courant/couple, saturation, retard)

**Mécanisme physique.** Le simulateur traite les moteurs comme des **sources de couple idéales** : le couple commandé s'applique instantanément. Un actionneur physique produit un profil de couple **moins précis** à cause de la **bande passante moteur/contrôle finie** et de la précision de suivi limitée de la boucle de couple bas niveau — décrit comme « une source dominante d'erreurs de modélisation ». La boucle électrique interne (courant→couple) a une bande passante haute mais finie (≈ 346 Hz mesuré), gain ≈ unité mais **déphasage** qui s'accumule (≤ 5° jusqu'à ~25 Hz). La **saturation de couple** est une non-linéarité dure (couple max décroissant avec la vitesse — back-EMF). Un terme d'**inertie virtuelle** de compensation firmware (offset ~constant) casse la généralisation s'il est ignoré. Enfin, **retard d'action** (communication + calcul) : la commande émise en `t` agit en `t+Δ` → MDP partiellement observable.

**Symptôme observable.** Le mouvement réel diverge de la trajectoire sim ; écart couple commandé/réalisé. Les séquences d'actions agressives/haute fréquence accumulent du déphasage → réponses déphasées, oscillation. La politique commande des manœuvres haut-couple que le moteur réel écrête → undershoot, appuis ratés, instabilité. Le retard induit oscillation et instabilité d'apprentissage ; une politique sim optimale tend vers du **bang-bang** que l'actionneur réel (retardé, à bande passante limitée) ne peut pas suivre → chattering, saturation, échec.

**Mitigations.**
- **Actuator net** (ANYmal, Hwangbo et al., Science Robotics 2019) : un réseau prédit le couple réalisé à partir d'un **historique** d'erreurs de position et de vitesses (vérifié : « historique de l'état courant et de deux états passés à t−0,01 et t−0,02 s »), capturant retards logiciels et dynamique SEA. Politique déployée : erreur de suivi de vitesse 0,143 m/s (mieux que le contrôleur model-based), −29,7 % couple et −19,8 % puissance.
- **Modèle de saturation** type tanh (préserve le linéaire aux faibles commandes, sature en douceur) ; écrêter le couple à l'enveloppe puissance-max du moteur ; **identifier la bande passante de boucle interne** et garder la fréquence de commande bien en dessous.
- **Modéliser explicitement les retards** ; augmenter l'état avec l'historique d'actions ; **randomiser les retards** (latence de contrôle ~0–20 ms) + **pénalité d'actions erratiques** (anti bang-bang) → performances proches du niveau sim.
- **MMDR** (Multi-Modal Delay Randomization) pour la latence asynchrone proprioception vs vision.

**Sources.**
- https://arxiv.org/abs/1901.08652 (actuator net ANYmal, métriques vérifiées)
- https://arxiv.org/html/2509.06342v1 ; https://arxiv.org/pdf/2509.06342 (couple idéal = erreur dominante, bande passante 346 Hz, saturation tanh, inertie virtuelle)
- https://arxiv.org/pdf/2303.14870 (retard + pénalité bang-bang)
- https://arxiv.org/html/2602.00399v1 (survey RL retards) ; https://arxiv.org/abs/2109.14549 (MMDR)

---

## 5. Compliance / flexibilité (élasticité harmonic drive, flexion des liens, vibrations, câblage)

**Mécanisme physique.** Le **harmonic drive** brise l'hypothèse de transmission rigide : le flexspline à paroi mince se déforme élastiquement → la position de sortie n'est pas un rapport rigide de l'entrée mais `entrée − torsion élastique dépendante de la charge`, avec **hystérésis**. Trois effets distincts : (a) **windup torsionnel** sous charge (~1,5 arc-min au couple nominal sur un CSF-25) ; (b) **erreur de transmission cinématique** périodique par tour (ellipticité du générateur d'onde) et **dépendante de la charge** ; (c) **friction elle-même dépendante de la charge** (couplage friction × compliance) — un coefficient de friction unique est invalide. Les joints à câbles/tendons ajoutent **friction de routage + hystérésis** (effet tendon-gaine), accentués quand le tendon est routé dans les doigts. Les manipulateurs sont simultanément **flexible-joint et flexible-link** (flexion structurelle + élasticité articulaire).

**Symptôme observable.** Décalage commande↔mouvement et boucle d'hystérésis dans la relation couple-angle ; erreur de position dépendant de l'historique de charge/direction. La sortie retarde sur la commande proportionnellement à la charge → les tâches de précision (insertion, alignement) ratent de la valeur du windup. Ripple périodique de position ; erreur de suivi variable avec la charge. Sur mains dextres, les gaps résiduels deviennent « prononcés pour les tâches dynamiques riches en contact » ; angle articulaire commandé vs réel divergent sous charge de câble. Vibrations/fluctuation de vitesse absentes d'un sim rigide.

**Mitigations.**
- Modèles explicites de **compliance torsionnelle non linéaire + hystérésis** dans le modèle d'articulation ; modèle de raideur dépendant de la charge / **estimation de couple à partir des mesures de position** pour compenser le windup.
- Modélisation/prédiction de l'**erreur de transmission** (consciente de la forme du générateur d'onde) ; framework de **friction adaptative variant dans le temps** (estimation modèle + apprentissage d'erreur adaptatif).
- **Compensation d'hystérésis adaptative à la forme** + system ID de la friction de câble pour les mains à tendons.
- Modéliser la compliance (raideur joint/pied, flexibilités de liens, élasticité série) **explicitement dans l'optimisation/sim** ; suppression de fluctuation de vitesse par feedback servo.

**Sources.**
- https://www.researchgate.net/publication/273396727 (compliance torsionnelle + hystérésis, windup ~1,5 arc-min)
- https://www.researchgate.net/publication/274461269 (estimation couple par position)
- https://www.sciencedirect.com/science/article/abs/pii/S0094114X20303402 ; https://spj.science.org/doi/10.34133/space.0233 (erreur de transmission dépendante de la charge)
- https://www.sciencedirect.com/science/article/abs/pii/S0094114X25001995 (friction dépendante de la charge)
- https://arxiv.org/html/2509.23075v1 ; https://arxiv.org/pdf/2109.06907 (tendon-gaine, hystérésis câbles)

---

## 6. Gravité & compensation

**Mécanisme physique.** Le couple de gravité `g(q)` dépend de la configuration. Si le modèle de masse est faux, le couple de gravité **résiduel non compensé** agit comme une **perturbation constante**. Sans action intégrale, cela produit une **erreur statique non nulle** ; en mouvement lent, la gravité domine le budget d'erreur. Sur bras réels à câbles/tendons (ex. dVRK), des **forces de perturbation non linéaires** dépendantes de la configuration (tension de câble, friction de routage) ne sont pas captées par un modèle de gravité analytique → le couple de compensation feedforward est faux.

**Symptôme observable.** Le bras se stabilise **sous/à côté de la cible** ; offset statique persistant ; dérive dans les directions pilotées par la gravité. Affaissement résiduel ou mouvement involontaire quand le contrôleur suppose la gravité totalement annulée.

**Mitigations.**
- Contrôle PD avec **compensation de gravité dynamique précise** ; **action intégrale** pour annuler l'erreur statique ; modèle de masse de charge correct.
- Stratégie de compensation **consciente de la perturbation** qui estime et ajoute le terme non linéaire (cas câbles/tendons).
- Recouvre les remèdes du § 3 (system ID des paramètres inertiels, calibration du modèle de masse).

**Sources.**
- https://www.sciencedirect.com/science/article/abs/pii/S0094114X09002249 (erreur statique, gravité en mouvement lent)
- https://arxiv.org/pdf/2001.06156 (perturbations non linéaires dVRK)

---

## 7. Dynamique des contacts (rigide vs réel, restitution, micro-glissements, insertion de précision)

**Mécanisme physique — le gap dynamique de contact, pas la vision, domine les tâches riches en contact.** Les écarts de modèle/paramètres de contact produisent un **fort décalage de la réponse force-mouvement** lors de l'interaction robot-objet. Détail des artefacts de simulateur :
- **MuJoCo** : contact mou régularisé (`R = αG`, type Tikhonov) qui garantit l'unicité de la solution mais **décale la solution hors de la dynamique rigide réelle** ; la compliance tangentielle **relâche la loi de Coulomb** (friction fantôme) → le peg interpénètre légèrement la paroi et glisse là où un peg réel rigide coincerait. La politique apprend des **raccourcis de pénétration** indisponibles sur le matériel.
- **LCP rigide** (ODE/DART) : le time-stepping permet inévitablement la **pénétration** ; la **stabilisation de Baumgarte** injecte des forces ∝ profondeur de pénétration (énergie non physique) → rebonds/pop-out parasites à l'entrée du trou. Le modèle LCP d'ODE/DART **échoue à simuler le glissement** sous certaines conditions de friction (Bullet le reproduit).
- **Linéarisation du cône de friction** (pyramide polyédrique) : **biaise la force de friction vers les coins de la pyramide** → perte d'isotropie de Coulomb. Le peg glisse dans des directions alignées sur les axes en sim, mais les micro-glissements réels partent ailleurs → les comportements de recherche latérale ne transfèrent pas.
- **PhysX (Isaac Gym/Lab)** : compromis itérations solveur — trop peu d'itérations de position → jitter+pénétration ; les itérations de vitesse absorbent l'énergie de pénétration mais **adoucissent** le contact, l'éloignant du rigide réel.
- **Insertions industrielles serrées** : clearance < 0,1 mm (parfois négative) avec réceptacle **déformable** ; le sim rigide suppose aucune compliance de surface → le **mode de contact** (coincer vs céder-et-glisser) est faux.

**Symptôme observable.** Une politique qui insère fiablement en sim **cale, coince ou sur-pousse** sur le bras réel, même avec une vision parfaite. La friction au premier contact **verrouille** le peg sur un robot rigide sans compliance, empêchant le ré-alignement latéral. Avec un poignet souple 6-DoF passif, le peg s'auto-aligne : **100 %, 95 %, 80 %** de succès (peg circulaire Ø40 mm / trou Ø42 mm, ±10 mm, ±5°) zero-shot, là où le rigide « colle » (limité par la bande passante du servo).

**Mitigations.**
- **Réduction de contact + raideur nette bornée** (Vuong & Pham, IROS 2023) : borner la raideur *nette* du système → permet d'entraîner une politique RL d'insertion serrée (double-pin) et de la déployer sur robot rigide position-contrôlé.
- **Poignet mécanique souple** + entraînement teacher-student (estimation de pose du peg via historique capteur).
- **Feedback force/couple** comme observation manquante : seuillage de force + conscience de la **direction de force** (FORGE : force threshold + DR → transfert zero-shot robuste ; « Direction Matters »).
- **Residual RL** : un contrôleur classique gère approche/retrait, la politique résiduelle lit la F/T au poignet et émet des deltas de pose sub-mm + modulation de raideur *seulement* pendant le contact (problème d'apprentissage bien plus petit).
- **Modèle de contact appris** (Online Linear Model Learning) pour connecteurs sub-mm où les modèles analytiques mis-prédisent la force ; **contrôle d'impédance/compliance variable**.
- **Sim de contact dédiée** : Factory (contact SDF rapide pour assemblage serré écrou-boulon/engrenages) ; **sim tactile** (TacSL 83–91 % succès réel, Taxim, TACTO, hydroélastique).
- **TRANSIC** : correction humaine en ligne → politique résiduelle, ~**77 %** moyen sur 5 paires sim-réel avec gaps exagérés (identifie 5 gaps coexistants : perception, contrôleur sous-actionné, embodiment, dynamique, asset objet).

**Sources.**
- https://arxiv.org/html/2304.06372v2 (analyse comparative des modèles de contact — MuJoCo R/Tikhonov, Baumgarte, biais pyramide ; **vérifié**)
- https://arxiv.org/abs/2306.06675 (raideur nette bornée ; **vérifié**)
- https://arxiv.org/html/2408.17061v1 (poignet souple, 100/95/80 % ; **vérifié**)
- https://arxiv.org/html/2405.10315v1 (TRANSIC, 77 % ; **vérifié**)
- https://arxiv.org/html/2602.14174v1 (Direction Matters) ; https://arxiv.org/pdf/2408.04587 (FORGE)
- https://arxiv.org/html/2311.07499v3 (compliance dynamique, clearance déformable) ; https://arxiv.org/abs/2312.09190 (modèle contact connecteur appris)
- https://arxiv.org/pdf/2205.03532 (Factory) ; https://arxiv.org/pdf/2501.08077 (TacSL) ; https://arxiv.org/pdf/2110.00541 (validation simulateurs sur impacts réels)
- https://isaac-sim.github.io/IsaacLab/main/source/migration/comparing_simulation_isaacgym.html (itérations solveur PhysX)

---

## 8. Limites & non-linéarités (couple/vitesse/accélération, butées, singularités, redondance)

**Mécanisme physique.**
- **Limites couple-vitesse** : le sim suppose un contrôle de couple idéal, mais le couple max réel **chute quand la vitesse monte** (back-EMF/saturation courant). Une RL qui n'impose qu'un pic de couple constant laisse la politique demander des couples irréalisables à la vitesse d'opération.
- **Limites vitesse/accélération articulaires** : si le sim laisse les joints bouger arbitrairement vite, la politique apprend des cibles exigeant des `q̇/q̈` infaisables que le contrôleur réel écrête/retarde.
- **Forces excessives** : sim sans protection matérielle → la politique apprend des stratégies haut-couple qui, en réel, dépassent les limites moteur et **déclenchent des fautes/E-stop**.
- **Singularités cinématiques** : quand `rang(J) < min(m,n)`, `‖J†‖ → ∞` et `κ(J) → ∞` → de petites commandes en espace tâche se transforment en vitesses articulaires énormes.
- **Redondance** : le mouvement dans l'espace nul dépend du schéma de résolution ; si sim et réel résolvent la redondance différemment, les joints réels dérivent vers leurs **butées** et se bloquent.

**Symptôme observable.** Sous-actionnement à vitesse ; le bras ne peut pas suivre la trajectoire. Mouvement saccadé, retard de suivi, contacts ratés, fautes. Fautes/E-stop pendant des comportements « réussis » en sim. Près des singularités : oscillation, suivi dégradé, divergence du solveur. Le bras se **bloque sur une butée** en milieu de trajectoire malgré une cible en espace tâche valide.

**Mitigations.**
- **Limite de couple dynamique dépendante de la vitesse** + reward conscient du couple (respecter l'enveloppe couple-vitesse) ; écrêter au moteur max-power + reward négatif sur violation.
- Imposer des limites strictes de vitesse/accélération **dans le sim** et clipper les sorties.
- **Least squares amorti** `q̇ = Jᵀ(JJᵀ+λ²I)⁻¹ẋ` ; projection dans l'espace nul du gradient de manipulabilité ; **warm-start classique + apprentissage** (jusqu'à 98,6 % de succès là où l'apprentissage pur échoue).
- Poids auto-adaptatifs pour évitement de butées/singularités ; évitement de butée comme tâche secondaire en espace nul.

**Sources.**
- https://doi.org/10.3390/math13152466 (limite couple dépendante vitesse)
- https://arxiv.org/html/2502.10894v1 (limites vitesse/accélération en sim)
- https://arxiv.org/pdf/2303.14870 (clip couple max-power + reward)
- https://arxiv.org/html/2604.13405v1 (singularités, DLS, warm-start 98,6 %)
- https://www.sciencedirect.com/science/article/abs/pii/S073658451000102X ; https://arxiv.org/pdf/2411.17052 (redondance, butées)

---

## 9. Écart contrôleur (PD/gains/interpolation/OSC : sim vs réel)

**Mécanisme physique.** Les **gains du contrôleur PD font implicitement partie de l'espace d'action** : un espace d'action position/vitesse convertit la sortie de la politique en couple via `τ = Kp(q_des − q) − Kd·q̇`. Si `Kp/Kd` diffèrent entre sim et réel, **les mêmes commandes produisent des couples/mouvements différents**. L'**OSC** (Operational Space Control) calcule les couples via les matrices de masse/dynamique du robot : des erreurs sur ces paramètres → l'OSC produit de mauvaises forces en bout d'outil, donc le mouvement tâche voulu n'est pas réalisé. Enfin, **mismatch de fréquence de contrôle** : politique entraînée à bas débit mais matériel contrôlant à haut débit → tenir l'action crée des discontinuités en marche et une incohérence temporelle vs l'entraînement.

**Symptôme observable (résultat clé vérifié — arXiv:2604.02523, MIT Improbable AI, avril 2026).** Les gains **raides + suramortis (stiff + overdamped) transfèrent le PIRE** : ils répondent à de petites déviations par de grands couples, amplifiant erreurs de modèle et bruit de politique, poussant le système hors distribution. Le **mode d'échec dominant est l'oscillation haute fréquence**, qui **persiste même avec domain randomization** (10 % de perturbation ne l'élimine pas) — l'instabilité naît de l'**interaction politique-contrôleur**, pas du contrôleur seul. Symptôme concret : jitter haute fréquence sur Franka Research 3 ; plus grande erreur de trajectoire sim-réel.

**Mitigations.**
- **Baisser la fréquence de la politique** : passer de 100 Hz → 10 Hz fait chuter les échecs de jitter de **21,8 % → 5,0 %** sur la grille de gains (à attribuer à ce papier 2026, résultat d'une grille d'expériences unique).
- **Utiliser des gains compliants** (pas raides/suramortis) ; **faire correspondre les gains sim↔réel** ; traiter les gains comme variable de conception ; les randomiser.
- **Espace d'action Cartésien/impédance** (OSC, impédance variable) qui abstrait la dynamique articulaire et fournit la compliance — transfère mieux que couple/vitesse articulaire bruts sur les tâches riches en contact ; ajouter de la **saturation** explicite en espace articulaire et Cartésien.
- **OSCAR** apprend les paramètres de dynamique de l'OSC en ligne, supprimant le fardeau de modélisation.
- **Interpoler** les actions bas débit en commandes haut débit (linéaire, spline cubique, first-order-hold + passe-bas) et utiliser **la même interpolation en sim et en réel**.

**Sources.**
- https://arxiv.org/abs/2604.02523 ; https://arxiv.org/html/2604.02523 (gains raides+suramortis pires, oscillation HF persistante sous DR, 21,8 %→5,0 % ; **vérifié, papier réel MIT 2026**)
- https://arxiv.org/pdf/2110.00704 (OSCAR)
- https://arxiv.org/pdf/2312.03673 ; https://arxiv.org/html/2606.18594v1 (espaces d'action, impédance Cartésienne, saturation)
- https://arxiv.org/pdf/2002.11635 (sim2real par correspondance de contrôleur, sans DR)

---

## 10. Calibration cinématique (longueurs de liens, offsets zéro encodeur, TCP)

**Mécanisme physique.** Une petite erreur géométrique se **propage et s'amplifie** le long de la chaîne cinématique en une grande erreur cartésienne (`erreur ∝ offset × longueur de bras`).
- **Offsets zéro d'encodeur** : dominent l'inexactitude pré-calibration — les offsets articulaires identifiés sont **bien plus grands** que les offsets géométriques (longueurs de liens) ; chaque offset fait tourner toute la chaîne aval.
- **Paramètres DH** : corrigibles, mais directement dans le contrôleur **seulement** en convention DH standard, sinon une couche de compensation est nécessaire.
- **Repère outil (TCP)** : une erreur d'estimation TCP couple l'erreur d'**orientation** en erreur de **position** — chaque réorientation du poignet balaie le point de contact réel sur un arc ∝ longueur d'outil.
- **Repère bout-d'outil observable seulement en sim** : avec IK, les positions articulaires existent en sim et en réel, mais le **vrai TCP n'est connu qu'en simulation** ; hystérésis et friction non modélisées font diverger le TCP réel → casse l'IK de précision.
- **Hand-eye** : une erreur sur la transformée flange↔caméra **décale toute pose perçue** → la politique vision commande la mauvaise cible cartésienne.

**Symptôme observable.** Erreur de positionnement bout-d'outil systématique ; échec des tâches de précision (insertion, alignement) ; saisies guidées vision décalées ; ripple/déviation le long du chemin. Politique IK qui « rate » de l'offset sim-réel résiduel.

**Mitigations.**
- **Calibration DH/offsets** par mesure 1D (encodeur à fil) minimisant l'erreur de position jusqu'à la limite de répétabilité ; **calibration cinématique en ligne par réseau de neurones**.
- **Calibration TCP automatique** par capteur laser (6-DoF) remplaçant le teach multi-points manuel.
- **Hand-eye markerless** depuis les features géométriques du flange (ajustement de cercle 3D) ; calibration hand-eye automatique par vision 3D apprise.
- **Calibration par feedback haptique** (le robot touche des points connus d'un écran tactile) + transformation linéaire/NN reconstruisant les variables bout-d'outil manquantes.
- **Interféromètre laser** pour mesurer/compenser l'erreur de pose le long des chemins.

**Sources.**
- https://arxiv.org/html/2502.14983v1 (offsets dominants, DH par mesure 1D)
- https://www.sciencedirect.com/science/article/abs/pii/S0263224124011667 (calibration NN en ligne)
- https://arxiv.org/html/2507.08572v1 (TCP observable seulement en sim, feedback haptique)
- https://ieeexplore.ieee.org/document/9910448/ ; https://www.mdpi.com/2076-0825/12/3/107 (calibration TCP laser)
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10892941/ ; https://arxiv.org/pdf/2311.01335 (hand-eye markerless / vision 3D)

---

## Synthèse des mitigations transversales (familles de méthodes)

| Famille | Principe | Méthodes / papiers | URL |
|---|---|---|---|
| **Robustifier (DR)** | Randomiser masse/inertie/friction/amortissement/gains/retards : la dynamique réelle tombe dans la distribution entraînée | Dynamics Randomization (Peng ICRA'18) ; Automatic DR (Rubik's Cube, OpenAI) ; DR consciente de la stiction | https://arxiv.org/abs/1710.06537 ; https://arxiv.org/abs/1910.07113 ; https://arxiv.org/abs/2503.01255 |
| **Identifier / calibrer (sysID, real2sim)** | Mesurer/estimer les paramètres réels pour réduire le gap que la DR doit couvrir | SimOpt « Closing the Sim-to-Real Loop » (Chebotar) ; DROPO (offline) ; system ID en sim différentiable ; real-to-sim-to-real | https://arxiv.org/abs/1810.05687 ; https://arxiv.org/pdf/2201.08434 ; https://arxiv.org/html/2411.00554v1 ; https://arxiv.org/html/2403.03949v3 |
| **Modèle d'actionnement appris** | Remplacer le modèle moteur idéal par un réseau capturant friction/retard/backlash/SEA | Actuator Net (Hwangbo, Science Robotics'19) ; Neural-Augmented Sim (Golemo CoRL'18) ; PACE | https://arxiv.org/abs/1901.08652 ; https://proceedings.mlr.press/v87/golemo18a/golemo18a.pdf ; https://github.com/leggedrobotics/pace-sim2real |
| **Grounding du simulateur** | Modifier les actions sim pour que la transition sim égale la transition réelle attendue | Grounded Action Transformation (GAT) ; SGAT (stochastique) ; RGAT (par RL) | https://link.springer.com/article/10.1007/s10994-021-05982-z ; https://arxiv.org/pdf/2008.01281 ; https://arxiv.org/abs/2008.01279 |
| **Adapter en ligne** | Inférer les paramètres réels (masse/terrain/friction/usure) depuis l'historique proprioceptif | RMA Rapid Motor Adaptation ; politiques récurrentes (sysID implicite) | https://arxiv.org/abs/2107.04034 ; https://arxiv.org/abs/1710.06537 |
| **Correction résiduelle / fine-tuning réel** | Apprendre seulement l'erreur contrôleur-vs-réalité avec peu de données réelles | Residual off-policy RL (BC figé, ~14→64 % en 15–76 min) ; TRANSIC (HITL) ; residual contact/assemblage | https://arxiv.org/abs/2509.19301 ; https://arxiv.org/html/2405.10315v1 ; https://arxiv.org/abs/2310.10509 |

**Surveys.**
- Zhao, Queralta, Westerlund, *Sim-to-Real Transfer in Deep RL for Robotics: a Survey* (IEEE SSCI 2020) — https://arxiv.org/abs/2009.13303
- *Survey of Sim-to-Real Methods in RL* (foundation-model era) — https://github.com/LongchaoDa/AwesomeSim2Real
- Survey reality-gap / fidélité contrôleur — https://arxiv.org/pdf/2510.20808

---

## Trois enseignements pour ton projet (bras 5 axes + pince, datasets LeRobot)

1. **Pour un bras bas coût, la stiction et le backlash dominent le gap** (ratio friction/couple-max disproportionné sur petites articulations). Le piège « articulation morte » en imitation learning est directement pertinent : *vérifier la variance par articulation du dataset avant entraînement* — une loss qui converge ne garantit pas qu'une articulation à fort jeu soit contrôlée.
2. **L'espace d'action et le contrôleur sont décisifs.** Un espace d'action Cartésien/impédance transfère mieux que couple/vitesse articulaire ; les gains raides+suramortis sont le pire cas et l'oscillation HF qui en résulte **ne se règle pas par DR** — baisser la fréquence de commande aide nettement. Faire correspondre exactement le contrôleur (gains, interpolation) sim↔réel est un levier sous-estimé.
3. **Pour les tâches riches en contact (pince, insertion), la DR seule plafonne (~60–80 %).** Le levier complémentaire le plus rentable est la **correction résiduelle / fine-tuning réel** (quelques dizaines de minutes de données réelles peuvent faire passer ~14→64 %), de préférence avec un retour force/couple ou une compliance mécanique passive (poignet souple) qui pardonne les erreurs de calibration sub-mm.

---

## Note méthodologique (fiabilité)

- **Recherche** : 5 agents en parallèle, ~25 requêtes WebSearch couvrant les 10 axes, ~70 affirmations falsifiables collectées avec URL.
- **Vérification contradictoire** : les affirmations quantitatives porteuses ont été re-vérifiées sur la source primaire (HTML arXiv) par 3 agents indépendants. **Toutes SUPPORTED**, notamment :
  - Stiction (arXiv:2503.01255) : exclusion par la DR conventionnelle, ratios 0,13 %/0,98 %, chute marche avant/escaliers, range [0 ; 1,2] — vérifiés mot à mot.
  - Actuator net (arXiv:1901.08652), Dynamics Randomization (arXiv:1710.06537 : LSTM 0,89 réel vs 0 sans DR), contact models (arXiv:2304.06372), poignet souple (arXiv:2408.17061 : 100/95/80 %), TRANSIC (arXiv:2405.10315 : 77 %), raideur bornée (arXiv:2306.06675) — vérifiés.
  - **arXiv:2604.02523** (gains PD, 21,8 %→5,0 %) : ID 2604 inhabituel (avril 2026) **explicitement re-vérifié comme papier réel** (Bronars, Park, Agrawal, MIT Improbable AI). Chiffres = grille d'expériences unique → à attribuer « selon ce papier ».
- **Caveats d'attribution** : (a) le « ~7,5× » de la stiction est dérivé de 0,98/0,13, non un chiffre étiqueté tel quel par le papier ; (b) « le robot rigide colle » est une paraphrase (le papier cite la bande passante limitée du servo) ; (c) les chiffres 2026 (2604.x) sont des résultats récents à citer prudemment. Quelques préprints à préfixe 2602/2603/2606 reflètent des soumissions 2026 remontées par le moteur de recherche.
