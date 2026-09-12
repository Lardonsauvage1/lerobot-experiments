# Donner une mémoire à la policy — état de l'art et plan d'expériences en sim

**Date** : 2026-09-08 · **Origine** : note atomman `2026-09-08_premier_rollout_autonome.md` (§2)
**Question posée** : *que faire quand la cible n'est plus visible ?*

---

## 0. Le problème, formulé proprement

Sur le vrai bras (5/14 = ~36 %), le mode d'échec dominant n'est **pas** l'imprécision :
la trajectoire allait toujours dans la bonne direction. C'est ceci :

> le bras passe au-dessus de la pomme → la caméra extérieure ne la voit plus →
> le modèle « pense » l'avoir dans la pince et enchaîne la remontée → il rate →
> il recule → il ne la voit toujours pas → il redescend → boucle infinie.

Deux constats qui structurent tout le reste :

1. **La policy est markovienne.** Elle calcule `a ~ P(a | o_t, o_{t-1})` avec
   `n_obs_steps = 2`, soit **133 ms de passé à 15 Hz**. Quand la pomme disparaît, il
   n'existe **aucun canal** par lequel l'information « elle était là » puisse atteindre le
   dénoiseur. Ce n'est pas un manque de capacité — 263 M de paramètres n'y changent rien —
   c'est un manque de **structure** : le problème est un POMDP traité comme un MDP.
2. **L'occlusion est aussi dans les données.** L'oracle scripté connaissait la position de
   la pomme et ne s'en souciait pas. Le réseau a donc appris, sur ces séquences, à
   continuer d'agir alors que la cible n'est plus visible — c'est-à-dire à **exécuter la
   suite de la phase apprise en aveugle**. Le comportement observé est la reproduction
   fidèle de ce qu'on lui a montré.

Corollaire important : déplacer la caméra ne résout pas le fond. Sam l'a écrit —
il y aura toujours des instants d'occlusion. Le sujet est bien la mémoire.

### Classer la tâche avant de choisir un mécanisme

RMBench propose une mesure utile, la **Task Memory Complexity (TMC)** : quantité minimale
d'information passée nécessaire pour décider de façon optimale.

| classe | définition | exemple |
|---|---|---|
| M(0) | l'observation courante suffit | pick-and-place à vue dégagée |
| **M(1)** | **il faut retenir UNE information passée** | **← notre cas : où était la pomme** |
| M(n) | il faut retenir et retrouver plusieurs événements | « quels sous-buts ai-je déjà faits » |

**Notre tâche est M(1)**, et c'est une bonne nouvelle : c'est le régime où les mécanismes
de mémoire les plus simples et les moins chers suffisent. La plupart des travaux récents
(LIBERO-Mem, HALO, Mem-0) visent surtout M(n) — mémoire **sémantique** long-horizon, du
type « ai-je déjà appuyé sur ce bouton ». Nous avons besoin de mémoire **spatiale
court-terme**, c'est-à-dire de **permanence de l'objet**. Ne pas se laisser entraîner vers
la machinerie M(n) : elle est plus lourde et ne répond pas à notre question.

---

## 1. La contrainte de latence — où va réellement le temps

Rappel des mesures de la note : **368 ms** au banc (CPU, machine au repos), **428–588 ms**
en conditions réelles, pour un budget de **530 ms** (`n_action_steps` 8 à 15 Hz). Le pic
dépasse déjà le budget. Il n'y a donc **aucune marge**.

Mais il faut décomposer ce budget, parce que toutes les mémoires ne coûtent pas la même
chose — et la plupart ne coûtent **rien** :

| ce qu'on ajoute | coût paramètres | coût latence par inférence |
|---|---|---|
| Dénoiseur 263 M × 10 pas DDPM (l'existant) | — | **l'essentiel des 368 ms** |
| Encodeur ResNet-18 sur 1 image (l'existant) | 11 M | une passe |
| **memory tokens / GRU / LSTM 64–256 D** | **+0,1 à 0,5 M** | **< 1 ms — négligeable** |
| **CAMP (LSTM 64 + quantifieur)** | **+~0,2 M** | **< 2 ms — négligeable** |
| **mémoire objet explicite (3–6 scalaires)** | **~0** | **~0** |
| +1 image d'historique (`n_obs_steps` +1) | 0 | **+1 passe d'encodeur entière** |
| suivi de points (TAPIP3D + SAM-2) | gros | prohibitif en synchrone (0,93 Hz), **viable en asynchrone** (5,44 Hz) |
| récupération dans un buffer épisodique | moyen | non borné, reconnu comme limitation |

**Conclusion opérationnelle** : ajouter un état récurrent est **gratuit**. Ce qui coûte,
c'est d'encoder **plus d'images**. Donc la voie « empiler des frames » est à la fois la
plus chère et — voir §2 — la moins efficace. C'est un renversement utile : la contrainte
de temps réel n'interdit pas la mémoire, elle interdit une seule de ses formes.

**Et un levier qui libère du budget** : la note §9.4 signale qu'un **28 M sain s'infère en
77 ms** au lieu de 368, et que la taille du modèle n'a **jamais** été comparée à conditions
égales (le balayage de juillet portait sur des modèles cassés). Un 28 M **avec** mémoire
sera vraisemblablement meilleur ET 4× plus rapide qu'un 263 M sans. À vérifier — c'est une
des expériences les plus rentables du plan.

---

## 1bis. Les trois contraintes supplémentaires (Sam, 2026-09-08)

Elles ne sont pas des détails : elles **éliminent** des familles entières et changent le classement.

**C1 — le temps d'ENTRAÎNEMENT ne doit pas augmenter significativement.**
Correction d'une imprécision de la §1 : la mémoire est gratuite *à l'inférence*, pas
forcément à l'entraînement. Entraîner un état récurrent demande de le **dérouler** sur un
segment de L pas. Contre-intuitivement ce n'est pas une explosion du nombre d'opérations
(pour couvrir 16 instants : 16 passes d'encodeur en récurrent, contre 32 avec des fenêtres
indépendantes de 2 frames). Le surcoût réel est ailleurs :
- **la mémoire vive** — la BPTT garde les activations des L passes d'encodeur, ce qui
  force à baisser le batch sur des machines à 16 Go. C'est la contrainte qui mord ;
- **la qualité du gradient** — des segments contigus donnent des échantillons corrélés,
  donc souvent plus de pas pour converger ;
- **la plomberie** — le dataloader LeRobot tire des fenêtres indépendantes ; le faire tirer
  des segments avec un état caché par épisode est du travail, et du travail qui peut se
  planter silencieusement (cf. `roby_train_xpu.py`).

**C2 — le temps de CAPTURE du dataset ne doit pas augmenter.**
Élimine tout ce qui exige une recollecte (perception active) ou une annotation
supplémentaire (paires VQA du retrieval). Bonne surprise en revanche :
l'**augmentation par occlusion ne coûte rien** — c'est un masque sur des images déjà
enregistrées. En simulation, la position de la cible est même une vérité terrain gratuite.

**C3 — la solution doit SCALER vers des tâches plus dures.**
La plus tranchante. Elle **disqualifie la mémoire-objet en 5 scalaires** (§F5) : c'est un
a priori codé à la main pour une tâche à un objet ; sur une tâche multi-objets séquencée,
la question « que met-on dans les cinq cases ? » n'a plus de réponse. Passer à la version
générique (SAM-2 + suivi 3D) violerait C1 et C2.
→ **La mémoire-objet est rétrogradée : ce n'est plus un candidat, c'est un instrument de
mesure.** L'oracle-mémoire (§5) reste pleinement valable *parce qu'il n'est pas destiné à
être déployé* — un thermomètre n'a pas besoin de scaler.

### Classement révisé sous C1+C2+C3

| | temps d'entraînement | capture de données | scalabilité | verdict |
|---|---|---|---|---|
| **F2 — CAMP** | phase 1 légère (LSTM 64) séparée, **phase 2 = boucle inchangée** | **zéro** | ✅ tâche-agnostique | **candidat n°1** |
| **F1 — état latent (μVLA)** | ⚠️ déroulement + BPTT + batch réduit | zéro | ✅ le plus général | candidat n°2 |
| F0 — empiler images | ❌ ×N encodeur | zéro | ❌ + copycat | témoin |
| F5 — mémoire objet | négligeable | ⚠️ détecteur au réel | ❌ | **instrument** |
| F4 — retrieval | ❌ + génération VQA | ❌ annotations | ✅ mais latence | écarté |
| F6 — perception active | ❌ | ❌ recollecte | ✅ | hors périmètre |

**Pourquoi CAMP passe devant** : c'est le seul candidat qui **découple**. Son module mémoire
s'entraîne dans une phase séparée, auto-supervisée, sur un LSTM à 64 dimensions — et le
déroulement temporel ne concerne **jamais** le dénoiseur de 263 M. Si l'on précalcule les
features visuelles, dérouler ce LSTM sur un épisode entier tourne en secondes. Ensuite la
boucle d'entraînement de la policy est **inchangée** : `m_t` n'est qu'un vecteur de 32
nombres de plus dans le conditionnement. C1 est satisfaite par construction.
Et sur C3 : mémoriser **ses propres actions** est indépendant de la tâche — rien à
redéfinir quand le problème change. Leurs sept tâches d'évaluation sont d'ailleurs
multi-étapes et contact-rich (empiler du Lego, insérer une sonde, ouvrir une porte),
pas des pick-and-place à un objet.

### ⚠️ Vérifié le 2026-09-08 — et la réponse nuance l'argument C1

**Pas de code publié.** Le site projet `robo-camp.github.io` existe mais ne contient aucun
dépôt (auteurs : Wang, Yeom, Cao, Zhi, Shinde, Yip — UCSD ARCLAB). Tout est donc
ré-implémenté d'après le papier, dans `src/camp.py`.

**Le module n'est PAS simplement pré-entraîné puis gelé.** Le papier fait
**pré-entraînement, PUIS warm-up à mémoire gelée, PUIS finetuning CONJOINT** avec le LSTM à
`lr × α = 0,1`, et l'encodeur visuel est *partagé avec la tête d'action*. Le finetuning
conjoint oblige à dérouler le LSTM pendant l'entraînement de la policy — exactement ce que
C1 interdit. **Mon « C1 satisfaite par construction » était trop optimiste.**

D'où **deux variantes**, à faire dans cet ordre :

- **variante B — « CAMP-lite » (implémentée)** : module pré-entraîné puis **gelé
  définitivement**, `m_t` précalculé pour toutes les frames et concaténé à
  `observation.state`. La boucle d'entraînement de la policy est alors *rigoureusement*
  celle d'un run normal. **C1 satisfaite pour de vrai**, et sans plomberie risquée dans
  LeRobot.
- **variante A — fidèle au papier** (warm-up gelé puis finetuning conjoint α=0,1) : à ne
  tenter **que si B plafonne**. On saura alors que le finetuning conjoint compte vraiment,
  et on décidera en connaissance de cause s'il mérite son coût.

⚠️ Ne pas présenter B comme « CAMP » : c'est CAMP amputé de son finetuning conjoint, et
**un échec de B ne réfute pas CAMP**.

**Détails récupérés du papier** : `w_k = exp(−γk/(K−1))` (pondération fréquentielle) ;
`L` = longueur complète de l'épisode ; `ℒ_cons = Σ_N ‖â_t[:L−N] − â_{t+N}[N:]‖²` ;
AdamW, batch 16 pour le pré-entraînement mémoire, lr 1e-4, EMA sur les poids de la policy,
DDPM à l'entraînement / DDIM à l'inférence. `λ_cons` n'est pas spécifié → à caler chez nous.

### Conséquence sur le banc : il faut un ESCALIER, pas une marche

C3 impose de vérifier qu'un mécanisme tient sur plus dur. Un banc M(1) seul ne permet pas
de conclure.

| échelon | tâche | ce qu'il faut retenir |
|---|---|---|
| **1 — M(1)** | Can-Occluded | *où était la canette* |
| **2 — M(n)** | NutAssemblySquare / ToolHang (déjà dans `data_cache`) | *ce que j'ai déjà fait, ce qu'il reste* |

Un mécanisme ne se qualifie que s'il gagne **sur les deux échelons**.

---

## 2. Le piège à connaître AVANT de commencer : la confusion causale

C'est le point qui fait échouer la solution naïve, et il faut le poser en premier.

**Le problème « copycat »** (Wen et al., NeurIPS 2020) : en clonage de comportement à
partir d'un **historique** d'observations, quand les actions de l'expert sont fortement
corrélées dans le temps, le réseau apprend un raccourci — il **retrouve l'action
précédente dans l'historique et la recopie**, au lieu de prédire la bonne action suivante.
La perte d'entraînement chute, et la policy s'effondre en déploiement dès que la
distribution dérive.

Deux conséquences directes :

- **C'est exactement notre symptôme.** « Il relève le bras comme s'il pensait avoir la
  pomme », « il boucle indéfiniment » : ce sont des signatures de policy qui continue son
  geste par inertie plutôt que par décision.
- **Cela explique pourquoi Diffusion Policy fixe `n_obs_steps = 2`.** Chi et al. ont trouvé
  empiriquement que davantage d'historique **dégrade**. Ce n'est pas une limite de
  capacité, c'est la confusion causale. Donc : **augmenter `n_obs_steps` ne marchera
  probablement pas**, en plus de coûter cher.

**Remèdes connus**, à intégrer dès la conception :
- **goulot d'information** sur la mémoire : la compresser fortement (CAMP quantifie sur un
  codebook de 128 et ne garde que 32 dimensions) — le réseau ne *peut pas* y cacher
  l'action précédente en clair ;
- **ne pas fournir l'action précédente brute** en entrée du dénoiseur ;
- **abandon aléatoire (dropout) de l'historique** pendant l'entraînement ;
- prédiction du **résidu** d'action plutôt que de l'action (Chuang et al. 2022) ;
- retrait adversarial de l'information sur l'action précédente (Wen et al. 2020).

---

## 3. Les six familles de mémoire

### F0 — Empilement de frames (`n_obs_steps` ↑) — *le témoin, pas la solution*
Ce qu'on ferait spontanément. Coût : ×N passes d'encodeur. Bénéfice : quelques centaines
de ms de passé seulement. Risque : copycat (§2). **À faire comme point de comparaison
négatif**, pour objectiver que le problème n'est pas soluble par la force brute.

### F1 — Récurrence dans le réseau (état caché porté d'un pas au suivant)
La famille la plus rentable dans notre régime.

- **BC-RNN** (robomimic, Mandlekar et al. 2021) — l'ancêtre : LSTM entraîné sur des
  séquences de longueur T ; au test, déroulé pas à pas, **état caché rafraîchi tous les T
  pas**. Déjà présent dans notre historique de projet (run 37 Lift, `robomimic_bc_gmm`).
- **μVLA** (2606.12497) — *le résultat le plus important pour nous* : un petit ensemble de
  **jetons mémoire apprenables** transportés d'un pas de temps au suivant et mis à jour par
  self-attention. **Aucune loss auxiliaire, aucun changement d'architecture.** Entraînement
  par **BPTT tronquée**. Deux variantes de mise à jour testées : gradients inter-pas, ou
  **EMA détachée** (encore plus simple). Résultats : MIKASA-Robo **0,42 → 0,84** en
  observabilité partielle ; LIBERO (pleine observabilité) **96,2 %, aucune régression**.
  Nuance honnête des auteurs : le gain disparaît quand la structure de mémoire ne
  correspond pas au besoin de la tâche.
- **TFP** (2607.08283) — croyance latente épisodique par réseau à **constante de temps
  liquide** (LTC) : la constante de temps dépend de l'entrée, donc la mémoire *retient*
  pendant les phases stables et *se met à jour vite* près des événements de manipulation
  (les auteurs mesurent un gain d'écriture ~6× plus grand près des événements). Croyance de
  256 D injectée dans le décodeur par **modulation AdaLN** — elle *façonne* la distribution
  d'actions au lieu d'être un contexte passif. Coût latence négligeable.
  Résultats : MIKASA ShellGameTouch 47 → **75 %** ; réel, échange d'objets **3/20 → 15/20**.

### F2 — Mémoire compressée de l'historique d'ACTIONS — **CAMP** (2606.21188)
*Le candidat le plus directement greffable sur notre pile.*

L'idée : **l'historique des actions du robot est un signal auto-supervisé riche** — il
encode « ce que j'ai déjà tenté », donc les échecs, les reprises, la progression. Pas
besoin d'annotation ni de prédiction visuelle coûteuse.

Architecture exacte :
- encodeur = **LSTM 2 couches**, entrée = [features visuelles partagées, proprioception
  `p_t`, **action précédente `a_{t-1}`**], **dimension cachée 64** ;
- tête linéaire → **K = 32 coefficients DCT** de la trajectoire d'actions passée ;
- **quantification vectorielle**, codebook 128 → code mémoire **`m_t` de 32 dimensions** ;
- `m_t` est simplement **concaténé au vecteur de conditionnement** de la diffusion policy :
  `concat[f(o_third), f(o_wrist), p_t, m_t]`.

Entraînement (pré-entraînement du module, auto-supervisé) :
- **perte de reconstruction** des coefficients DCT des actions passées, **pondérée en
  fréquence** pour écraser les hautes fréquences (on garde la forme du geste, pas le bruit) ;
- **perte de cohérence temporelle** : deux reconstructions à des instants différents
  doivent s'accorder sur le segment d'actions qu'elles partagent ;
- **BPTT sur des fenêtres de la longueur de l'épisode**.

Résultats : Memory-Manip-Bench **64,3 %** contre Diffusion Policy 40,9 %, π₀.₅ 26,8 %,
MemoryVLA 5,7 %. Memory-T-Bench **93,6 %** contre 66,8 %.

⚠️ **Réserve à garder en tête** : donner l'action précédente en entrée est *précisément*
ce qui déclenche le copycat. CAMP s'en protège par le goulot (DCT basse fréquence + VQ 32 D),
mais c'est une protection empirique. Surveiller le symptôme : perte qui s'effondre,
succès qui ne suit pas — le signe que nous connaissons déjà bien.

### F3 — Modèles espace-d'état (Mamba / SSM)
Compromis intéressant : coût **linéaire** en longueur de séquence, et surtout **O(1) par
pas en inférence** (forme récurrente) → compatible temps réel par construction.

- **MaIL** (2406.08234) : Mamba comme dorsale d'imitation, bat les Transformers sur toutes
  les tâches LIBERO, et surtout **tient bien sur petits jeux de données** (moins de
  surapprentissage). ← pertinent : nous avons **85 épisodes**.
- **Mamba Policy** (2409.07163) : SSM hybride pour diffusion policy 3D, **−80 % de
  paramètres** à performance égale. ← directement dans l'axe « pas plus gros ».
- **RoboSSM** (2509.19658), **DSSP** (2605.14598, encodage de l'historique complet).

### F4 — Mémoire externe / récupération — **HALO** (2606.25136)
Buffer épisodique non paramétrique (192–3840 entrées/épisode) d'observations et d'actions
encodées, avec récupération sélective. Entraînement **double objectif** : imitation +
**questions-réponses visuelles** générées automatiquement (« où est l'objet », « combien »,
« dans quel ordre ») qui biaisent la récupération vers l'information utile.

Résultats forts : bat **LSTM de 42 %**, **Mamba de 21 %**, **Transformer-XL de 29 %** —
l'argument étant que les états compressés *jettent* les détails nécessaires au rappel d'un
événement précis.

**Mais** : les auteurs reconnaissent la **latence de récupération** comme limitation
ouverte, et il faut fabriquer les paires VQA. → **écarté pour nous à ce stade** : c'est une
solution M(n), pour un problème que nous n'avons pas, à un coût que nous ne pouvons pas payer.

### F5 — Mémoire structurée objet-centrée — *la plus adaptée à notre échec précis*
Au lieu de mémoriser un vecteur latent opaque, on mémorise **où est l'objet**.

- **Historique par suivi de points** (2509.17141) : SAM-2 segmente les objets pertinents,
  **TAPIP3D** suit **K = 5 points par objet** en 3D, les trajectoires sont compressées en
  **un jeton par objet** et fusionnées aux jetons d'observation avant le dénoiseur.
  Le suivi tourne **en asynchrone** de l'inférence (5,44 Hz vs 0,93 Hz en synchrone), avec
  une **augmentation par abandon aléatoire** à l'entraînement pour rendre la policy robuste
  à cette asynchronie. Résultats : **85 %** contre 45–50 % pour les fenêtres fixes ;
  précision de décision > 90 %.
- **SAM2Act++ / approches object-centric** : ne stocker que les **coordonnées du centre de
  l'objet** — mémoire minuscule, a priori spatial fort, peu de bruit.
- **Embodied-SlotSSM / LIBERO-Mem** (2511.11478) : slots d'objets persistants + SSM.
- **« More Structure, Not More Capacity »** (2607.09825) : le titre résume l'argument
  central de cette section, et il s'applique mot pour mot à notre situation.

👉 **La version dégénérée, pour nous, tient en 5 scalaires** : `[x, y, z]` de la dernière
position connue de la pomme, un drapeau `visible / non visible`, et le **temps écoulé
depuis la dernière observation**. Concaténés à l'état. Coût nul, entièrement
interprétable, et c'est l'antidote exact à « il ne la voit plus donc il enchaîne ».
En simulation la position est une vérité terrain gratuite ; au réel il faut un détecteur
(la pomme est blanche sur fond contrasté — segmentation couleur plausible), ou :

**Variante sans détecteur — tête auxiliaire de permanence d'objet** : on ajoute une petite
tête qui prédit la position de la pomme à partir de l'état interne, entraînée avec la
vérité terrain **y compris pendant les frames où elle est occluse**. La perte auxiliaire
force la représentation interne (et donc la mémoire récurrente) à **transporter** la
position à travers l'occlusion. À l'inférence la tête peut être jetée. C'est la façon la
moins chère d'apprendre la permanence de l'objet, et elle se combine avec F1.

### F6 — Perception active — See2Act (2606.23625)
Plutôt que de se souvenir, **bouger pour revoir** : le débruitage d'action est couplé à un
raffinement de point de vue, appris depuis des poses de caméra ancrées sur des actions clés.
Jusqu'à +34 % sur RLBench sous occlusion sévère. Élégant et complémentaire, mais cela change
le schéma de contrôle et exige de nouvelles démonstrations. → **hors périmètre pour l'instant**,
à noter comme piste si la mémoire seule plafonne.

---

## 4. Comment on entraîne ça — le vrai savoir-faire

C'est ici que ça se joue, plus que dans le choix d'architecture. Passer d'une policy
markovienne à une policy récurrente change la **boucle d'entraînement**, pas seulement le
modèle.

1. **Échantillonnage par séquences.** Aujourd'hui on tire des fenêtres indépendantes
   (`delta_timestamps` de LeRobot). Une policy récurrente exige des **segments
   temporellement contigus**, et le batch doit garder l'identité de l'épisode.
2. **BPTT tronquée.** Rétropropager sur tout un épisode (~297 frames) est trop coûteux.
   Recette **EATB** de TFP : entraîner sur des morceaux contigus de plusieurs épisodes en
   parallèle, **un état caché par épisode**, gradient **tronqué** aux frontières de
   segment mais état **conservé numériquement** (`detach()`). On garde le crédit
   long-horizon sans en payer le coût. Longueur de segment typique : **16 à 64 pas**.
   Alternative encore plus simple (μVLA) : mise à jour de la mémoire par **EMA détachée**,
   donc aucun gradient inter-pas.
3. **Gestion de l'état caché au test.** BC-RNN **rafraîchit l'état tous les T pas** —
   sinon il dérive hors distribution. À décider et à mesurer. Et **remise à zéro en début
   d'épisode** au déploiement (notre `policy.reset()` existant).
4. **Anti-copycat** (§2) : goulot d'information, dropout d'historique, ne pas donner
   l'action précédente en clair.
5. **Augmentation par occlusion.** Masquer aléatoirement la cible pendant l'entraînement
   pour *forcer* l'usage de la mémoire — sans ça, le réseau apprend à ignorer le module
   mémoire tant que la vue est dégagée. C'est le pendant du « random drop » du suivi de points.
   Probablement **le levier le plus important de toute la liste**, et il est gratuit.
6. **Le juge reste le rollout**, jamais la perte (notre règle habituelle). Ajouter une
   métrique dédiée : **taux de récupération après occlusion** — proportion d'épisodes où
   la policy retrouve la cible après l'avoir perdue de vue (l'équivalent de la *Decision
   Accuracy* du papier suivi-de-points). C'est cette métrique-là qui mesure la mémoire ;
   le succès global la dilue.
7. **Méthodologie inchangée par ailleurs** : LR constant + EMA + cooldown évalué.
   ⚠️ Rappel du §8 de la note atomman : le scheduler demandé a été **silencieusement
   remplacé** par `cosine`. Vérifier le `train_config.json` du checkpoint après quelques
   centaines de pas, systématiquement.

---

## 5. Sur quel banc en simulation ?

### Les bancs publics
| banc | ce qu'il teste | pour nous |
|---|---|---|
| **MIKASA-Robo** (2502.10550, ManiSkill3) | 32 tâches, 4 catégories de mémoire, **objets occlus**, rappel de configurations | le plus proche de notre besoin ; utilisé par μVLA et TFP → chiffres comparables |
| **LIBERO-Mem** (2511.11478) | observabilité partielle **au niveau objet**, instances visuellement identiques, sous-buts séquencés | plutôt M(n), sémantique |
| **RMBench** (2603.01229) | 9 tâches classées M(1)/M(n) | surtout utile pour son **cadre TMC** |
| **Memory-T/Manip-Bench** (CAMP) | 4 variantes PushT + 7 tâches 3D contact-riche | référence si on reproduit CAMP |
| **ReMemBench** (HALO) | rappel long-horizon | hors périmètre |

### Ce que je recommande quand même : **notre propre banc, « Can-Occluded »**

Raison : nous avons déjà un **harnais d'évaluation mûr** sur robomimic Can — 500 rollouts,
IC95 de Wilson, étude de convergence auto-extensible, graphe 4-panneaux, calibrage des
temps, EMA/SWA/cooldown. Repartir sur MIKASA-Robo, c'est **jeter tout cela** et
re-débugger un environnement pour mesurer un effet qu'on peut mesurer chez nous.

Construction, par coût croissant :
1. **Occlusion synthétique scriptée** (le plus contrôlé, et le plus rapide) : on masque la
   canette dans l'image d'observation sur des fenêtres temporelles choisies — par exemple
   dès que le préhenseur passe à moins de X cm au-dessus d'elle, ce qui **reproduit
   exactement la géométrie de l'échec réel**. Avantage décisif : le **degré d'occlusion
   devient un paramètre continu** (0 % → 100 %), donc on trace une *courbe* de robustesse
   au lieu d'un point, exactement comme on l'a fait pour le décalage caméra (0/5/10/20 cm).
2. **Occlusion géométrique** : déplacer la caméra `birdview` de sorte que le bras
   s'interpose réellement pendant l'approche — plus fidèle, moins contrôlable.
3. **Validation externe** sur MIKASA-Robo, seulement une fois qu'un mécanisme gagne chez
   nous, pour comparer aux chiffres publiés.

Et surtout, une expérience à faire **en premier** parce qu'elle est presque gratuite et
qu'elle **borne tout le reste** :

> **L'oracle-mémoire.** Entraîner la policy en lui donnant la position vraie de la canette
> **en permanence, même occluse** (5 scalaires, §F5). Ce n'est pas déployable — c'est une
> **borne supérieure** : elle dit combien de points la mémoire *peut au mieux* rapporter
> sur ce banc. Si l'oracle ne récupère pas l'essentiel de l'écart, le problème n'est pas la
> mémoire et tout le programme est à revoir. Si l'oracle récupère tout, on sait exactement
> ce qu'on poursuit, et le reste du travail n'est plus que d'apprendre à *estimer* ce que
> l'oracle recevait gratuitement.

Note : nous avons déjà validé qu'on peut retirer une béquille de ce type (Lift, vision pure
sans coordonnées du cube → 100 %). Ici c'est l'inverse : on **rajoute** la béquille pour
mesurer un plafond, pas pour la garder.

---

## 6. Le plan retenu (validé le 2026-09-08)

| # | expérience | ce qu'elle répond | état |
|---|---|---|---|
| **0** | **Banc Can-Occluded** + métrique de récupération ; courbe d'occlusion 0→100 % sur le modèle témoin | de combien l'occlusion fait chuter une policy markovienne, et à partir de quel seuil | 🔵 **lancé le 08/09** |
| **0b** | **Dataset Can occlus** (re-rendu des démos avec occlusion) | indispensable : le robot réel a été entraîné SUR des données occluses — entraîner en clair et tester en occlus mesurerait autre chose | à faire (rayon choisi d'après 0) |
| **1** | **Oracle-mémoire** : position vraie de la canette donnée en permanence, même occluse | **borne supérieure du gain** — décide si le programme vaut la peine | à faire |
| **2** | Témoin `n_obs_steps` = 8 | objective que la force brute échoue (copycat) et à quel prix en latence | à faire |
| **3** | **CAMP-lite** — candidat n°1 sous C1/C2/C3 | la mémoire des **actions** capture-t-elle « j'ai déjà tenté et raté » ? | à faire |
| **4** | **État latent façon μVLA** — candidat n°2 | le mécanisme le plus général suffit-il, et à quel coût d'entraînement ? | si 3 plafonne |
| **5** | **Test de scalabilité** : le vainqueur sur l'échelon M(n) (Square / ToolHang) | **valide C3** — le seul test qui répond à l'inquiétude « ça ne scalera pas » | à faire |
| **6** | **Taille × mémoire** : 28 M avec mémoire vs 263 M sans, à conditions égales | résout en prime le point ouvert §9.4 de la note atomman, et potentiellement la latence | à faire |

L'**augmentation par occlusion** est présente dans 3, 4 et 5 — ce n'est pas une expérience,
c'est une condition d'entraînement.

### Le banc, tel qu'il a été construit (étape 0, faite)

`src/can_occlusion.py`. On ne peint **pas** un masque sur l'image : un rectangle noir serait
un artefact jamais vu à l'entraînement, et on mesurerait une chute due au
hors-distribution plutôt qu'à la perte d'information. À la place, **on retire la canette de
la scène et on re-rend** (`sim.forward()`, jamais `sim.step()`, état restauré à
l'identique — vérifié Δqpos = Δqvel = 0). L'image obtenue est photométriquement parfaite et
ne diffère que par l'absence de la canette. Contrôle visuel :
`results/runs/can/occluded/test_occlusion.png`.

Déclenchement géométrique, un seul paramètre continu :
`||eef_xy − can_xy|| < radius` **et** `eef_z > can_z` — la géométrie exacte de l'échec réel.
`radius = 0` = témoin, `radius = ∞` = jamais visible (borne basse).

Modèle témoin : **`wristcap_A_agentview`** (cooldown 5k), mono-caméra, ResNet34 +
U-Net[128,256,512], `n_obs_steps = 2`, horizon 16 / 8 exécutées — **la même structure que le
modèle déployé sur le vrai bras**, donc les conclusions se transposent. Référence sans
occlusion : **337/500 = 67,4 %**. Le point `radius = 0` sert de **test de non-régression**.

Métriques ajoutées (`OcclusionProbe`) — le succès global dilue l'effet cherché :
`occl_fraction` (temps passé sans voir la cible), `n_occl_events`, **`n_approaches`**
(nombre de descentes dans la zone de saisie = marqueur direct de la boucle observée sur le
vrai robot), et **`recovery_rate`** = succès parmi les seuls épisodes réellement occlus.

### ⭐ Résultat de l'étape 0 (2026-09-09, en cours) — le banc reproduit l'échec réel

| rayon | succès (250 rollouts) | |
|---|---|---|
| **0 (témoin)** | **68,8 %** [62,8-74,2] | référence 67,4 % @500 dans l'IC → **non-régression validée** |
| 1 cm | ~59 % (en cours) | |
| **4 cm** | **23,2 %** [18,4-28,8] | **−45 points** |

**Le plus petit rayon de la grille initiale consommait déjà les deux tiers du succès** → la
grille a été révisée en cours de route (7/10/15 cm remplacés par 1/2/3 cm) pour capturer la
**pente** plutôt que trois planchers.

**Diagnostic sur les trajectoires enregistrées à 4 cm** (`115_diag_failure_mode.py`) :

| | réussis | ratés |
|---|---|---|
| cycles descente/remontée | 1,16 | **3,70** |
| fermetures de pince | 1,16 | **3,41** |
| distance XY minimale | 0,007 m | **0,017 m** |
| distance XY finale | 0,023 m | **0,422 m** |
| durée | 117 pas | **300** (maximum) |

**100 % des échecs font ≥2 cycles haut/bas, 66 % en font ≥3.** Le bras descend à 1,7 cm de
la canette — presque aussi près que lors des réussites (0,7 cm) — ferme la pince sur du
vide, remonte, redescend, puis finit à 42 cm : il a *reculé*. C'est mot pour mot la note
atomman §2, et mot pour mot *« la trajectoire allait TOUJOURS dans la bonne direction »*.
→ **l'échec est PERCEPTIF, pas décisionnel** : le modèle sait où viser, il perd la cible au
moment de la préhension. C'est exactement ce qu'une mémoire peut réparer.

### ⚠️ Erreur de métrique, corrigée — à ne pas refaire

`n_approaches` comptait les **entrées** dans la zone de saisie, et donnait 1,09 sur les
échecs contre 1,03 au témoin : j'en ai conclu (à tort, et je l'ai annoncé) que la boucle ne
se reproduisait pas. Le modèle qui boucle **ne sort jamais de la zone** — il descend, ferme,
remonte de quelques centimètres, redescend. Une seule transition comptée pour trois
tentatives réelles.
→ le bon marqueur est **`z_cycles`** : les cycles verticaux de l'effecteur, comptés **sans
aucune référence à la position de la cible**, donc insensibles au réglage des seuils de zone.
Ajouté à la sonde, angle mort documenté dans le code. **Aucun calcul perdu** : les
trajectoires étant enregistrées, `z_cycles` se recalcule a posteriori sur tous les rayons.

### ✅ ÉTAPE 0 TERMINÉE (nuit du 2026-09-09) — courbe complète

| rayon | succès (250 rollouts) | McNemar apparié vs témoin | survie des exposés |
|---|---|---|---|
| 0 (témoin) | **68,8 %** [62,8-74,2] | — | — |
| 1 cm | 60,4 % | 49 perdus / 33 gagnés, **p = 0,097 → NON significatif** | 86,2 % |
| 2 cm | 52,8 % | 68 / 33, p = 6,4e-4 | 65,1 % |
| **3 cm** | **37,6 %** | 94 / 21, p = 3,2e-12 | **45,1 %** |
| 4 cm | 23,2 % | 132 / 23, p = 8,8e-20 | 21,5 % |
| ∞ | 1,2 % | 166 / 2, p = 7,6e-47 | 0,6 % |

Dégradation **régulière, sans seuil** (−8, −8, −15, −14 points). L'analyse appariée (mêmes
250 états initiaux figés → appariement exact par numéro d'épisode) corrige la lecture naïve :
**à 1 cm l'effet n'est pas démontrable** (33 épisodes changent d'issue par pur hasard, ce qui
calibre le bruit du rollout à ~13 %). Reproductibilité mesurée : témoin 68,8 % puis 66,8 %
au rejeu → **±2 points**.

⚠️ `recovery_rate` est CONFONDU aux petits rayons (80,5 % à 1 cm, au-dessus du succès global) :
l'occlusion ne se déclenche que si le préhenseur passe très près, donc « être occlus » est
un marqueur de bonne visée. La mesure non biaisée est la **survie des exposés** ci-dessus.

### ❌ CORRECTION MAJEURE : la boucle n'est PAS causée par l'occlusion

Le contrôle témoin (rejeu du rayon 0 avec enregistrement) tranche, et **infirme ce que
j'avais affirmé** : les échecs font **3,6 cycles descente/remontée AU TÉMOIN**, contre 3,7 à
4 cm. Les réussites en font 1,1 partout. La boucle est donc la **signature générique d'un
échec de policy markovienne**, pas une conséquence de la perte de vue.

→ Ce que l'occlusion fait : elle **augmente le nombre d'échecs** (prouvé, McNemar), pas leur
*nature*. Mon intuition « le témoin bouclera nettement moins » était fausse — d'où l'utilité
d'avoir armé le contrôle plutôt que de conclure.
→ Conséquence favorable pour la suite : une mémoire qui brise la boucle aiderait **aussi
sans occlusion**. Les deux bénéfices (retenir la position, briser la boucle) sont distincts.

L'échec reste **perceptif** : à tous les rayons finis, les ratés atteignent 1,6-2,3 cm de la
canette (réussis : 0,6-0,8 cm) — le modèle vise juste et rate la préhension. Seul le cas
∞ est décisionnel (ratés à 15,5 cm : il ne sait pas où aller).

### ✅ CAMP-lite : module pré-entraîné et testé

- **Pré-entraînement** (4000 steps, ~2 min) : reconstruction DCT 0,46 → **0,034**, val 0,040
  (pas de surapprentissage), **26,3 codes actifs / 128 → aucun collapse du codebook**.
- **LE test de discrimination** (`114`) : **95,3 % de codes différents** entre la 1re
  tentative et les suivantes, sur **957 épisodes** et 2423 comparaisons. Distance latente
  médiane 0,34. → **la mémoire distingue les tentatives, la boucle est brisable.**
  ⚠️ Première version du test : 78 % sur ~30 épisodes seulement — elle découpait les
  tentatives avec `appr_flags`, c'est-à-dire **la métrique aveugle déjà corrigée dans la
  sonde**. L'erreur s'était propagée. Segmentation refaite sur les creux de la trajectoire
  verticale.
- **Pipeline validé de bout en bout** : `CAMP=1` charge la mémoire gelée, étend l'état
  9D → 41D, précalcule 23 207 codes, et l'entraînement tourne — **val-loss incluse**
  (0,93 → 0,54), ce qui valide le correctif d'index global.

### 📦 Étape 0b faite — datasets occlus à 3 cm

Rayon retenu **3 cm** (37,6 % : chute franche de −31 points, et il reste 45 % de survie donc
de la marge à récupérer). Deux datasets, 35 Mo chacun :
`local/can_ph_proprio_occ03` (9D vision pure = baseline) et `local/can_ph_occ03`
(12D avec `can_pos` = **ORACLE**, position donnée même occluse). L'écart entre les deux
bornera ce que la mémoire peut au mieux rapporter.

### ⚠️ Vérifié le 2026-09-08 — et la réponse nuance l'argument C1

**Pas de code publié.** Le site projet `robo-camp.github.io` existe mais ne contient aucun
dépôt (auteurs : Wang, Yeom, Cao, Zhi, Shinde, Yip — UCSD ARCLAB). Tout est donc
ré-implémenté d'après le papier, dans `src/camp.py`.

**Le module n'est PAS simplement pré-entraîné puis gelé.** Le papier fait
**pré-entraînement, PUIS warm-up à mémoire gelée, PUIS finetuning CONJOINT** avec le LSTM à
`lr × α = 0,1`, et l'encodeur visuel est *partagé avec la tête d'action*. Le finetuning
conjoint oblige à dérouler le LSTM pendant l'entraînement de la policy — exactement ce que
C1 interdit. **Mon « C1 satisfaite par construction » était trop optimiste.**

D'où **deux variantes**, à faire dans cet ordre :

- **variante B — « CAMP-lite » (implémentée)** : module pré-entraîné puis **gelé
  définitivement**, `m_t` précalculé pour toutes les frames et concaténé à
  `observation.state`. La boucle d'entraînement de la policy est alors *rigoureusement*
  celle d'un run normal. **C1 satisfaite pour de vrai**, et sans plomberie risquée dans
  LeRobot.
- **variante A — fidèle au papier** (warm-up gelé puis finetuning conjoint α=0,1) : à ne
  tenter **que si B plafonne**. On saura alors que le finetuning conjoint compte vraiment,
  et on décidera en connaissance de cause s'il mérite son coût.

⚠️ Ne pas présenter B comme « CAMP » : c'est CAMP amputé de son finetuning conjoint, et
**un échec de B ne réfute pas CAMP**.

**Détails récupérés du papier** : `w_k = exp(−γk/(K−1))` (pondération fréquentielle) ;
`L` = longueur complète de l'épisode ; `ℒ_cons = Σ_N ‖â_t[:L−N] − â_{t+N}[N:]‖²` ;
AdamW, batch 16 pour le pré-entraînement mémoire, lr 1e-4, EMA sur les poids de la policy,
DDPM à l'entraînement / DDIM à l'inférence. `λ_cons` n'est pas spécifié → à caler chez nous.

### Conséquence sur le banc : il faut un ESCALIER, pas une marche

C3 impose de vérifier qu'un mécanisme tient sur plus dur. Un banc M(1) seul ne permet pas
de conclure.

| échelon | tâche | ce qu'il faut retenir |
|---|---|---|
| **1 — M(1)** | Can-Occluded | *où était la canette* |
| **2 — M(n)** | NutAssemblySquare / ToolHang (déjà dans `data_cache`) | *ce que j'ai déjà fait, ce qu'il reste* |

Un mécanisme ne se qualifie que s'il gagne **sur les deux échelons**.

---

## 2. Le piège à connaître AVANT de commencer : la confusion causale

C'est le point qui fait échouer la solution naïve, et il faut le poser en premier.

**Le problème « copycat »** (Wen et al., NeurIPS 2020) : en clonage de comportement à
partir d'un **historique** d'observations, quand les actions de l'expert sont fortement
corrélées dans le temps, le réseau apprend un raccourci — il **retrouve l'action
précédente dans l'historique et la recopie**, au lieu de prédire la bonne action suivante.
La perte d'entraînement chute, et la policy s'effondre en déploiement dès que la
distribution dérive.

Deux conséquences directes :

- **C'est exactement notre symptôme.** « Il relève le bras comme s'il pensait avoir la
  pomme », « il boucle indéfiniment » : ce sont des signatures de policy qui continue son
  geste par inertie plutôt que par décision.
- **Cela explique pourquoi Diffusion Policy fixe `n_obs_steps = 2`.** Chi et al. ont trouvé
  empiriquement que davantage d'historique **dégrade**. Ce n'est pas une limite de
  capacité, c'est la confusion causale. Donc : **augmenter `n_obs_steps` ne marchera
  probablement pas**, en plus de coûter cher.

**Remèdes connus**, à intégrer dès la conception :
- **goulot d'information** sur la mémoire : la compresser fortement (CAMP quantifie sur un
  codebook de 128 et ne garde que 32 dimensions) — le réseau ne *peut pas* y cacher
  l'action précédente en clair ;
- **ne pas fournir l'action précédente brute** en entrée du dénoiseur ;
- **abandon aléatoire (dropout) de l'historique** pendant l'entraînement ;
- prédiction du **résidu** d'action plutôt que de l'action (Chuang et al. 2022) ;
- retrait adversarial de l'information sur l'action précédente (Wen et al. 2020).

---

## 3. Les six familles de mémoire

### F0 — Empilement de frames (`n_obs_steps` ↑) — *le témoin, pas la solution*
Ce qu'on ferait spontanément. Coût : ×N passes d'encodeur. Bénéfice : quelques centaines
de ms de passé seulement. Risque : copycat (§2). **À faire comme point de comparaison
négatif**, pour objectiver que le problème n'est pas soluble par la force brute.

### F1 — Récurrence dans le réseau (état caché porté d'un pas au suivant)
La famille la plus rentable dans notre régime.

- **BC-RNN** (robomimic, Mandlekar et al. 2021) — l'ancêtre : LSTM entraîné sur des
  séquences de longueur T ; au test, déroulé pas à pas, **état caché rafraîchi tous les T
  pas**. Déjà présent dans notre historique de projet (run 37 Lift, `robomimic_bc_gmm`).
- **μVLA** (2606.12497) — *le résultat le plus important pour nous* : un petit ensemble de
  **jetons mémoire apprenables** transportés d'un pas de temps au suivant et mis à jour par
  self-attention. **Aucune loss auxiliaire, aucun changement d'architecture.** Entraînement
  par **BPTT tronquée**. Deux variantes de mise à jour testées : gradients inter-pas, ou
  **EMA détachée** (encore plus simple). Résultats : MIKASA-Robo **0,42 → 0,84** en
  observabilité partielle ; LIBERO (pleine observabilité) **96,2 %, aucune régression**.
  Nuance honnête des auteurs : le gain disparaît quand la structure de mémoire ne
  correspond pas au besoin de la tâche.
- **TFP** (2607.08283) — croyance latente épisodique par réseau à **constante de temps
  liquide** (LTC) : la constante de temps dépend de l'entrée, donc la mémoire *retient*
  pendant les phases stables et *se met à jour vite* près des événements de manipulation
  (les auteurs mesurent un gain d'écriture ~6× plus grand près des événements). Croyance de
  256 D injectée dans le décodeur par **modulation AdaLN** — elle *façonne* la distribution
  d'actions au lieu d'être un contexte passif. Coût latence négligeable.
  Résultats : MIKASA ShellGameTouch 47 → **75 %** ; réel, échange d'objets **3/20 → 15/20**.

### F2 — Mémoire compressée de l'historique d'ACTIONS — **CAMP** (2606.21188)
*Le candidat le plus directement greffable sur notre pile.*

L'idée : **l'historique des actions du robot est un signal auto-supervisé riche** — il
encode « ce que j'ai déjà tenté », donc les échecs, les reprises, la progression. Pas
besoin d'annotation ni de prédiction visuelle coûteuse.

Architecture exacte :
- encodeur = **LSTM 2 couches**, entrée = [features visuelles partagées, proprioception
  `p_t`, **action précédente `a_{t-1}`**], **dimension cachée 64** ;
- tête linéaire → **K = 32 coefficients DCT** de la trajectoire d'actions passée ;
- **quantification vectorielle**, codebook 128 → code mémoire **`m_t` de 32 dimensions** ;
- `m_t` est simplement **concaténé au vecteur de conditionnement** de la diffusion policy :
  `concat[f(o_third), f(o_wrist), p_t, m_t]`.

Entraînement (pré-entraînement du module, auto-supervisé) :
- **perte de reconstruction** des coefficients DCT des actions passées, **pondérée en
  fréquence** pour écraser les hautes fréquences (on garde la forme du geste, pas le bruit) ;
- **perte de cohérence temporelle** : deux reconstructions à des instants différents
  doivent s'accorder sur le segment d'actions qu'elles partagent ;
- **BPTT sur des fenêtres de la longueur de l'épisode**.

Résultats : Memory-Manip-Bench **64,3 %** contre Diffusion Policy 40,9 %, π₀.₅ 26,8 %,
MemoryVLA 5,7 %. Memory-T-Bench **93,6 %** contre 66,8 %.

⚠️ **Réserve à garder en tête** : donner l'action précédente en entrée est *précisément*
ce qui déclenche le copycat. CAMP s'en protège par le goulot (DCT basse fréquence + VQ 32 D),
mais c'est une protection empirique. Surveiller le symptôme : perte qui s'effondre,
succès qui ne suit pas — le signe que nous connaissons déjà bien.

### F3 — Modèles espace-d'état (Mamba / SSM)
Compromis intéressant : coût **linéaire** en longueur de séquence, et surtout **O(1) par
pas en inférence** (forme récurrente) → compatible temps réel par construction.

- **MaIL** (2406.08234) : Mamba comme dorsale d'imitation, bat les Transformers sur toutes
  les tâches LIBERO, et surtout **tient bien sur petits jeux de données** (moins de
  surapprentissage). ← pertinent : nous avons **85 épisodes**.
- **Mamba Policy** (2409.07163) : SSM hybride pour diffusion policy 3D, **−80 % de
  paramètres** à performance égale. ← directement dans l'axe « pas plus gros ».
- **RoboSSM** (2509.19658), **DSSP** (2605.14598, encodage de l'historique complet).

### F4 — Mémoire externe / récupération — **HALO** (2606.25136)
Buffer épisodique non paramétrique (192–3840 entrées/épisode) d'observations et d'actions
encodées, avec récupération sélective. Entraînement **double objectif** : imitation +
**questions-réponses visuelles** générées automatiquement (« où est l'objet », « combien »,
« dans quel ordre ») qui biaisent la récupération vers l'information utile.

Résultats forts : bat **LSTM de 42 %**, **Mamba de 21 %**, **Transformer-XL de 29 %** —
l'argument étant que les états compressés *jettent* les détails nécessaires au rappel d'un
événement précis.

**Mais** : les auteurs reconnaissent la **latence de récupération** comme limitation
ouverte, et il faut fabriquer les paires VQA. → **écarté pour nous à ce stade** : c'est une
solution M(n), pour un problème que nous n'avons pas, à un coût que nous ne pouvons pas payer.

### F5 — Mémoire structurée objet-centrée — *la plus adaptée à notre échec précis*
Au lieu de mémoriser un vecteur latent opaque, on mémorise **où est l'objet**.

- **Historique par suivi de points** (2509.17141) : SAM-2 segmente les objets pertinents,
  **TAPIP3D** suit **K = 5 points par objet** en 3D, les trajectoires sont compressées en
  **un jeton par objet** et fusionnées aux jetons d'observation avant le dénoiseur.
  Le suivi tourne **en asynchrone** de l'inférence (5,44 Hz vs 0,93 Hz en synchrone), avec
  une **augmentation par abandon aléatoire** à l'entraînement pour rendre la policy robuste
  à cette asynchronie. Résultats : **85 %** contre 45–50 % pour les fenêtres fixes ;
  précision de décision > 90 %.
- **SAM2Act++ / approches object-centric** : ne stocker que les **coordonnées du centre de
  l'objet** — mémoire minuscule, a priori spatial fort, peu de bruit.
- **Embodied-SlotSSM / LIBERO-Mem** (2511.11478) : slots d'objets persistants + SSM.
- **« More Structure, Not More Capacity »** (2607.09825) : le titre résume l'argument
  central de cette section, et il s'applique mot pour mot à notre situation.

👉 **La version dégénérée, pour nous, tient en 5 scalaires** : `[x, y, z]` de la dernière
position connue de la pomme, un drapeau `visible / non visible`, et le **temps écoulé
depuis la dernière observation**. Concaténés à l'état. Coût nul, entièrement
interprétable, et c'est l'antidote exact à « il ne la voit plus donc il enchaîne ».
En simulation la position est une vérité terrain gratuite ; au réel il faut un détecteur
(la pomme est blanche sur fond contrasté — segmentation couleur plausible), ou :

**Variante sans détecteur — tête auxiliaire de permanence d'objet** : on ajoute une petite
tête qui prédit la position de la pomme à partir de l'état interne, entraînée avec la
vérité terrain **y compris pendant les frames où elle est occluse**. La perte auxiliaire
force la représentation interne (et donc la mémoire récurrente) à **transporter** la
position à travers l'occlusion. À l'inférence la tête peut être jetée. C'est la façon la
moins chère d'apprendre la permanence de l'objet, et elle se combine avec F1.

### F6 — Perception active — See2Act (2606.23625)
Plutôt que de se souvenir, **bouger pour revoir** : le débruitage d'action est couplé à un
raffinement de point de vue, appris depuis des poses de caméra ancrées sur des actions clés.
Jusqu'à +34 % sur RLBench sous occlusion sévère. Élégant et complémentaire, mais cela change
le schéma de contrôle et exige de nouvelles démonstrations. → **hors périmètre pour l'instant**,
à noter comme piste si la mémoire seule plafonne.

---

## 4. Comment on entraîne ça — le vrai savoir-faire

C'est ici que ça se joue, plus que dans le choix d'architecture. Passer d'une policy
markovienne à une policy récurrente change la **boucle d'entraînement**, pas seulement le
modèle.

1. **Échantillonnage par séquences.** Aujourd'hui on tire des fenêtres indépendantes
   (`delta_timestamps` de LeRobot). Une policy récurrente exige des **segments
   temporellement contigus**, et le batch doit garder l'identité de l'épisode.
2. **BPTT tronquée.** Rétropropager sur tout un épisode (~297 frames) est trop coûteux.
   Recette **EATB** de TFP : entraîner sur des morceaux contigus de plusieurs épisodes en
   parallèle, **un état caché par épisode**, gradient **tronqué** aux frontières de
   segment mais état **conservé numériquement** (`detach()`). On garde le crédit
   long-horizon sans en payer le coût. Longueur de segment typique : **16 à 64 pas**.
   Alternative encore plus simple (μVLA) : mise à jour de la mémoire par **EMA détachée**,
   donc aucun gradient inter-pas.
3. **Gestion de l'état caché au test.** BC-RNN **rafraîchit l'état tous les T pas** —
   sinon il dérive hors distribution. À décider et à mesurer. Et **remise à zéro en début
   d'épisode** au déploiement (notre `policy.reset()` existant).
4. **Anti-copycat** (§2) : goulot d'information, dropout d'historique, ne pas donner
   l'action précédente en clair.
5. **Augmentation par occlusion.** Masquer aléatoirement la cible pendant l'entraînement
   pour *forcer* l'usage de la mémoire — sans ça, le réseau apprend à ignorer le module
   mémoire tant que la vue est dégagée. C'est le pendant du « random drop » du suivi de points.
   Probablement **le levier le plus important de toute la liste**, et il est gratuit.
6. **Le juge reste le rollout**, jamais la perte (notre règle habituelle). Ajouter une
   métrique dédiée : **taux de récupération après occlusion** — proportion d'épisodes où
   la policy retrouve la cible après l'avoir perdue de vue (l'équivalent de la *Decision
   Accuracy* du papier suivi-de-points). C'est cette métrique-là qui mesure la mémoire ;
   le succès global la dilue.
7. **Méthodologie inchangée par ailleurs** : LR constant + EMA + cooldown évalué.
   ⚠️ Rappel du §8 de la note atomman : le scheduler demandé a été **silencieusement
   remplacé** par `cosine`. Vérifier le `train_config.json` du checkpoint après quelques
   centaines de pas, systématiquement.

---

## 5. Sur quel banc en simulation ?

### Les bancs publics
| banc | ce qu'il teste | pour nous |
|---|---|---|
| **MIKASA-Robo** (2502.10550, ManiSkill3) | 32 tâches, 4 catégories de mémoire, **objets occlus**, rappel de configurations | le plus proche de notre besoin ; utilisé par μVLA et TFP → chiffres comparables |
| **LIBERO-Mem** (2511.11478) | observabilité partielle **au niveau objet**, instances visuellement identiques, sous-buts séquencés | plutôt M(n), sémantique |
| **RMBench** (2603.01229) | 9 tâches classées M(1)/M(n) | surtout utile pour son **cadre TMC** |
| **Memory-T/Manip-Bench** (CAMP) | 4 variantes PushT + 7 tâches 3D contact-riche | référence si on reproduit CAMP |
| **ReMemBench** (HALO) | rappel long-horizon | hors périmètre |

### Ce que je recommande quand même : **notre propre banc, « Can-Occluded »**

Raison : nous avons déjà un **harnais d'évaluation mûr** sur robomimic Can — 500 rollouts,
IC95 de Wilson, étude de convergence auto-extensible, graphe 4-panneaux, calibrage des
temps, EMA/SWA/cooldown. Repartir sur MIKASA-Robo, c'est **jeter tout cela** et
re-débugger un environnement pour mesurer un effet qu'on peut mesurer chez nous.

Construction, par coût croissant :
1. **Occlusion synthétique scriptée** (le plus contrôlé, et le plus rapide) : on masque la
   canette dans l'image d'observation sur des fenêtres temporelles choisies — par exemple
   dès que le préhenseur passe à moins de X cm au-dessus d'elle, ce qui **reproduit
   exactement la géométrie de l'échec réel**. Avantage décisif : le **degré d'occlusion
   devient un paramètre continu** (0 % → 100 %), donc on trace une *courbe* de robustesse
   au lieu d'un point, exactement comme on l'a fait pour le décalage caméra (0/5/10/20 cm).
2. **Occlusion géométrique** : déplacer la caméra `birdview` de sorte que le bras
   s'interpose réellement pendant l'approche — plus fidèle, moins contrôlable.
3. **Validation externe** sur MIKASA-Robo, seulement une fois qu'un mécanisme gagne chez
   nous, pour comparer aux chiffres publiés.

Et surtout, une expérience à faire **en premier** parce qu'elle est presque gratuite et
qu'elle **borne tout le reste** :

> **L'oracle-mémoire.** Entraîner la policy en lui donnant la position vraie de la canette
> **en permanence, même occluse** (5 scalaires, §F5). Ce n'est pas déployable — c'est une
> **borne supérieure** : elle dit combien de points la mémoire *peut au mieux* rapporter
> sur ce banc. Si l'oracle ne récupère pas l'essentiel de l'écart, le problème n'est pas la
> mémoire et tout le programme est à revoir. Si l'oracle récupère tout, on sait exactement
> ce qu'on poursuit, et le reste du travail n'est plus que d'apprendre à *estimer* ce que
> l'oracle recevait gratuitement.

Note : nous avons déjà validé qu'on peut retirer une béquille de ce type (Lift, vision pure
sans coordonnées du cube → 100 %). Ici c'est l'inverse : on **rajoute** la béquille pour
mesurer un plafond, pas pour la garder.

---

## 6. Le plan retenu (validé le 2026-09-08)

| # | expérience | ce qu'elle répond | état |
|---|---|---|---|
| **0** | **Banc Can-Occluded** + métrique de récupération ; courbe d'occlusion 0→100 % sur le modèle témoin | de combien l'occlusion fait chuter une policy markovienne, et à partir de quel seuil | 🔵 **lancé le 08/09** |
| **0b** | **Dataset Can occlus** (re-rendu des démos avec occlusion) | indispensable : le robot réel a été entraîné SUR des données occluses — entraîner en clair et tester en occlus mesurerait autre chose | à faire (rayon choisi d'après 0) |
| **1** | **Oracle-mémoire** : position vraie de la canette donnée en permanence, même occluse | **borne supérieure du gain** — décide si le programme vaut la peine | à faire |
| **2** | Témoin `n_obs_steps` = 8 | objective que la force brute échoue (copycat) et à quel prix en latence | à faire |
| **3** | **CAMP-lite** — candidat n°1 sous C1/C2/C3 | la mémoire des **actions** capture-t-elle « j'ai déjà tenté et raté » ? | à faire |
| **4** | **État latent façon μVLA** — candidat n°2 | le mécanisme le plus général suffit-il, et à quel coût d'entraînement ? | si 3 plafonne |
| **5** | **Test de scalabilité** : le vainqueur sur l'échelon M(n) (Square / ToolHang) | **valide C3** — le seul test qui répond à l'inquiétude « ça ne scalera pas » | à faire |
| **6** | **Taille × mémoire** : 28 M avec mémoire vs 263 M sans, à conditions égales | résout en prime le point ouvert §9.4 de la note atomman, et potentiellement la latence | à faire |

L'**augmentation par occlusion** est présente dans 3, 4 et 5 — ce n'est pas une expérience,
c'est une condition d'entraînement.

### Le banc, tel qu'il a été construit (étape 0, faite)

`src/can_occlusion.py`. On ne peint **pas** un masque sur l'image : un rectangle noir serait
un artefact jamais vu à l'entraînement, et on mesurerait une chute due au
hors-distribution plutôt qu'à la perte d'information. À la place, **on retire la canette de
la scène et on re-rend** (`sim.forward()`, jamais `sim.step()`, état restauré à
l'identique — vérifié Δqpos = Δqvel = 0). L'image obtenue est photométriquement parfaite et
ne diffère que par l'absence de la canette. Contrôle visuel :
`results/runs/can/occluded/test_occlusion.png`.

Déclenchement géométrique, un seul paramètre continu :
`||eef_xy − can_xy|| < radius` **et** `eef_z > can_z` — la géométrie exacte de l'échec réel.
`radius = 0` = témoin, `radius = ∞` = jamais visible (borne basse).

Modèle témoin : **`wristcap_A_agentview`** (cooldown 5k), mono-caméra, ResNet34 +
U-Net[128,256,512], `n_obs_steps = 2`, horizon 16 / 8 exécutées — **la même structure que le
modèle déployé sur le vrai bras**, donc les conclusions se transposent. Référence sans
occlusion : **337/500 = 67,4 %**. Le point `radius = 0` sert de **test de non-régression**.

Métriques ajoutées (`OcclusionProbe`) — le succès global dilue l'effet cherché :
`occl_fraction` (temps passé sans voir la cible), `n_occl_events`, **`n_approaches`**
(nombre de descentes dans la zone de saisie = marqueur direct de la boucle observée sur le
vrai robot), et **`recovery_rate`** = succès parmi les seuls épisodes réellement occlus.

### ⭐ Résultat de l'étape 0 (2026-09-09, en cours) — le banc reproduit l'échec réel

| rayon | succès (250 rollouts) | |
|---|---|---|
| **0 (témoin)** | **68,8 %** [62,8-74,2] | référence 67,4 % @500 dans l'IC → **non-régression validée** |
| 1 cm | ~59 % (en cours) | |
| **4 cm** | **23,2 %** [18,4-28,8] | **−45 points** |

**Le plus petit rayon de la grille initiale consommait déjà les deux tiers du succès** → la
grille a été révisée en cours de route (7/10/15 cm remplacés par 1/2/3 cm) pour capturer la
**pente** plutôt que trois planchers.

**Diagnostic sur les trajectoires enregistrées à 4 cm** (`115_diag_failure_mode.py`) :

| | réussis | ratés |
|---|---|---|
| cycles descente/remontée | 1,16 | **3,70** |
| fermetures de pince | 1,16 | **3,41** |
| distance XY minimale | 0,007 m | **0,017 m** |
| distance XY finale | 0,023 m | **0,422 m** |
| durée | 117 pas | **300** (maximum) |

**100 % des échecs font ≥2 cycles haut/bas, 66 % en font ≥3.** Le bras descend à 1,7 cm de
la canette — presque aussi près que lors des réussites (0,7 cm) — ferme la pince sur du
vide, remonte, redescend, puis finit à 42 cm : il a *reculé*. C'est mot pour mot la note
atomman §2, et mot pour mot *« la trajectoire allait TOUJOURS dans la bonne direction »*.
→ **l'échec est PERCEPTIF, pas décisionnel** : le modèle sait où viser, il perd la cible au
moment de la préhension. C'est exactement ce qu'une mémoire peut réparer.

### ⚠️ Erreur de métrique, corrigée — à ne pas refaire

`n_approaches` comptait les **entrées** dans la zone de saisie, et donnait 1,09 sur les
échecs contre 1,03 au témoin : j'en ai conclu (à tort, et je l'ai annoncé) que la boucle ne
se reproduisait pas. Le modèle qui boucle **ne sort jamais de la zone** — il descend, ferme,
remonte de quelques centimètres, redescend. Une seule transition comptée pour trois
tentatives réelles.
→ le bon marqueur est **`z_cycles`** : les cycles verticaux de l'effecteur, comptés **sans
aucune référence à la position de la cible**, donc insensibles au réglage des seuils de zone.
Ajouté à la sonde, angle mort documenté dans le code. **Aucun calcul perdu** : les
trajectoires étant enregistrées, `z_cycles` se recalcule a posteriori sur tous les rayons.

**Contrôle en attente** (`111c_temoin_recheck.sh`, armé) : le témoin a tourné avant
l'enregistrement, donc on ignore les `z_cycles` de SES échecs. Sans ce chiffre, impossible
d'attribuer la boucle à l'occlusion plutôt qu'au modèle. Le script rejoue le témoin avec
enregistrement dès la fin de la courbe.

## 7. Ce que je retiens en trois lignes

1. **La mémoire est gratuite en latence** — tant qu'on récurre un état caché (64–256 D) et
   qu'on n'empile pas des images. La contrainte temps réel n'interdit pas la mémoire.
2. **Notre tâche est M(1)** (retenir *où était la pomme*), le régime le plus facile ; viser
   la mémoire **spatiale** structurée, pas la machinerie sémantique long-horizon.
3. **Le risque principal n'est pas l'architecture, c'est la confusion causale** — et le
   remède principal n'est pas non plus l'architecture, c'est **l'augmentation par occlusion
   à l'entraînement**, qui force le réseau à se servir de la mémoire qu'on lui donne.

---

## Sources

**Mémoire & observabilité partielle en manipulation**
- CAMP — *Remember what you did? Learning Behavioral Memories for Partially Observable Object Manipulation* — arXiv 2606.21188
- μVLA — *On Recurrent Memory for Partially Observable Manipulation in VLA Models* — arXiv 2606.12497
- TFP — *Temporally Conditioned Memory-Fusion Policies for Visuomotor Learning* — arXiv 2607.08283
- HALO — *Memory Retrieval in Visuomotor Policies for Long-Horizon Robot Control* — arXiv 2606.25136
- MEMBOT — *Memory-Based Robot in Intermittent POMDP* — arXiv 2509.11225
- *Resolving State Ambiguity in Robot Manipulation via Adaptive Working Memory Recoding* — arXiv 2512.24638

**Mémoire structurée / objet-centrée**
- *History-Aware Visuomotor Policy Learning via Point Tracking* — arXiv 2509.17141
- *Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective* (LIBERO-Mem, Embodied-SlotSSM) — arXiv 2511.11478
- *More Structure, Not More Capacity: Object-Centric Representations for Visuomotor Imitation Learning* — arXiv 2607.09825
- *3D-Anchored Lookahead Planning for Persistent Robotic Scene Memory* — arXiv 2604.11302

**Modèles espace-d'état**
- MaIL — *Improving Imitation Learning with Mamba* — arXiv 2406.08234
- Mamba Policy — *Towards Efficient 3D Diffusion Policy with Hybrid Selective State Models* — arXiv 2409.07163
- RoboSSM — *Scalable In-context Imitation Learning via State-Space Models* — arXiv 2509.19658
- DSSP — *Diffusion State Space Policy with Full-History Encoding* — arXiv 2605.14598

**Confusion causale / copycat**
- Wen et al. — *Fighting Copycat Agents in Behavioral Cloning from Observation Histories* — NeurIPS 2020, arXiv 2010.14876
- Chuang et al. — *Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction* — arXiv 2207.09705

**Bancs d'essai**
- MIKASA-Robo — *Memory, Benchmark & Robots* — arXiv 2502.10550
- RMBench — *Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design* — arXiv 2603.01229
- robomimic — *What Matters in Learning from Offline Human Demonstrations* (BC-RNN) — arXiv 2108.03298

**Perception active**
- See2Act — *Learning to See While Learning to Act: Diffusion Models for Active Perception in Robot Imitation* — arXiv 2606.23625

> ⚠️ Les travaux 25xx/26xx ont été lus via résumé et version HTML arXiv, pas intégralement.
> Les chiffres cités sont ceux annoncés par les auteurs, non reproduits par nous.

---

## 🏁 BILAN DU CHANTIER (2026-09-09 → 12)

### Le problème est chiffré

| | succès (150 rollouts appariés) |
|---|---|
| **plafond sans occlusion** (clair/clair) | **72,7 %** |
| baseline sous occlusion 3 cm | **56,0 %** |
| **coût de l'occlusion** | **+16,7 pts** IC95 [+6,6 ; +26,7] · p = 0,002 ★ |

Et au-dessus du plafond, **27 % d'échecs de PRÉHENSION** que ni l'occlusion ni la mémoire n'expliquent.

### ⭐ Le vrai goulot : la préhension, pas la mémoire

Au moment de la fermeture de pince, avec la position exacte de la canette fournie en continu :

| | hauteur au-dessus de la canette | décalage XY |
|---|---|---|
| réussites (n=116) | 1,7 cm | 1,4 cm |
| **échecs (n=34)** | **3,5 cm** | **2,9 cm** |

62 % des échecs ferment à plus de 3 cm au-dessus, 88 % sont décalés de plus de 2 cm.
**Une fois la canette soulevée, les deux modèles réussissent à 95 %** — transport et dépôt ne posent aucun problème.
→ C'est exactement la note atomman §3 (« il fermait trop haut »), laissée en *cause non établie*. Elle l'est maintenant.

### Aucun mécanisme de mémoire ne récupère les 17 points

12 comparaisons appariées, **aucune significative** : CAMP variantes A et B, spatial softmax à 32/64/128/256
keypoints, signal d'absence (KPAMP). La variance entre entraînements fait basculer **~55 épisodes sur 150**,
ce qui rend indétectable tout écart < ~12 points.

⚠️ **Réserve majeure sur CAMP** : le module mémoire ne fonctionnait pas quand il a servi à juger — reconstruction
à 75,5 % d'erreur pour une borne de 10,7 %, arrêté à moins d'un quart de son apprentissage. Convergé et élargi
(`h=256`) il descend à 41,1 %. **Le verdict CAMP porte donc sur un module cassé** et n'a jamais été refait.

### Les 3 résultats significatifs

1. **oracle 12D** : 73,3 % (+17,3 pts, p = 0,0007) — mais c'est la béquille `can_pos`, et elle donne
   *exactement* le même plafond que l'absence d'occlusion (72,7 %) → **donner la position compense
   exactement l'occlusion**, ce qui confirme que l'information perdue est bien la position.
2. **réduire le U-Net** [64,128,256]→[32,64,128] coûte **−18 pts** (p = 0,002) → le décodeur porte la
   performance, il n'est PAS compressible. ⚠️ important pour la latence : il est déroulé **10×** par inférence.
3. **coût de l'occlusion** : +16,7 pts (p = 0,002).

### Ce qu'on a appris sur le spatial softmax

- keypoints **non interprétables** (confirmé par la littérature : Soare/LeRobot, et arXiv 2606.15232) ;
- softmax **saturé** — p_max médian 0,85, 63 % des keypoints à moins de 0,2 d'un point de grille :
  il se comporte comme un **argmax sur 9 cellules**, pas comme un localisateur continu ;
- **aucune notion d'absence** : motif absent → activation plate → sortie (0,0), le centre, indiscernable
  d'une vraie détection. Quand la canette disparaît les keypoints **se replacent** (jusqu'à 42,8 px) ;
- l'information sur la canette est présente mais **distribuée** (R² 0,76, ~6,6 cm d'erreur), jamais portée
  par un keypoint dédié.

### Recommandation

**Attaquer la préhension** : 27 points de gisement contre 17, défaut mesuré précisément, et il limitera le
robot réel exactement comme ici. Par ordre de coût : **résolution** (le projet a déjà mesuré 6 % → 42 % en
passant à 224 px sur le delta-actions), **horizon d'exécution** (8 actions jouées d'avance → fermeture décidée
sur une observation périmée), **moyennage des modes** de la diffusion.

### ⚠️ Leçons de méthode

- **Poser le contrôle sur données saines EN PREMIER.** Le plafond clair/clair aurait dû être la 1re expérience :
  il donne le gisement en 1 h. Faute de ce point, 4 hypothèses fausses ont été testées (banc, dataset, décodeur,
  schedule LR), ~20 h de calcul.
- **Vérifier qu'un module atteint SON objectif avant de juger ce qu'il apporte** (la reconstruction CAMP n'a
  été regardée qu'après deux jours de conclusions).
- **`pgrep -f` / `pkill -f` mentent** : ils matchent la ligne de commande des scripts et commandes SSH qui
  *mentionnent* le motif. 4 incidents : un run jamais lancé mais rapporté « armé », 2 moniteurs tués, un script
  d'attente bloqué par un autre. Tester le **contenu** (fichier, compteur), jamais la présence d'un processus.
- **`test -d dossier` est faux pendant un rsync** (arborescence créée dès la 1re seconde) → un run a démarré
  sur 86 épisodes sur 200.
