# Phase 5 — Can (PickPlaceCan) : généralise-t-on à une tâche plus dure ?

> ⏳ **Phase en cours.** Après Lift (résolu + compressé), on teste si les conclusions tiennent sur une tâche **pick-and-place** plus longue : attraper une canette et la déposer dans un bac. But : voir si les planchers (capacité U-Net, pas de diffusion, nb démos) **montent** quand la tâche se complexifie.
> Liens : [◀ Phase 4 Compression](COMPRESSION.md) · **Phase 5 (ici)** · [index](README.md)

## Contexte

**Robomimic Can / `PickPlaceCan`** : le bras Panda saisit une canette sur une table et la **dépose dans le bon compartiment** d'un bac — donc saisie **+ transport + dépôt** (vs Lift = juste soulever). ~2× plus d'étapes, horizon doublé (démos ~118 steps vs 59).

Dataset : **200 démos humaines** (ph), action 7D. Observations quasi identiques à Lift sauf `object` = **14D** (canette : pose absolue + position/orientation relative à la pince) → vecteur d'état **23D** (vs 19D pour Lift).

⚠️ **Single-object, pas du tri.** Le bac a 4 compartiments car la scène est héritée de la tâche complète **PickPlace** (4 objets : canette, lait, céréales, pain → 4 bacs). `PickPlaceCan` n'en garde **qu'un** (la canette, toujours vers son compartiment attribué ; les 3 autres restent vides). Le tri à 4 objets est une tâche **encore plus dure**, gardée en réserve.

**Pourquoi Can** (plutôt que Square/insertion ou le tri complet) : la marche incrémentale la plus sûre depuis Lift — pipeline réutilisable presque tel quel, baseline probablement atteignable.

## Le parcours

| Étape | État |
|---|---|
| 1. Télécharger Can ph (200 démos, `low_dim_v141.hdf5`) | ✅ |
| 2. Convertir → LeRobot (replay états → rendu agentview 96px) | ✅ `src/can_to_lerobot.py` |
| 3. Adapter le harnais d'éval (`make_env`, succès, **`fix_obs_sign`**) | ⏳ |
| 4. Entraîner un baseline diffusion (pleine capacité, pleines données) | ⏳ |
| 5. Compression + grilles 500 rollouts (réutilise l'outillage de la phase 4) | ⏳ |

## Résultats

⏳ *À venir — baseline en préparation.* (Démos expertes : voir `data_cache/can_demos/`.)

## Leçons clés

⏳ *À venir.*

## Détails techniques

- **Pipeline** : `src/can_to_lerobot.py` (replay des états → rendu agentview → LeRobot, `repo_id=local/can_ph`, `root=data_cache/lerobot_can_ph`). Réutilise `STATE_KEYS` (générique) et `lift_eval.rollout_eval_chunked` (env recréé par tranche, anti-dégradation renderer — cf. phase 4).
- **Risque connu** : même mismatch **robosuite 1.4 (dataset) ↔ 1.5 (installé)** qu'à Lift. Le hack `fix_obs_sign` (flip `object[7:10]`) est **spécifique à Lift** → à **re-dériver empiriquement** pour Can (object 14D, sémantique différente) en comparant obs dataset vs obs env live.
- **Données** : `data_cache/robomimic_can_ph/` (HDF5 source), `data_cache/lerobot_can_ph/` (converti), `data_cache/can_demos/` (vidéos d'exemples).

## Suite

Si Can est résolu puis compressé : paliers plus durs envisageables — le **tri complet** (PickPlace 4 objets) ou **Square** (insertion précise, multi-étapes).
