# Évaluation / sélection offline de policies d'imitation (multimodales, diffusion) — état de l'art 2023-2026

> Recherche multi-sources vérifiée (fan-out 5 angles, 19 sources fetchées, 90 claims extraits,
> 25 vérifiés de façon adversariale → 23 confirmés / 2 réfutés). Question : comment évaluer,
> sélectionner et comparer des policies SANS rollouts réels, et savoir quand l'entraînement est fini —
> cas cible : bras 5 axes réel, démos seules (vidéo + consignes), pas de simulateur.

## Synthèse

Consensus fort : **les métriques offline classiques (val-loss BC/MSE, distances distributionnelles) ne prédisent PAS le succès en boucle fermée**. Le meilleur checkpoint en validation peut être **50-100 % moins bon** que le meilleur en rollout. Raisons : (a) **mode-averaging** sur des démos multimodales, (b) **covariate shift / compounding errors** prouvés exponentiels en horizon et invisibles aux pertes par-état. Pour « quand arrêter / quel checkpoint » sans rollout, le signal dédié le plus fort est l'**incertitude par quantile de loss de diffusion** (Diff-DAgger, +39 % F1 vs ensembles) ; le **désaccord d'ensemble échoue** aux points multimodaux. L'OPE (FQE, IS, DR, DICE, model-based ; benchmark DOPE) classe les policies mais aucun n'est oracle et tous se dégradent en long-horizon. Réponse pratique = **hybride** : pré-filtre offline + **petit budget de rollouts réels** alloué intelligemment (**A-OPS**) + comparaison à arrêt séquentiel (**STEP**, −40 % d'essais).

## Constats vérifiés (avec sources)

1. **Découplage loss↔succès (high).** Robomimic « What Matters » (Mandlekar et al., CoRL 2021) : *« the best validation policy is 50 to 100% worse than the best performing policy… policies are trained with surrogate losses »*. [robomimic.github.io/study](https://robomimic.github.io/study/)

2. **Mode-averaging (high).** La BC régression/MSE donne un point estimate → moyenne les modes → biais vers les actions fréquentes ; d'où la montée spurieuse des pertes avec 1 action/état. Nuance : c'est une propriété des classes unimodales/gaussiennes/MSE, pas du max-likelihood (la diffusion est ELBO/max-likelihood et multimodale). [Deep Generative Models for Offline Policy Learning, 2402.13777] ; [Behavior Transformers, 2206.11251]

3. **Compounding prouvé exponentiel (high).** Simchowitz/Pfrommer/Jadbabaie, « The Pitfalls of Imitation Learning when Actions are Continuous » (2503.09722, 2025) : même système exponentiellement stable + expert lisse → tout imitateur lisse déterministe subit une erreur d'exécution **exponentiellement plus grande en H** que l'erreur sur les données. Échappatoires : policies improper/non-lisses/stochastiques (diffusion), données expertes étalées. (Le caractère « algorithm-agnostic » a été RÉFUTÉ 1-2 → ces échappatoires comptent.)

4. **Ensembles : échec au multimodal (high).** Diff-DAgger (2410.14868, 2024) : *« ensemble policies may disagree due to multiple viable strategies… high action variance »* → indistinguable de l'OOD. Le désaccord d'ensemble n'est PAS un signal fiable sur tâche multimodale. [arxiv 2410.14868](https://arxiv.org/html/2410.14868v4)

5. **Incertitude par loss de diffusion (medium).** Diff-DAgger : OOD si la loss de débruitage espérée (sur bruit+timestep) dépasse le quantile ~95 % des loss sur le train → **+39 % F1** vs ensembles. *Caveat : sim ManiSkill seulement, détection OOD en ligne, PAS sélection de checkpoint offline.* [diffdagger.github.io](https://diffdagger.github.io/)

6. **OPE / DOPE (high).** Benchmark DOPE (Fu et al., ICLR 2021) : FQE-L2/D, MB-FF/AR, IS, DR, DICE, VPM. Model-based & value-based > importance sampling, mais **aucun oracle**, tous se dégradent en high-dim/long-horizon (curse of horizon). [openreview kWSeGEeHvF8](https://openreview.net/pdf?id=kWSeGEeHvF8) ; [STITCH-OPE 2505.20781]

7. **A-OPS (high).** Active Offline Policy Selection (Konyushkova et al., NeurIPS 2021) : warm-start OPE + petit budget online choisi par optim bayésienne → bat OPE-seul et online-seul sous budget, validé en robotique réelle. Ne supprime pas les rollouts. [arxiv 2106.10251](https://arxiv.org/pdf/2106.10251)

8. **STEP (high).** « Policy Comparison with Near-Optimal Stopping » (2503.10966, 2025) : comparaison réelle limitée à ~10-60 essais ; STEP fait varier le nombre d'essais → **−40 %** en gardant le contrôle d'erreur de type I. Toujours des rollouts (A/B). [arxiv 2503.10966](https://arxiv.org/html/2503.10966v1)

## Non confirmés / à traiter comme incertains

- **World-models depuis les démos seules** (SIMPLER, WorldGym, Real-is-Sim, AutoEval) : aucun claim n'a survécu à la vérification → faisabilité **non prouvée** pour remplacer les rollouts.
- **Vraisemblance explicite / normalizing-flows / energy-based / coverage de modes** : aucune preuve de pouvoir prédictif réel sur le succès.

## Questions ouvertes

1. L'incertitude par loss de diffusion classe-t-elle vraiment les CHECKPOINTS par succès offline, ou seulement la détection OOD par-état en ligne ?
2. Peut-on apprendre un world-model utile uniquement depuis des logs vidéo+consignes (sans sim) et classer les policies avec une fidélité acceptable (Pearson/MMRV) ?
3. Les métriques à vraisemblance explicite / coverage de modes prédisent-elles le succès mieux que MMD/val-loss ?
4. Quel budget minimal de rollouts réels (A-OPS warm-start + STEP) sépare de façon fiable une shortlist pré-filtrée ?

## Recommandation (bras réel, démos seules)

1. **Pré-filtre sans rollout** : incertitude par quantile de loss de diffusion (PAS les ensembles) → shortlist 2-3 checkpoints + détection OOD.
2. **Décision finale** : budget minimal de rollouts réels, alloué **A-OPS** + comparaison **STEP** (−40 % d'essais).
3. **Aucun nombre offline seul** ne décide « c'est bon » — établi théoriquement (compounding exponentiel) et empiriquement (Robomimic).
