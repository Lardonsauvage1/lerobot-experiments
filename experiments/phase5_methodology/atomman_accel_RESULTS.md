# Résultats — Test d'accélération atomman (2026-06-11, autonome)

## ✅ Test de configs : RÉUSSI
Mesuré sur le continue train (resume depuis checkpoint 10500), ~240 s/config, machine chaude (régime réaliste) :

| Config | OMP_NUM_THREADS | Cœurs (taskset) | s/step |
|---|---|---|---|
| base22 (avant) | 22 | tous 0-21 | 16.9 |
| **p12 ⭐ OPTIMAL** | **12** | **P-cores 0-11** | **11.2** |
| p8 | 8 | 0-7 | 13.7 |
| p6 | 6 | 6 P-cores phys | 14.8 |

**Conclusion : p12 = 11.2 s/step, soit ~1.5× plus rapide que la baseline.**
- Les E-cores *ralentissent* (base22 = le pire) → il faut s'en tenir aux P-cores.
- Il faut bien les 12 threads P-core (p8/p6 sous-utilisent les 6 P-cores + HT).
- Gain ETA : ~45 h restantes → **~30 h** (9500 steps × 11.2 s). Bénéfice thermique probable en plus sur la durée (moins de cœurs allumés = moins de throttling).

## ✅ Script mis à jour
`07_continue_train_atomman.sh` édité (local + atomman) : `OMP_NUM_THREADS=12 MKL_NUM_THREADS=12` + `taskset -c 0-11` devant les deux invocations python. Donc **toute relance via 07 utilisera désormais la config optimale**.

## ✅ Relance RÉUSSIE en config optimale
Après le test, le `07` a été relancé une fois : **RESUME depuis checkpoint 010500**, `cfg.steps=20000`, vitesse **~9.7 s/step à froid** (se stabilise ~11). ETA : ~9500 steps → **~29 h** (au lieu de ~45 h).

## ⚠️ Fausse alerte que j'ai corrigée (note pour transparence)
J'ai d'abord cru à une « guerre d'agents » : un process `claude` (PID 8640) tournait sur atomman et je pensais qu'il relançait le train. **C'était FAUX** :
- 8640 travaille sur un **autre projet** (`/home/sam/.../brasRobot`), sans rapport avec l'entraînement.
- Les « 2 instances qui respawnent » étaient un **artefact de mes propres `pgrep -f`** : mes commandes SSH contenaient les chaînes `50_train_valloss` / `07_continue_train`, donc pgrep matchait mes propres commandes (faux positifs).
- Le `[07] ÉCHOUÉ` venait simplement de **mon kill** du train original (exit non-zéro), pas d'une erreur de resume.

→ En réalité : aucun conflit, le train était juste arrêté, et une relance propre a tout réglé. Leçon : utiliser des patterns pgrep qui ne s'auto-matchent pas (`val[l]oss`).

## État (sûr)
- `checkpoints/last -> 010500` intact, aucune donnée perdue.
- Train relancé en OMP=12 + P-cores, tourne vers 20000.
