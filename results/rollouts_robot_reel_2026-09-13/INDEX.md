# Essais du modèle sur le bras — enregistrements

Un dossier par essai (du clic DÉMARRER au STOP), enregistré par `roby_rec_rollout.sh` :
bag MCAP (2 caméras, `/joint_states`, sorties du modèle, consignes vers le garde), vidéo MP4
(caméras left | right), fiche `.fiche.md` (lisible) et `.fiche.json` (source, où saisir le résultat).

Les champs marqués [DÉDUIT] ne sont pas enregistrés dans le bag. Le résultat (réussite ou non)
est observé par l'opérateur : il n'est jamais déduit des données.

| Essai | Début | Modèle | Mode | Latence (ms) | Actions neuves | Lumière | Résultat |
|---|---|---|---|---|---|---|---|
| [rollout_20260913_000156](rollout_20260913_000156.fiche.md) | 00:01:56 | propre_long_20260908 / 036000 | async | 260.5 | 74 % | 105.0 | **à renseigner** |
| [rollout_20260913_001146](rollout_20260913_001146.fiche.md) | 00:11:46 | propre_long_20260908 / 036000 | async | 257.0 | 74 % | 106.4 | **à renseigner** |
| [rollout_20260913_001227](rollout_20260913_001227.fiche.md) | 00:12:28 | propre_long_20260908 / 036000 | async | 256 | 74 % | 105.3 | **à renseigner** |
| [rollout_20260913_003107](rollout_20260913_003107.fiche.md) | 00:31:08 | propre_long_20260908 / 036000 | RTC | 241 | 100 % | 107.4 | **à renseigner** |
| [rollout_20260913_003208](rollout_20260913_003208.fiche.md) | 00:32:09 | propre_long_20260908 / 036000 | RTC | 258.0 | 100 % | 106.3 | **à renseigner** |
| [rollout_20260913_003313](rollout_20260913_003313.fiche.md) | 00:33:13 | propre_long_20260908 / 036000 | RTC | 250.0 | 100 % | 109.1 | **à renseigner** |
| [rollout_20260913_005522](rollout_20260913_005522.fiche.md) | 00:55:22 | pince_cooldown_20260909 / 004000 | RTC | 243.5 | 100 % | 111.0 | **à renseigner** |
| [rollout_20260913_005707](rollout_20260913_005707.fiche.md) | 00:57:08 | pince_cooldown_20260909 / 004000 | RTC | 244.0 | 100 % | 106.9 | **à renseigner** |
| [rollout_20260913_005802](rollout_20260913_005802.fiche.md) | 00:58:02 | pince_cooldown_20260909 / 004000 | RTC | 243.0 | 100 % | 108.2 | **à renseigner** |
| [rollout_20260913_010850](rollout_20260913_010850.fiche.md) | 01:08:50 | pince_cooldown_20260909 / 004000 | RTC | 245 | 100 % | 111.9 | **à renseigner** |
| [rollout_20260913_011026](rollout_20260913_011026.fiche.md) | 01:10:26 | pince_cooldown_20260909 / 004000 | RTC | 252.0 | 100 % | 107.0 | **à renseigner** |
| [rollout_20260913_011151](rollout_20260913_011151.fiche.md) | 01:11:51 | pince_cooldown_20260909 / 004000 | RTC | 258 | 100 % | 107.6 | **à renseigner** |

## Essais enregistrés par le panneau (case « Enregistrer les essais »)

Fiche relevée au lancement : `.run.md` / `.run.json`.

| Essai | Modèle | Mode | Série | Résultat |
|---|---|---|---|---|
| [rollout_20260913_011951_pince_cooldown_20260909-004000_RTC](rollout_20260913_011951_pince_cooldown_20260909-004000_RTC.run.md) | pince_cooldown_20260909 / 004000 | RTC | [SERIE-20260913-0117](SERIE-20260913-0117.md) 1/3 | ECHEC (observe par Niels, 2026-09-13 01:33) |
| [rollout_20260913_012103_pince_cooldown_20260909-004000_RTC](rollout_20260913_012103_pince_cooldown_20260909-004000_RTC.run.md) | pince_cooldown_20260909 / 004000 | RTC | [SERIE-20260913-0117](SERIE-20260913-0117.md) 2/3 | ECHEC (observe par Niels, 2026-09-13 01:33) |
| [rollout_20260913_012211_pince_cooldown_20260909-004000_RTC](rollout_20260913_012211_pince_cooldown_20260909-004000_RTC.run.md) | pince_cooldown_20260909 / 004000 | RTC | — | **à renseigner** |
| [rollout_20260913_012315_pince_cooldown_20260909-004000_RTC](rollout_20260913_012315_pince_cooldown_20260909-004000_RTC.run.md) | pince_cooldown_20260909 / 004000 | RTC | [SERIE-20260913-0117](SERIE-20260913-0117.md) 3/3 | ECHEC (observe par Niels, 2026-09-13 01:33) |
| [rollout_20260913_012418_pince_cooldown_20260909-004000_RTC](rollout_20260913_012418_pince_cooldown_20260909-004000_RTC.run.md) | pince_cooldown_20260909 / 004000 | RTC | — | **à renseigner** |

## Séries

- [SERIE-20260913-0117](SERIE-20260913-0117.md) — 3 essais liés, recuit 36 000 : même position de pomme, même échec.

## Bilans par modèle

Comptages de Niels, indépendants des enregistrements. Même soirée, même inférence (RTC + iGPU).

| Modèle | Réussites | Remarque |
|---|---|---|
| [pince_cooldown_20260909 / 004000 (recuit 36 000)](BILAN-pince_cooldown_20260909-004000.md) | **9 / 14** | + 3 échecs de la SERIE-20260913-0117, hors comptage |
| [propre_long_20260908 / 036000 (premier modèle)](BILAN-propre_long_20260908-036000.md) | **5 / 10** | aucun essai enregistré ; « bien moins bien que le précédent » (Niels) |
| [pince_cooldown_030000 / 004000 (recuit 30 000)](BILAN-pince_cooldown_030000-004000.md) | **3 / 8** | aucun essai enregistré ; moins bon que le recuit 36 000 (Niels) |
| [pince_cooldown_024000 / 004000 (recuit 24 000)](BILAN-pince_cooldown_024000-004000.md) | **3 / 8** | aucun essai enregistré ; erreurs évidentes, lâche dans la ligne droite ; moins bon que le 30 000 (Niels) |
