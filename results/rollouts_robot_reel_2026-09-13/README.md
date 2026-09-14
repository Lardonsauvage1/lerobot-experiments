# Essais des modèles sur le vrai bras Roby — nuit du 12 au 13 septembre 2026

Copie de `~/roby_datasets/rollouts/` (PC sam-AtomMan), envoyée le 2026-09-14 à la demande de Niels.

- **Vidéos MP4** (1280×480, caméras gauche = extérieure | droite = poignet, 15 i/s) : ajoutées en
  **exception** à la règle `*.mp4` du `.gitignore` de ce dépôt (les vidéos restent locales en général).
- **Fiches** `.fiche.md` / `.fiche.json`, `.run.md` / `.run.json`, `.info`, `INDEX.md` (tableau
  récapitulatif), `BILAN-*.md` (comptages de Niels par modèle), `SERIE-*.md`.
- **Non inclus** : les bags MCAP (~800 Mo, dont un de 197 Mo > limite GitHub) et leurs journaux.
  Ils restent sur le PC dans `~/roby_datasets/rollouts/`. Les liens de l'INDEX vers les dossiers de
  bags ne fonctionnent donc pas ici.
- 2 essais sans vidéo (`012211`, `012418`) : leurs bags contiennent les images, vidéo non générée.
- La vidéo de `000156` ne couvre que 28 s sur les 201 s du bag.
