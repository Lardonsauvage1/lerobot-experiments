---
name: travail-autonome
description: Travailler SEUL sur un sujet pendant l'absence de Sam (carte blanche) — explorer un maximum de pistes, tenir un journal d'expériences, viser le meilleur taux de réussite, sans rien casser et sans poser de question. À utiliser dès que Sam dit « je te laisse la main », « carte blanche », « travaille seul sur X », « continue tant que je ne t'arrête pas ».
---

# But

Sam s'absente et confie un sujet (ex. : « trouve le RTC optimal »). À son retour, il veut **le meilleur résultat
possible, un journal clair de tout ce qui a été essayé, et rien de cassé**.

# Les consignes de Sam (non négociables)

1. **Aucune question.** En cas de doute, choisir au mieux, noter le choix et sa raison dans le journal, continuer.
   « Je ne veux surtout pas que tu m'annonces que tu en as fait la moitié car tu avais une question. »
2. **Ne rien casser, tout doit être réversible.** Toute nouveauté passe par une variable d'environnement dont la valeur
   par défaut garde le comportement actuel ; sauvegarder le fichier avant de le modifier (`results/sauvegardes/`).
   Ne jamais modifier un modèle, un résultat ou un script bash en cours d'exécution (écrire un nouveau fichier puis `mv`).
3. **Continuer tant que Sam n'arrête pas** : analyser les résultats → proposer une amélioration → la lancer → recommencer.
4. **Tenir un tableau des expériences** (`docs/<SUJET>_OPTIMISATION.md`) : une ligne par essai — réglage, échecs,
   % de réussite, p contre la référence, lecture en une phrase. Le meilleur réglage confirmé est écrit EN TÊTE.
5. **Viser le taux de réussite maximal** : chaque nouvelle série doit chercher à battre le meilleur du moment.
6. **Privilégier les expériences courtes** pour en faire plus (éval seule plutôt que réentraînement ; 200 épisodes).
7. **Tout doit être applicable au réel.** Respecter les contraintes physiques données par Sam (ex. : inférence jusqu'à
   300 ms = 3 décisions, une seule inférence à la fois, jamais « inférer à chaque mouvement »). Une piste qui ne
   respecte pas ces contraintes peut servir de diagnostic, jamais de recommandation.
8. **Réveil toutes les 30 min** (`CronCreate`, minutes décalées, ex. `7,37 * * * *`) pour ne pas rester bloqué sur une
   erreur silencieuse : script de contrôle qui teste le CONTENU (épisodes qui avancent, erreurs dans les journaux),
   pas seulement les processus (voir le skill `surveiller-les-runs`).
9. **Arrêter ses propres runs : autorisé en autonomie.** Un entraînement ou une évaluation que J'AI lancé pendant
   cette autonomie peut être arrêté sans attendre la fin si une nouvelle information le rend moins pertinent, ou
   s'il va visiblement dans le mur (perte qui diverge, échecs qui s'accumulent bien au-delà de la référence…).
   Noter l'arrêt et sa raison dans le journal. ⚠️ Les runs lancés AVANT l'autonomie (ou par Sam) restent intouchables.
10. Règles permanentes : ne jamais toucher au réseau ni au système des machines ; vérifier chaque transfert au md5 ; résultats en échecs ET en % avec la référence ;
   `TACHE=square` ; lancement détaché ; mettre à jour `RUNS_EN_COURS.md`.

# ⭐ Explorer d'abord, répliquer à la fin

Sam préfère **6 pistes différentes en une heure** à **6 fois la même piste sur des graines différentes**.

- **Cribler chaque piste d'abord une fois**, sur les MÊMES états (comparaison appariée, McNemar).
- **Un petit écart n'est PAS une preuve de nullité.** ~15 échecs sur 200 est l'ordre de grandeur du bruit d'une
  graine, rien de plus : une meilleure méthode peut faire moins bien sur une graine donnée. Une piste à petit écart
  est classée « non tranchée », pas « nulle » ; on la garde en tête si elle est plausible, on ne la jette pas.
- **Doubler les graines est permis** quand c'est ce qui permet de trancher une piste qui compte (piste candidate au
  meilleur réglage, désaccord entre graines, ou décision à fort enjeu pour le réel). Ce qu'il faut éviter, c'est
  de passer son temps à refaire des essais identiques par principe : chaque réplique doit répondre à une question.
- **Une réplique du réglage gagnant à la fin**, au minimum (2e graine, ou états neufs).
- **Ne pas dépenser de calcul à expliquer un nul.** Une piste clairement nulle = une ligne dans le journal, et on avance.

# Corrections des erreurs du 27/10 (à ne pas refaire)

1. **Ne pas annoncer comme acquis un résultat d'une seule graine.** L'écrire « piste (1 graine) » dans le journal et
   dans les messages. Deux fois, une conclusion de la graine 1 a été démentie par la graine 2. Pas besoin de
   répliquer plus pour autant : il suffit de ne pas sur-vendre.
2. **Fixer dès le départ la comparaison principale et sa référence** (ex. : « RTC réglé contre saut de 3, latence
   variable ») et la référence de chaque modèle. Le reste du journal se lit par rapport à elle.
3. **Ne jamais laisser la machine vide.** À chaque réveil : si moins de 4 évaluations sont en file, lancer la série
   suivante IMMÉDIATEMENT (préparer la série N+1 pendant que la série N tourne). Le 27/10, la machine a été vide
   jusqu'à 30 min.
4. **Si la seule réplique finale exige un modèle qui n'existe pas encore** (2e graine…), le lancer dès le DÉBUT
   (16 min sur gb10), en parallèle du criblage, pour qu'il soit prêt quand on en a besoin.
5. **Un résultat surprenant → UN diagnostic direct**, pas une série de variantes autour (le 27/10 : 4 pistes de
   « décalage » empilées au lieu de mesurer pourquoi l'indice de Roby aide).
6. **Mécanique** : lancer à distance avec `(setsid nohup bash script ... < /dev/null &)` entre parenthèses (sinon le
   ssh reste bloqué) ; jamais de `ssh` imbriqué dans un `ssh` ; écrire dans le journal par UN SEUL chemin (pas de
   commande qui ajoute des lignes en arrière-plan) ; tester le filtre du script de contrôle sur un cas réel.

# Déroulé type

1. **Cadrer (10 min)** : relire le sujet, fixer la comparaison principale, lister 6 à 10 pistes ordonnées par gain
   espéré / coût ; créer le journal et le script de contrôle ; armer le réveil 30 min ; lancer la 1re série.
2. **Boucle à chaque réveil** : contrôle → noter chaque essai fini (échecs, %, p, lecture) → mettre à jour le meilleur
   réglage → lancer la série suivante (nouvelles pistes, ou combinaison des pistes gagnantes) → `RUNS_EN_COURS.md`.
3. **Fin (ou quand les pistes s'épuisent)** : une réplique du meilleur réglage, puis un bilan pour Sam :
   meilleur réglage et son gain, ce qui ne marche pas, ce qu'il faut appliquer au réel, ce qui reste ouvert.
   Si toutes les pistes du sujet sont épuisées avant le retour de Sam, le dire dans le bilan plutôt que de lancer des
   vérifications inutiles.
