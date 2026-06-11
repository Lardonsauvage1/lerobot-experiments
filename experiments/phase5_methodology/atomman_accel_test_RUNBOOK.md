# Runbook — Test d'accélération atomman (AUTONOME, ne rien demander à l'user)

**Autorisé par l'user (2026-06-11) :** dès que le continue train atomman sauvegarde le **step 10500**, stopper → tester des configs → relancer dans la config optimale. L'user est absent : décider seul, ne rien demander.
**Limite :** process-level uniquement (OMP, affinité taskset, duty-cycle SIGSTOP/CONT). NE PAS toucher gouverneur/BIOS/réseau/système.

## Contexte machine
- atomman = Intel Ultra 9 185H (laptop, throttle thermique). Hôte SSH `atomman`, projet `/home/sam/Documents/projet VScode/experience_Le`.
- Topologie : **P-cores = CPU logiques 0-11** (6 cœurs phys, HT) ; E-cores 12-19 ; LP-E 20-21.
- Continue train = `26_resnet34_dense_continue`, save_freq=500, resume via `07_continue_train_atomman.sh` (auto-resume depuis `checkpoints/last`).
- Baseline observée : **OMP=22, ~16.9 s/step** (n'utilise que ~12 cœurs, throttle).

## Étapes

### 1. Confirmer la sauvegarde 10500 puis STOPPER
```
ssh atomman 'ls .../26_resnet34_dense_continue/checkpoints/ | grep 010500'   # doit exister
ssh atomman 'pkill -f 50_train_valloss; sleep 3; pgrep -f 50_train_valloss || echo STOPPED'
# vérifier last -> 010500
```

### 2. Commande de test (resume, courte, tuée avant 500 steps -> aucune sauvegarde, état reste à 10500)
Pour chaque config, lancer ~4 min, lire les s/step du log, puis tuer.
CKPT cfg = `.../26_resnet34_dense_continue/checkpoints/last/pretrained_model/train_config.json`
```
ssh atomman 'cd PROJET; OMP_NUM_THREADS=<N> MKL_NUM_THREADS=<N> nohup taskset -c <CPUS> \
  venv312/bin/python -u experiments/lift/50_train_valloss.py \
  --config_path="<CKPT cfg>" --resume=true > /tmp/accel_<tag>.log 2>&1 & echo $!'
sleep 240
ssh atomman 'tr "\r" "\n" < /tmp/accel_<tag>.log | grep -aoE "[0-9.]+s/step" | tail -8'
ssh atomman 'pkill -f 50_train_valloss; sleep 2'   # tuer avant prochain test
```

### Configs à tester (machine déjà chaude -> régime throttlé réaliste)
| tag | OMP | taskset | but |
|---|---|---|---|
| base22 | 22 | 0-21 | référence |
| p12 | 12 | 0-11 | P-cores only (HT) |
| p6 | 6 | 0,2,4,6,8,10 | 1 thread/P-core phys (pas de HT) |
| p8 | 8 | 0-7 | intermédiaire |

Mesurer le **s/step médian des 4-5 dernières lignes** de chaque (ignorer la 1ère, cold). Garder le plus bas.

### 3. (option) duty-cycling
Si throttling fort et qu'une config tourne nettement plus vite à froid puis ralentit : envisager wrapper run 8min / pause 2min (SIGSTOP/CONT). Sinon ignorer.

### 4. RELANCER en config optimale
Éditer `07_continue_train_atomman.sh` : remplacer `export OMP_NUM_THREADS=22 MKL_NUM_THREADS=22` par la valeur gagnante, et préfixer la commande python par `taskset -c <CPUS gagnants>`. Puis :
```
ssh atomman 'cd PROJET; nohup bash experiments/phase5_methodology/07_continue_train_atomman.sh > /tmp/run_07_continue.log 2>&1 & echo $!'
sleep 15 ; vérifier ALIVE + resume depuis 10500 + nouveau s/step
```
Ré-armer un Monitor "continue train DONE/offline" + un Monitor de retour si offline.

### 5. RAPPORT
Écrire `experiments/phase5_methodology/atomman_accel_RESULTS.md` : tableau config x s/step, config retenue, gain, nouvelle ETA. Laisser un message récap pour l'user (il lira à son retour). NE PAS attendre de réponse.
