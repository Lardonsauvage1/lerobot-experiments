#!/usr/bin/env bash
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le
CFG=experiments/hilserl_pusht/pusht_hilserl.json
BC=results/runs/hilserl_pusht/bc_actor
pkill -9 -f run_learner_pusht 2>/dev/null; pkill -9 -f run_actor_pusht 2>/dev/null; sleep 3
for i in $(seq 1 20); do lsof -nP -iTCP:8000 2>/dev/null | grep -q LISTEN || break; sleep 0.5; done
rm -rf results/runs/hilserl_pusht/run1 results/runs/hilserl_pusht/run1_actor
rm -f results/logs/hilserl_pusht/learner_live.log results/logs/hilserl_pusht/actor_live.log
# LEARNER warm-starté depuis le BC
PYTHONUNBUFFERED=1 venv312/bin/python experiments/hilserl_pusht/run_learner_pusht.py --config_path=$CFG --policy.pretrained_path=$BC > results/logs/hilserl_pusht/learner_live.log 2>&1 &
LP=$!
until grep -q "Starting learner thread" results/logs/hilserl_pusht/learner_live.log 2>/dev/null; do sleep 3; kill -0 $LP 2>/dev/null || { echo "LEARNER CRASH"; tail -6 results/logs/hilserl_pusht/learner_live.log; exit 1; }; done
echo "[warm] learner prêt (warm-starté BC)"
# ACTOR warm-starté aussi (policy locale = BC en attendant les params gRPC)
REWARD_CLF=results/runs/hilserl_pusht/reward_clf.pt PUSHT_BROWSER=1 PUSHT_PORT=8000 PUSHT_STEP=45 PYTHONUNBUFFERED=1 venv312/bin/python experiments/hilserl_pusht/run_actor_pusht.py --config_path=$CFG --policy.pretrained_path=$BC --output_dir=results/runs/hilserl_pusht/run1_actor > results/logs/hilserl_pusht/actor_live.log 2>&1 &
until grep -qE "OUVRE http|Connection with Learner established" results/logs/hilserl_pusht/actor_live.log 2>/dev/null; do sleep 2; pgrep -f run_actor_pusht >/dev/null || { echo "ACTOR CRASH"; tail -6 results/logs/hilserl_pusht/actor_live.log; exit 1; }; done
sleep 3
curl -s -m3 -o /dev/null -w "[warm] page HTTP %{http_code}\n" http://127.0.0.1:8000/ 2>/dev/null
echo "[warm] PRÊT"
