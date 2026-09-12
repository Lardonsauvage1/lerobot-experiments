#!/usr/bin/env bash
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le
CFG=experiments/hilserl_pusht/pusht_hilserl.json
pkill -f run_learner_pusht 2>/dev/null; pkill -f run_actor_pusht 2>/dev/null; sleep 3
rm -rf results/runs/hilserl_pusht/run1 results/runs/hilserl_pusht/run1_actor
rm -f results/logs/hilserl_pusht/learner_live.log results/logs/hilserl_pusht/actor_live.log
PYTHONUNBUFFERED=1 venv312/bin/python experiments/hilserl_pusht/run_learner_pusht.py --config_path=$CFG > results/logs/hilserl_pusht/learner_live.log 2>&1 &
LP=$!
until grep -q "Starting learner thread" results/logs/hilserl_pusht/learner_live.log 2>/dev/null; do sleep 3; kill -0 $LP 2>/dev/null || { echo "LEARNER CRASH"; tail -5 results/logs/hilserl_pusht/learner_live.log; exit 1; }; done
echo "[fresh] learner prêt (fresh)"
PUSHT_BROWSER=1 PUSHT_PORT=8000 PUSHT_STEP=45 PYTHONUNBUFFERED=1 venv312/bin/python experiments/hilserl_pusht/run_actor_pusht.py --config_path=$CFG --output_dir=results/runs/hilserl_pusht/run1_actor > results/logs/hilserl_pusht/actor_live.log 2>&1 &
until grep -qE "OUVRE http|Connection with Learner established" results/logs/hilserl_pusht/actor_live.log 2>/dev/null; do sleep 2; pgrep -f run_actor_pusht >/dev/null || { echo "ACTOR CRASH"; tail -5 results/logs/hilserl_pusht/actor_live.log; exit 1; }; done
sleep 2
curl -s -o /dev/null -w "[fresh] page HTTP %{http_code}\n" http://127.0.0.1:8000/ 2>/dev/null
echo "[fresh] PRÊT — learner PID $(pgrep -f run_learner_pusht), actor PID $(pgrep -f run_actor_pusht)"
