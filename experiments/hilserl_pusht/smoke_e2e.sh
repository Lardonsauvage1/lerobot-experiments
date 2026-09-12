#!/usr/bin/env bash
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le
CFG=experiments/hilserl_pusht/pusht_hilserl.json
L=results/logs/hilserl_pusht/learner.log; A=results/logs/hilserl_pusht/actor.log
rm -f "$L" "$A"
rm -rf results/runs/hilserl_pusht/run1 results/runs/hilserl_pusht/run1_actor   # sinon FileExistsError (resume=False)
PYTHONUNBUFFERED=1 venv312/bin/python experiments/hilserl_pusht/run_learner_pusht.py --config_path=$CFG > "$L" 2>&1 &
LPID=$!
until grep -q "Starting learner thread" "$L" 2>/dev/null; do sleep 3; kill -0 $LPID 2>/dev/null || { echo "LEARNER CRASH avant prêt:"; tail -5 "$L"; exit 1; }; done
echo "[smoke] learner prêt, lance actor headless $(date '+%H:%M:%S')"
PYTHONUNBUFFERED=1 PUSHT_HEADLESS=1 venv312/bin/python experiments/hilserl_pusht/run_actor_pusht.py --config_path=$CFG --output_dir=results/runs/hilserl_pusht/run1_actor > "$A" 2>&1 &
APID=$!
sleep 100
echo "===== ACTOR ====="; grep -ivE "WARNING|warn|objc|pygame|Hello|pkg_resources|dylib|duplicate|resource_stream|deprecat|Fetching|it/s|Map:" "$A" | grep -iE "connect|rollout|episode|transition|interaction|Traceback|Error|reward|intervention|sent" | tail -10
echo "===== LEARNER ====="; grep -ivE "WARNING|warn|objc|pygame|Hello|pkg|dylib|duplicate|resource_stream|deprecat" "$L" | grep -iE "optimization|update|step:|loss|Traceback|Error|received|online" | tail -8
kill -0 $LPID 2>/dev/null && echo "[smoke] learner VIVANT" || echo "[smoke] learner MORT"
kill -0 $APID 2>/dev/null && echo "[smoke] actor VIVANT" || echo "[smoke] actor MORT"
kill $APID $LPID 2>/dev/null
echo "[smoke] FINI"
