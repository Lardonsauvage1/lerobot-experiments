#!/usr/bin/env bash
# Arret de l'entrainement long a 18h00 puis RECUIT, demande Sam (2026-09-08).
#
# L'entrainement tourne a LR CONSTANT : on peut donc l'arreter n'importe quand sans
# obtenir un modele pris au milieu d'une descente de LR. C'est tout l'interet du choix
# de Sam. Le recuit est ensuite une phase SEPAREE, en cosinus vers zero, repartant du
# dernier checkpoint -- meme structure que cart_combined_128 (40000 pas + 5000 de recuit).
set -u
LOG="$HOME/plan_cooldown.log"
OUT="$HOME/lerobot-experiments/outputs/propre_long_20260908"
CD="$HOME/lerobot-experiments/outputs/propre_long_cooldown"
dit() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

dit "=== plan arme ; arret prevu a 18h00 (PID entrainement $1) ==="
FIN=$(date -d "today 17:57" +%s)
while [ "$(date +%s)" -lt "$FIN" ]; do
  kill -0 "$1" 2>/dev/null || { dit "!! l'entrainement s'est arrete tout seul avant l'heure"; break; }
  sleep 60
done

if kill -0 "$1" 2>/dev/null; then
  dit "17h57 : arret de l entrainement"
  kill -INT "$1" 2>/dev/null; sleep 20
  kill -9 "$1" 2>/dev/null
fi
sleep 10

CK=$(ls -d "$OUT"/checkpoints/[0-9]*/pretrained_model 2>/dev/null | sort | tail -1)
if [ -z "$CK" ]; then dit "!! AUCUN checkpoint trouve -> recuit impossible"; exit 1; fi
dit "dernier checkpoint : $CK"

# Config de recuit : cosinus vers zero, 5000 pas (meme proportion que la reference).
"$HOME/ipex_test_venv/bin/python" - "$CK" "$CD" <<'PY' >> "$LOG" 2>&1
import json, sys
ck, out = sys.argv[1], sys.argv[2]
c = json.load(open("/home/sam/train_config_long.json"))
c["scheduler"] = {"type": "diffuser", "name": "cosine", "num_warmup_steps": 0}
c["steps"] = 1200
c["save_freq"] = 600
c["output_dir"] = out
c["job_name"] = "propre_long_cooldown"
c["policy"]["pretrained_path"] = ck
json.dump(c, open("/home/sam/train_config_cooldown.json", "w"), indent=2)
print("config de recuit ecrite (5000 pas, cosinus, depuis %s)" % ck)
PY

dit "--- recuit : 1200 pas en cosinus (~32 min) ---"
rm -rf "$CD"
cd "$HOME/lerobot-experiments" && bash -c 'source ~/roby_xpu_env.sh; source /opt/ros/jazzy/setup.bash 2>/dev/null; exec ~/ipex_test_venv/bin/python -m lerobot.scripts.lerobot_train --config_path=/home/sam/train_config_cooldown.json' >> "$LOG" 2>&1
dit "=== RECUIT TERMINE -> $CD ==="
