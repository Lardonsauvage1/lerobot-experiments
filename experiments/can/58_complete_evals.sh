#!/bin/bash
# Complète les évals : (1) joint 42-48k (combler 40->50k) + régénérer le 4-panneaux joint propre
# (avec le segment loss +10k) ; (2) convergence du look-at SCRATCH (n=50, tous les 4k — modèle raté).
# Attend la fin de la convergence fine-tune (Principal libre). En FICHIER.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le || exit 1
PY=venv312/bin/python
TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale; IP=100.110.237.53

echo "[58] $(date '+%H:%M') attente fin convergence fine-tune (Principal libre)..."
while pgrep -f "32_eval_dense|56_eval_finetune" >/dev/null; do sleep 120; done
echo "[58] Principal libre."

xfer() { # $1 = run-dir relative, $2 = step pad
  local ck="$1/checkpoints/$2/pretrained_model"
  [ -s "$ck/model.safetensors" ] && return 0
  "$TS" ping -c 2 "$IP" >/dev/null 2>&1
  ssh -o BatchMode=yes mac2 "cd ~/lerobot-experiments && tar czf - $1/checkpoints/$2/pretrained_model" | tar xzf - -C . 2>/dev/null
}

# --- 1. JOINT 42-48k (combler 40->50k) ---
RDJ=results/runs/can/joint_r34_bigunet
for s in 42000 44000 46000 48000; do
  xfer "$RDJ" "$(printf %06d $s)"
  [ -s "$RDJ/checkpoints/$(printf %06d $s)/pretrained_model/model.safetensors" ] && \
    $PY -u experiments/can/37_eval_joint_birdview.py --run-dir "$RDJ" --steps "$s" --n 50 --kp 50 --out "$RDJ/rollouts_50.csv"
done
echo "[58] joint 42-48k fait. Régénération 4-panneaux joint (log complet 0-50k)..."
scp -o BatchMode=yes mac2:/tmp/run_joint.log results/logs/can/run_joint_full.log 2>/dev/null
mv "$RDJ/rollouts_500.csv" /tmp/j500.bak 2>/dev/null
$PY experiments/can/40_plot_model.py "$RDJ" "results/logs/can/run_joint_full.log" \
  --title "JOINT 61M (moteur, 0->50k LR constant) — 4 panneaux" --out "$RDJ/full_curves_complet.png"
mv /tmp/j500.bak "$RDJ/rollouts_500.csv" 2>/dev/null

# --- 2. SCRATCH look-at convergence (n=50, tous les 4k) ---
RDS=results/runs/phase6_camera/lookat_r34_bigunet
for s in 4000 8000 12000 16000 20000 24000 28000 32000 36000 40000; do
  xfer "$RDS" "$(printf %06d $s)"
  [ -s "$RDS/checkpoints/$(printf %06d $s)/pretrained_model/model.safetensors" ] && \
    $PY -u experiments/can/32_eval_dense_birdview.py --run-dir "$RDS" --steps "$s" --n 50 --out "$RDS/rollouts_50.csv"
done
echo "[58] scratch convergence faite. 4-panneaux scratch..."
$PY experiments/can/40_plot_model.py "$RDS" "results/logs/phase6_camera/run_lookat_mac2.log" \
  --title "look-at SCRATCH 61M — 4 panneaux (echec ~2%)" --out "$RDS/full_curves.png"
echo "[58] DONE"
