#!/usr/bin/env bash
# Éval JOINT agentview 500 rollouts SUR GB10 (rendu EGL GPU headless). Arg = nom du run sous results/runs/can/.
# Setup requis (déjà fait) : robosuite 1.5.2 + robomimic 0.3.0 + mujoco 3.6.0 + stub egl_probe + patch mujoco_py.
set -u
RUN=$1; N=${2:-500}; STEP=${3:-005000}
cd ~/lerobot-experiments
. venv/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
NVLIBS=$(ls -d venv/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH=$HOME/ffmpeg-env/lib:${NVLIBS}${LD_LIBRARY_PATH:-}
echo "[eval-gb10 $(date '+%H:%M')] $RUN/cooldown step $STEP, n=$N (EGL GPU)"
python -u experiments/can/54_eval_joint_agent.py \
  --run-dir results/runs/can/$RUN/cooldown --steps $STEP --n $N \
  --out results/runs/can/$RUN/rollouts_${N}.csv
echo "[eval-gb10 $(date '+%H:%M')] DONE -> results/runs/can/$RUN/rollouts_${N}.csv"
