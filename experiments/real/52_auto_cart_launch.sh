#!/usr/bin/env bash
# AUTONOME (Mac) : attend fin conversion -> split train/val -> transfert gb10 -> lance train+cooldown en tmux -> nettoie bags.
set -u
cd /Users/nielsmurawka/Documents/VScodeProject/experience_Le
CONVLOG=results/logs/real/convert_cart_combined.log
echo "===== [auto-cart] début $(date '+%H:%M:%S') ====="

# 1) attendre la fin de la conversion
while pgrep -f "47_convert_cart_combined" >/dev/null 2>&1; do sleep 20; done
echo "[auto-cart] conversion terminée $(date '+%H:%M:%S')"
grep -E "SOURCE|TERMINE|ECHEC" "$CONVLOG" | tail -6

# 2) parse plages (classique 0..C-1 ; total N) et calcule EPS (train = classique 0..C-7 + TOUTES corrections ; val = 6 derniers classiques)
CLASSIC_END=$(grep "SOURCE 'classique'" "$CONVLOG" | grep -oE "[0-9]+\.\.[0-9]+" | tail -1 | sed 's/.*\.\.//')
TOTAL=$(grep "TERMINE" "$CONVLOG" | grep -oE ": [0-9]+ ép" | grep -oE "[0-9]+")
if [ -z "$CLASSIC_END" ] || [ -z "$TOTAL" ]; then echo "[auto-cart] ECHEC parse (CLASSIC_END=$CLASSIC_END TOTAL=$TOTAL)"; exit 1; fi
C=$((CLASSIC_END+1))
/Users/nielsmurawka/Documents/VScodeProject/experience_Le/venv312/bin/python - "$C" "$TOTAL" > /tmp/eps.txt <<'PY'
import sys
C=int(sys.argv[1]); TOTAL=int(sys.argv[2])
train=list(range(0,C-6))+list(range(C,TOTAL))   # tient 6 classiques en val
print(str(train).replace(' ',''))
PY
EPS=$(cat /tmp/eps.txt)
echo "[auto-cart] classic=$C total=$TOTAL -> val=classique[$((C-6))..$((C-1))], train=$((C-6+TOTAL-C)) ép"

# 3) transfert dataset + scripts + eps vers gb10 (paths absolus, pas de --info=progress2)
rsync -a -e "ssh -o ConnectTimeout=20" data_cache/lerobot_apple_cart_combined_128 gb10:/home/guest1/lerobot-experiments/data_cache/ || { echo "[auto-cart] ECHEC transfert dataset"; exit 1; }
scp -o ConnectTimeout=20 experiments/real/48_train_cart_combined.sh experiments/real/49_cooldown_cart_combined.sh experiments/real/50_cart_combined_run_all.sh experiments/real/51_cart_launch_gb10.sh gb10:/home/guest1/lerobot-experiments/experiments/real/
scp -o ConnectTimeout=20 /tmp/eps.txt gb10:/tmp/eps.txt
echo "[auto-cart] dataset+scripts+eps transférés gb10 $(date '+%H:%M:%S')"

# 4) lancer train+cooldown en tmux 'cart' sur gb10
ssh -o ConnectTimeout=20 gb10 'cd ~/lerobot-experiments && chmod +x experiments/real/4[89]_*.sh experiments/real/5[01]_*.sh && tmux kill-session -t cart 2>/dev/null; tmux new-session -d -s cart "bash experiments/real/51_cart_launch_gb10.sh 2>&1 | tee /tmp/cart_combined_run_all.log"'
sleep 5
ssh -o ConnectTimeout=20 gb10 'tmux has-session -t cart 2>/dev/null && echo "[auto-cart] tmux cart VIVANT" || echo "[auto-cart] ECHEC lancement tmux"'
echo "[auto-cart] entraînement lancé sur gb10 $(date '+%H:%M:%S')"

# 5) nettoyage bags (libère disque Mac)
rm -rf data_cache/_cart_raw
echo "===== [auto-cart] TOUT LANCÉ, fini $(date '+%H:%M:%S') ====="
