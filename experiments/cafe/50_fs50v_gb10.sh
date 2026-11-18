#!/bin/bash
# 50 (18/11, Sam) : FS50V = les 40 démos S1/S2/S4 du jeu FP50V SEULES (sans les 40 S3) -> contrôle qualité -> 2 graines -> éval 200.
# Contrôle : seul KO admis = « depart » (règle de 20 : pose LUE sous biais), vraie pose vérifiée par 12b sur le jeu SOURCE (mêmes démos).
cd ~/lerobot-experiments || exit 1; source env_sim.sh
J=data_cache/cafeFS50V80_fermer_tiroir_10hz
C=results/logs/cafe_demos/controle_cafeFS50V80_fermer_tiroir_10hz.log
echo "[$(date "+%d/%m %H:%M")] contrôle"
$PY experiments/cafe/12_controle_jeu.py --jeu $J --n 40 > $C 2>&1
tail -3 $C
autres=$(grep -a "\[KO\]" $C | grep -vc "\[KO\] depart")
if grep -aq "CONTROLE OK" $C; then
  echo "contrôle OK"
elif [ "$autres" = 0 ] && grep -aq "DEPART VRAI OK" results/logs/cafe_demos/depart_vrai_cafeFP50V80_fermer_tiroir.log; then
  echo "contrôle OK (KO « depart » expliqué par le biais, vraie pose OK sur la source)"
else
  echo "ECHEC_SCRIPT CONTROLE KO"; exit 1
fi
echo "[$(date "+%d/%m %H:%M")] entraînement"
LISTE="FS50V:fermer_tiroir" bash experiments/cafe/30_entrainement_cafe_gb10.sh > results/logs/cafe_entrainement_FS50V.log 2>&1
grep -a "s4[23] OK\|ECHEC" results/logs/cafe_entrainement_FS50V.log
echo "[$(date "+%d/%m %H:%M")] éval"
ORDO_DEVICE=cpu ORDO_RENDU_PROPRE=1 ORDO_RENDU_SAUTER=1 ORDO_REESSAIS_BLOCS=6 ORDO_GARDE=600 MODE=eval \
  LISTE="FS50V:fermer_tiroir:0,1:6:etats_v2" bash experiments/cafe/47_rattrapage_gb10.sh
echo "FS50V DONE $(date "+%d/%m %H:%M")"
