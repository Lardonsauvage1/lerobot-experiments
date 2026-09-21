"""Estimer le temps d'entraînement SANS le mesurer — modèle de coût calibré.

POURQUOI. Les durées au chronomètre dépendent de ce que la machine faisait en même
temps : les bras F/G/H ont tourné pendant que le Mac évaluait, A/B/C/E non. Comparer
leurs horloges revient à comparer des charges, pas des configurations.

L'UNITÉ. On compte les FLOPs de la passe AVANT par échantillon, module par module, puis :
    module entraîné  -> 3 x  (1 avant + 2 arrière)
    module GELÉ      -> 1 x  (avant seul, aucun gradient de poids)
    taux réduit      -> 3 x  (le gradient est calculé quand même : AUCUNE économie)
Coût C = somme des contributions, en GFLOP-équivalents par échantillon.

LE MODÈLE.  updt_s = a + b x C      (a = frais fixes par pas, b = s/GFLOP-eq)
Calibré sur `updt_s` — le temps de calcul PUR par pas, que LeRobot journalise et qui
est insensible à la charge (on prend son p10). Jamais sur l'horloge murale.

VÉRIFICATION (atomman, Arc + IPEX, batch 32, ajusté sur A et E seulement) :
    bras            C       prédit   mesuré
    A sans gel     8.118    0.488    0.488   (ajustement)
    E deux gelés   2.786    0.250    0.250   (ajustement)
    B gel poignet  5.452    0.369    0.369   <- 0,1 % d'écart, NON ajusté
    C LR/10        8.118    0.488    0.494   <- 1,3 %, NON ajusté
Deux prédictions à l'aveugle à ~1 %. Le modèle tient.

⚠️ a et b sont PROPRES À LA MACHINE (et au batch). Les recalibrer ailleurs avec
--calibrer updt_A,C_A,updt_E,C_E.

Usage : venv312/bin/python experiments/can/169_estimer_temps.py
"""
import argparse

# GFLOPs de la passe AVANT, mesurés avec torch.utils.flop_counter (cf. en-tête du dépôt)
ENCODEUR = {("resnet18", 96): 0.667, ("resnet18", 224): 3.629,
            ("resnet34", 96): 1.346, ("resnet34", 224): 7.328}
DECODEUR = {(64, 128, 256): 0.040, (128, 256, 512): 0.151, (256, 512, 1024): 0.593}

# Calibrage atomman (Arc A770 + IPEX bf16, batch 32)
A_FIXE, B_PENTE = 0.1256, 0.04464


def cout(backbone="resnet18", px=96, cameras=2, n_obs=2, down_dims=(64, 128, 256),
         geles=(), part_gelee=0.0):
    """Coût C en GFLOP-équivalents par échantillon.

    geles : indices des caméras dont l'encodeur est GELÉ, p.ex. (0,) ou (0,1)
    part_gelee : pour un gel PARTIEL, fraction des FLOPs de l'encodeur qui est gelée
                 (⚠️ en FLOPs, pas en paramètres — les premiers étages pèsent peu en
                 poids et beaucoup en calcul, cf. F/G)
    """
    f_enc = ENCODEUR[(backbone, px)] * n_obs        # une caméra, tous les pas d'obs
    c = 0.0
    for i in range(cameras):
        if i in geles:
            c += f_enc * (1 if part_gelee == 0.0 else (1 * part_gelee + 3 * (1 - part_gelee)))
        else:
            c += f_enc * 3
    return c + DECODEUR[tuple(down_dims)] * 3


def heures(C, steps, a=A_FIXE, b=B_PENTE):
    return (a + b * C) * steps / 3600


def fmt(h):
    m = int(round(h * 60))
    return f"{m // 60}h{m % 60:02d}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000, help="10000 + 5000 de recuit")
    ap.add_argument("--mono-steps", type=int, default=15000, help="le mono préalable")
    a = ap.parse_args()

    print(f"Modèle : updt_s = {A_FIXE} + {B_PENTE} x C   (atomman, batch 32)")
    print(f"Chaîne = mono ({a.mono_steps} pas) + bi-caméra ({a.steps} pas)\n")
    mono = cout(cameras=1)
    print(f"{'configuration':34s} {'C':>7s} {'bi-cam':>8s} {'+ mono':>8s}   échecs")
    print(f"{'mono 1 caméra (prérequis)':34s} {mono:7.2f} {fmt(heures(mono, a.mono_steps)):>8s} {'':>8s}   —")
    print()
    ECHECS = {"A sans gel": 62, "B gel du poignet": 68, "C LR/10 sur l'extérieur": 71,
              "F gel partiel (65 % des FLOPs)": 70, "G gel partiel (49 % des FLOPs)": 77,
              "E les deux gelés": 126}
    for lab, kw in [("A sans gel", {}),
                    ("B gel du poignet", {"geles": (1,)}),
                    ("C LR/10 sur l'extérieur", {}),
                    ("F gel partiel (65 % des FLOPs)", {"geles": (0,), "part_gelee": 0.655}),
                    ("G gel partiel (49 % des FLOPs)", {"geles": (0,), "part_gelee": 0.487}),
                    ("E les deux gelés", {"geles": (0, 1)})]:
        C = cout(**kw)
        t = heures(C, a.steps)
        print(f"{lab:34s} {C:7.2f} {fmt(t):>8s} {fmt(t + heures(mono, a.mono_steps)):>8s}"
              f"   {ECHECS[lab]:4d}")

    print("\n--- et si on change d'architecture (sans gel, chaîne complète) ---")
    for lab, kw in [("R18 96px, décodeur [64,128,256]", {}),
                    ("R18 96px, décodeur [128,256,512]", {"down_dims": (128, 256, 512)}),
                    ("R18 96px, décodeur [256,512,1024]", {"down_dims": (256, 512, 1024)}),
                    ("R34 96px, décodeur [64,128,256]", {"backbone": "resnet34"}),
                    ("R18 224px, décodeur [64,128,256]", {"px": 224}),
                    ("R34 224px, décodeur [128,256,512]", {"backbone": "resnet34", "px": 224,
                                                           "down_dims": (128, 256, 512)})]:
        C = cout(**kw)
        m = cout(cameras=1, **kw)
        print(f"{lab:36s} C={C:7.2f}  bi-cam {fmt(heures(C, a.steps)):>6s}"
              f"  + mono {fmt(heures(m, a.mono_steps)):>6s} = {fmt(heures(C, a.steps) + heures(m, a.mono_steps)):>6s}")
