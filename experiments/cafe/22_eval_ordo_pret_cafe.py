"""22 (15/11) — ÉVALUATION café par la machine PRÊT / FINI en ORACLE (src/cafe/ordonnanceur_cafe.py) = 04_eval_chaine +
04q_eval_ordo_pret du marteau, paramétrés par la tâche café (docs/CAFE_EVAL_INFRA.md). Un modèle seul = une chaîne d'une
étape ; une chaîne = des étapes contiguës (gestes ou étapes virtuelles g<i>_<j>).

  CFG.json : {"etapes": [{"nom": "ouvrir", "membres": [{"nom": "s42", "ck": …/pretrained_model}, {"nom": "s43", "ck": …}]},
                         {"nom": "g2_4", "membres": [...]}, …]}
             membres FACTICES pour les tests sans modèle : {"nom": "expert", "type": "expert"} / {"nom": "nulle", "type": "nulle"}
  ORDO_PRET_ORACLE=1 ORDO_PRET_DEPARTS=departs.json [ORDO_PRET_DUREE_<etape>=s ORDO_PRET_MAX=6 ORDO_PRET_SUITE=4
  ORDO_PRET_K=5 ORDO_PRET_VOTE=majorite EVAL_GRAINE=0|1|2 + variables de la sim durcie (23_tableau_oracle_gb10.sh)] \
  venv/bin/python -u experiments/cafe/22_eval_ordo_pret_cafe.py --config CFG.json --etats ETATS.npz --tag TAG --n 100 \
      [--biais-joint 0.06 --paralleles 8 --chunk 13 --sortie results/runs/cafe/tab_<nom>.json]
États : experiments/cafe/20_etats_cafe.py (champ `pince` = commande en vigueur, sinon ORDO_PINCE_DEPART, déf. −1).
Garde anti-blocage par épisode (ORDO_GARDE_S, déf. 3600 s). Marqueurs : « [ORDO_CAFE] installé », « CAFE EVAL DONE », « ECHEC ».
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("SIM_NOSLIP", "5")
os.environ.setdefault("CAFE_BUTEES_SANS_REBOND", "1")
os.environ.setdefault("ROBY_PINCE_ELARGIE_MM", "0")

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RACINE)
os.chdir(RACINE)

# 17/11 — RENDU SAIN (docs/CAFE_CADRE.md 16-17/11) : sur gb10, CUDA dans le même processus que le rendu EGL fait que
# mjr_readPixels N'ÉCRIT PLUS rien (robosuite renvoie np.empty = mémoire recyclée). ORDO_DEVICE=cpu ne suffisait PAS (un autre
# module initialise CUDA : 100 % de lectures vides mesurées) -> on masque CUDA à torch AVANT tout import.
if os.environ.get("ORDO_DEVICE", "").strip() == "cpu":
    import torch as _torch
    _torch.cuda.is_available = lambda: False
    _torch.cuda.device_count = lambda: 0
# GARDE-FOU (actif par défaut, ne change rien quand le rendu est sain) : tampon prérempli d'une sentinelle ; une lecture que
# mjr_readPixels n'a pas écrite est détectée. Sous charge (≥ ~10 processus EGL), même sans CUDA, une lecture échoue parfois :
# l'image est RE-RENDUE (≤ ORDO_RENDU_ESSAIS, déf. 5) ; toujours vide -> « ECHEC RENDU » et arrêt. Bilan [RENDU] en fin de bloc.
_RENDU = {"n": 0, "vides": 0, "rerendus": 0, "vide": False}
# 17/11 20h, ORDO_RENDU_SAUTER=1 (inactif par défaut) : rendu perdu -> l'épisode en cours est noté « rendu_perdu » (À EXCLURE du
# décompte), le partiel est écrit (incomplet) et le processus sort en code 4 ; relancé, le bloc REPREND à l'épisode suivant.
_COURS = {"per": None, "ep": None, "partiel": None}


def _installer_garde_rendu():
    if os.environ.get("ORDO_RENDU_SANS_GARDE") == "1":
        return
    import time as _time
    import mujoco
    import numpy as _np
    import robosuite.utils.binding_utils as _BU
    _rp0 = _BU.MjRenderContext.read_pixels

    def _rp(self, width, height, depth=False, segmentation=False):
        if depth or segmentation:
            return _rp0(self, width, height, depth, segmentation)
        rgb = _np.full((height, width, 3), 7, _np.uint8)
        mujoco.mjr_readPixels(rgb=rgb, depth=None, viewport=mujoco.MjrRect(0, 0, width, height), con=self.con)
        _RENDU["n"] += 1
        _RENDU["vide"] = bool((rgb == 7).all())
        _RENDU["vides"] += _RENDU["vide"]
        return rgb
    _BU.MjRenderContext.read_pixels = _rp
    propre = os.environ.get("ORDO_RENDU_PROPRE") == "1"
    if propre:   # 17/11 23h — CAUSE des « rendus perdus » : quand le GC détruit un ANCIEN MjRenderContext, son __del__ fait
        # mjr_freeContext dans le contexte GL COURANT (celui en service) et efface ses framebuffers -> images vides jusqu'au
        # reset suivant (scène café : 376/1500 vides, reproduit sans modèle). Correctif : libérer dans SON propre contexte,
        # puis rendre le contexte en service courant avant chaque rendu (test : 0/6000 vides).
        def _del(self):
            try:
                self.gl_ctx.make_current(); self.con.free()
            except Exception:
                pass
            try:
                self.gl_ctx.free()
            except Exception:
                pass
        _BU.MjRenderContext.__del__ = _del
    _r0 = _BU.MjSim.render

    def _render(self, *a, **k):
        for essai in range(int(os.environ.get("ORDO_RENDU_ESSAIS", "5")) + 1):
            if propre and self._render_context_offscreen is not None:
                self._render_context_offscreen.gl_ctx.make_current()
            out = _r0(self, *a, **k)
            if not _RENDU["vide"]:
                return out
            _RENDU["rerendus"] += 1
            _time.sleep(0.002 * (essai + 1))
            if essai >= 1 and os.environ.get("ORDO_RENDU_RECREER", "0") == "1":   # 17/11 : RECRÉER fausse les couleurs du robot
                # (bras vert, pince bleue mesurés) -> désactivé par défaut : l'image vide persistante ARRÊTE le bloc
                old = self._render_context_offscreen
                dev = getattr(old, "device_id", -1)
                self.add_render_context(_BU.MjRenderContextOffscreen(self, device_id=dev))
                _RENDU["recrees"] = _RENDU.get("recrees", 0) + 1
        try:   # 17/11 20h : diagnostic (scène explosée ou rendu perdu ?)
            _q = _np.asarray(self.data.qpos); _v = _np.asarray(self.data.qvel)
            print(f"  [DIAG RENDU] t={float(self.data.time):.2f} s ; qpos NaN {int(_np.isnan(_q).sum())}/{_q.size} ; "
                  f"max|qpos| {float(_np.nanmax(_np.abs(_q))):.3g} ; max|qvel| {float(_np.nanmax(_np.abs(_v))):.3g} ; "
                  f"ncon {int(self.data.ncon)}", flush=True)
        except Exception as _e:
            print(f"  [DIAG RENDU] indisponible : {_e!r}", flush=True)
        print(f"ECHEC RENDU : image toujours vide après {essai} re-rendus ({_RENDU['vides']} vides / {_RENDU['n']}) -> arrêt",
              flush=True)
        if os.environ.get("ORDO_RENDU_SAUTER") == "1" and _COURS["partiel"] and _COURS["ep"] is not None:
            per = _COURS["per"] + [{"ep": _COURS["ep"], "etapes": [], "ok": False, "arret": "rendu_perdu", "verite_fin": {}}]
            Path(_COURS["partiel"]).write_text(json.dumps({"per": per, "incomplet": True}))
            print(f"  [SAUT] épisode {_COURS['ep']} noté rendu_perdu (exclu), reprise au suivant", flush=True)
            os._exit(4)
        os._exit(3)
    _BU.MjSim.render = _render


class _Bloque(Exception):
    pass


def _alarme(*_):
    raise _Bloque()


def gardes():
    assert os.environ.get("ROBOT") == "Roby", f"ROBOT={os.environ.get('ROBOT')} (attendu Roby)"
    assert os.environ.get("ORDO_PRET_ORACLE") == "1", "café : ORDO_PRET_ORACLE=1 obligatoire (pas d'observateurs)"
    for k in ("ROBY_RG", "ROBY_BM", "ROBY_RETRY0", "ROBY_RETRY1", "ROBY_RETRY5", "RELANCE", "ENSEMBLE_CKPTS", "HD_MODE",
              "ORDO_PRET", "ORDO_OBS", "TIROIR_SANS_REBOND"):
        assert not os.environ.get(k, "").strip(), f"{k} défini : variable du marteau / de 10_vision, pas de la machine café"


def jouer(a, cfg, etats, debut, n):
    from src.quiet_robosuite import silence_robosuite
    silence_robosuite()
    _installer_garde_rendu()
    from src.cafe import ordonnanceur_cafe as OC
    V = OC.initialiser(a.biais_joint)
    ordo = OC.OrdonnanceurCafe(cfg, pas_debruitage=a.pas, taille=V.RENDER_SIZE)
    env = OC.fabriquer_env_eval(V.RENDER_SIZE)
    garde = int(os.environ.get("ORDO_GARDE_S", "3600"))
    per, t0 = [], time.time()
    for ep in range(debut, debut + n):
        etat = {"xml": etats["xml"][ep], "etat": etats["etat"][ep]}
        if "pince" in etats and not os.environ.get("ORDO_PINCE_DEPART"):
            etat["pince"] = float(etats["pince"][ep])
        _COURS["ep"] = ep; _COURS["per"] = list(_COURS.get("avant") or []) + per
        signal.signal(signal.SIGALRM, _alarme); signal.alarm(garde)
        try:
            r = ordo.episode(env, etat, ep)
        except _Bloque:
            print(f"  [GARDE] épisode {ep} BLOQUÉ > {garde} s -> échec", flush=True)
            r = {"ep": ep, "etapes": [], "ok": False, "arret": "bloque", "verite_fin": {}}
        finally:
            signal.alarm(0)
        per.append(r)
        print(f"  ép {ep:4d} {'OK     ' if r['ok'] else 'échec  '} arrêt {r['arret']}  [{time.time() - t0:.0f} s]", flush=True)
    print(f"  [RENDU] {_RENDU['n']} lectures d'image, {_RENDU['vides']} vide(s), {_RENDU['rerendus']} re-rendu(s), {_RENDU.get('recrees', 0)} contexte(s) recréé(s)", flush=True)
    try:
        env.env.close()
    except Exception:
        pass
    return per, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--etats", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--debut", type=int, default=0)
    ap.add_argument("--biais-joint", type=float, default=0.06)
    ap.add_argument("--pas", type=int, default=5, help="pas de débruitage (5, comme la base)")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--paralleles", type=int, default=1)
    ap.add_argument("--partiel", default=None)
    ap.add_argument("--sortie", default="results/runs/cafe/eval_ordo_pret.json")
    a = ap.parse_args()
    gardes()
    cfg = json.load(open(a.config))
    from src.cafe import tache as T
    L = [e["nom"] for e in cfg["etapes"]]
    T.verifier_chaine(L)
    from src.marteau import etats as ET
    etats = ET.charger(a.etats)
    assert len(etats["etat"]) >= a.debut + a.n, f"{a.etats} : {len(etats['etat'])} états, {a.debut + a.n} demandés"
    if a.paralleles > 1:
        argv, saute = [], False
        for x in sys.argv[1:]:
            if saute:
                saute = False; continue
            if x in ("--paralleles", "--n", "--debut"):
                saute = True; continue
            argv.append(x)
        d = Path(tempfile.mkdtemp(prefix=f"cafe_{a.tag}_"))
        blocs = [(b, min(a.chunk, a.n - b)) for b in range(0, a.n, a.chunk)]
        print(f"  EN PARALLÈLE : {len(blocs)} blocs de {a.chunk}, {a.paralleles} processus ({d})", flush=True)
        en_cours, tous, t0 = [], [], time.time()
        for b, m in blocs:
            while len(en_cours) >= a.paralleles:
                time.sleep(5); en_cours = [p for p in en_cours if p.poll() is None]
            p = subprocess.Popen([sys.executable, "-u", os.path.abspath(sys.argv[0]), *argv, "--debut", str(a.debut + b),
                                  "--n", str(m), "--partiel", str(d / f"{b:05d}.json")],
                                 stdout=open(d / f"{b:05d}.log", "w"), stderr=subprocess.STDOUT)
            en_cours.append(p); tous.append((b, p))
        per = []
        reessais = int(os.environ.get("ORDO_REESSAIS_BLOCS", "0"))   # 17/11, inactif par défaut : bloc échoué (rendu EGL
        for b, p in tous:                                          # perdu…) RELANCÉ en entier dans un processus neuf
            k = 0
            while (rc := p.wait()) != 0 and (k < reessais or rc == 4):
                k += rc != 4                                       # code 4 = saut d'un épisode au rendu perdu : progrès, non compté
                os.replace(d / f"{b:05d}.log", d / f"{b:05d}.echec{k}{'s' if rc == 4 else ''}{int(time.time()) % 100000}.log")
                m = min(a.chunk, a.n - b)
                print(f"  bloc {b} échoué -> relance {k}/{reessais}", flush=True)
                p = subprocess.Popen([sys.executable, "-u", os.path.abspath(sys.argv[0]), *argv, "--debut", str(a.debut + b),
                                      "--n", str(m), "--partiel", str(d / f"{b:05d}.json")],
                                     stdout=open(d / f"{b:05d}.log", "w"), stderr=subprocess.STDOUT)
            if p.wait() != 0:
                raise SystemExit(f"ECHEC bloc {b} : voir {d / f'{b:05d}.log'}")
            per += json.loads((d / f"{b:05d}.json").read_text())["per"]
        el = time.time() - t0
    else:
        avant = []
        if a.partiel and os.environ.get("ORDO_RENDU_SAUTER") == "1":
            _COURS["partiel"] = a.partiel
            if Path(a.partiel).exists():
                prec = json.loads(Path(a.partiel).read_text())
                if prec.get("incomplet"):
                    avant = prec["per"]
        _COURS["avant"] = avant
        reprise = (avant[-1]["ep"] + 1) if avant else a.debut
        if avant:
            print(f"  [REPRISE] {len(avant)} épisode(s) déjà faits, reprise à l'épisode {reprise}", flush=True)
        per, el = jouer(a, cfg, etats, reprise, a.debut + a.n - reprise) if reprise < a.debut + a.n else ([], 0.0)
        per = avant + per
        if a.partiel:
            Path(a.partiel).write_text(json.dumps({"per": per}))
            return
    from src.cafe.ordonnanceur_cafe import bilan
    B, txt = bilan(per, L)
    print(f"\n  === {a.tag} : {len(per)} épisodes, {el / 60:.1f} min ===\n{txt}", flush=True)
    out = Path(a.sortie)
    res = json.loads(out.read_text()) if out.exists() else {}
    res[a.tag] = {"config": cfg, "etats": a.etats, "debut": a.debut, "n": len(per), "biais_joint": a.biais_joint,
                  "pas_debruitage": a.pas, "bilan": B, "per": per,
                  "env": {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith((
                      "SIMR_", "ROBY_", "JEU_", "EXEC_", "MAX_STEPS", "ORDO_", "ACT_", "SIM_", "CAFE_", "EVAL_"))}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"  -> {out}\nCAFE EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
