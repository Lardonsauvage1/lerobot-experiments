"""Calibrage AUTO-EXTENSIBLE du temps d'entraînement (M1) — se précise à chaque run.

Scanne tous les logs d'entraînement, extrait par run : batch + p10 du `updt_s`
(= s/step calcul pur, immunisé contre PC-non-dédié) + archi (backbone, séparé, down_dims, params).
**Regroupe par signature d'archi** : plus il y a de runs d'une même archi, plus le p10 médian
se resserre (la précision du modèle de temps augmente avec les runs).

Prédiction : t_total ≈ N_steps × (s/step) × 1.15 (val-loss + checkpoints).

Usage : venv312/bin/python experiments/can/41_calibrate_train_time.py
Sortie : results/runs/phase5_methodology/train_time_calib.md
"""
import glob, re
import numpy as np

LOG_GLOBS = ["results/logs/*/run_*.log"]
OUT = "results/runs/phase5_methodology/train_time_calib.md"


def parse(path):
    txt = open(path, errors="ignore").read()
    u = [float(x) for x in re.findall(r"updt_s:([0-9.]+)", txt)]
    if len(u) < 3:
        return None
    b = re.search(r"Effective batch size:\s*(\d+)", txt)
    bk = re.search(r"vision_backbone'?\s*[:=]\s*'?(\w+)", txt)
    sep = re.search(r"use_separate_rgb_encoder_per_camera'?\s*[:=]\s*(True|true|False|false)", txt)
    dd = re.search(r"down_dims'?\s*[:=]\s*(\[[0-9,\s]+\])", txt)
    pm = re.search(r"num_learnable_params=(\d+)", txt)
    p10 = float(np.percentile(sorted(u), 10))
    batch = int(b.group(1)) if b else None
    return {
        "p10": p10, "batch": batch,
        "backbone": bk.group(1) if bk else "?",
        "sep": (sep.group(1).lower() == "true") if sep else None,
        "down_dims": re.sub(r"\s", "", dd.group(1)) if dd else "?",
        "params_M": round(int(pm.group(1)) / 1e6, 1) if pm else None,
        "s_per_sample": (p10 / batch) if batch else None,
    }


rows = []
for g in LOG_GLOBS:
    for f in sorted(glob.glob(g)):
        r = parse(f)
        if r:
            r["log"] = f.split("/")[-1]
            rows.append(r)

# --- regroupement par signature d'archi (backbone + sep + down_dims + batch) ---
def sig(r):
    return f"{r['backbone']} {'sep' if r['sep'] else 'shared' if r['sep'] is not None else '?'} U{r['down_dims']} B{r['batch']}"

groups = {}
for r in rows:
    groups.setdefault(sig(r), []).append(r)

lines = ["# Calibrage du temps d'entraînement (M1) — auto-extensible\n",
         "*Régénéré par `experiments/can/41_calibrate_train_time.py`. Se précise à chaque nouveau run "
         "(plus de runs d'une même archi → p10 médian plus resserré). `updt_s` p10 = s/step calcul pur.*\n",
         "## Par signature d'architecture",
         "| archi (backbone · enc · U-Net · batch) | params | n runs | s/step (p10 médian) | écart | s/exemple |",
         "|---|---|---|---|---|---|"]
for s, rs in sorted(groups.items(), key=lambda kv: np.median([r["p10"] for r in kv[1]])):
    p10s = [r["p10"] for r in rs]
    pm = next((r["params_M"] for r in rs if r["params_M"]), "?")
    med = np.median(p10s)
    spread = f"±{(max(p10s)-min(p10s))/2:.2f}" if len(p10s) > 1 else "—"
    sps = med / rs[0]["batch"] if rs[0]["batch"] else float("nan")
    lines.append(f"| {s} | {pm} M | {len(rs)} | {med:.2f} s | {spread} | {sps*1000:.1f} ms |")

lines += ["\n## Prédiction",
          "`t_total ≈ N_steps × (s/step p10) × 1,15` (overhead val-loss + checkpoints). "
          "Exemple : 40k steps à 1,1 s/step ≈ 12,7 h.",
          "\n## Détail par run", "| log | archi | batch | params | s/step p10 |", "|---|---|---|---|---|"]
for r in sorted(rows, key=lambda r: r["p10"]):
    lines.append(f"| {r['log']} | {r['backbone']} {'sep' if r['sep'] else 'shared' if r['sep'] is not None else '?'} "
                 f"U{r['down_dims']} | {r['batch']} | {r['params_M']} M | {r['p10']:.2f} s |")

open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines[:6 + len(groups)]))
print(f"\n→ {len(rows)} runs analysés, {len(groups)} signatures d'archi. Sauvé : {OUT}")
