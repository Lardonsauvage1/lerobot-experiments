"""Variance FINE : succes 500-rollouts tous les 10 pas sur [57000, 57200] (constant 1e-4).
Zone jamais entrainee avant (premier passage, densement sauvee) -> pas d'ambiguite de trajectoire.
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path("results/runs/phase5_methodology/mini_constant_finevar/rollouts_500.csv")
st, sr, lo, hi = [], [], [], []
for r in csv.DictReader(open(P)):
    st.append(int(r["step"])); sr.append(float(r["success_rate"]) * 100)
    lo.append(float(r["ci95_low"]) * 100); hi.append(float(r["ci95_high"]) * 100)

mean = sum(sr) / len(sr)
fig, ax = plt.subplots(figsize=(11, 6))
ax.fill_between(st, lo, hi, color="tab:red", alpha=0.15, label="IC95 Wilson (n=500)")
ax.plot(st, sr, "-o", color="tab:red", ms=5)
ax.axhline(mean, ls="--", c="gray", lw=1, label=f"moyenne {mean:.1f}%")
ax.set_title(f"Variance FINE — constant 1e-4, 1 ckpt / 10 pas sur [57000, 57200]\n"
             f"succès 500-rollouts : {min(sr):.1f}%→{max(sr):.1f}% (amplitude {max(sr)-min(sr):.1f} pts) en 200 pas",
             fontsize=12, weight="bold")
ax.set_xlabel("step"); ax.set_ylabel("succès (%)"); ax.set_ylim(0, 85)
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
out = Path("results/runs/phase5_methodology/courbe_finevar_57k.png")
fig.savefig(out, dpi=150)
print(f"écrit : {out}")

# combien de sauts >2x IC95 entre voisins (10 pas) ?
big = 0
for i in range(1, len(st)):
    if hi[i] < lo[i-1] or hi[i-1] < lo[i]:   # IC95 disjoints
        big += 1
print(f"sauts a IC95 disjoints entre voisins (10 pas): {big}/{len(st)-1}")
