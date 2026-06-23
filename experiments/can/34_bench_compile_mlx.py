"""Benchmark overhead d'entraînement sur M1 — 4 conditions, MÊME architecture des deux côtés.

Architecture représentative de notre modèle (conv-lourd) : 2× backbone ResNet18 séparés
(96² RGB, une par caméra) -> features concat avec état 9D -> petite tête 1D (≈ U-Net).
forward + backward + Adam. On mesure la médiane s/step (temps mur).

SWEEP DE BATCH [8,16,32] par condition -> ajuste t_step = O + c·B :
  O = coût fixe par step (overhead, indépendant du batch)
  c = coût variable par exemple
=> prédiction d'un entraînement : t_total(N,B) ≈ N × (O + c·B)   (+ ~15% val/checkpoints).
Alimente et affine docs/PERF_TEMPS_ENTRAINEMENT.md (par backend).

Conditions :
  1. pt_eager     : PyTorch MPS, eager, avec .item() par step (= la vraie boucle actuelle)
  2. pt_nosync    : PyTorch MPS, eager, SANS synchro par step (synchro unique en fin)
  3. pt_compile   : PyTorch MPS, torch.compile(model)
  4. mlx_compile  : MLX, mx.compile (graphe fusionné natif Apple)

Usage :
  venv312/bin/python -u experiments/can/34_bench_compile_mlx.py            # vrai bench (MPS/GPU)
  venv312/bin/python -u experiments/can/34_bench_compile_mlx.py --smoke    # validation CPU
"""
import argparse, time
from functools import partial

import numpy as np

IMG, STATE_DIM, HORIZON, ACTION_DIM = 96, 9, 16, 7
BATCHES = [8, 16, 32]
WARMUP, MEASURE = 20, 100

# ----------------------------------------------------------------------------- PyTorch
def build_pt():
    import torch, torch.nn as nn

    class Block(nn.Module):
        def __init__(self, cin, cout, stride=1):
            super().__init__()
            self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
            self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
            self.down = None
            if stride != 1 or cin != cout:
                self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

        def forward(self, x):
            idt = x if self.down is None else self.down(x)
            y = torch.relu(self.b1(self.c1(x)))
            y = self.b2(self.c2(y))
            return torch.relu(y + idt)

    class ResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64),
                                      nn.ReLU(), nn.MaxPool2d(3, 2, 1))
            def layer(cin, cout, n, stride):
                return nn.Sequential(Block(cin, cout, stride), *[Block(cout, cout) for _ in range(n - 1)])
            self.l1 = layer(64, 64, 2, 1); self.l2 = layer(64, 128, 2, 2)
            self.l3 = layer(128, 256, 2, 2); self.l4 = layer(256, 512, 2, 2)

        def forward(self, x):
            x = self.l4(self.l3(self.l2(self.l1(self.stem(x)))))
            return x.mean(dim=(2, 3))  # [B,512]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1 = ResNet(); self.e2 = ResNet()
            self.proj = nn.Linear(512 + 512 + STATE_DIM, 256)
            self.head = nn.Sequential(nn.Conv1d(256, 256, 3, 1, 1), nn.ReLU(),
                                      nn.Conv1d(256, 256, 3, 1, 1), nn.ReLU())
            self.out = nn.Conv1d(256, ACTION_DIM, 1)

        def forward(self, img1, img2, state):
            f = torch.cat([self.e1(img1), self.e2(img2), state], dim=1)
            c = self.proj(f).unsqueeze(-1).repeat(1, 1, HORIZON)  # [B,256,H]
            return self.out(self.head(c)).transpose(1, 2)  # [B,H,A]

    return torch, Model()


def bench_pt(device, compile_model, per_step_sync, n_measure, n_warmup, batch_size):
    import torch
    torch_mod, model = build_pt()
    model = model.to(device).train()
    if compile_model:
        model = torch.compile(model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    img1 = torch.randn(batch_size, 3, IMG, IMG, device=device)
    img2 = torch.randn(batch_size, 3, IMG, IMG, device=device)
    state = torch.randn(batch_size, STATE_DIM, device=device)
    target = torch.randn(batch_size, HORIZON, ACTION_DIM, device=device)

    def one_step():
        opt.zero_grad()
        pred = model(img1, img2, state)
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        opt.step()
        return loss

    sync = (lambda: torch.mps.synchronize()) if device == "mps" else (lambda: None)
    for _ in range(n_warmup):
        l = one_step()
        if per_step_sync:
            float(l.item())
    sync()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        l = one_step()
        if per_step_sync:
            float(l.item())  # synchro par step (comme la vraie boucle)
    sync()
    return (time.perf_counter() - t0) / n_measure


# ----------------------------------------------------------------------------- MLX
def bench_mlx(n_measure, n_warmup, batch_size, cpu=False):
    import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
    if cpu:
        mx.set_default_device(mx.cpu)

    class Block(nn.Module):
        def __init__(self, cin, cout, stride=1):
            super().__init__()
            self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm(cout)
            self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm(cout)
            self.down = stride != 1 or cin != cout
            if self.down:
                self.dc = nn.Conv2d(cin, cout, 1, stride, bias=False); self.db = nn.BatchNorm(cout)

        def __call__(self, x):
            idt = self.db(self.dc(x)) if self.down else x
            y = mx.maximum(self.b1(self.c1(x)), 0)
            y = self.b2(self.c2(y))
            return mx.maximum(y + idt, 0)

    def layer(cin, cout, n, stride):
        return [Block(cin, cout, stride)] + [Block(cout, cout) for _ in range(n - 1)]

    class ResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem_c = nn.Conv2d(3, 64, 7, 2, 3, bias=False); self.stem_b = nn.BatchNorm(64)
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.blocks = layer(64, 64, 2, 1) + layer(64, 128, 2, 2) + layer(128, 256, 2, 2) + layer(256, 512, 2, 2)

        def __call__(self, x):  # x: NHWC
            x = self.pool(mx.maximum(self.stem_b(self.stem_c(x)), 0))
            for b in self.blocks:
                x = b(x)
            return mx.mean(x, axis=(1, 2))  # [B,512]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1 = ResNet(); self.e2 = ResNet()
            self.proj = nn.Linear(512 + 512 + STATE_DIM, 256)
            self.h1 = nn.Conv1d(256, 256, 3, 1, 1); self.h2 = nn.Conv1d(256, 256, 3, 1, 1)
            self.out = nn.Conv1d(256, ACTION_DIM, 1)

        def __call__(self, img1, img2, state):
            f = mx.concatenate([self.e1(img1), self.e2(img2), state], axis=1)
            c = self.proj(f)[:, None, :]  # [B,1,256]
            c = mx.broadcast_to(c, (c.shape[0], HORIZON, 256))  # [B,H,256]
            c = mx.maximum(self.h1(c), 0); c = mx.maximum(self.h2(c), 0)
            return self.out(c)  # [B,H,A]

    model = Model(); mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=1e-4)
    img1 = mx.random.normal((batch_size, IMG, IMG, 3))  # NHWC
    img2 = mx.random.normal((batch_size, IMG, IMG, 3))
    state = mx.random.normal((batch_size, STATE_DIM))
    target = mx.random.normal((batch_size, HORIZON, ACTION_DIM))

    def loss_fn(model, i1, i2, s, t):
        return mx.mean((model(i1, i2, s) - t) ** 2)

    lag = nn.value_and_grad(model, loss_fn)
    st = [model.state, opt.state]

    @partial(mx.compile, inputs=st, outputs=st)
    def step(i1, i2, s, t):
        loss, grads = lag(model, i1, i2, s, t)
        opt.update(model, grads)
        return loss

    for _ in range(n_warmup):
        l = step(img1, img2, state, target); mx.eval(l, st)
    t0 = time.perf_counter()
    for _ in range(n_measure):
        l = step(img1, img2, state, target); mx.eval(l, st)
    return (time.perf_counter() - t0) / n_measure


# ----------------------------------------------------------------------------- main
CONDITIONS = {
    "pt_eager (item/step)": lambda b: bench_pt("mps", False, True, MEASURE, WARMUP, b),
    "pt_nosync":            lambda b: bench_pt("mps", False, False, MEASURE, WARMUP, b),
    "pt_compile":           lambda b: bench_pt("mps", True, False, MEASURE, WARMUP, b),
    "mlx_compile":          lambda b: bench_mlx(MEASURE, WARMUP, b),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="validation CPU")
    a = ap.parse_args()

    if a.smoke:
        print("=== SMOKE CPU (validation correctness) ===", flush=True)
        t = bench_pt("cpu", False, True, 2, 1, 8)
        print(f"  PyTorch CPU OK : {t*1000:.0f} ms/step", flush=True)
        t = bench_mlx(2, 1, 8, cpu=True)
        print(f"  MLX CPU OK     : {t*1000:.0f} ms/step", flush=True)
        print("smoke OK", flush=True); return

    print(f"=== BENCH M1 — sweep batch {BATCHES}, image {IMG}², ResNet18 ×2 + tête 1D ===", flush=True)
    print(f"(warmup {WARMUP}, mesure {MEASURE} steps/point ; t_step = O + c·B)\n", flush=True)

    fits = {}  # condition -> (O, c, {B: t})
    for name, fn in CONDITIONS.items():
        times = {}
        for b in BATCHES:
            try:
                t = fn(b)
                times[b] = t
                print(f"  {name:22s} B={b:>3d} : {t:.3f} s/step", flush=True)
            except Exception as e:
                print(f"  {name:22s} B={b:>3d} : ÉCHEC ({str(e)[:60]})", flush=True)
        if len(times) >= 2:
            bs = np.array(sorted(times)); ts = np.array([times[b] for b in bs])
            c, O = np.polyfit(bs, ts, 1)  # pente=c (s/ex), ordonnée=O (s/step fixe)
            fits[name] = (float(O), float(c), times)
        else:
            fits[name] = (None, None, times)

    # ---- décomposition fixe/variable ----
    print(f"\n{'condition':22s} {'O fixe/step':>12s} {'c /exemple':>12s} {'part fixe @B16':>14s}")
    for name, (O, c, _) in fits.items():
        if O is None:
            print(f"{name:22s} {'(échec)':>12s}"); continue
        frac = O / (O + c * 16) * 100
        print(f"{name:22s} {O:>10.3f}s {c*1000:>9.2f}ms {frac:>12.0f}%")

    # ---- prédiction d'un entraînement ----
    print("\n=== Prédiction t_total = N × (O + c·B) (+~15% val/ckpt) ===")
    for N in (20000, 40000):
        for B in (16, 32):
            print(f"  N={N}, B={B} :", end="")
            for name, (O, c, _) in fits.items():
                if O is None:
                    print(f"  {name.split()[0]}=n/a", end=""); continue
                h = N * (O + c * B) * 1.15 / 3600
                print(f"  {name.split()[0]}={h:.1f}h", end="")
            print(flush=True)


if __name__ == "__main__":
    main()
