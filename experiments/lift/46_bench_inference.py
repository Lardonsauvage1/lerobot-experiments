"""Bench inference speed de la Diffusion Policy sur MPS vs CPU.

Mesure :
  1. Temps d'une diffusion sample complète (DDIM N steps → 16 actions)
  2. Temps d'une 'queue pop' (action déjà dans le buffer)

À partir de ça on peut extrapoler le temps total d'un episode (200 env steps,
n_action_steps=8 → 25 samples de diffusion + 175 pops).
"""

import sys
sys.path.insert(0, ".")

import time
import torch
import numpy as np

CHECKPOINT = "results/runs/lift/46_diffusion_official/checkpoints/012000/pretrained_model"
N_WARMUP = 2
N_TRIALS = 5
N_ENV_STEPS_PER_EP = 200


def make_fake_obs(device):
    img = torch.rand(1, 3, 96, 96, device=device)
    state = torch.rand(1, 19, device=device)
    return {"observation.image": img, "observation.state": state}


def bench_device(device_str: str, num_inference_steps: int = None):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    device = torch.device(device_str)

    print(f"\n{'='*60}")
    print(f"DEVICE = {device_str}    DDIM steps = {num_inference_steps or 'default(10)'}")
    print(f"{'='*60}")
    print(f"Loading policy...", flush=True)
    t0 = time.time()
    policy = DiffusionPolicy.from_pretrained(CHECKPOINT).to(device).eval()
    if num_inference_steps is not None:
        policy.config.num_inference_steps = num_inference_steps
    print(f"  Loaded in {time.time()-t0:.1f}s | n_action_steps={policy.config.n_action_steps} | "
          f"horizon={policy.config.horizon}", flush=True)

    # Warmup
    print(f"Warmup ({N_WARMUP} samples)...", flush=True)
    for _ in range(N_WARMUP):
        policy.reset()
        obs = make_fake_obs(device)
        with torch.no_grad():
            _ = policy.select_action(obs)
        if device_str == "mps":
            torch.mps.synchronize()

    # Bench full diffusion sample (queue empty → triggers full denoising)
    print(f"Bench diffusion sample (queue empty)...", flush=True)
    samples_times = []
    for _ in range(N_TRIALS):
        policy.reset()
        obs = make_fake_obs(device)
        if device_str == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = policy.select_action(obs)
        if device_str == "mps":
            torch.mps.synchronize()
        samples_times.append(time.time() - t0)

    # Bench queue pop (queue full → just dequeue, no diffusion)
    print(f"Bench queue pop (queue full)...", flush=True)
    policy.reset()
    obs = make_fake_obs(device)
    with torch.no_grad():
        _ = policy.select_action(obs)  # primes the queue
    pop_times = []
    for _ in range(N_TRIALS):
        if device_str == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = policy.select_action(obs)
        if device_str == "mps":
            torch.mps.synchronize()
        pop_times.append(time.time() - t0)

    t_sample = np.median(samples_times)
    t_pop = np.median(pop_times)
    n_action = policy.config.n_action_steps
    # Per episode: 200 env steps = (200 / n_action) full samples + (200 - 200/n_action) pops
    n_samples_per_ep = N_ENV_STEPS_PER_EP // n_action
    n_pops_per_ep = N_ENV_STEPS_PER_EP - n_samples_per_ep
    t_per_ep = n_samples_per_ep * t_sample + n_pops_per_ep * t_pop
    t_for_200_eps = t_per_ep * 200

    print(f"\nRésultats sur {N_TRIALS} essais :")
    print(f"  Full diffusion sample (median) : {t_sample*1000:.0f} ms  "
          f"(min={min(samples_times)*1000:.0f}, max={max(samples_times)*1000:.0f})")
    print(f"  Queue pop (median)             : {t_pop*1000:.1f} ms")
    print(f"  Per episode (200 steps)        : {t_per_ep:.1f} s "
          f"= {n_samples_per_ep}×sample + {n_pops_per_ep}×pop")
    print(f"  Pour 200 episodes              : {t_for_200_eps/60:.1f} min")
    return {"sample_ms": t_sample*1000, "pop_ms": t_pop*1000, "ep_s": t_per_ep, "total_min": t_for_200_eps/60}


if __name__ == "__main__":
    results = {}
    # MPS d'abord (DDIM 10 = config par défaut)
    results["mps_ddim10"] = bench_device("mps")
    # CPU même config
    results["cpu_ddim10"] = bench_device("cpu")
    # MPS DDIM 4 (réduit)
    results["mps_ddim4"] = bench_device("mps", num_inference_steps=4)

    print(f"\n{'='*60}")
    print("RÉCAP (temps total pour 200 episodes) :")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f"  {name:20s} : sample={r['sample_ms']:.0f}ms  pop={r['pop_ms']:.1f}ms  "
              f"→ ep={r['ep_s']:.1f}s  → 200ep={r['total_min']:.1f}min")
