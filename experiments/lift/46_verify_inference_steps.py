"""Test isolé : vérifier que num_inference_steps=10 produit bien 10 itérations dans le scheduler."""

import sys
sys.path.insert(0, ".")

import time
import torch

CHECKPOINT = "results/runs/lift/46_diffusion_official/checkpoints/012000/pretrained_model"

def main():
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    print("=== TEST 1 : valeur par défaut ===")
    policy = DiffusionPolicy.from_pretrained(CHECKPOINT).to("mps").eval()
    print(f"  policy.diffusion.num_inference_steps (avant): {policy.diffusion.num_inference_steps}")
    print(f"  scheduler type: {type(policy.diffusion.noise_scheduler).__name__}")
    print(f"  num_train_timesteps: {policy.diffusion.noise_scheduler.config.num_train_timesteps}")

    # Run 1 sample, check actual timesteps used
    obs = {"observation.image": torch.rand(1, 3, 96, 96, device="mps"),
           "observation.state": torch.rand(1, 19, device="mps")}
    policy.reset()
    torch.mps.synchronize()
    t0 = time.time()
    with torch.no_grad():
        _ = policy.select_action(obs)
    torch.mps.synchronize()
    dt_default = time.time() - t0
    print(f"  sample time : {dt_default*1000:.0f} ms")
    print(f"  scheduler.timesteps len: {len(policy.diffusion.noise_scheduler.timesteps)}")

    print("\n=== TEST 2 : num_inference_steps = 10 ===")
    policy2 = DiffusionPolicy.from_pretrained(CHECKPOINT).to("mps").eval()
    policy2.diffusion.num_inference_steps = 10
    print(f"  policy2.diffusion.num_inference_steps : {policy2.diffusion.num_inference_steps}")

    policy2.reset()
    torch.mps.synchronize()
    t0 = time.time()
    with torch.no_grad():
        _ = policy2.select_action(obs)
    torch.mps.synchronize()
    dt_10 = time.time() - t0
    print(f"  sample time : {dt_10*1000:.0f} ms")
    print(f"  scheduler.timesteps len: {len(policy2.diffusion.noise_scheduler.timesteps)}")

    print(f"\n  Speedup theorique attendu : 10×")
    print(f"  Speedup observé : {dt_default / dt_10:.1f}×")

if __name__ == "__main__":
    main()
