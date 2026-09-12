#!/usr/bin/env python
"""Learner HIL-SERL. Patch torch.load(weights_only=False) : la déserialisation gRPC des transitions
contient des scalaires numpy que le défaut weights_only=True (torch récent) refuse. Sûr = boucle locale."""
import torch
_orig = torch.load
def _load(*a, **k):
    k["weights_only"]=False
    return _orig(*a, **k)
torch.load = _load
import lerobot.policies.sac.configuration_sac  # noqa: F401 (enregistre sac)
import lerobot.rl.learner as L
L.train_cli()
