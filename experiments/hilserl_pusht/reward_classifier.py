#!/usr/bin/env python
"""Classifieur de récompense binaire HIL-SERL (phase 1) pour Push-T image.

Petit CNN : image 96x96 -> P(succès). Sert de signal de récompense EN LIGNE
(remplace le coverage triché de l'env — répétition exacte de ce qu'il faudra sur le vrai robot).
Entraîné par train_reward_classifier.py sur les démos (positifs = coverage>0.8).
"""
import numpy as np
import torch
import torch.nn as nn


class RewardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),   # 96->48
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 48->24
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True), # 24->12
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),# 12->6
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)   # logit


def _to_chw01(img):
    """Accepte uint8/float, HWC ou CHW -> tensor float (3,96,96) dans [0,1]."""
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] != 3 and a.shape[2] == 3:   # HWC -> CHW
        a = a.transpose(2, 0, 1)
    t = torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)
    if t.max() > 1.5:
        t = t / 255.0
    return t


class RewardClassifier:
    """Charge les poids et donne success(image)->bool (prob>seuil)."""
    def __init__(self, path, device="cpu", thresh=0.5):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.model = RewardCNN().to(device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.device = device
        self.thresh = float(ckpt.get("thresh", thresh))

    @torch.no_grad()
    def prob(self, img):
        x = _to_chw01(img).unsqueeze(0).to(self.device)
        return float(torch.sigmoid(self.model(x)).item())

    def success(self, img):
        return self.prob(img) >= self.thresh
