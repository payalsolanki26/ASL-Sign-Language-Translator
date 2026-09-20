"""
Model explainability  (Data Visualization + Deep Learning)  -  Grad-CAM

Shows *where* the network looked. In this project it is also a data-quality tool: if the
heat sits on the magenta box, the white canvas or the background instead of the hand /
skeleton, the model is using a shortcut and will fail on new users.

Requires PyTorch (not executed in the sandbox this project was authored in).
"""
from __future__ import annotations
import numpy as np


def grad_cam(model, x, target_layer, class_idx=None):
    """x: (1,3,H,W) tensor. Returns (heatmap[H,W] in 0..1, predicted_class)."""
    import torch
    import torch.nn.functional as F

    acts, grads = {}, {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: acts.setdefault("a", o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: grads.setdefault("g", go[0]))
    model.eval()
    logits = model(x)
    cls = int(logits.argmax(1)) if class_idx is None else class_idx
    model.zero_grad()
    logits[0, cls].backward()
    h1.remove(); h2.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy(), cls


def overlay(rgb_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib.cm as cm
    heat = (cm.jet(cam)[..., :3] * 255).astype(np.uint8)
    return (rgb_uint8 * (1 - alpha) + heat * alpha).astype(np.uint8)
