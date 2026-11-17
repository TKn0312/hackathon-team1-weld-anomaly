"""Record what happens inside the autoencoder during one real forward pass, using PyTorch forward hooks.

A forward hook is a small function PyTorch calls right after a layer has computed its output. We use it
to copy that output (the 'activations' or 'feature maps') without changing anything in the model."""
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

# The three encoder blocks are Conv 3x3 stride 2 -> BatchNorm -> ReLU. We record each block's ReLU output,
# i.e. the feature maps that the next block actually receives.
BLOCKS = ["block1", "block2", "block3"]


@dataclass
class Inference:
    x: np.ndarray            # input tile, (64, 64), values 0-1
    recon: np.ndarray        # reconstruction, (64, 64)
    error_map: np.ndarray    # (x - recon)^2 per pixel, (64, 64)
    score: float             # mean of error_map = the anomaly score used by the team
    latent: np.ndarray       # the 32-number code, (32,)
    blocks: dict = field(default_factory=dict)   # name -> (channels, H, W) feature maps after ReLU


def encoder_relus(model: nn.Module):
    return [m for m in model.encoder if isinstance(m, nn.ReLU)]


@torch.no_grad()
def run(model: nn.Module, x: torch.Tensor) -> Inference:
    """One forward pass of ONE tile (shape (1, 1, H, W)) with temporary hooks; hooks are always removed."""
    assert x.ndim == 4 and x.shape[0] == 1, "run() expects a single tile of shape (1, 1, H, W)"
    model.eval()
    captured = {}

    def keep(name):
        def hook(module, inputs, output):
            captured[name] = output.detach().cpu().clone()
        return hook

    relus = encoder_relus(model)
    assert len(relus) == len(BLOCKS), f"expected {len(BLOCKS)} encoder blocks, found {len(relus)}"
    handles = [m.register_forward_hook(keep(name)) for name, m in zip(BLOCKS, relus)]
    handles.append(model.encoder.register_forward_hook(keep("latent")))  # encoder output = the code
    try:
        device = next(model.parameters()).device
        recon = model(x.to(device)).cpu()
    finally:
        for h in handles:
            h.remove()

    err = (recon - x) ** 2
    return Inference(
        x=x[0, 0].numpy(), recon=recon[0, 0].numpy(), error_map=err[0, 0].numpy(), score=float(err.mean()),
        latent=captured["latent"][0].numpy(), blocks={b: captured[b][0].numpy() for b in BLOCKS},
    )


def strongest_channels(maps: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k channels with the highest mean activation for this tile (a ranking, not a meaning)."""
    return np.argsort(-maps.mean(axis=(1, 2)), kind="stable")[:k]
