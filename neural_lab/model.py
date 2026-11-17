"""The team's ConvAE, copied unchanged from 01_autoencoder/notebook/autoencoder.ipynb (SECTION 7),
plus loading of the published weights and the exact training/server preprocessing."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

HF_REPO = "iaac-maai/hackathon-team1-weld-anomaly-ae"
REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "02_model_server"


# ---- Architecture: verbatim copy of the notebook class (do not edit, or the weights will not fit) ----
class ConvAE(nn.Module):
    def __init__(self, image_size=128, latent_dim=32, channels=(32, 64, 128, 128)):
        super().__init__()
        n_down = int(np.log2(image_size // 8))           # halve until 8 x 8: 4 layers at 128 px, 3 at 64 px
        chans = [1] + list(channels[:n_down])
        encoder = []
        for c_in, c_out in zip(chans[:-1], chans[1:]):
            encoder += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1), nn.BatchNorm2d(c_out), nn.ReLU()]
        self.encoder = nn.Sequential(*encoder, nn.Flatten(), nn.Linear(chans[-1] * 8 * 8, latent_dim))

        decoder = [nn.Linear(latent_dim, chans[-1] * 8 * 8), nn.ReLU(), nn.Unflatten(1, (chans[-1], 8, 8))]
        rev = chans[::-1]
        for i, (c_in, c_out) in enumerate(zip(rev[:-1], rev[1:])):
            decoder.append(nn.ConvTranspose2d(c_in, c_out, 3, stride=2, padding=1, output_padding=1))
            is_last = i == len(rev) - 2
            decoder += [nn.Sigmoid()] if is_last else [nn.BatchNorm2d(c_out), nn.ReLU()]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ---- Published files ----------------------------------------------------------------------------------
def hf_file(filename: str) -> Path:
    """Download (once) a file of the published model; it lands in the Hugging Face cache, outside the repo."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(HF_REPO, filename))


def load_config() -> dict:
    return json.loads(hf_file("anomaly_ae_config.json").read_text())


def load_pretrained(device: str = "cpu"):
    """Build ConvAE(image_size, latent_dim) from the published config and load the published weights.
    strict=True: every weight must match the architecture exactly. Eval mode (BatchNorm uses its stored
    statistics, as at inference time) and gradients off: the weights are never changed here."""
    config = load_config()
    model = ConvAE(config["image_size"], config["latent_dim"])
    state = torch.load(hf_file("anomaly_ae.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device), config


# ---- Preprocessing: identical to load_images() in the notebook and preprocess() in 02_model_server/app.py ----
def preprocess(image, size: int = 64) -> torch.Tensor:
    """greyscale ('L') -> resize size x size, bilinear -> pixels / 255 -> tensor (1, 1, size, size)."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    grey = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    x = np.asarray(grey, dtype=np.float32) / 255.0
    return torch.from_numpy(x[None, None].copy())


def risk_level(score: float, config: dict) -> str:
    """Same rule as the model server."""
    return "high" if score > config["t_high"] else "check" if score > config["t_low"] else "low"
