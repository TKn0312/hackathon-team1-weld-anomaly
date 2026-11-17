"""Prepare NEURAL / LAB: export the published ConvAE as two ONNX halves for the browser and precompute
the measured facts the page shows. Everything comes from inference with the published weights (nothing retrained).

    python -m neural_lab.playground.export

Writes to neural_lab/.cache/web/ (git-ignored):
- encoder.onnx  image (1,1,64,64) -> block1 (1,32,32,32), block2 (1,64,16,16), block3 (1,128,8,8), latent (1,32)
                (the block outputs are the encoder's ReLU outputs: the same tensors the forward hooks record)
- decoder.onnx  latent (1,32) -> dec1 (1,128,8,8), dec2 (1,64,16,16), dec3 (1,32,32,32), reconstruction (1,1,64,64)
- meta.json     latent ranges (1st-99th percentile over the test tiles), measured neuron influence, PCA of the
                latent codes, real score quartiles per class
- map.json      2-D PCA position, class, weld and score of each of the 3,019 test tiles
Checks: decoder(encoder(x)) == deployed ONNX == PyTorch; block outputs == forward hooks.
"""
import json

import numpy as np
import torch
from torch import nn

from .. import activations
from ..data import CACHE_DIR, ORDER
from ..explorer import Explorer
from ..model import SERVER_DIR
from ..verify import load_tiles, onnx_session

WEB_DIR = CACHE_DIR / "web"
OPSET = 18                       # widely supported by onnxruntime-web
EXPORT_VERSION = 2               # bump when the exported files change (2: decoder taps, 3-D PCA)
FILES = ["encoder.onnx", "decoder.onnx", "meta.json", "map.json"]


class EncoderTaps(nn.Module):
    """The same encoder layers, returning the three block outputs (after ReLU) and the latent code."""

    def __init__(self, encoder: nn.Sequential):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        taps, h = [], x
        for layer in self.encoder:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                taps.append(h)
        return taps[0], taps[1], taps[2], h


class DecoderTaps(nn.Module):
    """The same decoder layers, returning its three 4-D intermediate tensors and the reconstruction:
    dec1 = 128x8x8 (after Linear, ReLU, Unflatten), dec2 = 64x16x16 and dec3 = 32x32x32 (after BatchNorm, ReLU)."""

    def __init__(self, decoder: nn.Sequential):
        super().__init__()
        self.decoder = decoder

    def forward(self, z):
        taps, h = [], z
        for layer in self.decoder:
            h = layer(h)
            if isinstance(layer, nn.Unflatten) or (isinstance(layer, nn.ReLU) and h.dim() == 4):
                taps.append(h)
        return taps[0], taps[1], taps[2], h


def _export(module, example, path, in_names, out_names):
    """torch.onnx.export as in the team notebook (single file); classic exporter as fallback."""
    kwargs = dict(input_names=in_names, output_names=out_names, opset_version=OPSET)
    try:
        torch.onnx.export(module, (example,), path, external_data=False, **kwargs)
    except Exception as err:  # noqa: BLE001
        print(f"  new exporter failed ({type(err).__name__}: {err}); using dynamo=False")
        torch.onnx.export(module, (example,), path, dynamo=False, **kwargs)


def export_halves(model, out_dir):
    _export(EncoderTaps(model.encoder).eval(), torch.zeros(1, 1, 64, 64), out_dir / "encoder.onnx",
            ["image"], ["block1", "block2", "block3", "latent"])
    _export(DecoderTaps(model.decoder).eval(), torch.zeros(1, 32), out_dir / "decoder.onnx", ["latent"],
            ["dec1", "dec2", "dec3", "reconstruction"])


def run_halves(enc, dec, X: np.ndarray, keep_blocks=()):
    """Run the two halves tile by tile (batch 1, as in the browser). Returns reconstructions, latents and the
    block maps of the tiles listed in keep_blocks (all of them would not fit in memory)."""
    rec, lat, blocks = [], [], {}
    for i in range(len(X)):
        b1, b2, b3, z = enc.run(None, {"image": X[i:i + 1]})
        rec.append(dec.run(None, {"latent": z})[-1])
        lat.append(z)
        if i in keep_blocks:
            blocks[i] = (b1[0], b2[0], b3[0])
    return np.concatenate(rec), np.concatenate(lat), blocks


@torch.no_grad()
def decoder_taps_check(model, dec, Z):
    """Compare the ONNX decoder taps with the PyTorch decoder, layer by layer (forward hooks)."""
    worst = 0.0
    for z in Z:
        got = dec.run(None, {"latent": z[None].astype(np.float32)})
        want, hooks = [], []
        for m in model.decoder:
            if isinstance(m, nn.Unflatten) or isinstance(m, nn.ReLU):
                hooks.append(m.register_forward_hook(lambda mod, i, o: want.append(o) if o.dim() == 4 else None))
        try:
            rec = model.decoder(torch.from_numpy(z[None]).float())
        finally:
            for h in hooks:
                h.remove()
        for g, w in zip(got, [*want, rec]):
            worst = max(worst, float(np.abs(g - w.numpy()).max()))
    return worst


@torch.no_grad()
def neuron_influence(model, Z_sound: np.ndarray, lo: np.ndarray, hi: np.ndarray, n_base: int = 24):
    """For each latent number: mean |pixel change| of the decoder output when that number alone is moved from
    its 1st to its 99th percentile, averaged over sound tiles. A measurement, not an interpretation."""
    idx = np.linspace(0, len(Z_sound) - 1, n_base).astype(int)
    base = torch.from_numpy(Z_sound[idx]).float()
    effect = []
    for d in range(base.shape[1]):
        a, b = base.clone(), base.clone()
        a[:, d], b[:, d] = float(lo[d]), float(hi[d])
        effect.append(float((model.decoder(b) - model.decoder(a)).abs().mean()))
    return effect


def main():
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    ex = Explorer("cpu")
    model, test = ex.model, ex.test
    export_halves(model, WEB_DIR)
    print(f"exported encoder.onnx (+ block taps) and decoder.onnx, opset {OPSET}")

    X = load_tiles(test["path"]).numpy()
    enc, dec = onnx_session(WEB_DIR / "encoder.onnx"), onnx_session(WEB_DIR / "decoder.onnx")
    check_ids = (0, len(X) // 2, len(X) - 1)
    rec, Z, blocks = run_halves(enc, dec, X, keep_blocks=check_ids)
    full = onnx_session(SERVER_DIR / "anomaly_ae.onnx")
    rec_full = np.concatenate([full.run(None, {"image": X[i:i + 256]})[0] for i in range(0, len(X), 256)])
    with torch.no_grad():
        rec_torch = model(torch.from_numpy(X)).numpy()
    d_full, d_torch = float(np.abs(rec - rec_full).max()), float(np.abs(rec - rec_torch).max())
    d_taps = max(float(np.abs(blocks[i][k] - activations.run(model, torch.from_numpy(X[i:i + 1])).blocks[f"block{k + 1}"]).max())
                 for i in check_ids for k in range(3))
    print(f"on {len(X)} test tiles: |halves - deployed ONNX| {d_full:.2e}, |halves - PyTorch| {d_torch:.2e}, "
          f"|block taps - forward hooks| {d_taps:.2e}")
    d_dec = decoder_taps_check(model, dec, Z[list(check_ids)])
    print(f"|decoder taps - PyTorch decoder layers| {d_dec:.2e}")
    d_taps = max(d_taps, d_dec)
    assert max(d_full, d_torch, d_taps) < 1e-4, "exported model does not match the published one"

    labels = test["label"].to_numpy()
    lo, hi = np.percentile(Z, 1, axis=0), np.percentile(Z, 99, axis=0)
    effect = neuron_influence(model, Z[labels == "no_defect"], lo, hi)
    mean = Z.mean(axis=0)
    _, svals, vt = np.linalg.svd(Z - mean, full_matrices=False)
    coords = (Z - mean) @ vt[:3].T
    var = (svals ** 2) / (svals ** 2).sum()

    meta = {
        "export_version": EXPORT_VERSION,
        "latent_lo": lo.tolist(), "latent_hi": hi.tolist(),
        "latent_range_rule": f"1st-99th percentile of the encoder output over the {len(Z)} test tiles",
        "influence": effect,
        "influence_rule": "mean |pixel change| of the decoder output when this number alone goes from its 1st to "
                          "99th percentile, averaged over 24 sound test tiles",
        "pca_mean": mean.tolist(), "pca_components": vt[:3].tolist(),
        "pca_explained": [float(var[0]), float(var[1]), float(var[2])],
        "class_scores": {c: np.percentile(test.loc[labels == c, "score"], [25, 50, 75]).tolist() for c in ORDER},
        "checks": {"halves_vs_deployed_onnx": d_full, "halves_vs_pytorch": d_torch, "taps_vs_hooks": d_taps},
    }
    (WEB_DIR / "meta.json").write_text(json.dumps(meta))
    points = [{"x": round(float(c[0]), 3), "y": round(float(c[1]), 3), "z": round(float(c[2]), 3), "label": l, "score": float(s), "weld": w}
              for c, l, s, w in zip(coords, labels, test["score"], test["weld"])]
    (WEB_DIR / "map.json").write_text(json.dumps(points))
    top = np.argsort(effect)[::-1][:6]
    print(f"most influential latent numbers (measured): {', '.join(f'z{d} ({effect[d]:.4f})' for d in top)}")
    print(f"PCA: first three components explain {var[0]:.0%} + {var[1]:.0%} + {var[2]:.0%} of the latent variance")


if __name__ == "__main__":
    main()
