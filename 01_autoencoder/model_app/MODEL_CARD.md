---
license: mit
library_name: onnx
tags:
  - anomaly-detection
  - autoencoder
  - onnx
  - pytorch
  - radiography
  - x-ray
  - weld-inspection
  - construction
  - built-environment
---

# Weld anomaly autoencoder (RIAWELC)

A small **convolutional autoencoder** that flags defects in **X-ray tiles of steel welds**: porosity, cracks and lack of penetration. It was trained **only on defect-free (sound) welds**. A tile it cannot rebuild well is anomalous, and the error map shows where.

Made by Team 1 (Jerome, Kazuko, Amelia, Sahil) for the *AI Methods I* hackathon, IAAC Master in AI for Architecture & the Built Environment, November 2026.
Code, notebook and figures: [github.com/iaac-maai/hackathon-team1-weld-anomaly](https://github.com/iaac-maai/hackathon-team1-weld-anomaly).

## What it does

```
X-ray tile ─► greyscale, 64×64, 0–1 ─► ENCODER ─► 32 numbers ─► DECODER ─► rebuilt tile
                                                                              │
                     score = mean((input − rebuilt)²) ◄──────────────────────┘
                     error map = (input − rebuilt)² per pixel
```

The decoder only learned to draw sound welds, so a defect is missing from the rebuilt tile and the error is high exactly there.

| Risk level | Score | Meaning |
|---|---|---|
| 🟢 low | ≤ 0.000255 | looks like a sound weld |
| 🟠 check | 0.000255 – 0.001442 | unusual, an inspector should look |
| 🔴 high | > 0.001442 | strongly abnormal, inspect first |

**Intended use:** a screening aid that ranks X-ray tiles so an inspector reviews the riskiest first. **Not** an accept/reject decision under welding standards.

## Files

| File | Contents |
|---|---|
| `anomaly_ae.onnx` | The model (architecture + weights), for ONNX Runtime. Input `image` (N, 1, 64, 64) float32 0–1 → output `reconstruction`, same shape |
| `anomaly_ae.pt` | PyTorch `state_dict` of the `ConvAE` class in the notebook (`ConvAE(64, 32)`) |
| `anomaly_ae_config.json` | Preprocessing, score definition, thresholds and results. The server must follow it exactly |

## How to use

```python
import json
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

repo = "iaac-maai/hackathon-team1-weld-anomaly-ae"
session = ort.InferenceSession(hf_hub_download(repo, "anomaly_ae.onnx"))
config = json.load(open(hf_hub_download(repo, "anomaly_ae_config.json")))

size = config["image_size"]                                     # 64
img = Image.open("weld_tile.png").convert("L").resize((size, size), Image.Resampling.BILINEAR)
x = (np.asarray(img, dtype=np.float32) / 255.0)[None, None]     # (1, 1, 64, 64)

rebuilt = session.run(None, {"image": x})[0]
error_map = (rebuilt - x)[0, 0] ** 2                            # where to look
score = float(error_map.mean())
level = "high" if score > config["t_high"] else "check" if score > config["t_low"] else "low"
print(f"score {score:.6f} -> {level}")
```

The input should be a radiograph tile like those in RIAWELC: greyscale, about 227 × 227 px, showing a section of a weld.

## Training data

[RIAWELC](https://github.com/stefyste/RIAWELC): 24,407 greyscale X-ray tiles (227 × 227 px) of steel welds from 29 radiographs, in four classes: no defect, porosity, crack, lack of penetration.
- **2,514 duplicates removed:** every original test image was also in training. That leaves 21,893 unique tiles.
- **New split by weld**, so no weld appears in two splits: 20 welds for training, 5 for validation, 4 for test.
- **Training used only the 3,783 sound tiles** of the training welds.

The images are **not redistributed** here. Please cite the dataset authors (below).

## Training

- **Model:** `ConvAE`. Encoder: 3 × (Conv 3×3 stride 2 + BatchNorm + ReLU), 64 → 8 px, then a linear layer to a 32-number code. The decoder mirrors it with transposed convolutions and a sigmoid. 718,369 parameters.
- **Training:** MSE loss, Adam (lr 0.001), batch 64, random flips, early stopping on the error of sound validation tiles (best epoch 21). 8 s on an RTX 5080 Laptop GPU.
- **Selection:** best of 4 setups by validation ROC AUC: Conv 128 px latent 32, Conv 128 px latent 128, Dense 128 px latent 32, and Conv 64 px latent 32 (this model).

## Results

| | Validation (5 welds) | Test (4 unseen welds) |
|---|---|---|
| ROC AUC (defect vs sound) | 0.990 | 0.999 |
| AUC porosity / crack / lack of penetration | 0.994 / 0.985 / 0.998 | 0.999 / 0.999 / 0.999 |
| Precision / recall at the 95% threshold | 0.981 / 0.958 | 0.994 / 0.984 |
| Contrast-only baseline, ROC AUC | 0.946 | 0.936 |

Risk levels on the test set:
- **Sound tiles:** 90.9% low, 9.1% check, 0% high.
- **Defect tiles:** only 0.5% end up in "low".

The fair summary is **about 0.99 ROC AUC on unseen welds**: the test set has only 4 welds.

## Limitations

- **Small dataset:** 29 welds, all from one lab and one X-ray setup. It may not transfer to other equipment or exposures.
- **Tile-level labels:** whether the error map points at the right spot was not measured.
- **Grain counts too:** on grainy radiographs part of the score comes from texture, not from the defect.
- **Cracks are the hardest defect:** thin and faint.
- **Screening aid only:** acceptance under standards such as ISO 5817 depends on defect size and plate thickness, which the model does not measure.

## Citation (dataset)

- B. Totino, F. Spagnolo, S. Perri, "RIAWELC: A Novel Dataset of Radiographic Images for Automatic Weld Defects Classification", ICMECE 2022.
- S. Perri, F. Spagnolo, F. Frustaci, P. Corsonello, "Welding Defects Classification Through a Convolutional Neural Network", Manufacturing Letters, Elsevier.
