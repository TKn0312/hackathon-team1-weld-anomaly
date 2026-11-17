# Neural Network Lab · inside the weld-anomaly autoencoder

An isolated, local experiment by Sahil on top of Team 1's project. It loads the **published, pretrained**
`ConvAE(64, 32)` from Hugging Face ([iaac-maai/hackathon-team1-weld-anomaly-ae](https://huggingface.co/iaac-maai/hackathon-team1-weld-anomaly-ae)),
runs real tiles through it and shows what happens inside. Nothing is retrained. The deployed app
(`02_model_server/`) is neither changed nor imported.

## Run it

From the repository root, in the `iaac-ai` environment:

```powershell
python -m neural_lab.data                  # once: get + extract RIAWELC, check counts, rebuild the team split
python -m neural_lab.verify                # PyTorch == ONNX, published threshold and ROC AUC reproduced
python -m pytest neural_lab/tests -q       # tests
python -m neural_lab.app                   # the explorer: http://127.0.0.1:7861
python -m neural_lab.render_examples       # optional: save the figures to neural_lab/outputs/
python -m neural_lab                       # NEURAL / LAB: the main app (http://127.0.0.1:7870)
```

### NEURAL / LAB — Exploring the Black Box

The main app. The published model runs **in the browser** (ONNX Runtime Web), so it reacts live.

- **Defect experiment**: start from a test-weld tile or upload one, draw on a large canvas; reconstruction, error
  heatmap (overlay on the weld), score, LOW / CHECK / HIGH and a before/after wipe update while you draw.
  **Sensitivity experiment**: one synthetic pore / crack / band on 12 sound test welds, sweep its size or darkness,
  plot the real scores (mean and range) against real defect scores of the test welds; CSV export.
- **Neural explorer**: six latent numbers preselected by *measured* influence (all 32 in advanced mode, ranges =
  1st-99th percentile of real test tiles), original vs edited code and redraw, the difference they make, real
  block 1-3 feature maps (and how your drawing changed them), and a PCA map of the 3,019 test tiles where your
  weld and your edited code move live.

`playground/export.py` prepares it: exports the encoder (with its three block outputs) and decoder from the published
weights and checks them against the deployed ONNX model, PyTorch and the forward hooks; measures neuron influence;
computes the PCA. The page lists these checks at the bottom. A drawn mark is not a real defect: the page never
claims the model will flag it.

The Gradio explorer (`python -m neural_lab.app`) still works but is superseded by NEURAL / LAB.

## Files

| File | What it does |
|---|---|
| `model.py` | `ConvAE` copied unchanged from the team notebook; loads the published weights (`strict=True`); the exact preprocessing (greyscale, 64×64 bilinear, ÷255) |
| `activations.py` | forward hooks that record the 3 encoder blocks and the 32-number latent code during one real forward pass |
| `data.py` | downloads/extracts RIAWELC (19 RAR parts), checks the counts, rebuilds the team's split by weld and checks it against the notebook |
| `verify.py` | the model is the deployed one: same ONNX file, same outputs, same threshold and ROC AUC |
| `plots.py` | figures: input / reconstruction / error map, feature maps, latent code, sound-vs-defect comparison |
| `explorer.py`, `app.py` | the local Gradio interface |
| `playground/` | NEURAL / LAB: export + checks, a small local server (127.0.0.1 only), the in-browser page |
| `tests/` | pytest checks for all of the above |

Git-ignored: `neural_lab/.cache/` (split table, cached test scores) and `neural_lab/outputs/` (figures contain
RIAWELC pixels, which must not be redistributed). The weights stay in the Hugging Face cache, outside the repo.

## The ideas, in plain words

- **Neural network.** A big function made of simple steps (layers). Each step multiplies its input by learned
  numbers (weights) and passes the result on. Training adjusted those weights once; here we only *use* them.
- **Autoencoder.** Two halves. The **encoder** squeezes a 64×64 tile (4,096 pixels) into **32 numbers**; the
  **decoder** redraws the tile from those 32 numbers alone. Ours learned only from sound welds.
- **Convolution and feature maps.** A convolution slides a small 3×3 filter over the image and asks "does this
  little pattern appear here?" at every position. Each filter produces one **feature map**: a picture of where
  it responded. A block has many filters (32, 64, 128 **channels**), so many feature maps.
- **Stride 2.** The filter jumps 2 pixels at a time, so each block halves the size: 64 → 32 → 16 → 8 px. Deeper
  maps are smaller but each pixel "sees" a bigger area of the tile.
- **BatchNorm and ReLU.** BatchNorm rescales values to a stable range; ReLU keeps positives and sets negatives to
  0. That is why the feature maps are never negative and often have dark (zero) regions.
- **Activation.** The value a layer outputs for a given input. "Strong activation" = large value. We do not claim
  what a channel *means*: "strongest" only ranks channels by their average value for that tile.
- **Forward hook.** A small function PyTorch calls right after a layer runs. We use it to copy the layer's output
  without changing the model. The tests check that the output is identical with and without hooks.
- **Latent code.** The 32 numbers in the middle. Everything the decoder knows about the tile is in there.
- **Reconstruction error.** (input − reconstruction)² per pixel = the **error map**; its mean = the **anomaly
  score**. The model never learned to draw pores or cracks, so it fails exactly there and the score rises.
- **ONNX.** A portable copy of the model used by the deployed server. `verify.py` checks it gives the same answers.

## Data and citation

RIAWELC (no formal licence, free with citation, not redistributed): B. Totino, F. Spagnolo, S. Perri, "RIAWELC:
A Novel Dataset of Radiographic Images for Automatic Weld Defects Classification", ICMECE 2022; S. Perri,
F. Spagnolo, F. Frustaci, P. Corsonello, "Welding Defects Classification Through a Convolutional Neural Network",
Manufacturing Letters.
