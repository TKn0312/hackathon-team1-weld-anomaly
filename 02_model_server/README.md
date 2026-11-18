# Weld X-ray anomaly detector: model server

The model server of Team 1's hackathon project (AI Methods I, IAAC MaAI 2026/27). One Render service, three things:

| Address | What |
|---|---|
| **https://hackathon-team1-weld-anomaly.onrender.com** | **NEURAL / LAB** (`neural_lab/`): draw a defect on a real weld and watch the network react live, the architecture, the latent universe, a 90-second demo. The model runs **in the browser** |
| **…/gradio** + `/predict` API | **API only, for programs:** send an image, get JSON back (no page for people any more; the app is NEURAL / LAB). The Cloudflare web app uses it (`MODEL_URL` = `https://hackathon-team1-weld-anomaly.onrender.com/gradio`) |

Free Render service: it sleeps after 15 minutes without requests, and the first request then takes about a minute.

**The lab online:** `build_lab.py` packs it once on a laptop into `lab/`: the page (copied from `neural_lab/playground/static`), the model split into encoder and decoder ONNX files, the PCA map, and **115 test tiles at 64 px** (not the dataset, which must not be re-published). A click on another point of the map loads the nearest packed tile of the same class. Re-run `python 02_model_server/build_lab.py` after changing `neural_lab/`.
**Model on Hugging Face:** [iaac-maai/hackathon-team1-weld-anomaly-ae](https://huggingface.co/iaac-maai/hackathon-team1-weld-anomaly-ae)
**How it was made:** [../01_autoencoder/](../01_autoencoder/) (notebook, comparison of setups, figures)

## The autoencoder

| | |
|---|---|
| **Type** | Convolutional autoencoder. Encoder: 3 × (3×3 convolution, stride 2, BatchNorm, ReLU), 64 → 8 px, then a linear layer to a **code of 32 numbers**. The decoder mirrors it with transposed convolutions and ends with a sigmoid. 718,369 parameters. |
| **Data** | [RIAWELC](https://github.com/stefyste/RIAWELC): X-ray tiles of steel welds in 4 classes (no defect, porosity, crack, lack of penetration). 2,514 duplicates removed; split **by weld** (20 / 5 / 4 welds), so test welds are unseen. |
| **Trained on** | Only the **3,783 sound (no-defect) tiles** of the training welds. It never saw a defect. |
| **Training** | Mean squared error, Adam (lr 0.001), batch 64, random flips, early stopping on sound validation tiles (best epoch 21). 8 s on a GPU. |
| **Chosen how** | Best of 4 setups on validation ROC AUC (128 vs 64 px, code 32 vs 128, convolutional vs dense). |
| **Format** | ONNX (`anomaly_ae.onnx`, 2.8 MB), run with ONNX Runtime: no PyTorch on the server. |

**Why it finds defects:** the model only learned to redraw sound welds. When a tile contains a defect, it redraws a sound weld anyway, so the defect is missing from the rebuilt tile. The difference between the input and the rebuilt tile is large exactly there.

## Input and output

**Input:** an X-ray tile of a weld (PNG or JPEG), like the RIAWELC tiles (about 227 × 227 px). The server applies **the same preprocessing as in training**: greyscale → resized to 64 × 64 (bilinear) → pixels divided by 255.

**Output:**

| Field | Meaning |
|---|---|
| `score` | Anomaly score: the mean squared difference between the input and its reconstruction. Sound tiles score about 0.0002 |
| `risk_level` | `low` (score ≤ 0.000255): looks like a sound weld · `check`: unusual, an inspector should look · `high` (score > 0.001442): strongly abnormal, inspect first |
| `is_anomaly` | `true` above the 95% threshold (0.000345) |
| `score_vs_threshold` | The score as a multiple of that threshold |
| `threshold`, `t_low`, `t_high` | The thresholds used |
| `reconstruction_png_base64` | The tile as the model rebuilds it (a sound weld), as a base64 PNG (API only) |
| `error_map_png_base64` | The error map as a heatmap: bright = where the model failed = where to look (API only) |

**How the thresholds were chosen** (validation welds only):
- `t_low`: 98% of defect tiles score above it, so few defects are missed.
- `t_high`: only 0.5% of sound tiles score above it, so "high" almost always means a real defect.

**Results on 4 unseen test welds:** ROC AUC 0.999 (≈ 0.99 is the fair claim), recall 0.984. 91% of sound tiles are "low", and only 0.5% of defect tiles end up in "low".

## Call it from a program

```python
import base64
from gradio_client import Client

client = Client("https://hackathon-team1-weld-anomaly.onrender.com/gradio")
image_b64 = base64.b64encode(open("weld_tile.png", "rb").read()).decode()
result = client.predict(image_b64, api_name="/predict")
print(result["risk_level"], result["score"])
```

Over plain HTTP, which is what the Cloudflare app does:
1. `POST /gradio/gradio_api/call/predict` with `{"data": ["<base64 image>"]}` returns an `event_id`.
2. `GET /gradio/gradio_api/call/predict/<event_id>` returns the result after the line `event: complete`.

`call_server.py` does both, and first wakes the server up if it is asleep:

```bash
python call_server.py path/to/tile.png                              # the Render server
python call_server.py path/to/tile.png --url http://127.0.0.1:7860  # a local server
```

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\activate           # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py                    # then open http://127.0.0.1:7860
```

## Files

| File | Contents |
|---|---|
| `app.py` | Loading the model, the preprocessing, the page and the `/predict` API |
| `anomaly_ae.onnx` | The model (architecture and weights) |
| `config.json` | Image size, preprocessing, thresholds and risk levels, results |
| `call_server.py` | Wakes the server and calls it with `gradio_client` and with plain HTTP |
| `requirements.txt`, `.python-version` | What Render installs: Gradio and ONNX Runtime, Python 3.12, no PyTorch |
| `render.yaml` | Render settings from the course template (our service was set up in the Render dashboard: root directory `02_model_server`, start command `python app.py`) |
| `examples/` | Test tiles for `call_server.py`. **Local only:** they come from RIAWELC, which must not be re-uploaded |

## Limitations

- **Trained on 29 welds from one lab and one X-ray setup.** Tiles from other equipment or exposures may get unreliable scores.
- **Faint cracks are the hardest defect:** thin lines change few pixels, so their score stays lower.
- **Unusual sound tiles** (very grainy or odd brightness) can land in "check": false alarms, not dangerous.
- **It does not name the defect type.** It says how abnormal a tile is, not whether it is a crack or a pore.
- **A screening aid, not a verdict.** Acceptance under welding standards depends on defect size and plate thickness, which the model does not measure. A qualified inspector decides.
