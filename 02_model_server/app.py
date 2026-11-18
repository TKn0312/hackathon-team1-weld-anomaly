"""Model server: weld X-ray anomaly detection with a convolutional autoencoder (Gradio + ONNX Runtime).

The page has a form for people (upload an X-ray tile, see the risk level and where to look) and an
API for programs (`/predict`: send a base64 image, get JSON back). The Cloudflare app uses the API.
Runs locally (`python app.py`) and on Render.

The model is the ONNX file exported in 01_autoencoder (setup D: convolutional autoencoder, 64 px,
code of 32 numbers). It contains the architecture AND the weights, so no PyTorch is needed here.
"""
import base64
import io
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = Path(__file__).parent
CONFIG = json.loads((HERE / "config.json").read_text())
SIZE = CONFIG["image_size"]            # 64
THRESHOLD = CONFIG["threshold"]        # 95th percentile of sound validation tiles
T_LOW = CONFIG["t_low"]                # below: low risk  (98% of validation defects score above it)
T_HIGH = CONFIG["t_high"]              # above: high risk (only 0.5% of sound validation tiles score above it)

# Per-pixel squared error shown as full brightness in the error map. Fixed, so a sound tile stays dark
# and maps can be compared between tiles (a pore reaches ~0.25, faint cracks ~0.02).
ERROR_MAP_MAX = 0.05

session = ort.InferenceSession(str(HERE / "anomaly_ae.onnx"), providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name

RISK_TEXT = {
    "low": "LOW: looks like a sound weld",
    "check": "CHECK: unusual, an inspector should look",
    "high": "HIGH: strongly abnormal, inspect first",
}


def preprocess(image: Image.Image) -> np.ndarray:
    # The SAME preprocessing as in training: greyscale, 64 x 64 (bilinear), pixels 0-1.
    # No inversion: X-ray tiles always look the same way (dark = less metal).
    grey = image.convert("L").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    x = np.asarray(grey, dtype=np.float32) / 255.0
    return x[None, None]                 # shape (1, 1, 64, 64): one image, one channel


def risk_level(score: float) -> str:
    return "high" if score > T_HIGH else "check" if score > T_LOW else "low"


def analyse(image: Image.Image):
    x = preprocess(image)
    rebuilt = session.run(None, {INPUT_NAME: x})[0]
    error_map = ((rebuilt - x) ** 2)[0, 0]        # where the model failed to rebuild the tile
    score = float(error_map.mean())               # the anomaly score: mean squared error
    level = risk_level(score)
    result = {
        "score": round(score, 6),
        "risk_level": level,
        "is_anomaly": score > THRESHOLD,
        "score_vs_threshold": round(score / THRESHOLD, 2),
        "threshold": round(THRESHOLD, 6),
        "t_low": round(T_LOW, 6),
        "t_high": round(T_HIGH, 6),
    }
    return result, rebuilt[0, 0], error_map


# ---- Images for display ---------------------------------------------------------
HEAT_STOPS = np.array([[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]], dtype=np.float32)


def heat_colours(values: np.ndarray) -> np.ndarray:
    # Values 0-1 -> RGB on a dark-to-yellow scale (similar to matplotlib's "inferno"), without matplotlib
    pos = np.clip(values, 0, 1) * (len(HEAT_STOPS) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(HEAT_STOPS) - 2)
    frac = (pos - lo)[..., None]
    return (HEAT_STOPS[lo] * (1 - frac) + HEAT_STOPS[lo + 1] * frac).astype(np.uint8)


def grey_image(array: np.ndarray, scale: int = 4) -> Image.Image:
    img = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))
    return img.resize((SIZE * scale, SIZE * scale), Image.Resampling.NEAREST)


def heatmap_image(error_map: np.ndarray, scale: int = 4) -> Image.Image:
    img = Image.fromarray(heat_colours(error_map / ERROR_MAP_MAX))
    return img.resize((SIZE * scale, SIZE * scale), Image.Resampling.NEAREST)


def png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# ---- For people: the web form ----------------------------------------------------
def analyse_for_page(image):
    if image is None:
        raise gr.Error("Upload an X-ray tile first.")
    result, rebuilt, error_map = analyse(image)
    return {RISK_TEXT[result["risk_level"]]: 1.0}, result, grey_image(rebuilt), heatmap_image(error_map)


# ---- For programs: base64 image in, JSON out --------------------------------------
def predict(image_b64: str) -> dict:
    """Score one weld X-ray tile. Send a base64-encoded PNG or JPEG (a 'data:image/...;base64,' prefix is allowed).
    Returns the score, the risk level (low / check / high), and the reconstruction and error map as base64 PNGs."""
    if "," in image_b64[:100]:
        image_b64 = image_b64.split(",", 1)[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    result, rebuilt, error_map = analyse(image)
    result["reconstruction_png_base64"] = png_base64(grey_image(rebuilt))
    result["error_map_png_base64"] = png_base64(heatmap_image(error_map))
    return result


EXAMPLES = sorted(str(p) for p in (HERE / "examples").glob("*.png"))   # local only: dataset tiles are not in git

with gr.Blocks(title="Weld X-ray anomaly detector") as demo:
    gr.Markdown(
        "# Weld X-ray anomaly detector\n"
        "A convolutional autoencoder trained **only on sound welds** (RIAWELC X-ray tiles). "
        "It redraws every tile as a sound weld: where it fails, there is probably a defect "
        "(porosity, crack, lack of penetration).\n\n"
        f"**Risk levels:** low (score ≤ {T_LOW:.6f}) · check · high (score > {T_HIGH:.6f}). "
        "A screening aid for an inspector, not an accept/reject decision."
    )
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="X-ray tile of a weld")
            button = gr.Button("Analyse", variant="primary")
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=image_in)
        with gr.Column():
            verdict_out = gr.Label(label="Risk level")
            details_out = gr.JSON(label="Details")
            with gr.Row():
                rebuilt_out = gr.Image(label="Rebuilt as a sound weld")
                error_out = gr.Image(label="Error map: bright = where to look")
    button.click(analyse_for_page, inputs=image_in, outputs=[verdict_out, details_out, rebuilt_out, error_out],
                 api_visibility="private")
    gr.api(predict, api_name="predict")

# ---- NEURAL / LAB (Sahil's lab, neural_lab/) at the main address ------------------------------------
# The lab runs the model in the browser (two ONNX halves); the server only hands out files, a small set of
# test tiles (64 px, packed by build_lab.py) and the preprocessing of uploads. Our Gradio app moves to /gradio.
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

LAB = HERE / "lab"
app = FastAPI(title="Weld X-ray anomaly detector")

if (LAB / "static" / "index.html").exists():
    _lab_cfg = json.loads((LAB / "data" / "config.json").read_text())
    _lab_meta = json.loads((LAB / "data" / "tiles.json").read_text())
    _npz = np.load(LAB / "data" / "tiles.npz")
    _lab_px = {int(i): p for i, p in zip(_npz["ids"], _npz["pixels"])}
    _map = json.loads((LAB / "data" / "map.json").read_text())
    _xyz = np.array([[p["x"], p["y"], p["z"]] for p in _map], np.float32)
    _labels = np.array([p["label"] for p in _map])
    _have = np.array(sorted(_lab_px))

    def _lab_tile(i: int) -> dict:
        if i not in _lab_px:                     # not packed: the nearest packed tile of the SAME class on the map
            same = _have[_labels[_have] == _labels[i]]
            i = int(same[np.argmin(((_xyz[same] - _xyz[i]) ** 2).sum(axis=1))])
        x = _lab_px[i].astype(np.float32) / 255.0
        return {"id": i, **_lab_meta["tiles"][str(i)], "x": x.ravel().round(6).tolist()}

    @app.get("/api/config")
    def lab_config():
        return _lab_cfg

    @app.get("/api/map")
    def lab_map():
        return FileResponse(LAB / "data" / "map.json", media_type="application/json")

    @app.get("/api/tile/{i}")
    def lab_tile(i: int):
        if not 0 <= i < len(_map):
            return JSONResponse({"error": "no such tile"}, status_code=404)
        return _lab_tile(i)

    @app.get("/api/sound_set")
    def lab_sound_set():
        return [{"id": i, "x": _lab_tile(i)["x"], "score": _lab_meta["tiles"][str(i)]["score"]} for i in _lab_meta["sound_set"]]

    @app.post("/api/preprocess")
    async def lab_preprocess(request: Request):
        body = await request.body()
        if not 0 < len(body) <= 15 * 1024 * 1024:
            return JSONResponse({"error": "empty or too large"}, status_code=400)
        try:
            x = preprocess(Image.open(io.BytesIO(body)))
        except Exception as err:  # noqa: BLE001
            return JSONResponse({"error": f"could not read image: {err}"}, status_code=400)
        score = float(((session.run(None, {INPUT_NAME: x})[0] - x) ** 2).mean())
        return {"label": "uploaded", "weld": "–", "source": "your file", "x": x.ravel().round(6).tolist(), "python_score": score}

    @app.get("/models/{name}")
    def lab_model(name: str):
        if name not in ("encoder.onnx", "decoder.onnx"):
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(LAB / "models" / name, media_type="application/octet-stream")

app = gr.mount_gradio_app(app, demo, path="/gradio")          # our tile app + its /predict API
if (LAB / "static" / "index.html").exists():
    app.mount("/", StaticFiles(directory=LAB / "static", html=True), name="lab")   # last: the lab page itself

if __name__ == "__main__":
    # Render (and most hosts) tell the app which port to use in the PORT variable; locally it is 7860
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
