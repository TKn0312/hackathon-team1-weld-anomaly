"""Local server for NEURAL / LAB (installation at /, previous interface at /classic.html) — Exploring the Black Box (the model itself runs in the browser).

    python -m neural_lab            (or python -m neural_lab.playground)   ->  http://127.0.0.1:7870

Serves the page, the two ONNX halves, precomputed measurements, and tiles from YOUR local RIAWELC copy or
from an image you upload, always preprocessed in Python exactly like training (greyscale, 64x64 bilinear, /255).
Listens on 127.0.0.1 only: no image or model leaves your laptop or goes into Git.
"""
import io
import json
import os
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

from .. import activations
from ..data import ORDER, tile_path
from ..explorer import Explorer
from ..model import preprocess
from ..plots import ERROR_MAP_MAX
from .export import EXPORT_VERSION, FILES, WEB_DIR, main as export_main

STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 15 * 1024 * 1024
N_SOUND_SET = 12        # sound test tiles used as backgrounds in the sensitivity experiment


def catalogue(test):
    """Dropdown entries: per class the lowest / median / highest scoring test tile and 5 ranks in between."""
    out = []
    for label in ORDER:
        sub = test[test["label"] == label].sort_values("score", kind="stable")
        n = len(sub)
        picks = {"lowest": 0, "median": n // 2, "highest": n - 1}
        for r in np.linspace(0, n - 1, 7)[1:-1].astype(int):
            picks.setdefault(f"rank {r}/{n - 1}", int(r))
        for name, r in sorted(picks.items(), key=lambda kv: kv[1]):
            i = int(sub.index[r])
            out.append({"id": i, "label": label, "pick": name, "score": float(test.at[i, "score"])})
    return out


class Handler(SimpleHTTPRequestHandler):
    ex: Explorer = None
    payload: dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, *_):
        pass

    def _send(self, body, ctype="application/json", code=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _tile(self, x_tensor, extra):
        inf = activations.run(self.ex.model, x_tensor)
        return {**extra, "x": inf.x.ravel().round(6).tolist(), "python_score": inf.score}

    def do_GET(self):
        p = self.path.split("?")[0]
        test = self.ex.test
        if p.startswith("/models/") and p[8:] in ("encoder.onnx", "decoder.onnx"):
            return self._send((WEB_DIR / p[8:]).read_bytes(), "application/octet-stream")
        if p == "/api/config":
            return self._send(self.payload)
        if p == "/api/map":
            return self._send((WEB_DIR / "map.json").read_bytes())
        if p.startswith("/api/tile/"):
            i = int(p.rsplit("/", 1)[1])
            row = test.iloc[i]
            return self._send(self._tile(preprocess(tile_path(row["path"]), self.ex.size),
                                         {"id": i, "label": row["label"], "weld": row["weld"], "source": row["source"],
                                          "score": float(row["score"])}))
        if p == "/api/sound_set":
            sub = test[test["label"] == "no_defect"].sort_values("score", kind="stable")
            ids = [int(sub.index[r]) for r in np.linspace(0, len(sub) - 1, N_SOUND_SET + 2)[1:-1].astype(int)]
            tiles = [{"id": i, "x": preprocess(tile_path(test.at[i, "path"]), self.ex.size).numpy().ravel().round(6).tolist(),
                      "score": float(test.at[i, "score"])} for i in ids]
            return self._send(tiles)
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/preprocess":
            return self._send({"error": "not found"}, code=404)
        n = int(self.headers.get("Content-Length", 0))
        if not 0 < n <= MAX_UPLOAD:
            return self._send({"error": "empty or too large"}, code=400)
        try:
            image = Image.open(io.BytesIO(self.rfile.read(n)))
            return self._send(self._tile(preprocess(image, self.ex.size), {"label": "uploaded", "weld": "–", "source": "your file"}))
        except Exception as err:  # noqa: BLE001
            return self._send({"error": f"could not read image: {err}"}, code=400)


def main(port: int = int(os.environ.get("NEURAL_LAB_PORT", 7870))):
    def up_to_date():
        if not all((WEB_DIR / f).exists() for f in FILES):
            return False
        return json.loads((WEB_DIR / "meta.json").read_text()).get("export_version") == EXPORT_VERSION
    if not up_to_date():
        export_main()
    ex = Explorer("cpu")
    Handler.ex = ex
    c = ex.config
    Handler.payload = {"t_low": c["t_low"], "t_high": c["t_high"], "threshold": c["threshold"],
                       "error_map_max": ERROR_MAP_MAX, "catalogue": catalogue(ex.test),
                       **json.loads((WEB_DIR / "meta.json").read_text())}
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"NEURAL / LAB running at {url}   (Ctrl+C to stop)")
    webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
