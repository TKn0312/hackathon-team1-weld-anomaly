"""Pack Sahil's NEURAL / LAB (neural_lab/) for the online model server.     Run once on a laptop, from the repo root:

    python 02_model_server/build_lab.py

The lab runs the model in the browser, so online it only needs files. This script needs PyTorch and the local
RIAWELC copy (the Render server needs neither) and writes 02_model_server/lab/:
  static/   the lab page (copied unchanged from neural_lab/playground/static)
  models/   encoder.onnx, decoder.onnx (the published model split in two, checked by neural_lab's export)
  data/     config.json, map.json (PCA of the 3,019 test tiles), tiles.npz + tiles.json
Only ~150 test tiles are packed, at 64x64 px (what the model sees), not the dataset: RIAWELC must not be
re-published. A click on any other point of the map gets the nearest packed tile.
"""
import json, shutil, sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from neural_lab.data import ORDER, tile_path                                          # noqa: E402
from neural_lab.explorer import Explorer                                              # noqa: E402
from neural_lab.model import preprocess                                               # noqa: E402
from neural_lab.plots import ERROR_MAP_MAX                                            # noqa: E402
from neural_lab.playground.export import EXPORT_VERSION, FILES, WEB_DIR, main as export_main  # noqa: E402
from neural_lab.playground.server import N_SOUND_SET, STATIC, catalogue              # noqa: E402

OUT = REPO / "02_model_server" / "lab"
PER_CLASS = 25          # extra tiles per class, spread over the score range


def main():
    if not all((WEB_DIR / f).exists() for f in FILES) or json.loads((WEB_DIR / "meta.json").read_text()).get("export_version") != EXPORT_VERSION:
        export_main()
    ex = Explorer("cpu")
    test, c = ex.test, ex.config

    cat = catalogue(test)
    sound = test[test["label"] == "no_defect"].sort_values("score", kind="stable")
    sound_ids = [int(sound.index[r]) for r in np.linspace(0, len(sound) - 1, N_SOUND_SET + 2)[1:-1].astype(int)]
    ids = {e["id"] for e in cat} | set(sound_ids)
    for label in ORDER:
        sub = test[test["label"] == label].sort_values("score", kind="stable")
        ids |= {int(sub.index[r]) for r in np.linspace(0, len(sub) - 1, PER_CLASS).astype(int)}
    ids = sorted(ids)

    full = ort.InferenceSession(str(REPO / "02_model_server" / "anomaly_ae.onnx"), providers=["CPUExecutionProvider"])
    pixels, meta = [], {}
    for i in ids:
        row = test.iloc[i]
        x = preprocess(tile_path(row["path"]), ex.size).numpy()                   # (1, 1, 64, 64), pixels / 255
        u8 = np.rint(x[0, 0] * 255).astype(np.uint8)
        assert np.array_equal(u8.astype(np.float32) / 255.0, x[0, 0]), "64 px tile not exact in 8 bit"
        rec = full.run(None, {full.get_inputs()[0].name: x})[0]
        pixels.append(u8)
        meta[str(i)] = {"label": row["label"], "weld": row["weld"], "source": row["source"],
                        "score": float(row["score"]), "python_score": float(((rec - x) ** 2).mean())}

    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(STATIC, OUT / "static")
    (OUT / "models").mkdir(parents=True)
    for f in ("encoder.onnx", "decoder.onnx"):
        shutil.copy(WEB_DIR / f, OUT / "models" / f)
    (OUT / "data").mkdir()
    shutil.copy(WEB_DIR / "map.json", OUT / "data" / "map.json")
    config = {"t_low": c["t_low"], "t_high": c["t_high"], "threshold": c["threshold"], "error_map_max": ERROR_MAP_MAX,
              "catalogue": cat, **json.loads((WEB_DIR / "meta.json").read_text())}
    (OUT / "data" / "config.json").write_text(json.dumps(config))
    np.savez_compressed(OUT / "data" / "tiles.npz", ids=np.array(ids, np.int32), pixels=np.stack(pixels))
    (OUT / "data" / "tiles.json").write_text(json.dumps({"tiles": meta, "sound_set": sound_ids}))
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"lab packed: {len(ids)} tiles, {size / 1e6:.1f} MB in {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
