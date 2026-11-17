"""Checks that the lab's PyTorch model is the team's deployed model (run from the repository root):

    python -m neural_lab.verify            # all checks (needs the dataset for 3 and 4)

1. the published weights load into the notebook's ConvAE with strict=True (718,369 parameters)
2. the ONNX file on Hugging Face computes exactly what 02_model_server/anomaly_ae.onnx computes
   (they are separate exports of the same weights, so the bytes differ; the outputs must not)
3. PyTorch and ONNX Runtime give the same reconstructions and scores on the team's test tiles
4. re-scoring the validation/test welds reproduces the published threshold and ROC AUC
   (only inference: no retraining, same preprocessing, same metric definitions as the notebook)
"""
import hashlib
import json

import numpy as np
import torch

from .model import SERVER_DIR, hf_file, load_pretrained, n_params, preprocess

EXPECTED_PARAMS = 718_369


def sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_tiles(rel_paths, size=64) -> torch.Tensor:
    from .data import tile_path
    return torch.cat([preprocess(tile_path(p), size) for p in rel_paths])


@torch.no_grad()
def scores_torch(model, X, batch=256):
    out = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]
        out.append(((model(xb) - xb) ** 2).mean(dim=(1, 2, 3)))
    return torch.cat(out).numpy()


def onnx_session(path=None):
    import onnxruntime as ort
    return ort.InferenceSession(str(path or hf_file("anomaly_ae.onnx")), providers=["CPUExecutionProvider"])


def compare_onnx(model, X: torch.Tensor, session=None) -> dict:
    """Reconstructions and scores of PyTorch (CPU) vs ONNX Runtime on the same input batch."""
    session = session or onnx_session()
    xb = X.numpy()
    rec_onnx = np.concatenate([session.run(None, {"image": xb[i:i + 256]})[0] for i in range(0, len(xb), 256)])
    with torch.no_grad():
        rec_torch = torch.cat([model(X[i:i + 256]) for i in range(0, len(X), 256)]).numpy()
    s_onnx = ((rec_onnx - xb) ** 2).mean(axis=(1, 2, 3))
    s_torch = ((rec_torch - xb) ** 2).mean(axis=(1, 2, 3))
    return {"n": len(xb), "max_abs_diff_reconstruction": float(np.abs(rec_onnx - rec_torch).max()),
            "max_abs_diff_score": float(np.abs(s_onnx - s_torch).max()),
            "scores_allclose": bool(np.allclose(s_onnx, s_torch, rtol=1e-5, atol=1e-8))}


def compare_onnx_files(path_a, path_b, X: torch.Tensor) -> dict:
    """Two ONNX exports of the same weights are not byte-identical (protobuf serialisation differs between
    exports), so compare what they compute: same inputs/outputs and the same reconstructions on X."""
    a, b = onnx_session(path_a), onnx_session(path_b)
    io = lambda s: ([(i.name, i.shape) for i in s.get_inputs()], [(o.name, o.shape) for o in s.get_outputs()])
    xb = X.numpy()
    ra = np.concatenate([a.run(None, {"image": xb[i:i + 256]})[0] for i in range(0, len(xb), 256)])
    rb = np.concatenate([b.run(None, {"image": xb[i:i + 256]})[0] for i in range(0, len(xb), 256)])
    return {"n": len(xb), "same_io": io(a) == io(b), "max_abs_diff": float(np.abs(ra - rb).max()),
            "byte_identical": sha256(path_a) == sha256(path_b)}


def reproduce_metrics(model, config) -> dict:
    """Notebook SECTIONS 9-10, inference only: threshold = 95th percentile of sound validation scores."""
    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    from .data import evaluation_sets, load_split
    _, val, test = evaluation_sets(load_split())
    s_val = scores_torch(model, load_tiles(val["path"]))
    s_test = scores_torch(model, load_tiles(test["path"]))
    y_val = (val["label"] != "no_defect").to_numpy().astype(int)
    y_test = (test["label"] != "no_defect").to_numpy().astype(int)
    thr = float(np.quantile(s_val[y_val == 0], 0.95))
    pred = (s_test > config["threshold"]).astype(int)
    return {"n_val": len(val), "n_test": len(test),
            "threshold_reproduced": thr, "threshold_published": config["threshold"],
            "threshold_rel_diff": abs(thr - config["threshold"]) / config["threshold"],
            "val_roc_auc": float(roc_auc_score(y_val, s_val)), "test_roc_auc": float(roc_auc_score(y_test, s_test)),
            "test_precision": float(precision_score(y_test, pred)), "test_recall": float(recall_score(y_test, pred)),
            "published": {k: config[k] for k in ["validation_roc_auc", "test_roc_auc", "test_precision", "test_recall"]}}


def main():
    torch.set_grad_enabled(False)
    model, config = load_pretrained("cpu")
    print(f"1. weights loaded (strict): {n_params(model):,} parameters "
          f"({'ok' if n_params(model) == EXPECTED_PARAMS else 'EXPECTED 718,369'})")

    from .data import evaluation_sets, load_split
    _, _, test = evaluation_sets(load_split())
    X = load_tiles(test["path"])

    server_onnx = SERVER_DIR / "anomaly_ae.onnx"
    eq = compare_onnx_files(hf_file("anomaly_ae.onnx"), server_onnx, X)
    print(f"2. HF ONNX vs 02_model_server/anomaly_ae.onnx on {eq['n']} test tiles: same inputs/outputs {eq['same_io']}, "
          f"max |Δ reconstruction| {eq['max_abs_diff']:.2e} (byte-identical file: {eq['byte_identical']}, "
          f"separate exports of the same weights)")
    server_cfg = json.loads((SERVER_DIR / "config.json").read_text())
    print(f"   HF config == server config: {server_cfg == config}")

    res = compare_onnx(model, X, onnx_session(server_onnx))
    print(f"3. PyTorch vs ONNX on {res['n']} test tiles: max |Δ reconstruction| {res['max_abs_diff_reconstruction']:.2e}, "
          f"max |Δ score| {res['max_abs_diff_score']:.2e}, allclose {res['scores_allclose']}")

    m = reproduce_metrics(model, config)
    print(f"4. threshold reproduced {m['threshold_reproduced']:.9f} vs published {m['threshold_published']:.9f} "
          f"(rel. diff {m['threshold_rel_diff']:.2%})")
    print(f"   ROC AUC val {m['val_roc_auc']:.3f} (published {config['validation_roc_auc']}), "
          f"test {m['test_roc_auc']:.3f} (published {config['test_roc_auc']})")
    print(f"   test precision {m['test_precision']:.3f} / recall {m['test_recall']:.3f} "
          f"(published {config['test_precision']} / {config['test_recall']})")


if __name__ == "__main__":
    main()
