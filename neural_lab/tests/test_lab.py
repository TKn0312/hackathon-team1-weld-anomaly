"""Tests for the Neural Network Lab.   Run from the repository root:  python -m pytest neural_lab/tests -q

Fast tests need only the model download. Tests marked 'data' are skipped when RIAWELC is not extracted."""
import json

import numpy as np
import pytest
import torch
from PIL import Image

import neural_lab  # noqa: F401  (sets the Windows OpenMP flag before torch work)
from neural_lab import activations
from neural_lab.data import RAW_DIR, check_split, evaluation_sets, load_split
from neural_lab.model import SERVER_DIR, ConvAE, hf_file, load_config, load_pretrained, n_params, preprocess
from neural_lab.verify import (EXPECTED_PARAMS, compare_onnx, compare_onnx_files, load_tiles, onnx_session,
                               reproduce_metrics)

needs_data = pytest.mark.skipif(not RAW_DIR.is_dir(), reason="RIAWELC not extracted (python -m neural_lab.data)")


@pytest.fixture(scope="session")
def pretrained():
    torch.manual_seed(0)
    return load_pretrained("cpu")


@pytest.fixture(scope="session")
def tile():
    rng = np.random.default_rng(0)  # a synthetic 227x227 greyscale tile: no dataset needed
    return Image.fromarray(rng.integers(40, 220, (227, 227), dtype=np.uint8))


# ---- model -----------------------------------------------------------------------------------------
def test_architecture_matches_published_model(pretrained):
    model, config = pretrained
    assert config["image_size"] == 64 and config["latent_dim"] == 32
    assert n_params(model) == EXPECTED_PARAMS
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())


def test_weights_are_not_modified(pretrained):
    model, _ = pretrained
    published = torch.load(hf_file("anomaly_ae.pt"), map_location="cpu", weights_only=True)
    activations.run(model, torch.rand(1, 1, 64, 64))
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, published[name]), name


def test_config_matches_deployed_server():
    assert load_config() == json.loads((SERVER_DIR / "config.json").read_text())


def test_preprocessing_is_identical_to_server(tile):
    # copy of preprocess() in 02_model_server/app.py (not imported: importing it would start the server)
    grey = tile.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
    server_x = (np.asarray(grey, dtype=np.float32) / 255.0)[None, None]
    lab_x = preprocess(tile, 64).numpy()
    assert lab_x.shape == (1, 1, 64, 64) and lab_x.dtype == np.float32
    assert np.array_equal(lab_x, server_x)


# ---- activations -------------------------------------------------------------------------------------
def test_hooks_record_real_shapes_and_leave_model_clean(pretrained, tile):
    model, _ = pretrained
    x = preprocess(tile)
    inf = activations.run(model, x)
    assert inf.blocks["block1"].shape == (32, 32, 32)
    assert inf.blocks["block2"].shape == (64, 16, 16)
    assert inf.blocks["block3"].shape == (128, 8, 8)
    assert inf.latent.shape == (32,)
    assert all((v >= 0).all() for v in inf.blocks.values()), "ReLU outputs must be >= 0"
    assert all(len(m._forward_hooks) == 0 for m in model.modules()), "hooks must be removed"


def test_hooks_do_not_change_the_output(pretrained, tile):
    model, _ = pretrained
    x = preprocess(tile)
    inf = activations.run(model, x)
    with torch.no_grad():
        plain = model(x)[0, 0].numpy()
        assert np.array_equal(inf.recon, plain)
        assert np.allclose(model.encoder(x)[0].numpy(), inf.latent)
        assert np.allclose(model.decoder(torch.from_numpy(inf.latent)[None])[0, 0].numpy(), plain, atol=1e-6)
    assert np.isclose(inf.score, float(((plain - x[0, 0].numpy()) ** 2).mean()))


def test_onnx_matches_pytorch_on_synthetic_batch(pretrained):
    model, _ = pretrained
    X = torch.rand(16, 1, 64, 64, generator=torch.Generator().manual_seed(1))
    res = compare_onnx(model, X, onnx_session(SERVER_DIR / "anomaly_ae.onnx"))
    assert res["scores_allclose"], res
    assert res["max_abs_diff_reconstruction"] < 1e-4, res


def test_hf_onnx_is_the_deployed_onnx():
    # separate exports of the same weights: bytes differ, computation must not
    X = torch.rand(16, 1, 64, 64, generator=torch.Generator().manual_seed(2))
    res = compare_onnx_files(hf_file("anomaly_ae.onnx"), SERVER_DIR / "anomaly_ae.onnx", X)
    assert res["same_io"], res
    assert res["max_abs_diff"] < 1e-6, res


# ---- with the dataset -----------------------------------------------------------------------------------
@needs_data
def test_split_reproduces_team_notebook():
    check_split(load_split())


@needs_data
def test_onnx_matches_pytorch_on_real_test_tiles(pretrained):
    model, _ = pretrained
    _, _, test = evaluation_sets(load_split())
    res = compare_onnx(model, load_tiles(test["path"]), onnx_session(SERVER_DIR / "anomaly_ae.onnx"))
    assert res["n"] == 3019
    assert res["scores_allclose"], res


@needs_data
def test_published_metrics_are_reproduced(pretrained):
    model, config = pretrained
    m = reproduce_metrics(model, config)
    assert m["threshold_rel_diff"] < 0.01, m          # GPU (training) vs CPU maths: tiny differences allowed
    assert round(m["val_roc_auc"], 3) == config["validation_roc_auc"], m
    assert round(m["test_roc_auc"], 3) == config["test_roc_auc"], m
