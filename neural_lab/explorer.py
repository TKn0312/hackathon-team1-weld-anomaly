"""Shared state for the app and the example renderer: the pretrained model and the team's TEST tiles
(4 welds never used for training or for choosing the model), each with its real anomaly score."""
from functools import cached_property

import numpy as np
import pandas as pd

from . import activations
from .data import CACHE_DIR, ORDER, evaluation_sets, load_split, tile_path
from .model import load_pretrained, preprocess, risk_level


class Explorer:
    def __init__(self, device: str = "cpu"):
        self.model, self.config = load_pretrained(device)
        self.size = self.config["image_size"]

    @cached_property
    def test(self) -> pd.DataFrame:
        """Test tiles with their score (computed once with the model, cached in neural_lab/.cache/)."""
        from .verify import load_tiles, scores_torch
        _, _, test = evaluation_sets(load_split())
        cache = CACHE_DIR / "test_scores.npy"
        if cache.exists() and len(np.load(cache)) == len(test):
            scores = np.load(cache)
        else:
            scores = scores_torch(self.model, load_tiles(test["path"], self.size))
            CACHE_DIR.mkdir(exist_ok=True)
            np.save(cache, scores)
        test = test.copy()
        test["score"] = scores
        test["risk"] = [risk_level(s, self.config) for s in scores]
        return test

    def pick(self, label: str, which: str = "median", rank: int = 0) -> pd.Series:
        """A test tile of one class: 'median' = typical score, 'lowest' / 'highest' = extremes,
        'rank' = position in score order (0 = lowest)."""
        sub = self.test[self.test["label"] == label].sort_values("score", kind="stable").reset_index(drop=True)
        if which == "lowest":
            i = 0
        elif which == "highest":
            i = len(sub) - 1
        elif which == "rank":
            i = int(np.clip(rank, 0, len(sub) - 1))
        else:
            i = len(sub) // 2
        return sub.iloc[i]

    def infer_tile(self, row: pd.Series):
        inf = activations.run(self.model, preprocess(tile_path(row["path"]), self.size))
        return inf, risk_level(inf.score, self.config)

    def infer_image(self, image):
        inf = activations.run(self.model, preprocess(image, self.size))
        return inf, risk_level(inf.score, self.config)

    def class_sizes(self) -> dict:
        return self.test["label"].value_counts().reindex(ORDER).to_dict()
