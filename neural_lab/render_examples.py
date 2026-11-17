"""Save the lab's figures for one typical (median-score) test tile per class to neural_lab/outputs/
(git-ignored: the figures contain RIAWELC pixels, which must not be redistributed).

    python -m neural_lab.render_examples
"""
from pathlib import Path

from . import plots
from .activations import BLOCKS
from .data import ORDER
from .explorer import Explorer

OUT = Path(__file__).resolve().parent / "outputs"


def main():
    OUT.mkdir(exist_ok=True)
    ex = Explorer("cpu")
    print(f"test tiles per class: {ex.class_sizes()}")
    items = []
    for label in ORDER:
        row = ex.pick(label, "median")
        inf, risk = ex.infer_tile(row)
        assert abs(inf.score - row["score"]) < 1e-6, "single-tile score differs from batch score"
        print(f"{label:20s} {row['source']:24s} score {inf.score:.6f} -> {risk}")
        plots.overview(inf, risk, f"{label} (median test tile)").savefig(OUT / f"{label}_overview.png", dpi=90)
        for b in BLOCKS:
            plots.feature_maps(inf, b, 16).savefig(OUT / f"{label}_{b}.png", dpi=80)
        items.append((label, inf, risk))
    plots.latent([(n, i) for n, i, _ in items]).savefig(OUT / "latent_all_classes.png", dpi=90)
    plots.compare(items).savefig(OUT / "compare_classes.png", dpi=90)
    print(f"saved figures to {OUT}")


if __name__ == "__main__":
    main()
