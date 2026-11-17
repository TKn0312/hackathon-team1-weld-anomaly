"""Figures for the lab. Every value drawn here is an array returned by activations.run() (real inference).
Uses matplotlib's object API (no pyplot state), so it is safe inside the Gradio app."""
import numpy as np
from matplotlib.figure import Figure

from .activations import BLOCKS, Inference, strongest_channels

# Same fixed scale as 02_model_server/app.py (ERROR_MAP_MAX): a per-pixel squared error of 0.05 is full
# brightness, so error maps of different tiles can be compared directly.
ERROR_MAP_MAX = 0.05
RISK_COLOURS = {"low": "#55A868", "check": "#F0A030", "high": "#C44E52"}
BLOCK_INFO = {"block1": "block 1 · 32 channels · 32×32", "block2": "block 2 · 64 channels · 16×16",
              "block3": "block 3 · 128 channels · 8×8"}


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([])


def overview(inf: Inference, risk: str, title: str = "") -> Figure:
    fig = Figure(figsize=(13, 3.6), layout="constrained")
    axes = fig.subplots(1, 4)
    axes[0].imshow(inf.x, cmap="gray", vmin=0, vmax=1); axes[0].set_title("input (64×64, 0–1)")
    axes[1].imshow(inf.recon, cmap="gray", vmin=0, vmax=1); axes[1].set_title("reconstruction")
    im = axes[2].imshow(inf.error_map, cmap="inferno", vmin=0, vmax=ERROR_MAP_MAX)
    axes[2].set_title("error map (input − rebuilt)²")
    fig.colorbar(im, ax=axes[2], shrink=0.8, label="squared error")
    axes[3].imshow(inf.x, cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(inf.error_map, cmap="inferno", vmin=0, vmax=ERROR_MAP_MAX, alpha=0.6)
    axes[3].set_title("error over input")
    for ax in axes:
        _clean(ax)
    fig.suptitle(f"{title}   score {inf.score:.6f} → {risk.upper()}", color=RISK_COLOURS.get(risk, "black"),
                 fontsize=13, fontweight="bold")
    return fig


def feature_maps(inf: Inference, block: str, k: int = 16, order: str = "strongest") -> Figure:
    """k channels of one encoder block. 'strongest' = highest mean activation for THIS tile; 'index' = 0..k-1.
    One colour scale per figure (0 to the largest value shown), so channels can be compared."""
    maps = inf.blocks[block]
    k = int(min(k, maps.shape[0]))
    idx = strongest_channels(maps, k) if order == "strongest" else np.arange(k)
    cols = min(8, k)
    rows = int(np.ceil(k / cols))
    fig = Figure(figsize=(1.7 * cols + 1, 1.85 * rows + 0.8), layout="constrained")
    axes = np.atleast_1d(fig.subplots(rows, cols)).ravel()
    vmax = float(maps[idx].max()) or 1.0
    for ax in axes:
        _clean(ax); ax.axis("off")
    for ax, c in zip(axes, idx):
        im = ax.imshow(maps[c], cmap="viridis", vmin=0, vmax=vmax)
        ax.axis("on"); _clean(ax)
        ax.set_title(f"ch {c} · mean {maps[c].mean():.2f}", fontsize=8)
    fig.colorbar(im, ax=list(axes), shrink=0.7, label="activation (after ReLU)")
    active = int((maps.reshape(maps.shape[0], -1).max(axis=1) > 0).sum())
    fig.suptitle(f"{BLOCK_INFO[block]} — {'strongest' if order == 'strongest' else 'first'} {k} channels "
                 f"({active}/{maps.shape[0]} channels non-zero somewhere)", fontsize=11)
    return fig


def latent(items: list[tuple[str, Inference]]) -> Figure:
    """The 32-number code of one or more tiles, as grouped bars."""
    fig = Figure(figsize=(13, 3.4), layout="constrained")
    ax = fig.subplots()
    n = len(items)
    width = 0.8 / n
    dims = np.arange(len(items[0][1].latent))
    for i, (name, inf) in enumerate(items):
        ax.bar(dims + (i - (n - 1) / 2) * width, inf.latent, width, label=name)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(dims)
    ax.set(xlabel="latent dimension (0–31)", ylabel="value", title="Latent code: the 32 numbers the decoder receives")
    if n > 1:
        ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def compare(items: list[tuple[str, Inference, str]]) -> Figure:
    """One row per tile: input, reconstruction, error map, then the channel-average of each encoder block.
    Block images = mean over all channels (a summary, as in the team notebook's 'mechanism' figure)."""
    cols = ["input", "reconstruction", "error map"] + [BLOCK_INFO[b].split(" · ")[0] + "\n(mean of channels)" for b in BLOCKS]
    fig = Figure(figsize=(2.3 * len(cols), 2.5 * len(items) + 0.6), layout="constrained")
    axes = np.atleast_2d(fig.subplots(len(items), len(cols)))
    # shared scale per block across the rows, so sound and defect are directly comparable
    vmax = {b: max(float(inf.blocks[b].mean(axis=0).max()) for _, inf, _ in items) or 1.0 for b in BLOCKS}
    for r, (name, inf, risk) in enumerate(items):
        panels = [(inf.x, "gray", 0, 1), (inf.recon, "gray", 0, 1), (inf.error_map, "inferno", 0, ERROR_MAP_MAX)]
        panels += [(inf.blocks[b].mean(axis=0), "viridis", 0, vmax[b]) for b in BLOCKS]
        for c, (arr, cmap, lo, hi) in enumerate(panels):
            axes[r, c].imshow(arr, cmap=cmap, vmin=lo, vmax=hi)
            _clean(axes[r, c])
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=9)
        axes[r, 0].set_ylabel(f"{name}\nscore {inf.score:.5f}\n{risk.upper()}", fontsize=9,
                              color=RISK_COLOURS.get(risk, "black"))
    fig.suptitle("Sound vs defect: same model, same scales per column", fontsize=12)
    return fig
