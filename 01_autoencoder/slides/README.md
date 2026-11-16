# Tuesday slides: weld anomaly autoencoder

The homework for Tuesday from the teacher's notebook (`hackathon/docs/01_autoencoders.ipynb`, section 10): three slides on (1) the dataset and the setups compared, (2) one figure and its interpretation, (3) one app idea.

**Online deck (present, export to PowerPoint or PDF):** https://claude.ai/artifact/4pbiHwZwdUymJPP8wXH8ga. Private: the owner shares it from its Share menu.

| # | File | Slide |
|---|---|---|
| 0 | [slides/cover.html](slides/cover.html) | Cover: a real weld X-ray stitched from 87 dataset tiles |
| 1 | [slides/setups.html](slides/setups.html) | Dataset and setups: the 4 classes, a bar chart of validation ROC AUC, key numbers |
| 1b | [slides/table.html](slides/table.html) | The full comparison table: parameters, training time, errors, AUC, precision, recall |
| 2 | [slides/seeing.html](slides/seeing.html) | What we see: input, rebuilt tile and error map, with the interpretation |
| 3 | [slides/app.html](slides/app.html) | App idea: who, what they upload, what they get back (low / check / high plus heatmap) |

Each slide file has its **speaker notes** in the `<aside>` at the end.

## Files

- `deck.json`: the slide order, title and fonts
- `deck.css`: shared styles (white and grey, Inter + JetBrains Mono, one steel-blue accent)
- `slides/*.html`: one file per slide

The images (`/_blob/...`) are stored in the online deck, not in this repo: they show RIAWELC X-ray tiles, which must not be re-uploaded. The figures themselves are produced by the notebook in [../figures/](../figures/).

All numbers on the slides come from [../notebook/autoencoder.ipynb](../notebook/autoencoder.ipynb) and match the Monday README.
