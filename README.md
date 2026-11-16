# Weld anomaly detection: from autoencoder to live app

**Team 1:** Jerome, Kazuko, Amelia, Sahil
**Course:** AI Methods I – Hackathon (IAAC MaAI 2026/27), 16–20 November 2026
**Topic:** an autoencoder trained on defect-free weld radiographs flags porosity, cracks and lack of penetration by their reconstruction error, as a quality check for structural steel.

## Getting started

Setup, data download and the team's git workflow: see [CONTRIBUTING.md](CONTRIBUTING.md).

## Links

| Part | Link |
|---|---|
| Model on Hugging Face | [iaac-maai/hackathon-team1-weld-anomaly-ae](https://huggingface.co/iaac-maai/hackathon-team1-weld-anomaly-ae) |
| Model server (Render) | Tuesday |
| App (Cloudflare Pages) | Wednesday |
| Impulse talk slides | [impulse/LINK.md](impulse/LINK.md) |

## Days

| Day | Folder | Status |
|---|---|---|
| Mon 16 Nov | [01_autoencoder/](01_autoencoder/) | model trained, compared and published on Hugging Face; slides pending |
| Tue 17 Nov | `02_model_server/` | |
| Wed 18 Nov | `03_app/` | |
| Thu 19 Nov | `04_final/` | |

## Results so far (Monday)

A convolutional autoencoder (64 px, code of 32 numbers, 0.7 M parameters), trained **only on sound welds**, flags defects by how badly it rebuilds them.
- **About 0.99 ROC AUC on unseen welds.** On the 4 test welds: AUC 0.999, recall 0.984.
- **Risk levels:** 91% of sound tiles are "low" (can be skipped), and only 0.5% of defect tiles end up there.
- **The error map shows where the defect is.**

Details, comparison table and figures: [01_autoencoder/README.md](01_autoencoder/README.md).

## Data

[RIAWELC](https://github.com/stefyste/RIAWELC): 24,407 greyscale X-ray patches (224 × 224 px) of steel welds in four classes: no defect, porosity, crack, lack of penetration. No formal licence: the data is released freely with a request to cite Totino, Spagnolo and Perri (2023), "RIAWELC: a novel dataset of radiographic images for automatic weld defects classification", and the second paper listed in the dataset README. The images are not redistributed in this repository or on Hugging Face.

## AI usage disclosure

| | |
|---|---|
| **Tools** | ... |
| **Why** | ... |
| **Where** | Monday: ... Tuesday: ... Wednesday: ... Thursday: ... |
| **How** | ... |

**Lessons learned:** ...

## License

Code released under the [MIT License](LICENSE). The data keeps its own terms (see above).
