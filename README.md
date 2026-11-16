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
| Mon 16 Nov | [01_autoencoder/](01_autoencoder/) | done: model trained, compared, published on Hugging Face; Tuesday slides ready |
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

[RIAWELC](https://github.com/stefyste/RIAWELC): 24,407 greyscale X-ray patches (227 × 227 px; the dataset README says 224) of steel welds from 29 radiographs, in four classes: no defect, porosity, crack, lack of penetration. No formal licence: the data is released freely with a request to cite Totino, Spagnolo and Perri (2023), "RIAWELC: a novel dataset of radiographic images for automatic weld defects classification", and the second paper listed in the dataset README. The images are not redistributed in this repository or on Hugging Face.

## AI usage disclosure

| | |
|---|---|
| **Tools** | Claude Code (Anthropic, Claude Opus 5.5) as a coding assistant in VS Code. |
| **Why** | To spend the hackathon week on decisions and understanding rather than on boilerplate: the assistant wrote and ran code, and we directed the work, questioned it and decided. |
| **Where** | **Monday:** we used it to implement the steps we defined: repository setup, data cleaning, the split, the training and comparison code, the figures, code comments and documentation. **Tuesday:** ... **Wednesday:** ... **Thursday:** ... |
| **How** | **Our decisions:** <br>• **Look at the data before modelling.** We asked whether it needed cleaning, then had the duplicates removed and examples of every class shown. <br>• **Framing the problem** as a screening tool with **low / check / high** risk levels for an inspector, not an accept/reject verdict. <br>• **Comparing models instead of assuming:** we wanted the best fit, so instead of picking 128 px we had several setups (resolution, code size, dense vs convolutional) trained and compared on validation. <br>• **Checking the plan against the teacher's notebook** and its checklist. <br>• **Designing the pipeline figure** (workflow plus mechanism in one image) and the **encoder vs decoder comparison** of the same tile, to explain how the model works. <br>• **Choosing what to explore next** from the brainstormed experiments. <br>**The assistant's role:** it wrote and ran the code, proposed implementation options (e.g. the specific setups and thresholds) and explained each step. <br>**How we checked it:** we questioned the results and had every step explained (what the encoder and decoder do, what a "sound weld" means, why 64 px won) before moving on. All numbers come from the notebook, which runs from top to bottom with fixed seeds. <br>**Using it as a tutor:** we also had the assistant quiz us on the method: how an autoencoder works, how the anomaly score and the error map are computed, why the size of the bottleneck matters, data leakage and splitting by weld, the roles of train / validation / test, and the purpose of a baseline. |

**Lessons learned:**
- **Looking at the data first paid off.** Every test image was also in the training set, and tiles of the same weld were spread over all splits. We only found this because we explored the data before training; otherwise the scores would have been meaningless.
- **Questioning results catches mistakes.** Asking "why?" exposed several issues, all of them fixed:
  - results that changed between runs (a missing random seed)
  - an ONNX check that failed (GPU vs CPU precision)
  - example tiles that happened to be rare misses
- **Comparing beats assuming.** We expected 128 px to win; the comparison showed 64 px was better, because it averages away the X-ray grain.
- **Quizzing ourselves deepened our understanding**, both of how the model works and of how to evaluate it honestly.
- **AI writes code fast, but people make the decisions.** How many false alarms an inspector should accept, or what the tool may and may not claim, are human decisions.

## License

Code released under the [MIT License](LICENSE). The data keeps its own terms (see above).
