# NEURAL / LAB · Weld anomaly detection with an autoencoder

**Team 1:** Jerome, Kazuko, Amelia, Sahil
**Course:** AI Methods I – Hackathon (IAAC MaAI 2026/27), 16–20 November 2026
**Topic:** an autoencoder trained on defect-free weld radiographs flags porosity, cracks and lack of penetration by their reconstruction error, as a quality check for structural steel.

## ▶ The app: NEURAL / LAB, in two parts

**Final submission: [04_final/README.md](04_final/README.md)** (use case, live URLs, diagrams, the three statements, limitations) and [04_final/PROOFS.md](04_final/PROOFS.md).

| Part | Address | For |
|---|---|---|
| **NEURAL / LAB · Inspect** | **[hackathon-team1-weld-app.pages.dev](https://hackathon-team1-weld-app.pages.dev)** (Cloudflare, team password + your own account) | **Use** the model: upload a weld X-ray, get LOW / CHECK / HIGH and the error map; every check is saved (D1 + R2), and each user sees only their own. See [03_app/](03_app/) |
| **NEURAL / LAB · Look inside** | **[hackathon-team1-weld-anomaly.onrender.com](https://hackathon-team1-weld-anomaly.onrender.com)** (Render, public) | **Understand** the model: draw defects and look inside the network. "Look inside the network →" in Inspect opens it with the same X-ray (a signed link to it in R2) |

**NEURAL / LAB · Look inside, exploring the black box.** Our trained model runs live in your browser:
- **Defect experiment:** draw a pore, crack or band on a real weld X-ray and watch the score, the risk level (LOW / CHECK / HIGH) and the error heatmap react as you draw.
- **Architecture:** follow one weld through the encoder, the 32-number code and the decoder.
- **Latent universe:** all 3,019 test welds as points in the model's latent space.
- **90-second demo:** a guided tour.

Works on any phone or laptop, nothing to install. The free server sleeps after 15 minutes without visitors, so the first visit then takes about a minute. The lab's code is in [neural_lab/](neural_lab/).

## Getting started

Setup, data download and the team's git workflow: see [CONTRIBUTING.md](CONTRIBUTING.md).

## Links

| Part | Link |
|---|---|
| Model on Hugging Face | [iaac-maai/hackathon-team1-weld-anomaly-ae](https://huggingface.co/iaac-maai/hackathon-team1-weld-anomaly-ae) |
| **NEURAL / LAB · Look inside** | **[hackathon-team1-weld-anomaly.onrender.com](https://hackathon-team1-weld-anomaly.onrender.com)**: draw a defect on a real weld X-ray and watch the network react (sleeps after 15 min idle; first visit ~1 min) |
| Model API (`/predict`) | `https://hackathon-team1-weld-anomaly.onrender.com/gradio`: for programs and the Cloudflare app |
| **NEURAL / LAB · Inspect** (Cloudflare Pages) | **[hackathon-team1-weld-app.pages.dev](https://hackathon-team1-weld-app.pages.dev)**: upload a weld X-ray, get the risk level and the error map; results saved in D1 and R2. Password in the course hand-in |
| Impulse talk slides | [impulse/LINK.md](impulse/LINK.md) |

## Days

| Day | Folder | Status |
|---|---|---|
| Mon 16 Nov | [01_autoencoder/](01_autoencoder/) | done: model trained, compared, published on Hugging Face; Tuesday slides ready |
| Tue 17 Nov | [02_model_server/](02_model_server/) · [neural_lab/](neural_lab/) | done: Render serves NEURAL / LAB as the app, plus the `/predict` API |
| Wed 18 Nov | [03_app/](03_app/) | done: app live on Cloudflare (password, our Render API, D1 + R2), user stories and diagrams |
| Thu 19 Nov | [04_final/](04_final/) | final app with user accounts; diagrams, proofs and the three statements: **[04_final/README.md](04_final/README.md)** |

## Results so far (Monday)

A convolutional autoencoder (64 px, code of 32 numbers, 0.7 M parameters), trained **only on sound welds**, flags defects by how badly it rebuilds them.
- **About 0.99 ROC AUC on unseen welds.** On the 4 test welds: AUC 0.999, recall 0.984.
- **Risk levels:** 91% of sound tiles are "low" (can be skipped), and only 0.5% of defect tiles end up there.
- **The error map shows where the defect is.**

Details, comparison table and figures: [01_autoencoder/README.md](01_autoencoder/README.md).

## Data

[RIAWELC](https://github.com/stefyste/RIAWELC): 24,407 greyscale X-ray patches (227 × 227 px; the dataset README says 224) of steel welds from 29 radiographs, in four classes: no defect, porosity, crack, lack of penetration. No formal licence: the data is released freely with a request to cite Totino, Spagnolo and Perri, "RIAWELC: a novel dataset of radiographic images for automatic weld defects classification" (ICMECE 2022), and Perri et al., "Welding defects classification through a convolutional neural network" (Manufacturing Letters). The full images are not redistributed in this repository or on Hugging Face; only NEURAL / LAB packs 115 test tiles at 64 × 64 px (what the model sees, `02_model_server/lab/data/`).

## AI usage disclosure

| | |
|---|---|
| **Tools** | Claude Code (Anthropic, Claude Opus 5.5) as a coding assistant in VS Code. |
| **Why** | To spend the hackathon week on decisions and understanding rather than on boilerplate: the assistant wrote and ran code, and we directed the work, questioned it and decided. |
| **Where** | **Monday:** we used it to implement the steps we defined: repository setup, data cleaning, the split, the training and comparison code, the figures, code comments and documentation. **Tuesday:** the model server on Render (FastAPI + Gradio `/predict` API, `call_server.py`, `call_api.ipynb`), packing NEURAL / LAB for the same server, and the READMEs. Team members also used it while building the lab itself. **Wednesday:** the Cloudflare app (Pages Functions, D1, R2, password gate, wake-up retries) was built by team members with the assistant; we used it to merge the branches, check the app against our model server, set up deploying from the terminal, write the README and check our Miro diagrams against the code. **Thursday:** the "Look inside" link that opens the same X-ray (a signed link to it in R2), user accounts (hashed passwords, own data only), the database migration, the final diagrams (drawn with code from what the code does), `PROOFS.md` (commands run on the live services), the secret scan and the final README. The three statements are written by us, not by the assistant. |
| **How** | **Our decisions:** <br>• **Look at the data before modelling.** We asked whether it needed cleaning, then had the duplicates removed and examples of every class shown. <br>• **Framing the problem** as a screening tool with **low / check / high** risk levels for an inspector, not an accept/reject verdict. <br>• **Comparing models instead of assuming:** we wanted the best fit, so instead of picking 128 px we had several setups (resolution, code size, dense vs convolutional) trained and compared on validation. <br>• **Checking the plan against the teacher's notebook** and its checklist. <br>• **Designing the pipeline figure** (workflow plus mechanism in one image) and the **encoder vs decoder comparison** of the same tile, to explain how the model works. <br>• **Choosing what to explore next** from the brainstormed experiments. <br>• **One model server, as the teacher asked:** the Cloudflare app calls the same Render API as everything else; we checked that both give identical scores on the same tiles. <br>• **"Don't guess":** for the user stories we used our teammates' text instead of letting the assistant invent them, and had the assistant check a teammate's README against the code, which found and fixed wrong claims (e.g. which dataset images are in the repository). <br>• **Every diagram arrow must exist in the code:** we had each Miro diagram checked against the code. This found drift (calls to the model drawn from the page, which only the Worker makes), and the diagrams were redrawn. <br>• **Following the teacher's hints:** R2 to pass the same X-ray to NEURAL / LAB, and user accounts behind the shared password. <br>• **Product names:** "NEURAL / LAB · Inspect" for the app and "NEURAL / LAB · Look inside" for the lab. <br>• **Privacy over features:** we decided not to add session-replay analytics, because it would record X-rays and screens. <br>• **Nothing goes live unchecked:** deploys and database changes ran only after we approved them, and each was tested on the live site (e.g. user B does not see user A's checks). <br>**The assistant's role:** it wrote and ran the code, proposed implementation options (e.g. the specific setups and thresholds) and explained each step. <br>**How we checked it:** we questioned the results and had every step explained (what the encoder and decoder do, what a "sound weld" means, why 64 px won) before moving on. All numbers come from the notebook, which runs from top to bottom with fixed seeds. For the server and the app, we tested on the live URLs. <br>**What we rejected or changed:** assistant-written user stories (replaced by ours), diagram versions that did not match the code, wording that credited individuals instead of the team, and analytics that would record users' screens. <br>**Using it as a tutor:** we also had the assistant quiz us on the method: how an autoencoder works, how the anomaly score and the error map are computed, why the size of the bottleneck matters, data leakage and splitting by weld, the roles of train / validation / test, and the purpose of a baseline. |

**Lessons learned:**
- **Looking at the data first paid off.** Every test image was also in the training set, and tiles of the same weld were spread over all splits. We only found this because we explored the data before training; otherwise the scores would have been meaningless.
- **Questioning results catches mistakes.** Asking "why?" exposed several issues, all of them fixed:
  - results that changed between runs (a missing random seed)
  - an ONNX check that failed (GPU vs CPU precision)
  - example tiles that happened to be rare misses
- **Comparing beats assuming.** We expected 128 px to win; the comparison showed 64 px was better, because it averages away the X-ray grain.
- **Quizzing ourselves deepened our understanding**, both of how the model works and of how to evaluate it honestly.
- **Documents drift from the code.** Diagrams and READMEs, ours and the assistant's, said things the code did not do. Checking each line against the code, and testing on the live site, was the only reliable way to catch it.
- **Ask for proof, not for "done".** "It works" from the assistant was not enough: a deploy had once not happened at all. Running the live check ourselves (a 401 without login, the same scores from both apps) showed what was really true.
- **Explaining it back is the real test.** We had every part explained in plain words (why a sigmoid at the end of the decoder, why UMAP and not t-SNE, what the Worker does) so that we can explain it ourselves in the finals.
- **AI writes code fast, but people make the decisions.** How many false alarms an inspector should accept, or what the tool may and may not claim, are human decisions.

## License

Code released under the [MIT License](LICENSE). The data keeps its own terms (see above).
