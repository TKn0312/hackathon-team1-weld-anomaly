# Working on this repo (Team 1)

## 1. Get access

Accept the invitation to this repository (check your email or <https://github.com/iaac-maai/hackathon-team1-weld-anomaly/invitations>). You also need to be a member of the [iaac-maai](https://github.com/iaac-maai) organization.

## 2. Clone and set up (once)

Requires Git and Python **3.12** (see the [course README](https://github.com/iaac-maai/ai-methods-1-2026#2-before-you-start)).

```bash
git clone https://github.com/iaac-maai/hackathon-team1-weld-anomaly.git
cd hackathon-team1-weld-anomaly

# Windows
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1          # cmd: .venv\Scripts\activate.bat
# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
```

**PyTorch first**, then the rest (same rule as the course `requirements.txt`):

```bash
# NVIDIA GPU (Windows/Linux), incl. RTX 50xx:
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
# Older NVIDIA driver:
#   ... --index-url https://download.pytorch.org/whl/cu126
# No NVIDIA GPU / Mac:
#   pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu   (Mac: no --index-url)

pip install -r 01_autoencoder/requirements.txt
pip install ipykernel
```

Check: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

In VS Code: open a notebook, **Select Kernel → Python Environments → .venv**.

**Data** (not in git: ignored by `.gitignore`):

```bash
git clone https://github.com/stefyste/RIAWELC 01_autoencoder/data/RIAWELC
```

## 3. Daily workflow

```bash
git pull                                  # always before you start
git switch -c <your-name>/<what>          # e.g. kazuko/augmentation
# ... work ...
git status                                # no data, .venv, model files or secrets!
git add <files>
git commit -m "Short description"
git push -u origin <your-name>/<what>
```

Then open a Pull Request on GitHub into `main` and ask a teammate to merge it (or merge it yourself if it's small and nobody else is touching the same file).

## 4. Rules that avoid pain

- **Notebooks merge badly.** Only one person edits a given `.ipynb` at a time. Say in the group chat which notebook you are working on. To experiment, copy it (`autoencoder_<name>.ipynb`) and merge the good parts back.
- Before committing a notebook: **Restart & Run All** so it runs top to bottom, and keep the outputs.
- Never commit: `.venv/`, datasets (`data/`), model weights (`*.pt`, `*.onnx`, `*.joblib`, they go to Hugging Face), `.env`, `kaggle.json`, API keys. `.gitignore` covers these, but check `git status`.
- New package? Add it with its exact version to the day's `requirements.txt` (`pip freeze | findstr <name>`, or `grep` on macOS/Linux) and tell the team.
- Figures for the README go in the day's `figures/` folder as small PNGs. `*.png` is not ignored here, so they can be committed.

## 5. Folder layout

```
01_autoencoder/        Monday: notebook/, data/ (ignored), figures/, models/ (ignored weights), slides/, model_app/
02_model_server/       Tuesday
03_app/                Wednesday
04_final/              Thursday
impulse/               impulse talk link
```
