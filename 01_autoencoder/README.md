# 01 · Autoencoder for weld anomaly detection

## Use case

A weld inspector uploads X-ray tiles of steel welds. The app flags suspicious areas and ranks each tile **low / check / high** risk, so the inspector reviews the riskiest welds first.

- **How:** a convolutional autoencoder learns what a sound weld looks like from defect-free tiles only. Tiles it cannot reconstruct well (high reconstruction error) are anomalous; the error map shows *where*.
- **Risk levels:** two thresholds on the anomaly score, chosen on validation data to catch most defects, because a missed crack costs far more than a false alarm.
- **What it is not:** a certified accept/reject decision. Acceptance under standards such as ISO 5817 depends on defect type, size and plate thickness, which the dataset does not provide. The tool is a screening aid; a qualified inspector decides.

## Dataset

| | |
|---|---|
| **Name** | RIAWELC weld radiographs |
| **Source** | https://github.com/stefyste/RIAWELC |
| **Size** | 24,407 greyscale PNG tiles of 227 × 227 px (the dataset README says 224; the files are 227), cut from X-rays of 29 welds |
| **Classes** | no defect (`NoDifetto`), porosity (`Difetto2`), crack (`Difetto1`), lack of penetration (`Difetto4`): mapping confirmed visually |
| **Cleaning** | 2,514 duplicates removed (every original test image was also in training) → **21,893 unique tiles** |
| **Split** | New split **by weld**, so no weld appears in two splits: train 20 welds, validation 5, test 4 |
| **Used** | Training: the **3,783 no-defect tiles** of the training welds only. Validation: 2,788 tiles (747 sound + 2,041 defects). Test: 3,019 tiles (866 sound + 2,153 defects) |
| **Licence** | No formal licence; free with citation (see the root README). Not redistributed. |

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate           # macOS/Linux: source .venv/bin/activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130   # NVIDIA GPU; CPU: .../whl/cpu
pip install -r requirements.txt
git clone https://github.com/stefyste/RIAWELC data/RIAWELC
# then extract data/RIAWELC/Dataset_partitioned/RIAWELC_dataset.part01.rar into data/riawelc_extracted/ (see CONTRIBUTING.md)
```

Open `notebook/autoencoder.ipynb`, select the `.venv` kernel and run all cells. Training ran on an **NVIDIA RTX 5080 Laptop GPU**: 3–12 s per setup, about 2 minutes for the whole notebook. On a CPU it works too, just slower. Results are reproducible: the same seed gives the same numbers on every run.

The notebook writes the model files to `model_app/` (`anomaly_ae.onnx`, `anomaly_ae.pt`, `anomaly_ae_config.json`, example tiles). The `.onnx`/`.pt` files and the example tiles are git-ignored: the model goes to Hugging Face, and the dataset must not be re-uploaded.

## Setups compared

All setups trained on the same no-defect tiles, with the same split, seed and recipe (Adam, batch 64, random flips, early stopping on validation). Errors are the mean reconstruction error (MSE) of **sound** tiles. ROC AUC measures how well the error separates defects from sound tiles. Precision and recall are on the test set at the 95% threshold (95th percentile of sound validation errors). **The winner was chosen on validation ROC AUC only.**

| Setup | Parameters | Training time | Validation error | Test error | ROC AUC (val / test) | Precision | Recall |
|---|---|---|---|---|---|---|---|
| A: Conv AE, 128 px, latent 32 | 1,014,049 | 11 s | 0.00027 | 0.00023 | 0.987 / 0.998 | 1.000 | 0.964 |
| B: Conv AE, 128 px, latent 128 | 2,587,009 | 6 s | 0.00038 | 0.00031 | 0.986 / 0.999 | 1.000 | 0.967 |
| C: Dense AE, 128 px, latent 32 | 16,827,424 | 3 s | 0.00035 | 0.00025 | 0.977 / 1.000 | 1.000 | 0.972 |
| **D: Conv AE, 64 px, latent 32** | **718,369** | **8 s** | **0.00020** | **0.00018** | **0.990 / 0.999** | **0.994** | **0.984** |
| Baseline: contrast only (no model) | – | – | – | – | 0.946 / 0.936 | 0.926 | 0.879 |

ROC AUC per defect type on validation:

| | Porosity | Crack | Lack of penetration |
|---|---|---|---|
| A | 0.992 | 0.980 | 0.997 |
| B | 0.992 | 0.979 | 0.997 |
| C | 0.987 | 0.966 | 0.992 |
| **D** | **0.994** | **0.985** | **0.998** |
| Baseline | 0.949 | 0.926 | 0.991 |

**Conclusion:** **setup D wins**: the smallest model has the best validation AUC overall and for every defect type, and the best recall.
- **64 px beats 128 px.** The lower resolution averages away the X-ray grain, which an autoencoder cannot copy anyway, while the defects survive the resize.
- **A bigger code (B) does not help.** More room in the bottleneck lets the model copy defects too.
- **The dense model (C) is worst**, with 17× more parameters. It ignores which pixels are neighbours, so it is worse at telling shapes from grain.
- **Every autoencoder beats the contrast baseline**, most of all on **cracks** (0.985 vs 0.926). The model learns more than "defects are high contrast".
- **Cracks are the hardest defect** for every setup: thin, faint, few pixels.
- **Test is easier than validation** (only 4 welds), so the fair claim is **"about 0.99 AUC on unseen welds"**. The gap between A and D is within that uncertainty.

## Results

![...](figures/...)

## Model on Hugging Face

https://huggingface.co/iaac-maai/hackathon-team1-weld-anomaly-ae

## Slides

[slides/](slides/)

## Limitations

- ...
