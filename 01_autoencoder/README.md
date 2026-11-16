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

| Setup | Parameters | Training time | Validation error | Test error | ROC AUC | Precision | Recall |
|---|---|---|---|---|---|---|---|
| ... | | | | | | | |

**Conclusion:** ...

## Results

![...](figures/...)

## Model on Hugging Face

https://huggingface.co/iaac-maai/hackathon-team1-weld-anomaly-ae

## Slides

[slides/](slides/)

## Limitations

- ...
