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
| **Size** | 24,407 greyscale PNG patches, 224 × 224 px; ... used |
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

Open `notebook/autoencoder.ipynb` and run all cells. Training ran on ... and took ...

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
