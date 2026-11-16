# 01 · Autoencoder for weld anomaly detection

## Use case

...who uses it, what they upload, what the anomaly score means for them...

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
