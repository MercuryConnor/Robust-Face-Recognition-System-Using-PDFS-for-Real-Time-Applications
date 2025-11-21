# Purified Deep Features with Automated Feature Selection (PDFS)

This repository implements PDFS — Purified Deep Features with Automated Feature Selection — a face recognition framework designed for robustness to occlusion and real-time deployment.

**Abstract**

Face recognition is reliable, but occlusions, lighting changes, and limited hardware can reduce consistency. PDFS purifies learned features (Grad-CAM spatial masks, channel reweighting, frequency filtering) and applies automated feature selection (sparsity + evolutionary search) to produce compact, robust embeddings. The system is further optimized with FP16/INT8 quantization and simple graph fusion to achieve real-time inference on desktop GPUs.

**Repository Layout**
- `data/` : datasets and manifests (raw, processed, manifests).
- `models/` : training models, checkpoints, and export artifacts.
- `notebooks/` : analysis and visualization notebooks.
- `scripts/` : training, preprocessing, evaluation, export and quantization utilities.
- `paper/` : paper source and figures.

Key scripts:
- `scripts/train.py` — training and fine-tuning of the backbone with purification losses.
- `scripts/preprocess.py` — dataset preprocessing and alignment using RetinaFace outputs.
- `scripts/evaluate.py` — evaluation on LFW / IJB-B / MaskedFaceNet / RMFRD.
- `scripts/quantize_export.py` — export and quantize models (FP16 / INT8).
- `scripts/gradcam.py` — Grad-CAM utilities used for spatial purification.

Requirements
- Python 3.8+ (recommended)
- See `requirements.txt` for full list.

Installation (example, PowerShell)
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Quickstart — Inference (aligned faces)
1. Prepare an aligned face crop (112x112) using the included detector/alignment pipeline (`scripts/preprocess.py`).
2. Run a simple inference using the exported model (example):

```powershell
python -c "from models import inference; print(inference.embed('aligned_face.png'))"
```

(Replace with the actual inference utility in `models/` or `scripts/` if you use a different API.)

Training Overview
- Backbone: ResNet-IR pretrained on VGGFace2, fine-tuned with ArcFace + Triplet + purification-driven terms.
- Purification modules:
  - Spatial: Grad-CAM based masks to suppress occluded activations.
  - Channel: Lightweight SE/CBAM reweighting.
  - Frequency: Band-pass filtering to regularize feature frequencies.
  - Occlusion consistency: synthetic occlusions and the consistency loss Lcons.
- Optimization: sparsity constraints and evolutionary search to select compact channel subsets.

To train (example):
```powershell
python scripts/train.py --config configs/train_pdfsi.yaml
```

Replace `--config` with the path to your training config. See `scripts/train.py` docstring for options.

Evaluation
- Use `scripts/evaluate.py` to run verification/identification evaluations.

```powershell
python scripts/evaluate.py --dataset LFW --model models/checkpoints/latest.pth
```

Deployment & Optimization
- Export the trained model (TorchScript / ONNX) and apply the following:
  - FP16 conversion for faster GPU inference.
  - INT8 quantization for low-latency CPU/GPU where supported.
  - BatchNorm folding and simple graph-fusion before quantization.
- Script: `scripts/quantize_export.py` (see options for INT8 calibration data, batch sizes).

Performance Notes
- On a desktop GPU (RTX 3060), the optimized PDFS models typically run around 45–75 FPS depending on model size and quantization.
- Automated feature selection can reduce channels ~20–30% with small accuracy loss and large speed gains.

Results Summary (high level)
- LFW: ~99.7% (verification)
- MaskedFace-Net: ~92.5% (verification under mask)
- IJB-B: improved TAR @ FAR=0.001 and Rank-1 vs baselines
- RMFRD: marked robustness under occlusion

Method Details (losses and objectives)
- ArcFace loss for discriminative features.
- Triplet loss for intra-class compactness.
- Occlusion consistency Lcons and frequency regularization Lfreq.
- Sparsity constraint Lsparse for channel pruning.
- Total objective: Ltotal = LArcFace + λt Ltriplet + λc Lcons + λf Lfreq + λs Lsparse.

Reproducibility
- Set random seeds in training configs (see `scripts/train.py` options).
- Use `data/manifests/` to reproduce dataset splits and preprocessing.

Tips / Troubleshooting
- If evaluation is slower than expected, try FP16 export and ensure CUDA/CUDNN are up to date.
- For INT8, provide a representative calibration subset (few hundred images) to `scripts/quantize_export.py`.
- If Grad-CAM masks appear noisy, increase the smoothing on attribution maps or apply morphological post-processing in `scripts/gradcam.py`.

Citations
If you use this work, please cite the original PDFS paper and the following components used in this project:
- ArcFace: Deng et al., 2019.
- RetinaFace: Deng et al., 2019.
- Grad-CAM: Selvaraju et al., 2017.
- SE, CBAM: Hu et al., 2018; Woo et al., 2018.

License & Contact
- See `LICENSE` in the repo root.
- For questions or collaboration: open an issue or contact the authors (see `paper/` for author info).

— End of README —
# PDFS Project

Face detection and analysis pipeline.
