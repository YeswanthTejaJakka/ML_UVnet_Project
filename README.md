# UltraLight VM-UNet — Thyroid Nodule Segmentation (TN3K)

Adapting **UltraLight VM-UNet** (~0.049M params, ~0.060 GFLOPs) for thyroid ultrasound nodule segmentation on the **TN3K** dataset with lesion-scale-aware boundary supervision and corruption-aware consistency training.

---

## 🔬 What This Project Does

Ultrasound images suffer from speckle noise, acoustic shadowing, and low contrast — especially for small nodules. We introduce three training-time augmentations on top of the UltraLight VM-UNet baseline, with **zero extra cost at inference**:

| Experiment | Description |
|---|---|
| **E1 — Baseline** | Faithful reproduction of UltraLight VM-UNet on TN3K |
| **E2 — Boundary** | Scale-aware boundary loss; small nodules get higher supervision weight |
| **E3 — Consistency** | Speckle/contrast augmentation + latent-space consistency loss during training |
| **E4 — Combined** | E2 + E3 together (best overall for real-world noisy ultrasound) |

> [!IMPORTANT]
> All four experiments share the **exact same inference model** — 49,457 parameters, 0.060 GFLOPs, ~20–34 FPS. No auxiliary heads or extra layers at test time.

---

## 📊 Key Results (Seed 42, Fold 0, TN3K Test Set — 614 images)

| Model | Dice ↑ | IoU ↑ | Boundary F1 ↑ | HD95 ↓ | Speckle ΔAUC | Gaussian ΔAUC |
|---|---|---|---|---|---|---|
| E1 Baseline | **0.7775** | **0.6732** | 0.3611 | **27.56** | — | — |
| E2 Boundary | 0.7690 | 0.6659 | **0.3732** | 28.05 | −0.014 | −0.040 |
| E3 Consistency | 0.7677 | 0.6602 | 0.3393 | 28.54 | +0.021 ✅ | +0.134 ✅ |
| **E4 Combined** | 0.7703 | 0.6653 | 0.3729 | 29.16 | **+0.023 ✅** | **+0.140 ✅** |

E4 also achieves the best **small-lesion Dice (0.658)** and **small-lesion Boundary IoU (0.508)**.

---

## 📂 Repository Structure

```
ML_UVnet/
├── configs/
│   ├── base.yaml              # Shared hyperparameters (all experiments inherit this)
│   ├── e1_baseline.yaml
│   ├── e2_boundary.yaml
│   ├── e3_consistency.yaml
│   └── e4_combined.yaml
├── dataset/
│   ├── tn3k_dataset.py        # TN3K PyTorch Dataset with 5-fold CV
│   └── boundary_utils.py      # Morphological boundary extraction & scale weighting
├── losses/
│   └── seg_losses.py          # BceDiceLoss, ScaleAwareBoundaryLoss, ConsistencyLoss
├── models/
│   └── ultralight_vmunet.py   # UltraLight VM-UNet (dual-mode Mamba SSM)
├── utils/
│   ├── metrics.py             # Dice, IoU, Boundary IoU, HD95, FPS
│   └── corruptions.py         # Speckle & contrast augmentations
├── seed42_results/
│   ├── models/                # Trained checkpoints (E1–E4, best_model.pth + best_ema.pth)
│   └── results/               # All CSVs, PNGs, and analysis markdown files
├── tests/                     # Unit tests for metrics and pscan
├── tools/
│   ├── audit_dataset.py       # Dataset integrity checker
│   └── create_local_runner.py # Regenerates local_runner.ipynb
├── local_runner.ipynb         # ⭐ Main notebook — run this on a GPU machine/Colab
├── train.py                   # Training engine
├── evaluate.py                # Full test benchmark & robustness suite
├── robustness_suite.py        # Corruption sweep (speckle, Gaussian, contrast)
├── run_all.py                 # Runs E1→E4 sequentially
└── generate_deliverables.py   # Generates final report artifacts
```

---

## 🚀 How to Run

### Option A — Google Colab (Recommended, GPU required)

> [!IMPORTANT]
> **You need a GPU runtime.** Go to `Runtime → Change runtime type → GPU` (T4 / L4 / A100).

1. Clone or upload this repo to your Google Drive or Colab session.
2. **Place the TN3K dataset at this exact location inside the repo:**
   ```
   ML_UVnet/
   └── data/
       └── tn3k/
           ├── trainval-image/     ← training images
           ├── trainval-mask/      ← training masks
           ├── test-image/         ← test images
           ├── test-mask/          ← test masks
           └── tn3k-trainval-fold*.json  ← 5 fold JSON files
   ```
3. Open **`local_runner.ipynb`** and run all cells top to bottom.

That's it — the notebook handles installs, training, evaluation, and result generation automatically.

> [!TIP]
> The dataset folder (`data/`) is in `.gitignore` and is **not** included in this repo. Download TN3K from the [official source](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation) and place it as shown above.

---

### Option B — Local CLI (GPU laptop/desktop)

```bash
# Install dependencies
pip install torch torchvision timm einops thop pyyaml pandas numpy scikit-image scipy tqdm

# Train individual experiments (requires GPU)
python train.py --config configs/e1_baseline.yaml --data_root data/tn3k --fold 0
python train.py --config configs/e2_boundary.yaml --data_root data/tn3k --fold 0
python train.py --config configs/e3_consistency.yaml --data_root data/tn3k --fold 0
python train.py --config configs/e4_combined.yaml --data_root data/tn3k --fold 0

# Or run all experiments sequentially
python run_all.py --data_root data/tn3k

# Evaluate a trained model
python evaluate.py --weights seed42_results/models/e4_combined/best_model.pth --data_root data/tn3k
```

---

## 📦 Pre-trained Weights

Trained checkpoints (Seed 42, Fold 0) are included in `seed42_results/models/`:

```
seed42_results/models/
├── e1_baseline/   best_model.pth  best_ema.pth
├── e2_boundary/   best_model.pth  best_ema.pth
├── e3_consistency/ best_model.pth  best_ema.pth
└── e4_combined/   best_model.pth  best_ema.pth
```

To evaluate directly without retraining, just run `evaluate.py` pointing to any of these weights.

---

## 📈 Results & Analysis

All outputs from our Seed 42 run are in `seed42_results/results/`:

| File | Contents |
|---|---|
| `final_analysis_summary.md` | Per-model honest assessment with CI and p-values |
| `RESEARCH_FINDINGS.md` | High-level research conclusions |
| `all_runs_summary.csv` | Dice/IoU/HD95 across all experiments |
| `robustness_summary.csv` | AUC retention under speckle/Gaussian/contrast |
| `training_curves_E1_E4.png` | Training & validation curves |
| `qualitative_comparison_8_cases.png` | Side-by-side segmentation outputs |
| `robustness_curves.png` | Dice vs. corruption severity plots |
| `failure_analysis.md` | Failure case breakdown by lesion scale |

---

## ⚙️ Config Quick Reference

All experiments inherit from `configs/base.yaml`. Key defaults:

- **Image size:** 256×256
- **Epochs:** 200, batch 16, lr 1e-3 → 1e-6 (cosine)
- **EMA decay:** 0.999 (EMA weights used for evaluation)
- **Seed:** 42, Fold: 0

Modify `--fold 0..4` to run cross-validation folds.
