# Lesion-Scale-Aware Boundary Preservation under Speckle Corruption

Official research implementation adapting **UltraLight VM-UNet** (~0.049M parameters, ~0.060 GFLOPs) for thyroid ultrasound nodule segmentation on the **TN3K** dataset.

---

## 📌 Research Hypotheses & Key Novelties

Ultrasound imaging suffers from low acoustic impedance contrast, acoustic shadowing, and multiplicative speckle corruption. Small nodules are especially vulnerable to boundary erosion and loss vanishing under downsampling.

This framework introduces:
1. **Lesion-Scale-Aware Boundary Supervision (E2)**:
   Extracts boundary bands online from ground-truth masks and dynamically weights boundary loss inversely proportional to nodule area. Small nodules receive higher boundary supervision.
2. **Corruption-Aware Consistency Training (E3)**:
   Simulates realistic ultrasound multiplicative speckle noise ($I_{noisy} = I + I \cdot \eta$) and acoustic contrast attenuation, enforcing latent prediction consistency.
3. **Zero-Overhead Lightweight Inference Guarantee**:
   The inference model remains 100% strictly UltraLight VM-UNet (~0.049M params, ~0.060 GFLOPs). All boundary and consistency objectives operate purely during training.

---

## 📂 Repository Structure

```
ML_UVnet/
├── configs/
│   ├── e1_baseline.yaml       # UltraLight VM-UNet Baseline reproduction
│   ├── e2_boundary.yaml       # Scale-aware boundary supervision
│   ├── e3_consistency.yaml    # Corruption consistency training
│   └── e4_combined.yaml       # Combined proposed framework
├── dataset/
│   ├── __init__.py
│   ├── boundary_utils.py      # Morphological boundary & scale weighting
│   └── tn3k_dataset.py        # TN3K PyTorch Dataset with 5-fold cross validation
├── losses/
│   ├── __init__.py
│   └── seg_losses.py          # BceDiceLoss, ScaleAwareBoundaryLoss, ConsistencyLoss
├── models/
│   ├── __init__.py
│   └── ultralight_vmunet.py   # UltraLight VM-UNet with dual-mode Mamba
├── utils/
│   ├── __init__.py
│   ├── corruptions.py         # Multiplicative speckle & contrast attenuation
│   └── metrics.py             # Dice, IoU, Boundary IoU, HD95, latency, FPS
├── colab_runner.ipynb         # Master Google Colab notebook
├── train.py                   # Unified training engine
├── evaluate.py                # Full test benchmark & robustness suite
├── instructions.txt           # Project specification
└── PROJECT_UNDERSTANDING_AND_PLAN.txt
```

---

## 🚀 Fast Colab Setup (Zero Compilation Hangs)

To run the project on Google Colab:
1. Open Google Colab and upload `colab_runner.ipynb` (or open from GitHub).
2. Set Runtime to **GPU** (T4, L4, or A100).
3. Run the notebook cells sequentially.

---

## 💻 Local CLI Usage

```bash
# Experiment 1: Baseline Reproduction
python train.py --config configs/e1_baseline.yaml --data_root /path/to/tn3k --fold 0

# Experiment 2: Boundary-guided
python train.py --config configs/e2_boundary.yaml --data_root /path/to/tn3k --fold 0

# Experiment 3: Corruption consistency
python train.py --config configs/e3_consistency.yaml --data_root /path/to/tn3k --fold 0

# Experiment 4: Full Proposed Method
python train.py --config configs/e4_combined.yaml --data_root /path/to/tn3k --fold 0

# Evaluation on held-out test set
python evaluate.py --weights checkpoints/e4_combined/best_model.pth --data_root /path/to/tn3k
```
