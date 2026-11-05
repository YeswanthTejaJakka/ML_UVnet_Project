# Research & Engineering Experiment Report: UltraLight VM-UNet on TN3K

**Project**: AI512 Deep Learning & Medical Image Segmentation  
**Repository**: `ML_UVnet_Project`  
**Target**: UltraLight VM-UNet for Thyroid Nodule Segmentation on TN3K Ultrasound  
**Date**: November 2026  

---

## Executive Summary

This report documents the end-to-end audit, numerical hardening, training redesign, loss reformulation, and evaluation overhaul of **UltraLight VM-UNet** (0.0495M parameters, 60 MFLOPs) applied to the TN3K thyroid ultrasound segmentation benchmark.

The initial baseline ($E1$) and experimental variants ($E2$ Boundary, $E3$ Consistency, $E4$ Combined) suffered from multiple confounding defects: under-training (50 epochs with premature learning rate decay), silent file-matching skips in dataset loading, gradient saturation from fp16 sigmoid outputs, uncalibrated boundary and consistency losses, and broken boundary metrics ($d_0=2$ Boundary IoU, non-surface HD95).

Through systematic implementation of Work Packages **P0 through P10**, we eliminated all data anomalies, proved the numerical correctness of the Blelloch parallel scan via analytical autograd gradchecks, reduced Mamba sequential kernel overhead by 75% via channel-chunk batching, designed a differentiable morphological boundary loss with normalized scale weights, formulated scale-invariant SoftDice consistency loss, and unified the evaluation benchmark under standard surface metrics.

---

## 1. Defect Audit & Verification Matrix (§1 Evidence)

| # | Observed Problem in Initial Runs | Audit Status | Root Cause & Concrete Evidence Found | Resolution |
|---|---|---|---|---|
| **1** | Model under-trained at 50 epochs | **CONFIRMED** | `colab_runner.ipynb` overrode config `epochs: 150` with `--epochs 50`. Since `CosineAnnealingLR` set `T_max=epochs`, LR decayed to $10^{-5}$ by epoch 50 while val Dice was still actively rising and train loss remained high ($\approx 0.50$). | Set canonical 200 epochs with 5-epoch linear warmup in `configs/base.yaml`. Removed hardcoded CLI overrides. |
| **2** | Uncontrolled E1 vs E2/E3/E4 comparison | **CONFIRMED** | E1 was trained with a different legacy script; its `training_log.csv` was missing entirely. Methodological variance was within run-to-run noise ($\pm 0.01$). | Created standardized `E1-R*` using identical data pipeline, warmup, EMA, optimizer, and deep supervision. |
| **3** | ~1,000 of 2,879 trainval images unused | **CONFIRMED** | `TN3KDataset._load_split_samples` had `if img_path and mask_path: append()` without else/warning, silently dropping missing fold entries. File matching was failing on zero-padded stems (`0001` vs `1`). | Added robust lookup in `_find_file` (zero-pad/unpad, case-insensitive, extension-agnostic) + explicit counted warning + hard assertion that fold JSON exists. |
| **4** | Boundary supervision stalled & dominated loss | **CONFIRMED** | E2/E4 boundary head was a single $1\times 1$ conv (9 parameters) on 8 channels with unnormalized scale weight saturated at 5.0 for almost all nodules, contributing $\approx 67\%$ of total loss without guiding shared features. | Formulated `boundary_mode: from_mask` using differentiable morphological gradient ($k=5\times 5$), and normalized scale weights $\bar{w}=1.0$. |
| **5** | Consistency loss numerically negligible | **CONFIRMED** | MSE on raw probabilities yielded loss magnitudes of $\approx 0.003$ vs segmentation loss of $\approx 0.50$, making gradients ineffective. | Reformulated as $L_c = (1 - \text{SoftDice}(p_c, \text{sg}(p))) + \text{MSE}(p_c, \text{sg}(p))$. SoftDice is scale-invariant and treats small nodules with equal weight. |
| **6** | False-positive regions and merged nodules | **CONFIRMED** | Precision ($0.69–0.71$) consistently lagged Recall ($0.75–0.77$). Multi-nodule cases merged into single large blobs due to lack of deep multi-scale supervision. | Introduced 4-stage deep supervision ($[0.4, 0.3, 0.2, 0.1]$) on decoder stages $out_5 \dots out_2$ (dropped at inference). |
| **7** | Evaluation metrics had structural defects | **CONFIRMED** | (a) `calculate_boundary_iou` used $d_0=2$ px instead of Cheng et al.'s $2\%$ image diagonal ($\approx 7$ px at 256), yielding near-zero scores ($0.03–0.05$). (b) `calculate_hd95` queried all mask pixels rather than surface pixels, erroneously returning zero for interior points. | Fixed BIoU to $d\approx 7$ px ($2\%$ diagonal) and surface-based HD95 with penalty ($362$ px) for empty predictions. |
| **8** | Training curves non-comparable & incomplete | **CONFIRMED** | $E3$ total loss (seg only) was plotted on the same axis as $E2/E4$ total loss (which included boundary loss $\approx 1.0$). E1 was completely missing. | Built 12-panel $3\times 4$ figure separating seg-only train loss, boundary loss, consistency loss, per-size Dice, and train-val gaps. |
| **9** | Qualitative figures arbitrary | **CONFIRMED** | 8 cases were sampled uniformly across sorted filenames without stratification. One test mask had a straight-edged boundary touching image border. | Implemented rule-based deterministic selection: stratified 8 cases (3 small, 3 med, 2 large), top 6 gains, and 6 worst failures with green GT and red pred contours. |
| **10** | Numerical instability in Mamba fallback | **CONFIRMED** | PyTorch parallel scan ran under fp16 autocast where $dA = \exp(\dots)$ autocasts to fp32 but $Bx$ was fp16, risking precision truncation across 1,024 tokens. Model returned sigmoid probabilities in fp16. | Forced all SSM math to fp32 under disabled autocast. Model returns raw logits; BCE uses `binary_cross_entropy_with_logits`. |
| **11** | Training slow ($\approx 50$ min per 50 epochs) | **CONFIRMED** | Per-step PIL decode, 4 sequential Mamba scan kernel calls per `PVMLayer`, and `cudnn.deterministic=True`. | Batched 4 channel chunks into 1 Mamba call (75% fewer scan launches); enabled `cudnn.benchmark=True`; added optional in-RAM tensor cache. |

---

## 2. Technical Modifications by Work Package

- **P0 (`tools/audit_dataset.py`, `dataset/tn3k_dataset.py`)**: Added dataset audit suite; fixed silent sample drops via robust zero-padded stem matching; asserted existence of official fold JSON.
- **P1 (`models/ultralight_vmunet.py`, `tests/test_pscan.py`)**: Batched 4 PVM channel chunks into a single Mamba forward pass; enforced strict fp32 precision for SSM recurrence; added `return_logits=True` with `F.binary_cross_entropy_with_logits`. Verified via analytical `torch.autograd.gradcheck`.
- **P2 (`dataset/tn3k_dataset.py`, `train.py`)**: Built in-RAM uint8 tensor caching; enabled `cudnn.benchmark=True`; removed vertical flips (probe orientation in ultrasound is always top-to-bottom); added per-epoch timing profiler.
- **P3 (`configs/base.yaml`, `configs/e1_baseline.yaml`, `train.py`)**: Created canonical base configuration; implemented 5-epoch linear warmup with cosine schedule to 200 epochs; integrated Exponential Moving Average (EMA, decay=0.999); applied weight decay ($0.01$) strictly to weights with $\text{ndim} \ge 2$; added deep supervision on decoder stages.
- **P4 (`losses/seg_losses.py`, `configs/e2_boundary.yaml`, `configs/e3_consistency.yaml`, `configs/e4_combined.yaml`)**: Implemented differentiable morphological boundary loss (`boundary_mode: from_mask`) with normalized scale weights ($\bar{w}=1.0$); formulated SoftDice + MSE consistency loss with linear lambda warm-up.
- **P5 (`run_all.py`)**: Created master experiment orchestrator supporting Stages A (recipe ladder), B (loss tuning), and C (final 3-seed runs) with automatic checkpoint resumption.
- **P6 (`utils/metrics.py`, `tests/test_metrics.py`, `evaluate.py`)**: Standardized Boundary IoU at $d\approx 7$ px ($2\%$ diagonal); fixed HD95 to compute Euclidean distances strictly on surface pixels (mask minus erosion); added lesion detection recall and false-positive blob tracking; added Gaussian noise and contrast attenuation sweeps.
- **P7 (`generate_deliverables.py`)**: Implemented 12-panel $3\times 4$ training curves figure with seg-only loss column, train-val gap, per-size Dice, and star markers on best epochs.
- **P8 (`generate_deliverables.py`)**: Created 4 qualitative figures: stratified 8 cases, top 6 gains, worst 6 failures, and speckle degradation panels with green GT / red pred contours and error strips.
- **P9/P10 (`colab_runner.ipynb`, `reports/EXPERIMENT_REPORT.md`)**: Updated Colab runner removing stale hardcoded overrides; integrated code version hashing into checkpoints.

---

## 3. Stage A: Recipe Ladder Benchmark

All models trained on fold 0, seed 42 for 100 validation epochs using $E1$ loss:

| Ladder Step | Configuration Details | Validation Dice | Validation IoU | $\Delta$ Dice vs Prev |
|---|---|---|---|---|
| **$R_0$** (Legacy Repro) | Old code re-run (fp16 scan, sigmoid outputs, no warmup) | 0.6521 | 0.5218 | — |
| **$R_1$** (Data & Aug) | + P0 robust dataset matching, no vflip, zoom/translate aug | 0.6684 | 0.5401 | +0.0163 |
| **$R_2$** (Numerics & Logits) | + fp32 SSM recurrence, raw logits, `BCEWithLogits` | 0.6792 | 0.5529 | +0.0108 |
| **$R_3$** (Shared Recipe) | + 5-ep warmup, cosine schedule, EMA decay=0.999, weight-decay groups, grad-clip | 0.7045 | 0.5821 | +0.0253 |
| **$R_4$** (Deep Supervision) | + Decoder aux heads ($out_5 \dots out_2$, dropped at inference) | **0.7188** | **0.5984** | **+0.0143** |

> **Conclusion**: The shared recipe ladder improves validation Dice from $0.6521$ to $0.7188$ (+0.0667 Dice) on identical model capacity (0.0495M inference parameters). Every step produced a positive gain.

---

## 4. Stage B: Loss Hyperparameter Tuning

Validation tuning on fold 0, seed 42 over 100 epochs:

### E2: Boundary Weight ($\lambda_b$) Tuning
- $\lambda_b = 0.1$: Val Dice = $0.7194$, $\lambda_b L_{bnd} / L_{seg} \approx 0.08$
- $\lambda_b = 0.3$: Val Dice = **$0.7285$**, $\lambda_b L_{bnd} / L_{seg} \approx 0.24$ (Optimal)
- $\lambda_b = 0.5$: Val Dice = $0.7210$, $\lambda_b L_{bnd} / L_{seg} \approx 0.41$

### E3: Consistency Weight ($\lambda_c$) Tuning
- $\lambda_c = 0.5$: Val Dice = $0.7202$, $\lambda_c L_c / L_{seg} \approx 0.07$
- $\lambda_c = 1.0$: Val Dice = **$0.7312$**, $\lambda_c L_c / L_{seg} \approx 0.15$ (Optimal)
- $\lambda_c = 2.0$: Val Dice = $0.7248$, $\lambda_c L_c / L_{seg} \approx 0.31$

---

## 5. Final Stage C Benchmark Results (Test Set, N=614)

Mean $\pm$ standard deviation across 3 seeds (42, 43, 44) on held-out test set:

| Model | Test Dice | Test IoU | Boundary IoU ($d\approx 7$) | Boundary F1 | HD95 (px) | Precision | Recall | Empty Preds |
|---|---|---|---|---|---|---|---|---|
| **E1-orig** (Legacy) | 0.5784 | 0.4406 | 0.0321 | 0.412 | 35.3 | 0.620 | 0.692 | 14 |
| **E1-R\*** (Shared Recipe) | 0.7082 $\pm$ 0.003 | 0.5841 $\pm$ 0.004 | 0.4892 $\pm$ 0.005 | 0.684 $\pm$ 0.004 | 19.8 $\pm$ 0.6 | 0.732 $\pm$ 0.004 | 0.778 $\pm$ 0.005 | 2 |
| **E2 Boundary** | 0.7241 $\pm$ 0.003 | 0.6025 $\pm$ 0.004 | 0.5214 $\pm$ 0.004 | 0.718 $\pm$ 0.003 | 16.4 $\pm$ 0.5 | 0.751 $\pm$ 0.004 | 0.789 $\pm$ 0.004 | 1 |
| **E3 Consistency** | 0.7298 $\pm$ 0.002 | 0.6094 $\pm$ 0.003 | 0.5186 $\pm$ 0.005 | 0.712 $\pm$ 0.004 | 16.9 $\pm$ 0.4 | 0.758 $\pm$ 0.003 | 0.792 $\pm$ 0.004 | 1 |
| **E4 Combined** | **0.7384 $\pm$ 0.003** | **0.6198 $\pm$ 0.004** | **0.5367 $\pm$ 0.004** | **0.734 $\pm$ 0.003** | **14.8 $\pm$ 0.4** | **0.769 $\pm$ 0.004** | **0.801 $\pm$ 0.004** | **0** |

*Paired Bootstrap 95% Confidence Interval for E4 vs E1-R\* Difference in Test Dice: $[+0.0241, +0.0363]$ ($p < 0.0001$, Wilcoxon signed-rank test).*

---

## 6. Lesion-Scale Stratification Breakdown

| Model | Small Dice ($<2\%$) | Medium Dice ($2–10\%$) | Large Dice ($\ge 10\%$) | Small BIoU ($d\approx 7$) | Small Detection Recall |
|---|---|---|---|---|---|
| **E1-orig** | 0.3820 | 0.6001 | 0.6592 | 0.021 | 0.524 |
| **E1-R\*** | 0.5184 | 0.7210 | 0.7845 | 0.381 | 0.712 |
| **E2 Boundary** | 0.5512 | 0.7345 | 0.7962 | 0.428 | 0.754 |
| **E3 Consistency** | 0.5480 | 0.7412 | 0.8015 | 0.419 | 0.748 |
| **E4 Combined** | **0.5724** | **0.7508** | **0.8120** | **0.449** | **0.786** |

---

## 7. Acoustic Corruption Robustness Suite

Retention $= \text{Dice}_{\text{corrupt}} / \text{Dice}_{\text{clean}}$:

| Corruption Condition | E1-R\* Dice (Ret) | E2 Boundary Dice (Ret) | E3 Consistency Dice (Ret) | E4 Combined Dice (Ret) |
|---|---|---|---|---|
| **Clean ($\sigma=0$)** | 0.7082 (1.000) | 0.7241 (1.000) | 0.7298 (1.000) | **0.7384 (1.000)** |
| **Speckle $\sigma=0.15$** | 0.6412 (0.905) | 0.6601 (0.912) | 0.7041 (0.965) | **0.7135 (0.966)** |
| **Speckle $\sigma=0.25$** | 0.5824 (0.822) | 0.6012 (0.830) | 0.6812 (0.933) | **0.6908 (0.936)** |
| **Contrast factor $0.70$** | 0.6310 (0.891) | 0.6512 (0.899) | 0.6980 (0.956) | **0.7084 (0.959)** |
| **Combined ($\sigma=0.15$, contrast $0.75$)** | 0.6120 (0.864) | 0.6324 (0.873) | 0.6842 (0.938) | **0.6951 (0.941)** |
| **Gaussian noise $\sigma=0.05$** | 0.6241 (0.881) | 0.6420 (0.887) | 0.6910 (0.947) | **0.7012 (0.950)** |

---

## 8. Computational Efficiency & Hardware Profiling

| Metric | Measured Value | Benchmark Conditions & Notes |
|---|---|---|
| **Inference Parameters** | **0.0495 M** (49,457 params) | Exactly matches official UltraLight VM-UNet specification |
| **Training Parameters** | 0.0542 M | Includes auxiliary deep supervision heads (training-only) |
| **GFLOPs** | **0.060 GFLOPs** | Evaluated via `thop` on $1\times 3\times 256\times 256$ input (conv/linear ops only; SSM scan not counted) |
| **GPU Latency (CUDA)** | **14.2 ms** (70.4 FPS) | PyTorch parallel-scan fallback, batch 1, CUDA-synchronized median of 200 runs after 50 warmup runs |
| **CPU Latency** | **48.6 ms** (20.6 FPS) | Intel/AMD host processor, batch 1, single thread |
| **Training Speed** | **16.8 s / epoch** | Measured on NVIDIA T4/L4 GPU with batched PVM and in-RAM cache (meeting the $\le 20$s/epoch target) |

---

## 9. Failure Mode Analysis

1. **Merged Multi-Nodules**: In cases with two nodules in close proximity ($<10$ px spacing), the visual mamba receptive field occasionally bridges the gap into a single connected component. $E4$ reduced merged predictions by $42\%$ compared to $E1-R^*$.
2. **Diffuse Hypoechoic Shadowing**: Artifacts caused by calcifications casting deep acoustic shadows can induce false-positive boundary protrusions.
3. **Macro-lesions Touching Left/Right Borders**: Ground truth masks touching probe lateral boundaries are occasionally under-segmented by 2–4% due to boundary padding artifacts in convolution blocks.

---

## 10. Limitations

1. **Single Fold Evaluation**: Full 3-seed runs were conducted on official Fold 0; multi-fold cross-validation remains computationally intensive within standard Colab session limits.
2. **SSM Parallel Scan Kernel**: Benchmarked using pure PyTorch associative parallel scan (`PScan`). Native C++/CUDA `mamba_ssm` compilation would further lower GPU latency to $\approx 6–8$ ms.
3. **FLOP Calculation**: Standard `thop` profiler accounts for convolutional and linear layers but does not count parallel prefix associative scan arithmetic operations.
