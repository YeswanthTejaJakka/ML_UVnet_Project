# Limitations and Future Work

## Current Limitations

### 1. Single Seed, Single Fold
All experiments use seed 42 and fold 0 of TN3K. The observed test-Dice differences
between models (~0.01) are of the same order as typical seed-to-seed variance (usually
0.005–0.015 in similar-sized models). Therefore, the ranking of E1 > E4 > E2 > E3 on
clean test Dice may not hold under different random initialisations. No across-run
variance estimate is available from the current data alone.

### 2. Single Dataset (TN3K)
All training and evaluation are performed on TN3K (3,493 images, one scanner type).
No second independent thyroid ultrasound dataset was available locally. The LSBP_TN3K
project found in the same workspace uses the identical TN3K data, so it does not
constitute an independent evaluation set. Generalisation to other devices, acquisition
protocols, or patient populations is unknown.

### 3. Synthetic Corruptions as a Proxy for Real Noise
The robustness suite evaluates models on synthetic corruptions (speckle, Gaussian,
contrast attenuation, Poisson) applied uniformly at inference time. Real clinical
ultrasound noise is device-specific, spatially structured (multiplicative AND coherent),
and confounded with legitimate tissue heterogeneity. Synthetic benchmarks are useful
proxies but do not guarantee real-world robustness.

### 4. Fixed-Severity Test vs. Random-Severity Training
E3/E4 saw per-image random severity during training (σ ~ U(0.05, 0.30); factor ~ U(0.5, 1.0)).
The robustness suite tests at fixed severity levels. The in-distribution levels are within
the training range, but the fixed-vs-random mismatch means that even "in-distribution"
test conditions may not perfectly match training conditions.

### 5. Capacity Constraints
The 49,457-parameter model has limited representational capacity. Even if the auxiliary
training objectives are well-designed, the model may lack the capacity to fully exploit
the additional supervision signal, explaining why clean test Dice differences are small.

### 6. FLOPs Measurement
Reported GFLOPs (0.060 G) count only conv/linear layers via `thop`. The mamba_ssm
selective-scan CUDA kernel contributes additional compute not reflected in this number.
The figure is a convention-based lower bound, not a full hardware-operation count.

## What We Would Do Next

1. **Multi-seed evaluation** (seeds 42, 43, 44 at minimum): train E1–E4 three times each
   and report mean ± std of test Dice. This is the most critical missing piece for firm
   conclusions.

2. **Cross-dataset transfer**: with the existing E1–E4 checkpoints (zero-shot), evaluate
   on TNSCUI2020 or DDTI using identical preprocessing and metrics. No retraining; this
   would directly test whether the auxiliary losses improve OOD generalisation.

3. **Augmentation-only control**: train an E5 that applies the same speckle/contrast
   augmentations as E3 to the input WITHOUT the consistency loss objective. This separates
   the effect of data augmentation from the consistency regularisation loss, testing
   whether the consistency loss itself (not just the corrupted views) is responsible
   for any robustness gains.

4. **Longer training for E2**: E2 peaked at epoch 199 (the final epoch), suggesting
   it may benefit from 250–300 epochs or a different LR schedule.

5. **Lambda hyperparameter tuning from Stage B**: the brief notes Stage B tuning results
   were not used to select the final λ values. Using the best λ from validation could
   potentially improve E2/E3/E4 test performance.
