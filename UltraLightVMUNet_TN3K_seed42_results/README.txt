UltraLight VM-UNet on TN3K - trained models and results (fold 0, seed 42)
==========================================================================

Trained on an RTX 3050 6GB with the official mamba_ssm CUDA kernels
(mamba-ssm 2.3.2, causal-conv1d 1.7.0, torch 2.7.1+cu126), 200 epochs each.
Code commit: 24e8160 (+ the changes in code_changes.patch).

Start with results/RESEARCH_FINDINGS.md.

models/<experiment>/
  best_ema.pth       EMA weights at the best validation-Dice epoch.
                     THESE ARE THE WEIGHTS USED FOR ALL TEST RESULTS.
  best_model.pth     Same epoch: raw + EMA weights, metrics, config.
  latest_model.pth   Epoch 200: weights + optimizer/scheduler (to resume).
  config_used.yaml   Exact training config.
  training_log.csv   Per-epoch losses and validation metrics.

  e1_baseline     segmentation loss only
  e2_boundary     + scale-aware boundary loss (lambda 0.3)
  e3_consistency  + corruption-aware consistency loss (lambda 1.0)
  e4_combined     boundary + consistency

results/
  RESEARCH_FINDINGS.md               summary, ablation, significance, limitations
  failure_analysis.md                failure modes by lesion scale
  final_test_comparison.csv          test metrics (614 test images)
  lesion_scale_analysis.csv          Dice / BIoU by small / medium / large
  ablation_study_summary.csv         deltas vs E1 + paired-bootstrap 95% CI
  model_efficiency_benchmark.csv     params, FLOPs, GPU latency
  training_summary.csv, all_runs_summary.csv
  training_curves_E1_E4.png          12-panel training curves
  qualitative_*.png / qualitative_cases.csv   visual comparisons, per-case Dice

Loading a model (inside the repo):
  generate_deliverables.load_model_from_checkpoint("models/e4_combined/best_ema.pth", "cuda")
  Requires CUDA: the mamba_ssm kernels do not run on CPU.

code_changes.patch (apply in the repo with: git apply code_changes.patch)
  train.py              ScaleAwareBoundaryLoss argument r= -> radius=; NUM_WORKERS env override
  losses/seg_losses.py  missing "import random"
  utils/metrics.py      CPU latency benchmark no longer crashes with CUDA-only mamba_ssm
  generate_deliverables.py  missing MetricTracker import; now also writes
                        ablation_study_summary.csv, RESEARCH_FINDINGS.md, failure_analysis.md
                        (--reports_only regenerates just those)
