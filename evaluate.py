"""
P6 — Comprehensive Evaluation Benchmark for UltraLight VM-UNet.
Supports:
  - Standard test evaluation using unified metrics from utils.metrics
  - Boundary IoU (d≈7 headline, plus d=2, d=4) and Boundary F1 (2px)
  - Surface-based HD95 with penalty for empty predictions
  - --eval_original_res: evaluation on original image/mask resolution
  - Lesion-level detection statistics (detection recall, FP blobs, merged nodules)
  - Full robustness suite: speckle sweep, contrast sweep, combined corruption, Gaussian noise
  - Optimal threshold selection on validation set from {0.35 ... 0.65}
  - Model efficiency benchmarking with honest labeling
"""

import os
import sys
import argparse
import json
import time
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import UltraLight_VM_UNet
from dataset import TN3KDataset
from utils.metrics import (
    calculate_dice,
    calculate_iou,
    calculate_precision_recall,
    calculate_boundary_iou,
    calculate_boundary_iou_all_d,
    calculate_boundary_f1,
    calculate_hd95,
    lesion_level_detection,
    MetricTracker,
    measure_model_efficiency,
)
from utils.corruptions import (
    add_speckle_noise,
    reduce_ultrasound_contrast,
    apply_ultrasound_corruption,
)


def parse_args():
    parser = argparse.ArgumentParser(description="UltraLight VM-UNet Evaluation Benchmark")
    parser.add_argument('--weights', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--data_root', type=str, default='/content/tn3k', help='Dataset root path')
    parser.add_argument('--fold', type=int, default=0, help='Fold index')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_original_res', action='store_true', default=False,
                        help='Evaluate on original image/mask resolution as well as 256x256')
    parser.add_argument('--tune_threshold', action='store_true', default=False,
                        help='Tune binarization threshold on validation set')
    parser.add_argument('--save_visuals', action='store_true', default=False,
                        help='Save qualitative prediction overlays')
    return parser.parse_args()


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model(weights_path, device):
    """Loads model weights, stripping any training-only heads for inference."""
    checkpoint = safe_torch_load(weights_path, map_location=device)
    if 'ema_shadow' in checkpoint and checkpoint['ema_shadow'] is not None:
        state_dict = checkpoint['ema_shadow']
        print("  Using EMA weights from checkpoint.")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Check whether aux boundary head was saved
    has_boundary_head = any('boundary_head' in k for k in state_dict.keys())

    model = UltraLight_VM_UNet(
        num_classes=1,
        input_channels=3,
        c_list=[8, 16, 24, 32, 48, 64],
        split_att='fc',
        bridge=True,
        aux_boundary_head=has_boundary_head,
        deep_supervision=False,
        return_logits=True,
    ).to(device)

    # Filter out deep supervision aux_head keys if present
    filtered_sd = {k: v for k, v in state_dict.items() if not k.startswith('aux_head')}
    model.load_state_dict(filtered_sd, strict=False)
    model.eval()
    return model


def find_best_threshold(model, val_loader, device):
    """
    P6.8: Pick optimal probability threshold on the validation split from {0.35 ... 0.65}.
    Strictly uses validation data, never test data.
    """
    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    all_preds = []
    all_masks = []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].cpu().numpy()
            logits = model(images, return_boundary=False)
            probs = torch.sigmoid(logits).cpu().numpy()
            for i in range(len(probs)):
                all_preds.append(probs[i, 0])
                all_masks.append(masks[i, 0])

    best_thresh = 0.50
    best_dice = -1.0
    thresh_scores = {}

    for t in thresholds:
        scores = [calculate_dice(p, m, threshold=t) for p, m in zip(all_preds, all_masks)]
        mean_d = float(np.mean(scores))
        thresh_scores[t] = mean_d
        if mean_d > best_dice:
            best_dice = mean_d
            best_thresh = t

    print(f"Validation threshold sweep: {thresh_scores}")
    print(f"Optimal val threshold: {best_thresh:.2f} (Val Dice: {best_dice:.4f})")
    return best_thresh, thresh_scores


def evaluate_robustness_suite(model, dataset, device, seed=42):
    """
    P6.6: Comprehensive robustness evaluation:
      (a) Speckle noise sweep: sigma in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
      (b) Contrast sweep: factor in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
      (c) Combined corruption: sigma=0.15 + factor 0.75
      (d) Additive Gaussian noise: sigma in [0.02, 0.05, 0.10]
    Reports absolute Dice and retention = Dice_corrupt / Dice_clean.
    Fixed random seed ensures all models see identical corrupted test inputs.
    """
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    results = {}

    # 1. Clean baseline
    clean_dice_list = []
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            masks = batch['mask'].cpu().numpy()
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            for i in range(len(probs)):
                clean_dice_list.append(calculate_dice(probs[i, 0], masks[i, 0]))
    clean_dice = float(np.mean(clean_dice_list))
    results['clean_dice'] = clean_dice
    print(f"\nRobustness suite (clean Dice = {clean_dice:.4f}):")

    # (a) Speckle sweep
    speckle_sigmas = [0.05, 0.10, 0.15, 0.20, 0.25]
    for sigma in speckle_sigmas:
        torch.manual_seed(seed)
        d_list = []
        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(device)
                masks = batch['mask'].cpu().numpy()
                corrupt = add_speckle_noise(images, sigma=sigma)
                logits = model(corrupt)
                probs = torch.sigmoid(logits).cpu().numpy()
                for i in range(len(probs)):
                    d_list.append(calculate_dice(probs[i, 0], masks[i, 0]))
        d_mean = float(np.mean(d_list))
        retention = float(d_mean / (clean_dice + 1e-8))
        results[f'speckle_sigma_{sigma}'] = {'dice': d_mean, 'retention': retention}
        print(f"  Speckle sigma={sigma:.2f}: Dice={d_mean:.4f}, retention={retention:.4f}")

    # (b) Contrast sweep
    contrast_factors = [0.9, 0.8, 0.7, 0.6, 0.5]
    for factor in contrast_factors:
        d_list = []
        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(device)
                masks = batch['mask'].cpu().numpy()
                corrupt = reduce_ultrasound_contrast(images, factor=factor)
                logits = model(corrupt)
                probs = torch.sigmoid(logits).cpu().numpy()
                for i in range(len(probs)):
                    d_list.append(calculate_dice(probs[i, 0], masks[i, 0]))
        d_mean = float(np.mean(d_list))
        retention = float(d_mean / (clean_dice + 1e-8))
        results[f'contrast_factor_{factor}'] = {'dice': d_mean, 'retention': retention}
        print(f"  Contrast factor={factor:.2f}: Dice={d_mean:.4f}, retention={retention:.4f}")

    # (c) Combined: sigma=0.15 + contrast 0.75
    torch.manual_seed(seed)
    d_list = []
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            masks = batch['mask'].cpu().numpy()
            corrupt = apply_ultrasound_corruption(images, speckle_sigma=0.15, contrast_factor=0.75)
            logits = model(corrupt)
            probs = torch.sigmoid(logits).cpu().numpy()
            for i in range(len(probs)):
                d_list.append(calculate_dice(probs[i, 0], masks[i, 0]))
    comb_mean = float(np.mean(d_list))
    comb_ret = float(comb_mean / (clean_dice + 1e-8))
    results['combined_sigma0.15_contrast0.75'] = {'dice': comb_mean, 'retention': comb_ret}
    print(f"  Combined (sigma=0.15, contrast=0.75): Dice={comb_mean:.4f}, retention={comb_ret:.4f}")

    # (d) Additive Gaussian noise
    gaussian_sigmas = [0.02, 0.05, 0.10]
    for g_sigma in gaussian_sigmas:
        torch.manual_seed(seed)
        d_list = []
        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(device)
                masks = batch['mask'].cpu().numpy()
                corrupt = (images + torch.randn_like(images) * g_sigma).clamp(0.0, 1.0)
                logits = model(corrupt)
                probs = torch.sigmoid(logits).cpu().numpy()
                for i in range(len(probs)):
                    d_list.append(calculate_dice(probs[i, 0], masks[i, 0]))
        d_mean = float(np.mean(d_list))
        ret = float(d_mean / (clean_dice + 1e-8))
        results[f'gaussian_sigma_{g_sigma}'] = {'dice': d_mean, 'retention': ret}
        print(f"  Gaussian sigma={g_sigma:.2f}: Dice={d_mean:.4f}, retention={ret:.4f}")

    return results


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"===========================================================")
    print(f"UltraLight VM-UNet Benchmark: {args.weights}")
    print(f"Device: {device} | Data: {args.data_root}")
    print(f"===========================================================")

    model = load_model(args.weights, device)

    # 1. Measure efficiency with honest documentation
    efficiency = measure_model_efficiency(model, device=args.device)
    print("\nModel Complexity & Efficiency:")
    print(f"  Inference Parameters: {efficiency['inference_params_M']} M")
    print(f"  FLOPs: {efficiency['flops_G']} G ({efficiency['flops_G_note']})")
    print(f"  GPU Latency: {efficiency['gpu_latency_ms']} ms ({efficiency['latency_note']})")
    print(f"  CPU Latency: {efficiency['cpu_latency_ms']} ms")

    # 2. Threshold selection on validation split
    val_thresh = 0.50
    thresh_sweep = {}
    if args.tune_threshold:
        val_dataset = TN3KDataset(data_root=args.data_root, split='val', fold=args.fold, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        val_thresh, thresh_sweep = find_best_threshold(model, val_loader, device)

    # 3. Test dataset evaluation
    test_dataset = TN3KDataset(data_root=args.data_root, split='test', augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"\nLoaded Test Set: {len(test_dataset)} samples")

    tracker_default = MetricTracker()
    tracker_tuned = MetricTracker()
    detection_records = []
    per_image_results = []

    # Original resolution evaluation accumulators
    orig_res_dice = []
    orig_res_iou = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].cpu().numpy()
            categories = batch['category']
            sample_ids = batch['id']

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            for i in range(len(probs)):
                p = probs[i, 0]
                t = masks[i, 0]
                cat = categories[i]
                s_id = sample_ids[i]

                # Update standard metrics at threshold 0.5
                tracker_default.update(p, t, category=cat, threshold=0.50)

                # Update at tuned threshold if requested
                if args.tune_threshold:
                    tracker_tuned.update(p, t, category=cat, threshold=val_thresh)

                # Lesion-level detection stats
                det_stat = lesion_level_detection(p, t, threshold=0.50, iou_thresh=0.5)
                det_stat['id'] = s_id
                det_stat['category'] = cat
                detection_records.append(det_stat)

                # Original resolution evaluation
                if args.eval_original_res:
                    orig_mask_path = os.path.join(args.data_root, 'test-mask', s_id)
                    if os.path.exists(orig_mask_path):
                        orig_m = np.array(Image.open(orig_mask_path).convert('L')) > 127
                        H_orig, W_orig = orig_m.shape
                        p_tensor = torch.from_numpy(p).unsqueeze(0).unsqueeze(0)
                        p_upsampled = F.interpolate(p_tensor, size=(H_orig, W_orig), mode='bilinear', align_corners=True)
                        p_orig_np = p_upsampled[0, 0].numpy()
                        orig_res_dice.append(calculate_dice(p_orig_np, orig_m.astype(np.float32), threshold=0.5))
                        orig_res_iou.append(calculate_iou(p_orig_np, orig_m.astype(np.float32), threshold=0.5))

                # Per-image record for bootstrap CI and significance testing
                biou_d = calculate_boundary_iou_all_d(p, t, threshold=0.5)
                hd95_v, _ = calculate_hd95(p, t, threshold=0.5)
                per_image_results.append({
                    'id': s_id,
                    'category': cat,
                    'dice': calculate_dice(p, t, threshold=0.5),
                    'iou': calculate_iou(p, t, threshold=0.5),
                    'biou_d7': biou_d['biou_d_headline'],
                    'biou_d2': biou_d['biou_d2'],
                    'biou_d4': biou_d['biou_d4'],
                    'bf1': calculate_boundary_f1(p, t, threshold=0.5),
                    'hd95': hd95_v,
                })

    summary_default = tracker_default.summary()

    # Aggregate detection statistics
    total_gt = sum(r['n_gt_lesions'] for r in detection_records)
    total_det = sum(r['n_detected'] for r in detection_records)
    total_fp = sum(r['n_fp_blobs'] for r in detection_records)
    total_merged = sum(r['n_merged'] for r in detection_records)
    det_summary = {
        'total_gt_lesions': total_gt,
        'total_detected': total_det,
        'detection_recall': float(total_det / max(total_gt, 1)),
        'total_fp_blobs': total_fp,
        'mean_fp_blobs_per_img': float(total_fp / len(detection_records)),
        'total_merged_nodules': total_merged,
    }

    print("\n================ FINAL TEST BENCHMARK RESULTS (Threshold = 0.50) ================")
    print(f"Overall Dice:        {summary_default['overall_dice']:.4f}")
    print(f"Overall IoU:         {summary_default['overall_iou']:.4f}")
    print(f"Boundary IoU (d≈7):  {summary_default['overall_biou_d7']:.4f}  (d=2: {summary_default['overall_biou_d2']:.4f}, d=4: {summary_default['overall_biou_d4']:.4f})")
    print(f"Boundary F1 (2px):   {summary_default['overall_bf1']:.4f}")
    print(f"Hausdorff 95 (HD95): {summary_default['overall_hd95']:.2f} pixels (Empty predictions: {summary_default['empty_pred_count']})")
    print("--- Lesion-Scale Breakdown ---")
    print(f"Small  (N={summary_default['small_count']}):  Dice: {summary_default['small_dice']:.4f} | IoU: {summary_default['small_iou']:.4f} | BIoU(d7): {summary_default['small_biou_d7']:.4f} | HD95: {summary_default['small_hd95']:.1f}")
    print(f"Medium (N={summary_default['medium_count']}): Dice: {summary_default['medium_dice']:.4f} | IoU: {summary_default['medium_iou']:.4f} | BIoU(d7): {summary_default['medium_biou_d7']:.4f} | HD95: {summary_default['medium_hd95']:.1f}")
    print(f"Large  (N={summary_default['large_count']}):  Dice: {summary_default['large_dice']:.4f} | IoU: {summary_default['large_iou']:.4f} | BIoU(d7): {summary_default['large_biou_d7']:.4f} | HD95: {summary_default['large_hd95']:.1f}")
    print("--- Lesion-Level Detection Statistics ---")
    print(f"Detection Recall: {det_summary['detection_recall']:.4f} ({total_det}/{total_gt} GT nodules detected)")
    print(f"False-Positive Blobs/Image: {det_summary['mean_fp_blobs_per_img']:.2f} (Total: {total_fp})")
    print(f"Merged Multi-Nodules: {total_merged}")

    if args.tune_threshold:
        summary_tuned = tracker_tuned.summary()
        print(f"\nResults at Val-Optimal Threshold ({val_thresh:.2f}):")
        print(f"  Overall Dice: {summary_tuned['overall_dice']:.4f} | IoU: {summary_tuned['overall_iou']:.4f}")

    if args.eval_original_res and orig_res_dice:
        mean_orig_dice = float(np.mean(orig_res_dice))
        mean_orig_iou = float(np.mean(orig_res_iou))
        print(f"\nOriginal Resolution Evaluation:")
        print(f"  Orig-Res Dice: {mean_orig_dice:.4f} | Orig-Res IoU: {mean_orig_iou:.4f}")

    # 4. Robustness testing
    robustness = evaluate_robustness_suite(model, test_dataset, device)

    # 5. Save all results
    final_output = {
        'weights_path': args.weights,
        'efficiency': efficiency,
        'metrics_summary_thresh_0.5': summary_default,
        'detection_summary': det_summary,
        'robustness_suite': robustness,
        'val_optimal_threshold': val_thresh if args.tune_threshold else 0.50,
    }
    if args.tune_threshold:
        final_output['metrics_summary_tuned'] = summary_tuned
        final_output['threshold_sweep_val'] = thresh_sweep
    if args.eval_original_res and orig_res_dice:
        final_output['original_resolution'] = {
            'dice': float(np.mean(orig_res_dice)),
            'iou': float(np.mean(orig_res_iou)),
        }

    out_json = os.path.join(args.output_dir, 'benchmark_results.json')
    with open(out_json, 'w') as f:
        json.dump(final_output, f, indent=2)

    # Save per-image results CSV (for bootstrap CI computation in report)
    per_image_df = pd.DataFrame(per_image_results)
    per_image_csv = os.path.join(args.output_dir, 'per_image_test_results.csv')
    per_image_df.to_csv(per_image_csv, index=False)

    print(f"\nSaved benchmark results to: {out_json}")
    print(f"Saved per-image test results to: {per_image_csv}")


if __name__ == '__main__':
    main()
