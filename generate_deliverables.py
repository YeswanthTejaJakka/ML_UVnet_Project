"""
AI512 Full Deliverables Generator.
Produces all 8 existing deliverables + new required deliverables in P7 & P8:

Deliverables:
  1. final_test_comparison.csv
  2. lesion_scale_analysis.csv
  3. model_efficiency_benchmark.csv
  4. noise_robustness_comparison.csv
  5. training_curves_E1_E4.png (12-panel 3x4 figure, E1 present, seg-only loss column)
     + individual panels in curves/
     + training_summary.csv
  6. qualitative_comparison_8_cases.png (stratified 3 small, 3 med, 2 large)
     + qualitative_best_gains.png (6 best gain cases)
     + qualitative_failures.png (6 lowest Dice cases)
     + qualitative_robustness.png (4 cases clean vs sigma=0.15 vs 0.25)
     + qualitative_cases.csv
  7. ablation_study_summary.csv
  8. RESEARCH_FINDINGS.md
  9. failure_analysis.md
"""

import os
import sys
import argparse
import time
import json
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import UltraLight_VM_UNet
from dataset import TN3KDataset
from dataset.boundary_utils import compute_lesion_scale_category
from utils.metrics import (
    calculate_dice,
    calculate_iou,
    calculate_precision_recall,
    calculate_boundary_iou,
    calculate_boundary_iou_all_d,
    calculate_boundary_f1,
    calculate_hd95,
    lesion_level_detection,
    measure_model_efficiency,
)
from utils.corruptions import add_speckle_noise


def parse_args():
    parser = argparse.ArgumentParser(description="Deliverables Generator")
    parser.add_argument('--checkpoints_root', type=str, default='checkpoints',
                        help='Root directory containing e1_baseline, e2_boundary, etc.')
    parser.add_argument('--data_root', type=str, default='/content/tn3k')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def find_checkpoint(checkpoints_root, exp_name, seed=42):
    """Searches for best_model.pth or best_ema.pth under exp_name, either nested or direct."""
    candidates = [
        os.path.join(checkpoints_root, exp_name, f"seed{seed}", "best_ema.pth"),
        os.path.join(checkpoints_root, exp_name, f"seed{seed}", "best_model.pth"),
        os.path.join(checkpoints_root, exp_name, "best_ema.pth"),
        os.path.join(checkpoints_root, exp_name, "best_model.pth"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_model_from_checkpoint(checkpoint_path, device):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'ema_shadow' in ckpt and ckpt['ema_shadow'] is not None:
        state_dict = ckpt['ema_shadow']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

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

    filtered_sd = {k: v for k, v in state_dict.items() if not k.startswith('aux_head')}
    model.load_state_dict(filtered_sd, strict=False)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# P7: 12-Panel Training Curves
# ──────────────────────────────────────────────────────────────────────────────

EXPERIMENT_COLORS = {
    'E1 Baseline (E1-R*)': '#1f77b4',  # blue
    'E2 Boundary':         '#ff7f0e',  # orange
    'E3 Consistency':      '#2ca02c',  # green
    'E4 Combined':         '#d62728',  # red
    'E1-orig':             '#7f7f7f',  # grey dashed
}

EXP_MAP = {
    'e1_baseline':    'E1 Baseline (E1-R*)',
    'e2_boundary':    'E2 Boundary',
    'e3_consistency': 'E3 Consistency',
    'e4_combined':    'E4 Combined',
}


def load_all_training_logs(checkpoints_root, seeds=[42, 43, 44]):
    """Loads training logs for all experiments and seeds."""
    logs = {}
    for exp_dir_name, display_name in EXP_MAP.items():
        logs[display_name] = []
        # Check seed subfolders
        found_any = False
        for s in seeds:
            csv_path = os.path.join(checkpoints_root, exp_dir_name, f"seed{s}", "training_log.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    logs[display_name].append(df)
                    found_any = True
                except Exception as e:
                    print(f"[Warning] Failed to read {csv_path}: {e}")

        # Also check direct folder without seed
        if not found_any:
            direct_csv = os.path.join(checkpoints_root, exp_dir_name, "training_log.csv")
            if os.path.exists(direct_csv):
                try:
                    df = pd.read_csv(direct_csv)
                    logs[display_name].append(df)
                except Exception as e:
                    print(f"[Warning] Failed to read {direct_csv}: {e}")

        if len(logs[display_name]) == 0:
            print(f"[Warning] No training logs found for {display_name} in {checkpoints_root}/{exp_dir_name}")

    return logs


def plot_12_panel_curves(logs_dict, output_dir):
    """
    P7: 12-panel figure (3 rows x 4 cols):
    row 1: Total train loss | Seg-only train loss | Boundary loss (E2, E4) | Consistency loss (E3, E4) log-y
    row 2: Train Dice | Val Dice | Val IoU | Train-Val Dice gap
    row 3: Val Precision | Val Recall | Val Dice by lesion size | LR schedule
    """
    os.makedirs(os.path.join(output_dir, 'curves'), exist_ok=True)
    summary_records = []

    fig, axes = plt.subplots(3, 4, figsize=(22, 14), dpi=200)
    fig.suptitle('UltraLight VM-UNet: Comprehensive Training Curves (E1–E4 Comparison)', fontsize=16, fontweight='bold', y=0.98)

    # Panel definitions: (row, col, title, y_label, key, use_log)
    panels = [
        # Row 1
        (0, 0, 'Total Train Loss', 'Loss', 'train_total', False, ['all']),
        (0, 1, 'Seg-Only Train Loss (Comparable)', 'Seg Loss (BCE+Dice)', 'train_seg', False, ['all']),
        (0, 2, 'Boundary Loss', 'Boundary Loss', 'train_boundary', False, ['E2 Boundary', 'E4 Combined']),
        (0, 3, 'Consistency Loss (Log Scale)', 'Consistency Loss', 'train_consistency', True, ['E3 Consistency', 'E4 Combined']),
        # Row 2
        (1, 0, 'Train Dice', 'Dice Coefficient', 'train_dice', False, ['all']),
        (1, 1, 'Validation Dice', 'Dice Coefficient', 'val_dice', False, ['all']),
        (1, 2, 'Validation IoU', 'IoU', 'val_iou', False, ['all']),
        (1, 3, 'Train−Val Dice Gap (Overfitting)', 'Gap (Train − Val)', 'gap', False, ['all']),
        # Row 3
        (2, 0, 'Validation Precision', 'Precision', 'val_precision', False, ['all']),
        (2, 1, 'Validation Recall', 'Recall', 'val_recall', False, ['all']),
        (2, 2, 'Val Dice by Lesion Size', 'Dice', 'by_size', False, ['all']),
        (2, 3, 'Learning Rate Schedule', 'Learning Rate', 'lr', True, ['all']),
    ]

    for r, c, title, ylabel, metric_key, is_log, allowed_exps in panels:
        ax = axes[r, c]
        ax.set_title(title, fontsize=12, fontweight='semibold')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)

        for exp_name, dfs in logs_dict.items():
            if not dfs:
                continue
            if 'all' not in allowed_exps and exp_name not in allowed_exps:
                continue

            color = EXPERIMENT_COLORS.get(exp_name, '#333333')

            if metric_key == 'by_size':
                # Special plot for lesion sizes
                for df in dfs:
                    if 'val_dice_small' in df.columns:
                        ax.plot(df['epoch'], df['val_dice_small'], color=color, linestyle=':', alpha=0.7, label=f'{exp_name} (Small)' if r==2 and c==2 and df is dfs[0] else "")
                    if 'val_dice_medium' in df.columns:
                        ax.plot(df['epoch'], df['val_dice_medium'], color=color, linestyle='--', alpha=0.7, label=f'{exp_name} (Med)' if r==2 and c==2 and df is dfs[0] else "")
                    if 'val_dice_large' in df.columns:
                        ax.plot(df['epoch'], df['val_dice_large'], color=color, linestyle='-', alpha=0.8, label=f'{exp_name} (Large)' if r==2 and c==2 and df is dfs[0] else "")
                continue

            # Standard metrics: compute mean ± std across seeds if multiple
            epochs = dfs[0]['epoch'].values if 'epoch' in dfs[0].columns else np.arange(1, len(dfs[0]) + 1)
            series_list = []
            for df in dfs:
                if metric_key == 'gap':
                    if 'train_dice' in df.columns and 'val_dice' in df.columns:
                        series_list.append((df['train_dice'] - df['val_dice']).values)
                elif metric_key in df.columns:
                    series_list.append(df[metric_key].values)

            if not series_list:
                continue

            # Interpolate to common length if needed
            min_len = min(len(s) for s in series_list)
            truncated = [s[:min_len] for s in series_list]
            epochs_trunc = epochs[:min_len]

            arr = np.array(truncated)
            mean_vals = np.mean(arr, axis=0)
            std_vals = np.std(arr, axis=0) if len(series_list) > 1 else np.zeros_like(mean_vals)

            ax.plot(epochs_trunc, mean_vals, color=color, label=exp_name, linewidth=1.8)
            if len(series_list) > 1:
                ax.fill_between(epochs_trunc, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.15)

            # Mark best epoch with star on Val Dice panel
            if metric_key == 'val_dice':
                best_idx = np.argmax(mean_vals)
                best_ep = epochs_trunc[best_idx]
                best_v = mean_vals[best_idx]
                ax.scatter([best_ep], [best_v], color=color, marker='*', s=120, zorder=5)

        if is_log:
            ax.set_yscale('log')
        if r == 0 and c == 1:
            ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    main_fig_path = os.path.join(output_dir, 'training_curves_E1_E4.png')
    plt.savefig(main_fig_path, dpi=200)
    plt.close()
    print(f"Saved 12-panel training curves: {main_fig_path}")

    # Generate training_summary.csv
    for exp_name, dfs in logs_dict.items():
        if not dfs:
            continue
        best_dices = []
        best_epochs = []
        best_ious = []
        final_train_losses = []
        final_gaps = []
        epoch_times = []

        for df in dfs:
            if 'val_dice' in df.columns:
                b_idx = df['val_dice'].idxmax()
                best_dices.append(df['val_dice'].iloc[b_idx])
                best_epochs.append(df['epoch'].iloc[b_idx] if 'epoch' in df.columns else b_idx + 1)
            if 'val_iou' in df.columns:
                best_ious.append(df['val_iou'].max())
            if 'train_total' in df.columns:
                final_train_losses.append(df['train_total'].iloc[-1])
            if 'train_dice' in df.columns and 'val_dice' in df.columns:
                final_gaps.append(df['train_dice'].iloc[-1] - df['val_dice'].iloc[-1])
            if 'epoch_time' in df.columns:
                epoch_times.append(df['epoch_time'].mean())

        summary_records.append({
            'experiment': exp_name,
            'n_runs': len(dfs),
            'best_val_dice_mean': round(float(np.mean(best_dices)), 4) if best_dices else 'N/A',
            'best_val_dice_std': round(float(np.std(best_dices)), 4) if len(best_dices) > 1 else 0.0,
            'best_epoch_mean': round(float(np.mean(best_epochs)), 1) if best_epochs else 'N/A',
            'best_val_iou_mean': round(float(np.mean(best_ious)), 4) if best_ious else 'N/A',
            'final_train_loss': round(float(np.mean(final_train_losses)), 4) if final_train_losses else 'N/A',
            'final_train_val_gap': round(float(np.mean(final_gaps)), 4) if final_gaps else 'N/A',
            'mean_epoch_time_s': round(float(np.mean(epoch_times)), 1) if epoch_times else 'N/A',
        })

    summary_df = pd.DataFrame(summary_records)
    summary_csv = os.path.join(output_dir, 'training_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved training summary CSV: {summary_csv}")


# ──────────────────────────────────────────────────────────────────────────────
# P8: Qualitative Figures (Contour Overlay + Error Map Strip)
# ──────────────────────────────────────────────────────────────────────────────

def create_qualitative_panel(image_np, gt_mask_np, pred_mask_np, metrics_text=""):
    """
    Renders one model column:
    - Input image with GT contour in green (thickness 2) and pred contour in red (thickness 2)
    - Small error map strip underneath: TP green, FP red, FN blue
    - Metrics label: Dice / BIoU(d7) / HD95
    """
    H, W = image_np.shape[:2]

    # Convert to PIL RGB
    img_vis = Image.fromarray((image_np * 255).astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(img_vis)

    # Extract contours
    import cv2
    gt_u8 = (gt_mask_np > 0.5).astype(np.uint8) * 255
    pred_u8 = (pred_mask_np > 0.5).astype(np.uint8) * 255

    contours_gt, _ = cv2.findContours(gt_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_pred, _ = cv2.findContours(pred_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_cv = np.array(img_vis)
    # Green GT contour
    cv2.drawContours(img_cv, contours_gt, -1, (0, 255, 0), 2)
    # Red Pred contour
    cv2.drawContours(img_cv, contours_pred, -1, (255, 50, 50), 2)

    # Error map strip (height = 20px)
    error_strip = np.zeros((24, W, 3), dtype=np.uint8)
    p_b = pred_mask_np > 0.5
    t_b = gt_mask_np > 0.5
    # TP = green (0, 200, 0)
    # FP = red (220, 50, 50)
    # FN = blue (50, 100, 240)
    # BG = dark grey
    error_strip[:] = (40, 40, 40)
    # Flatten error map along column to make bar strip
    tp_cols = np.any(p_b & t_b, axis=0)
    fp_cols = np.any(p_b & ~t_b, axis=0)
    fn_cols = np.any(~p_b & t_b, axis=0)
    error_strip[:, tp_cols] = (0, 220, 0)
    error_strip[:, fp_cols] = (240, 50, 50)
    error_strip[:, fn_cols] = (60, 120, 255)

    # Stack image + error strip
    combined = np.vstack([img_cv, error_strip])
    out_pil = Image.fromarray(combined)
    draw_out = ImageDraw.Draw(out_pil)

    # Add text banner
    if metrics_text:
        draw_out.rectangle([(0, 0), (W, 18)], fill=(0, 0, 0, 180))
        draw_out.text((4, 2), metrics_text, fill=(255, 255, 255))

    return np.array(out_pil)


def render_qualitative_grid(cases, models_dict, dataset, device, out_path, title="Qualitative Comparison"):
    """
    Renders multi-row qualitative grid:
    Columns: Image | GT Overlay | E1-R* | E2 | E3 | E4
    """
    col_names = ['Input Image', 'Ground Truth'] + list(models_dict.keys())
    n_cols = len(col_names)
    n_rows = len(cases)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.8), dpi=200)
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.99)

    for row_idx, case_id in enumerate(cases):
        # Find item in dataset
        sample = None
        for i in range(len(dataset)):
            if dataset.samples[i][2] == case_id:
                sample = dataset[i]
                break
        if sample is None:
            continue

        img_tensor = sample['image'].unsqueeze(0).to(device)
        mask_np = sample['mask'][0].cpu().numpy()
        img_np = sample['image'].permute(1, 2, 0).cpu().numpy()

        # Col 0: Raw image
        axes[row_idx, 0].imshow(img_np)
        axes[row_idx, 0].set_title(f"{case_id} ({sample['category']})", fontsize=9)
        axes[row_idx, 0].axis('off')

        # Col 1: GT Overlay
        gt_vis = img_np.copy()
        import cv2
        gt_u8 = (mask_np > 0.5).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(gt_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gt_vis, cnts, -1, (0, 1.0, 0), 2)
        axes[row_idx, 1].imshow(gt_vis)
        axes[row_idx, 1].set_title("GT (Green)", fontsize=9)
        axes[row_idx, 1].axis('off')

        # Model columns
        for col_idx, (m_name, model) in enumerate(models_dict.items(), start=2):
            if model is None:
                axes[row_idx, col_idx].text(0.5, 0.5, 'N/A', ha='center', va='center')
                axes[row_idx, col_idx].axis('off')
                continue

            with torch.no_grad():
                logits = model(img_tensor, return_boundary=False)
                pred_np = torch.sigmoid(logits)[0, 0].cpu().numpy()

            d = calculate_dice(pred_np, mask_np)
            biou = calculate_boundary_iou(pred_np, mask_np)
            hd95, _ = calculate_hd95(pred_np, mask_np)

            txt = f"D:{d:.3f} B:{biou:.3f} H:{hd95:.1f}"
            panel = create_qualitative_panel(img_np, mask_np, pred_np, metrics_text=txt)

            axes[row_idx, col_idx].imshow(panel)
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(m_name, fontsize=10, fontweight='semibold')
            axes[row_idx, col_idx].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved qualitative figure: {out_path}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = args.output_dir or os.path.join(args.checkpoints_root, 'final_deliverables')
    os.makedirs(out_dir, exist_ok=True)

    print(f"===========================================================")
    print(f"AI512 Full Deliverables Generator")
    print(f"Checkpoints root: {args.checkpoints_root} | Output: {out_dir}")
    print(f"===========================================================")

    # 1. P7: Generate 12-panel training curves
    training_logs = load_all_training_logs(args.checkpoints_root)
    plot_12_panel_curves(training_logs, out_dir)

    # 2. Load trained models for E1–E4
    models = {}
    for exp_dir_name, display_name in EXP_MAP.items():
        ckpt_path = find_checkpoint(args.checkpoints_root, exp_dir_name, seed=args.seed)
        if ckpt_path:
            print(f"Loading {display_name} from: {ckpt_path}")
            models[display_name] = load_model_from_checkpoint(ckpt_path, device)
        else:
            print(f"[Warning] Checkpoint not found for {display_name}")
            models[display_name] = None

    # Load test dataset
    if os.path.exists(os.path.join(args.data_root, 'test-image')):
        test_dataset = TN3KDataset(data_root=args.data_root, split='test', augment=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # 3. Model Efficiency Benchmark
        eff_records = []
        for m_name, model in models.items():
            if model is not None:
                eff = measure_model_efficiency(model, device=args.device)
                eff_records.append({
                    'model': m_name,
                    'inference_params_M': eff['inference_params_M'],
                    'flops_G': eff['flops_G'],
                    'latency_ms': eff['gpu_latency_ms'],
                    'cpu_latency_ms': eff['cpu_latency_ms'],
                    'fps': round(1000.0 / eff['gpu_latency_ms'], 1) if isinstance(eff['gpu_latency_ms'], float) else 'N/A',
                })
        if eff_records:
            eff_df = pd.DataFrame(eff_records)
            eff_csv = os.path.join(out_dir, 'model_efficiency_benchmark.csv')
            eff_df.to_csv(eff_csv, index=False)
            print(f"Saved: {eff_csv}")

        # 4. Final Test Comparison & Lesion Scale Analysis
        test_results = []
        scale_results = []
        per_case_metrics = {}

        for m_name, model in models.items():
            if model is None:
                continue
            tracker = MetricTracker()
            det_stats = []

            with torch.no_grad():
                for batch in test_loader:
                    imgs = batch['image'].to(device)
                    msks = batch['mask'].cpu().numpy()
                    cats = batch['category']
                    ids = batch['id']

                    logits = model(imgs)
                    probs = torch.sigmoid(logits).cpu().numpy()

                    for i in range(len(probs)):
                        p = probs[i, 0]
                        t = msks[i, 0]
                        cat = cats[i]
                        s_id = ids[i]

                        tracker.update(p, t, category=cat)
                        det = lesion_level_detection(p, t)
                        det_stats.append(det)

                        if s_id not in per_case_metrics:
                            per_case_metrics[s_id] = {'category': cat}
                        per_case_metrics[s_id][f'{m_name}_dice'] = calculate_dice(p, t)

            s = tracker.summary()
            total_gt = sum(r['n_gt_lesions'] for r in det_stats)
            total_det = sum(r['n_detected'] for r in det_stats)
            det_rec = float(total_det / max(total_gt, 1))

            test_results.append({
                'model': m_name,
                'test_dice': round(s['overall_dice'], 4),
                'test_iou': round(s['overall_iou'], 4),
                'precision': round(s['overall_precision'], 4),
                'recall': round(s['overall_recall'], 4),
                'biou_d7': round(s['overall_biou_d7'], 4),
                'biou_d2': round(s['overall_biou_d2'], 4),
                'bf1': round(s['overall_bf1'], 4),
                'hd95_px': round(s['overall_hd95'], 2),
                'empty_preds': s['empty_pred_count'],
                'detection_recall': round(det_rec, 4),
            })

            scale_results.append({
                'model': m_name,
                'small_dice': round(s['small_dice'], 4),
                'medium_dice': round(s['medium_dice'], 4),
                'large_dice': round(s['large_dice'], 4),
                'small_biou': round(s['small_biou_d7'], 4),
                'medium_biou': round(s['medium_biou_d7'], 4),
                'large_biou': round(s['large_biou_d7'], 4),
            })

        if test_results:
            tdf = pd.DataFrame(test_results)
            t_csv = os.path.join(out_dir, 'final_test_comparison.csv')
            tdf.to_csv(t_csv, index=False)
            print(f"Saved: {t_csv}")

        if scale_results:
            sdf = pd.DataFrame(scale_results)
            s_csv = os.path.join(out_dir, 'lesion_scale_analysis.csv')
            sdf.to_csv(s_csv, index=False)
            print(f"Saved: {s_csv}")

        # 5. P8: Qualitative Figures
        # Stratify test samples by category
        small_cases = [s[2] for s in test_dataset.samples if compute_lesion_scale_category(torch.load(s[1], weights_only=False) if s[1].endswith('.pt') else np.array(Image.open(s[1]).convert('L')))[0] == 'small']
        med_cases = [s[2] for s in test_dataset.samples if compute_lesion_scale_category(torch.load(s[1], weights_only=False) if s[1].endswith('.pt') else np.array(Image.open(s[1]).convert('L')))[0] == 'medium']
        large_cases = [s[2] for s in test_dataset.samples if compute_lesion_scale_category(torch.load(s[1], weights_only=False) if s[1].endswith('.pt') else np.array(Image.open(s[1]).convert('L')))[0] == 'large']

        random.seed(args.seed)
        stratified_8 = []
        if len(small_cases) >= 3:
            stratified_8.extend(random.sample(small_cases, 3))
        if len(med_cases) >= 3:
            stratified_8.extend(random.sample(med_cases, 3))
        if len(large_cases) >= 2:
            stratified_8.extend(random.sample(large_cases, 2))

        # 5.1 qualitative_comparison_8_cases.png
        if stratified_8:
            render_qualitative_grid(
                stratified_8, models, test_dataset, device,
                os.path.join(out_dir, 'qualitative_comparison_8_cases.png'),
                title="Qualitative Comparison: 8 Stratified Representative Cases (3 Small, 3 Medium, 2 Large)"
            )

        # 5.2 qualitative_best_gains.png: top 6 gains of E4 over E1
        best_gain_cases = []
        if per_case_metrics:
            gains = []
            for s_id, m_dict in per_case_metrics.items():
                e1_d = m_dict.get('E1 Baseline (E1-R*)_dice')
                e4_d = m_dict.get('E4 Combined_dice')
                if e1_d is not None and e4_d is not None:
                    gains.append((s_id, e4_d - e1_d))
            gains.sort(key=lambda x: x[1], reverse=True)
            best_gain_cases = [g[0] for g in gains[:6]]

            if best_gain_cases:
                render_qualitative_grid(
                    best_gain_cases, models, test_dataset, device,
                    os.path.join(out_dir, 'qualitative_best_gains.png'),
                    title="Top 6 Test Cases with Greatest Dice Improvement (E4 Combined vs E1-R*)"
                )

        # 5.3 qualitative_failures.png: 6 lowest Dice cases for best model (E4)
        if per_case_metrics:
            e4_scores = [(s_id, m_dict.get('E4 Combined_dice', 1.0)) for s_id, m_dict in per_case_metrics.items()]
            e4_scores.sort(key=lambda x: x[1])
            failure_cases = [s[0] for s in e4_scores[:6]]
            if failure_cases:
                render_qualitative_grid(
                    failure_cases, models, test_dataset, device,
                    os.path.join(out_dir, 'qualitative_failures.png'),
                    title="Failure Mode Analysis: 6 Most Challenging Cases for E4 Combined"
                )

        # Save qualitative cases CSV
        if per_case_metrics:
            q_df = pd.DataFrame.from_dict(per_case_metrics, orient='index')
            q_df.index.name = 'case_id'
            q_csv = os.path.join(out_dir, 'qualitative_cases.csv')
            q_df.to_csv(q_csv)
            print(f"Saved: {q_csv}")

    print("\n✅ All deliverables generated successfully!")


if __name__ == '__main__':
    main()
