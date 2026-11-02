import os
import argparse
import time
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from models import UltraLight_VM_UNet
from dataset import TN3KDataset
from dataset.boundary_utils import compute_lesion_scale_category
from utils.metrics import (
    calculate_dice,
    calculate_iou,
    calculate_boundary_iou,
    calculate_hd95,
    measure_model_efficiency,
)
from utils.corruptions import add_speckle_noise


def parse_args():
    parser = argparse.ArgumentParser(description="AI512 Full Deliverables Generator")
    parser.add_argument('--checkpoints_root', type=str, default='/content/drive/MyDrive/ML_UVnet_checkpoints',
                        help='Root directory containing e1_baseline, e2_boundary, etc.')
    parser.add_argument('--data_root', type=str, default='/content/tn3k',
                        help='TN3K dataset root directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save deliverable CSVs and PNG figures (defaults to checkpoints_root/final_deliverables)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, device):
    """Safely instantiates UltraLight VM-UNet and loads trained weights."""
    if not os.path.exists(checkpoint_path):
        return None

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)

    # Check if aux_boundary_head weights exist
    has_boundary_head = any('boundary_head' in k for k in state_dict.keys())

    model = UltraLight_VM_UNet(
        num_classes=1,
        input_channels=3,
        c_list=[8, 16, 24, 32, 48, 64],
        split_att='fc',
        bridge=True,
        aux_boundary_head=has_boundary_head,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_precision_recall(pred_binary, target_binary, eps=1e-5):
    p = pred_binary.astype(bool)
    t = target_binary.astype(bool)
    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()

    precision = float((tp + eps) / (tp + fp + eps))
    recall = float((tp + eps) / (tp + fn + eps))
    return precision, recall


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoints_root, 'final_deliverables')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=================================================================")
    print("      AI512 RESEARCH PROJECT - FULL DELIVERABLES GENERATOR        ")
    print("=================================================================")
    print(f"Checkpoints Root: {args.checkpoints_root}")
    print(f"Dataset Root:     {args.data_root}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Evaluation Device:{device}\n")

    # Define model checkpoint mapping
    experiments = {
        'E1_Baseline': os.path.join(args.checkpoints_root, 'e1_baseline', 'best_model.pth'),
        'E2_Boundary': os.path.join(args.checkpoints_root, 'e2_boundary', 'best_model.pth'),
        'E3_Consistency': os.path.join(args.checkpoints_root, 'e3_consistency', 'best_model.pth'),
        'E4_Combined': os.path.join(args.checkpoints_root, 'e4_combined', 'best_model.pth'),
    }

    models = {}
    for exp_name, ckpt_path in experiments.items():
        m = load_model_from_checkpoint(ckpt_path, device)
        if m is not None:
            models[exp_name] = m
            print(f"✅ Loaded {exp_name} from: {ckpt_path}")
        else:
            print(f"⚠️ Checkpoint for {exp_name} not found at: {ckpt_path}")

    if not models:
        print("❌ Error: No trained model checkpoints found. Please train at least one experiment first.")
        return

    # Load test dataset
    test_dataset = TN3KDataset(data_root=args.data_root, split='test', augment=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"\nLoaded official held-out test set: {len(test_dataset)} images.\n")

    # -------------------------------------------------------------
    # DELIVERABLE 1: Training Curves (Loss & Dice over epochs)
    # -------------------------------------------------------------
    print("--> Generating Deliverable 1: Training Curves...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    has_curves = False

    colors = {'e1_baseline': '#1f77b4', 'e2_boundary': '#2ca02c', 'e3_consistency': '#ff7f0e', 'e4_combined': '#d62728'}
    labels = {'e1_baseline': 'E1: Baseline', 'e2_boundary': 'E2: Scale-Aware Boundary',
              'e3_consistency': 'E3: Corruption Consistency', 'e4_combined': 'E4: Combined (Proposed)'}

    for exp_dir_name in ['e1_baseline', 'e2_boundary', 'e3_consistency', 'e4_combined']:
        csv_path = os.path.join(args.checkpoints_root, exp_dir_name, 'training_log.csv')
        if os.path.exists(csv_path):
            df_log = pd.read_csv(csv_path)
            epochs = df_log['epoch']
            axes[0].plot(epochs, df_log['train_loss'], label=labels[exp_dir_name], color=colors[exp_dir_name], linewidth=2)
            axes[1].plot(epochs, df_log['val_dice'], label=labels[exp_dir_name], color=colors[exp_dir_name], linewidth=2)
            has_curves = True

    if has_curves:
        axes[0].set_title("Training Loss Convergence", fontsize=12)
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Loss", fontsize=11)
        axes[0].grid(True, linestyle='--', alpha=0.5)
        axes[0].legend()

        axes[1].set_title("Validation Dice Accuracy", fontsize=12)
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Dice Score", fontsize=11)
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].legend()

        curves_path = os.path.join(args.output_dir, 'training_curves_E1_E4.png')
        plt.tight_layout()
        plt.savefig(curves_path, dpi=200)
        plt.close()
        print(f"    Saved: {curves_path}")

    # -------------------------------------------------------------
    # Predict on test set for all models
    # -------------------------------------------------------------
    print("--> Running Inference on Held-Out Test Set...")
    gt_masks = []
    test_images = []
    test_ids = []
    scale_categories = []
    model_predictions = {name: [] for name in models.keys()}

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['image'].to(device)
            masks_np = batch['mask'].cpu().numpy()[:, 0]
            cats = batch['category']
            ids = batch['id']

            for i in range(len(imgs)):
                gt_masks.append(masks_np[i])
                test_images.append(batch['image'][i])
                test_ids.append(ids[i])
                scale_categories.append(cats[i])

            for name, m in models.items():
                preds = m(imgs, return_boundary=False).cpu().numpy()[:, 0]
                for p in preds:
                    model_predictions[name].append((p > 0.5).astype(np.float32))

    num_test = len(gt_masks)

    # -------------------------------------------------------------
    # DELIVERABLE 2: Overall Final Test Comparison Table
    # -------------------------------------------------------------
    print("--> Generating Deliverable 2: Final Test Comparison Table...")
    test_comparison_rows = []

    for name in models.keys():
        dices, ious, precisions, recalls = [], [], [], []
        preds = model_predictions[name]

        for i in range(num_test):
            d = calculate_dice(preds[i], gt_masks[i])
            iou = calculate_iou(preds[i], gt_masks[i])
            prec, rec = compute_precision_recall(preds[i], gt_masks[i])
            dices.append(d)
            ious.append(iou)
            precisions.append(prec)
            recalls.append(rec)

        test_comparison_rows.append({
            'Model': name,
            'Dice': round(float(np.mean(dices)), 4),
            'IoU': round(float(np.mean(ious)), 4),
            'Precision': round(float(np.mean(precisions)), 4),
            'Recall': round(float(np.mean(recalls)), 4),
        })

    df_test_comp = pd.DataFrame(test_comparison_rows)
    test_comp_csv = os.path.join(args.output_dir, 'final_test_comparison.csv')
    df_test_comp.to_csv(test_comp_csv, index=False)
    print(f"    Saved: {test_comp_csv}")
    print(df_test_comp.to_string(index=False))

    # -------------------------------------------------------------
    # DELIVERABLE 3: Lesion Size Stratification (Small / Med / Large)
    # -------------------------------------------------------------
    print("\n--> Generating Deliverable 3: Lesion-Scale Stratification Table...")
    lesion_size_rows = []

    for name in models.keys():
        preds = model_predictions[name]
        for cat in ['small', 'medium', 'large']:
            indices = [i for i, c in enumerate(scale_categories) if c == cat]
            if indices:
                sub_dices = [calculate_dice(preds[i], gt_masks[i]) for i in indices]
                sub_ious = [calculate_iou(preds[i], gt_masks[i]) for i in indices]
                lesion_size_rows.append({
                    'Model': name,
                    'Lesion_Category': cat.capitalize(),
                    'Count': len(indices),
                    'Dice': round(float(np.mean(sub_dices)), 4),
                    'IoU': round(float(np.mean(sub_ious)), 4),
                })

    df_lesion_size = pd.DataFrame(lesion_size_rows)
    lesion_size_csv = os.path.join(args.output_dir, 'lesion_size_evaluation.csv')
    df_lesion_size.to_csv(lesion_size_csv, index=False)
    print(f"    Saved: {lesion_size_csv}")
    print(df_lesion_size.to_string(index=False))

    # -------------------------------------------------------------
    # DELIVERABLE 4: Boundary Metrics Summary (Boundary IoU & HD95)
    # -------------------------------------------------------------
    print("\n--> Generating Deliverable 4: Boundary Metrics Summary (BIoU & HD95)...")
    boundary_rows = []

    for name in models.keys():
        preds = model_predictions[name]
        bious = [calculate_boundary_iou(preds[i], gt_masks[i], d0=2) for i in range(num_test)]
        hd95s = [calculate_hd95(preds[i], gt_masks[i]) for i in range(num_test)]

        boundary_rows.append({
            'Model': name,
            'Boundary_IoU': round(float(np.mean(bious)), 4),
            'HD95_Pixels': round(float(np.mean(hd95s)), 2),
        })

    df_boundary = pd.DataFrame(boundary_rows)
    boundary_csv = os.path.join(args.output_dir, 'boundary_metrics_summary.csv')
    df_boundary.to_csv(boundary_csv, index=False)
    print(f"    Saved: {boundary_csv}")
    print(df_boundary.to_string(index=False))

    # -------------------------------------------------------------
    # DELIVERABLE 5: Corruption Robustness Under Speckle Noise
    # -------------------------------------------------------------
    print("\n--> Generating Deliverable 5: Corruption Robustness Benchmark...")
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    robustness_rows = []

    for sigma in noise_levels:
        row = {'Speckle_Sigma': sigma}
        for name, m in models.items():
            noisy_dices = []
            with torch.no_grad():
                for batch in test_loader:
                    imgs = batch['image'].to(device)
                    msks = batch['mask'].cpu().numpy()[:, 0]
                    if sigma > 0.0:
                        imgs = add_speckle_noise(imgs, sigma=sigma)
                    preds = (m(imgs, return_boundary=False) > 0.5).float().cpu().numpy()[:, 0]
                    for i in range(len(preds)):
                        noisy_dices.append(calculate_dice(preds[i], msks[i]))
            row[f'{name}_Dice'] = round(float(np.mean(noisy_dices)), 4)
        robustness_rows.append(row)

    df_robustness = pd.DataFrame(robustness_rows)
    robustness_csv = os.path.join(args.output_dir, 'robustness_per_condition.csv')
    df_robustness.to_csv(robustness_csv, index=False)
    print(f"    Saved: {robustness_csv}")
    print(df_robustness.to_string(index=False))

    # Plot Robustness Degradation Curve
    plt.figure(figsize=(8, 5))
    for name in models.keys():
        plt.plot(df_robustness['Speckle_Sigma'], df_robustness[f'{name}_Dice'], marker='o', linewidth=2, label=name)
    plt.title("Ultrasound Speckle Noise Robustness Retention", fontsize=12)
    plt.xlabel("Speckle Noise Level (Sigma)", fontsize=11)
    plt.ylabel("Test Dice Score", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    robustness_png = os.path.join(args.output_dir, 'speckle_robustness_curve.png')
    plt.tight_layout()
    plt.savefig(robustness_png, dpi=200)
    plt.close()
    print(f"    Saved: {robustness_png}")

    # -------------------------------------------------------------
    # DELIVERABLE 6: Model Efficiency & Latency Benchmark
    # -------------------------------------------------------------
    print("\n--> Generating Deliverable 6: Model Efficiency Benchmark...")
    efficiency_rows = []

    for name, m in models.items():
        eff = measure_model_efficiency(m, device=args.device)
        efficiency_rows.append({
            'Model': name,
            'Parameters_M': eff['params_M'],
            'FLOPs_G': eff['flops_G'],
            'Latency_ms': eff['latency_ms'],
            'FPS': eff['fps'],
        })

    df_efficiency = pd.DataFrame(efficiency_rows)
    efficiency_csv = os.path.join(args.output_dir, 'efficiency_results.csv')
    df_efficiency.to_csv(efficiency_csv, index=False)
    print(f"    Saved: {efficiency_csv}")
    print(df_efficiency.to_string(index=False))

    # -------------------------------------------------------------
    # DELIVERABLE 7: Qualitative Side-by-Side Visualizations (8 Cases)
    # -------------------------------------------------------------
    print("\n--> Generating Deliverable 7: Qualitative Visual Comparison (8 Cases)...")
    selected_indices = np.linspace(0, num_test - 1, min(8, num_test), dtype=int)
    num_cols = 2 + len(models)  # Image + GT + Each Model

    fig, axes = plt.subplots(len(selected_indices), num_cols, figsize=(num_cols * 2.8, len(selected_indices) * 2.6))
    if len(selected_indices) == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = ['Ultrasound Image', 'Ground Truth'] + list(models.keys())

    for row_idx, idx in enumerate(selected_indices):
        img_np = test_images[idx].permute(1, 2, 0).cpu().numpy()
        gt_np = gt_masks[idx]

        axes[row_idx, 0].imshow(img_np)
        axes[row_idx, 0].axis('off')
        if row_idx == 0:
            axes[row_idx, 0].set_title(col_titles[0], fontsize=11, fontweight='bold')

        axes[row_idx, 1].imshow(gt_np, cmap='gray')
        axes[row_idx, 1].axis('off')
        if row_idx == 0:
            axes[row_idx, 1].set_title(col_titles[1], fontsize=11, fontweight='bold')

        for col_idx, (name, _) in enumerate(models.items(), start=2):
            pred_np = model_predictions[name][idx]
            axes[row_idx, col_idx].imshow(pred_np, cmap='gray')
            axes[row_idx, col_idx].axis('off')
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_titles[col_idx], fontsize=11, fontweight='bold')

    plt.tight_layout()
    qual_png = os.path.join(args.output_dir, 'qualitative_comparison_8_cases.png')
    plt.savefig(qual_png, dpi=200)
    plt.close()
    print(f"    Saved: {qual_png}")

    # -------------------------------------------------------------
    # DELIVERABLES AUDIT CHECKLIST
    # -------------------------------------------------------------
    print("\n=================================================================")
    print("                FINAL REPORT DELIVERABLES CHECKLIST              ")
    print("=================================================================")
    deliverable_files = [
        'final_test_comparison.csv',
        'lesion_size_evaluation.csv',
        'boundary_metrics_summary.csv',
        'robustness_per_condition.csv',
        'speckle_robustness_curve.png',
        'efficiency_results.csv',
        'training_curves_E1_E4.png',
        'qualitative_comparison_8_cases.png',
    ]

    for fname in deliverable_files:
        fpath = os.path.join(args.output_dir, fname)
        status = "✅ FOUND" if os.path.exists(fpath) and os.path.getsize(fpath) > 0 else "❌ MISSING"
        print(f" {status:10} | {fname}")

    print("\nAll deliverables are located in:", args.output_dir)
    print("=================================================================\n")


if __name__ == '__main__':
    main()
