import os
import argparse
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader

from models import UltraLight_VM_UNet
from dataset import TN3KDataset
from utils import (
    MetricTracker,
    calculate_dice,
    calculate_iou,
    calculate_boundary_iou,
    calculate_hd95,
    measure_model_efficiency,
    apply_ultrasound_corruption
)


def parse_args():
    parser = argparse.ArgumentParser(description="UltraLight VM-UNet Evaluation Benchmark")
    parser.add_argument('--weights', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--data_root', type=str, default='/content/tn3k', help='Dataset root path')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='Directory to save outputs and plots')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_visuals', action='store_true', default=True, help='Save qualitative visual overlays')
    return parser.parse_args()


def save_qualitative_overlay(image_tensor, gt_mask_np, pred_mask_np, save_path):
    """
    Saves a 3-panel comparison: [Input Ultrasound, Ground Truth Mask, Model Prediction].
    """
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)

    gt_vis = (gt_mask_np * 255.0).clip(0, 255).astype(np.uint8)
    pred_vis = (pred_mask_np * 255.0).clip(0, 255).astype(np.uint8)

    h, w, _ = img_np.shape
    combined = np.zeros((h, w * 3, 3), dtype=np.uint8)
    combined[:, 0:w, :] = img_np
    combined[:, w:2*w, :] = np.stack([gt_vis]*3, axis=-1)
    combined[:, 2*w:3*w, :] = np.stack([pred_vis]*3, axis=-1)

    Image.fromarray(combined).save(save_path)


def evaluate_robustness(model, dataset, device, noise_levels=[0.0, 0.05, 0.10, 0.20, 0.30]):
    """
    Evaluates performance degradation under varying speckle noise variances.
    """
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    robustness_curve = {}

    print("\n--- Running Noise Robustness Benchmark ---")
    for sigma in noise_levels:
        dice_scores = []
        for batch in loader:
            images = batch['image'].to(device)
            masks = batch['mask'].cpu().numpy()

            if sigma > 0.0:
                images = apply_ultrasound_corruption(images, speckle_sigma=sigma, contrast_factor=1.0)

            with torch.no_grad():
                preds = model(images, return_boundary=False).cpu().numpy()

            for i in range(len(preds)):
                d = calculate_dice(preds[i, 0], masks[i, 0])
                dice_scores.append(d)

        avg_dice = float(np.mean(dice_scores))
        robustness_curve[f'sigma_{sigma}'] = avg_dice
        print(f"  Speckle Sigma {sigma:.2f} -> Test Dice: {avg_dice:.4f}")

    return robustness_curve


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Evaluating Checkpoint: {args.weights} ===")
    checkpoint = torch.load(args.weights, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Check whether aux_boundary_head was present in state_dict
    has_aux_head = any('boundary_head' in k for k in state_dict.keys())

    # Instantiate model
    model = UltraLight_VM_UNet(
        num_classes=1,
        input_channels=3,
        c_list=[8, 16, 24, 32, 48, 64],
        split_att='fc',
        bridge=True,
        aux_boundary_head=has_aux_head,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    # Measure efficiency metrics
    efficiency = measure_model_efficiency(model, device=args.device)
    print("\n--- Model Complexity & Efficiency ---")
    print(f"Parameters:     {efficiency['params_M']} M")
    print(f"FLOPs:          {efficiency['flops_G']} GFLOPs")
    print(f"Latency:        {efficiency['latency_ms']} ms/image")
    print(f"Throughput:     {efficiency['fps']} FPS")

    # Load official test set
    test_dataset = TN3KDataset(data_root=args.data_root, split='test', augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"\nLoaded Test Set: {len(test_dataset)} samples")

    tracker = MetricTracker()
    vis_dir = os.path.join(args.output_dir, 'visual_predictions')
    if args.save_visuals:
        os.makedirs(vis_dir, exist_ok=True)

    sample_counter = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].cpu().numpy()
            categories = batch['category']
            sample_ids = batch['id']

            preds = model(images, return_boundary=False).cpu().numpy()

            for i in range(len(preds)):
                pred_i = preds[i, 0]
                mask_i = masks[i, 0]
                cat_i = categories[i]
                tracker.update(pred_i, mask_i, category=cat_i)

                # Save first 15 qualitative overlays
                if args.save_visuals and sample_counter < 15:
                    save_path = os.path.join(vis_dir, f"sample_{sample_counter:03d}_{sample_ids[i]}.png")
                    save_qualitative_overlay(batch['image'][i], mask_i, pred_i, save_path)
                    sample_counter += 1

    summary = tracker.summary()
    print("\n================ FINAL TEST BENCHMARK RESULTS ================")
    print(f"Overall Dice:        {summary['overall_dice']:.4f}")
    print(f"Overall IoU:         {summary['overall_iou']:.4f}")
    print(f"Boundary IoU (BIoU): {summary['overall_biou']:.4f}")
    print(f"Hausdorff 95 (HD95): {summary['overall_hd95']:.2f} pixels")
    print("---------------- Lesion-Scale Breakdown ----------------")
    print(f"Small Lesions  (N={summary['small_count']}): Dice: {summary['small_dice']:.4f} | IoU: {summary['small_iou']:.4f} | BIoU: {summary['small_biou']:.4f}")
    print(f"Medium Lesions (N={summary['medium_count']}): Dice: {summary['medium_dice']:.4f} | IoU: {summary['medium_iou']:.4f} | BIoU: {summary['medium_biou']:.4f}")
    print(f"Large Lesions  (N={summary['large_count']}): Dice: {summary['large_dice']:.4f} | IoU: {summary['large_iou']:.4f} | BIoU: {summary['large_biou']:.4f}")
    print("==============================================================")

    # Robustness stress-testing
    robustness_results = evaluate_robustness(model, test_dataset, device)

    # Save all results to JSON
    final_output = {
        'weights_file': args.weights,
        'efficiency': efficiency,
        'metrics_summary': summary,
        'noise_robustness': robustness_results,
    }

    out_file = os.path.join(args.output_dir, 'benchmark_results.json')
    with open(out_file, 'w') as f:
        json.dump(final_output, f, indent=2)

    print(f"\nAll benchmark results successfully saved to: {out_file}")


if __name__ == '__main__':
    main()
