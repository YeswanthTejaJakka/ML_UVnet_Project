"""
P6 — Unit tests for evaluation metrics.
Tests:
1. Identical masks -> Dice=1, BIoU=1, HD95=0
2. A disc shifted by k px -> HD95 approx k
3. Empty vs non-empty -> penalty path
4. BIoU monotonically decreases with shift
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from utils.metrics import (
    calculate_dice,
    calculate_iou,
    calculate_boundary_iou,
    calculate_hd95,
    calculate_boundary_f1,
    lesion_level_detection,
)


def make_disc(size=256, center=(128, 128), radius=30):
    Y, X = np.ogrid[:size, :size]
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    return (dist <= radius).astype(np.float32)


def test_identical_masks():
    m = make_disc(256, (128, 128), 30)
    dice = calculate_dice(m, m)
    iou = calculate_iou(m, m)
    biou = calculate_boundary_iou(m, m)
    hd95, is_empty = calculate_hd95(m, m)

    assert abs(dice - 1.0) < 1e-4, f"Dice should be 1.0, got {dice}"
    assert abs(iou - 1.0) < 1e-4, f"IoU should be 1.0, got {iou}"
    assert abs(biou - 1.0) < 1e-4, f"BIoU should be 1.0, got {biou}"
    assert hd95 == 0.0, f"HD95 should be 0.0, got {hd95}"
    assert not is_empty
    print("[PASS] test_identical_masks (Dice=1, BIoU=1, HD95=0).")


def test_shifted_disc_hd95():
    """A disc shifted by k pixels should have HD95 ≈ k."""
    m_gt = make_disc(256, (128, 128), 30)
    for shift in [3, 7, 12]:
        m_pred = make_disc(256, (128 + shift, 128), 30)
        hd95, is_empty = calculate_hd95(m_pred, m_gt)
        assert abs(hd95 - shift) <= 1.5, f"For shift={shift}, expected HD95 approx {shift}, got {hd95}"
    print("[PASS] test_shifted_disc_hd95 (HD95 scales with shift).")


def test_empty_penalty():
    """Empty prediction vs non-empty target triggers penalty path (diagonal = 362px for 256x256)."""
    m_gt = make_disc(256, (128, 128), 30)
    m_empty = np.zeros((256, 256), dtype=np.float32)

    hd95, is_empty = calculate_hd95(m_empty, m_gt)
    diag = np.sqrt(256**2 + 256**2)
    assert abs(hd95 - diag) < 1.0, f"Expected penalty {diag:.1f}, got {hd95}"
    assert is_empty
    print(f"[PASS] test_empty_penalty (HD95 penalty = {hd95:.1f}px).")


def test_biou_monotonic_decrease():
    """BIoU should monotonically decrease as shift increases."""
    m_gt = make_disc(256, (128, 128), 30)
    prev_biou = 1.0
    for shift in [1, 3, 6, 10, 15]:
        m_pred = make_disc(256, (128 + shift, 128), 30)
        biou = calculate_boundary_iou(m_pred, m_gt)
        assert biou < prev_biou, f"BIoU did not decrease: shift {shift} gave {biou} vs prev {prev_biou}"
        prev_biou = biou
    print("[PASS] test_biou_monotonic_decrease.")


def test_lesion_detection():
    """Test lesion detection on multi-nodule cases."""
    m_gt1 = make_disc(256, (64, 64), 20)
    m_gt2 = make_disc(256, (192, 192), 20)
    m_gt = (m_gt1 + m_gt2).clip(0, 1)

    # Prediction matches only nodule 1
    m_pred = make_disc(256, (64, 64), 20)
    stats = lesion_level_detection(m_pred, m_gt)
    assert stats['n_gt_lesions'] == 2
    assert stats['n_detected'] == 1
    assert stats['detection_recall'] == 0.5
    print("[PASS] test_lesion_detection.")


if __name__ == "__main__":
    print("Running metrics test suite...")
    test_identical_masks()
    test_shifted_disc_hd95()
    test_empty_penalty()
    test_biou_monotonic_decrease()
    test_lesion_detection()
    print("ALL METRICS TESTS PASSED SUCCESSFULLY!")

