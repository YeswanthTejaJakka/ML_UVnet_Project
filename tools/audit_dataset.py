"""
P0 — Dataset Integrity Audit Tool for TN3K
Prints and saves audit/audit_report.md with:
  - File counts in each directory
  - Fold JSON entry counts and match rates
  - Unmatched entries with reasons
  - Train/val disjointness and hash-overlap checks with test set
  - Image size distribution, mask value sets, empty masks, shape mismatches
  - Lesion-size category counts
  - 12 random image+mask overlays in audit/overlays/
"""

import os
import sys
import json
import hashlib
import argparse
import random
from collections import defaultdict

import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[WARNING] PIL not available; overlays will be skipped.")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    if not os.path.exists(folder):
        return []
    return sorted(f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts)


def find_file_robust(folder, name):
    stem = os.path.splitext(name)[0]
    exts = ["", ".jpg", ".jpeg", ".png", ".bmp"]
    for ext in exts:
        candidate = os.path.join(folder, stem + ext)
        if os.path.exists(candidate):
            return candidate, None
        candidate2 = os.path.join(folder, name + ext) if not name.endswith(ext) else os.path.join(folder, name)
        if os.path.exists(candidate2):
            return candidate2, None
    try:
        entries = {e.lower(): e for e in os.listdir(folder)}
    except Exception:
        return None, "folder not listable"
    for ext in exts:
        key = (stem + ext).lower()
        if key in entries:
            return os.path.join(folder, entries[key]), None
    if stem.isdigit():
        padded = stem.zfill(4)
        unpadded = str(int(stem))
        for try_stem in [padded, unpadded]:
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ""]:
                key = (try_stem + ext).lower()
                if key in entries:
                    return os.path.join(folder, entries[key]), None
    return None, f"no file matching '{name}' found in {os.path.basename(folder)}"


def compute_lesion_category(mask_np, small_thresh=0.02, large_thresh=0.10):
    total = mask_np.size
    area = float(np.sum(mask_np > 127)) / total
    if area < 1e-5:
        return "empty", area
    elif area < small_thresh:
        return "small", area
    elif area < large_thresh:
        return "medium", area
    else:
        return "large", area


def analyze_image(img_path, mask_path):
    result = {}
    try:
        img = Image.open(img_path)
        result["img_size"] = img.size
        result["img_mode"] = img.mode
    except Exception as e:
        result["img_error"] = str(e)
    try:
        msk = Image.open(mask_path).convert("L")
        msk_np = np.array(msk)
        values = set(np.unique(msk_np).tolist())
        result["mask_values"] = values
        result["mask_size"] = msk.size
        result["empty_mask"] = bool(np.sum(msk_np) == 0)
        cat, area = compute_lesion_category(msk_np)
        result["lesion_cat"] = cat
        result["lesion_area"] = area
        if "img_size" in result:
            result["shape_mismatch"] = (result["img_size"] != result["mask_size"])
    except Exception as e:
        result["mask_error"] = str(e)
    return result


def save_overlay(img_path, mask_path, out_path):
    if not PIL_AVAILABLE:
        return
    try:
        img = Image.open(img_path).convert("RGB").resize((256, 256))
        msk = Image.open(mask_path).convert("L").resize((256, 256), Image.NEAREST)
        msk_np = np.array(msk)
        overlay = np.array(img).copy()
        overlay[msk_np > 127, 1] = np.minimum(overlay[msk_np > 127, 1] + 100, 255)
        Image.fromarray(overlay).save(out_path)
    except Exception as e:
        print(f"  [overlay error] {e}")


def audit(data_root, fold, out_dir):
    report_lines = []
    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log(f"# TN3K Dataset Audit Report")
    log(f"data_root: {data_root}  |  fold: {fold}")
    log()

    overlay_dir = os.path.join(out_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    log("## 1. File Counts")
    dirs = {
        "trainval-image": os.path.join(data_root, "trainval-image"),
        "trainval-mask":  os.path.join(data_root, "trainval-mask"),
        "test-image":     os.path.join(data_root, "test-image"),
        "test-mask":      os.path.join(data_root, "test-mask"),
    }
    file_lists = {}
    for name, d in dirs.items():
        files = list_images(d)
        file_lists[name] = files
        log(f"  {name}: {len(files)} files  (exists={os.path.exists(d)})")
    log()

    log("## 2. Fold JSON Analysis")
    fold_file = os.path.join(data_root, f"tn3k-trainval-fold{fold}.json")
    if not os.path.exists(fold_file):
        log(f"  [ERROR] Fold JSON not found: {fold_file}")
        log("  AUDIT CANNOT CONTINUE FOR TRAIN/VAL SPLIT.")
        fold_train_names = []
        fold_val_names = []
    else:
        with open(fold_file, "r") as f:
            fold_data = json.load(f)
        if isinstance(fold_data, dict):
            train_key = next((k for k in fold_data if "train" in k.lower()), None)
            val_key   = next((k for k in fold_data if "val"   in k.lower()), None)
            fold_train_names = fold_data.get(train_key, []) if train_key else []
            fold_val_names   = fold_data.get(val_key,   []) if val_key   else []
        elif isinstance(fold_data, list):
            fold_train_names = fold_data
            fold_val_names = []
        else:
            fold_train_names = []
            fold_val_names = []
        log(f"  Fold JSON: {fold_file}")
        log(f"  Entries — train: {len(fold_train_names)}, val: {len(fold_val_names)}, total: {len(fold_train_names)+len(fold_val_names)}")
        log()

    log("## 3. File Matching")
    img_dir  = dirs["trainval-image"]
    mask_dir = dirs["trainval-mask"]

    matched_train = []
    unmatched_train = []
    for name in fold_train_names:
        img_p,  ir = find_file_robust(img_dir,  name)
        mask_p, mr = find_file_robust(mask_dir, name)
        if img_p and mask_p:
            matched_train.append((name, img_p, mask_p))
        else:
            reason = f"img: {ir or 'ok'}, mask: {mr or 'ok'}"
            unmatched_train.append((name, reason))

    matched_val = []
    unmatched_val = []
    for name in fold_val_names:
        img_p,  ir = find_file_robust(img_dir,  name)
        mask_p, mr = find_file_robust(mask_dir, name)
        if img_p and mask_p:
            matched_val.append((name, img_p, mask_p))
        else:
            reason = f"img: {ir or 'ok'}, mask: {mr or 'ok'}"
            unmatched_val.append((name, reason))

    total_matched = len(matched_train) + len(matched_val)
    total_entries = len(fold_train_names) + len(fold_val_names)
    log(f"  Train matched: {len(matched_train)} / {len(fold_train_names)}")
    log(f"  Val   matched: {len(matched_val)} / {len(fold_val_names)}")
    log(f"  Total matched: {total_matched} / {total_entries}")
    log(f"  Trainval-image on disk: {len(file_lists['trainval-image'])}")
    log()

    if unmatched_train or unmatched_val:
        log("### Unmatched Entries")
        for name, reason in (unmatched_train + unmatched_val)[:50]:
            log(f"  - '{name}': {reason}")
        extra = len(unmatched_train) + len(unmatched_val) - 50
        if extra > 0:
            log(f"  ... and {extra} more")
        log()
    else:
        log("  No unmatched entries — all fold entries matched successfully.")
        log()

    log("## 4. Train/Val Disjointness")
    train_names_set = set(n for n, _, _ in matched_train)
    val_names_set   = set(n for n, _, _ in matched_val)
    overlap = train_names_set & val_names_set
    if overlap:
        log(f"  [ERROR] Train ∩ Val overlap: {len(overlap)} items — {list(overlap)[:10]}")
    else:
        log(f"  Train ∩ Val = ∅ ✓")
    log()

    log("## 5. Test-Set Hash Overlap Check")
    test_img_dir = dirs["test-image"]
    if not os.path.exists(test_img_dir):
        log("  [SKIP] test-image directory not found.")
    else:
        test_hashes = {}
        for f in file_lists["test-image"]:
            p = os.path.join(test_img_dir, f)
            try:
                test_hashes[md5(p)] = f
            except Exception as e:
                log(f"  [hash error] {f}: {e}")
        log(f"  Test set: {len(test_hashes)} unique hashes")
        trainval_paths = [p for _, p, _ in matched_train] + [p for _, p, _ in matched_val]
        hash_overlaps = []
        for p in trainval_paths:
            try:
                h = md5(p)
                if h in test_hashes:
                    hash_overlaps.append((os.path.basename(p), test_hashes[h]))
            except Exception:
                pass
        if hash_overlaps:
            log(f"  [ERROR] {len(hash_overlaps)} trainval images have same MD5 as test images!")
            for tv, te in hash_overlaps[:10]:
                log(f"    trainval: {tv}  <->  test: {te}")
        else:
            log("  No hash overlap between trainval and test ✓")
    log()

    log("## 6. Image & Mask Statistics (from matched pairs)")
    all_matched = matched_train + matched_val
    sample_size = min(200, len(all_matched))
    sample = random.sample(all_matched, sample_size) if len(all_matched) > sample_size else all_matched

    img_widths, img_heights = [], []
    mask_value_union = set()
    empty_count = 0
    mismatch_count = 0
    cat_counts = defaultdict(int)

    for name, img_p, mask_p in sample:
        stats = analyze_image(img_p, mask_p)
        if "img_size" in stats:
            img_widths.append(stats["img_size"][0])
            img_heights.append(stats["img_size"][1])
        if "mask_values" in stats:
            mask_value_union |= stats["mask_values"]
        if stats.get("empty_mask"):
            empty_count += 1
        if stats.get("shape_mismatch"):
            mismatch_count += 1
        if "lesion_cat" in stats:
            cat_counts[stats["lesion_cat"]] += 1

    if img_widths:
        log(f"  Width  — min: {min(img_widths)}, median: {int(np.median(img_widths))}, max: {max(img_widths)}")
        log(f"  Height — min: {min(img_heights)}, median: {int(np.median(img_heights))}, max: {max(img_heights)}")
    log(f"  Mask values (union of sample): {sorted(mask_value_union)}")
    log(f"  Empty masks in sample: {empty_count} / {sample_size}")
    log(f"  Shape mismatches in sample: {mismatch_count} / {sample_size}")
    log()

    log("## 7. Lesion-Size Category Counts (sample)")
    for cat in ["small", "medium", "large", "empty"]:
        log(f"  {cat}: {cat_counts.get(cat, 0)}")
    log()

    log("## 8. Test Set Lesion Categories")
    test_mask_dir = dirs["test-mask"]
    test_cat_counts = defaultdict(int)
    if os.path.exists(test_mask_dir):
        for f in file_lists["test-mask"][:500]:
            mp = os.path.join(test_mask_dir, f)
            try:
                msk_np = np.array(Image.open(mp).convert("L"))
                cat, _ = compute_lesion_category(msk_np)
                test_cat_counts[cat] += 1
            except Exception:
                pass
        for cat in ["small", "medium", "large", "empty"]:
            log(f"  {cat}: {test_cat_counts.get(cat, 0)}")
    else:
        log("  [SKIP] test-mask not found")
    log()

    log("## 9. Data Coverage Summary & Decision")
    disk_count = len(file_lists["trainval-image"])
    log(f"  Images on disk (trainval): {disk_count}")
    log(f"  Images in fold JSON:       {total_entries}")
    log(f"  Successfully matched:      {total_matched}")
    if disk_count > total_entries:
        log(f"  EXTRA images not in fold JSON: {disk_count - total_entries}")
        log(f"  DECISION: Use only fold JSON entries for train/val splits.")
        log(f"  Extra images are not used to avoid any risk of test-set leakage.")
    elif disk_count == total_entries:
        log(f"  Disk count matches fold JSON — no extra images.")
    else:
        log(f"  [WARNING] Fewer images on disk than in fold JSON! Check data integrity.")
    log()

    log("## 10. Qualitative Overlays (12 samples)")
    if PIL_AVAILABLE and len(all_matched) > 0:
        random.seed(1234)
        overlay_sample = random.sample(all_matched, min(12, len(all_matched)))
        for i, (name, img_p, mask_p) in enumerate(overlay_sample):
            out_path = os.path.join(overlay_dir, f"overlay_{i:02d}_{os.path.splitext(name)[0]}.png")
            save_overlay(img_p, mask_p, out_path)
            log(f"  Saved: {os.path.basename(out_path)}")
    else:
        log("  [SKIP] PIL not available or no matched pairs")

    if file_lists["test-mask"]:
        log("\n  Checking for suspicious straight-edged masks in test set...")
        suspicious = []
        test_mdir = dirs["test-mask"]
        for f in file_lists["test-mask"]:
            mp = os.path.join(test_mdir, f)
            try:
                msk_np = np.array(Image.open(mp).convert("L"))
                cols = np.any(msk_np > 127, axis=0)
                if cols.any():
                    c_first = np.argmax(cols)
                    if c_first == 0:
                        suspicious.append(f)
            except Exception:
                pass
        if suspicious:
            log(f"  Found {len(suspicious)} masks touching left border: {suspicious[:5]}")
            for sf in suspicious[:3]:
                sip, _ = find_file_robust(dirs["test-image"], sf)
                smp = os.path.join(test_mdir, sf)
                if sip:
                    out_path = os.path.join(overlay_dir, f"suspicious_{sf}")
                    save_overlay(sip, smp, out_path)
                    log(f"  Overlay saved: suspicious_{sf}")
        else:
            log("  No masks found touching the left border.")
    log()

    report_path = os.path.join(out_dir, "audit_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"\nReport saved to: {report_path}")
    return total_matched, total_entries, disk_count


def main():
    parser = argparse.ArgumentParser(description="TN3K Dataset Integrity Audit")
    parser.add_argument("--data_root", type=str, default="/content/tn3k")
    parser.add_argument("--fold",      type=int, default=0)
    parser.add_argument("--out_dir",   type=str, default="audit")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    audit(args.data_root, args.fold, args.out_dir)


if __name__ == "__main__":
    main()
