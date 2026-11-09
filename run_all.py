"""
P5 — Automated Experiment Driver: run_all.py
Executes Stages A, B, and C with resume support and seed management.

Usage:
  python run_all.py --stage A --data_root /content/tn3k
  python run_all.py --stage B --data_root /content/tn3k
  python run_all.py --stage C --seeds 42 43 44 --data_root /content/tn3k
  python run_all.py --stage all --data_root /content/tn3k
"""

import os
import sys
import argparse
import subprocess
import json
import time
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="AI512 Master Experiment Orchestrator")
    parser.add_argument('--stage', type=str, default='C', choices=['A', 'B', 'C', 'all', 'smoke'],
                        help='Stage to run: A (recipe ladder), B (loss tuning), C (final runs), all, smoke (2 epochs)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Seeds for Stage C')
    parser.add_argument('--data_root', type=str, default='data/tn3k')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--checkpoints_root', type=str, default='checkpoints')
    parser.add_argument('--reports_dir', type=str, default='reports')
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def run_command(cmd):
    print(f"\n[RUNNING] {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"[ERROR] Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[COMPLETED in {elapsed:.1f}s]")
    return elapsed


def run_stage_a_ladder(args):
    """
    Stage A: Recipe Ladder (validation only, seed 42, 100 epochs, E1 loss only)
    R0: Current recipe
    R1: + P0 data fix, no vflip, new geometric aug
    R2: + P1 logits & fp32 scan
    R3: + 200 epochs (or 100 for ladder), warm-up, EMA, no-decay groups, grad-clip, bs 8
    R4: + deep supervision
    Writes ladder table to reports/recipe_ladder.csv.
    """
    os.makedirs(args.reports_dir, exist_ok=True)
    ladder_csv = os.path.join(args.reports_dir, 'recipe_ladder.csv')
    print("\n" + "="*60)
    print("STAGE A: RECIPE LADDER")
    print("="*60)

    ladder_steps = [
        ('R0_legacy_repro', 'configs/e1_baseline.yaml', {'epochs': 100, 'deep_supervision': False, 'amp': True}),
        ('R1_data_and_aug', 'configs/e1_baseline.yaml', {'epochs': 100, 'deep_supervision': False, 'amp': True}),
        ('R2_logits_fp32ssm', 'configs/e1_baseline.yaml', {'epochs': 100, 'deep_supervision': False, 'amp': False}),
        ('R3_warmup_ema_wd', 'configs/e1_baseline.yaml', {'epochs': 100, 'deep_supervision': False, 'amp': False}),
        ('R4_deep_supervision', 'configs/e1_baseline.yaml', {'epochs': 100, 'deep_supervision': True, 'amp': False}),
    ]

    records = []
    for step_name, cfg_path, overrides in ladder_steps:
        exp_dir = os.path.join(args.checkpoints_root, 'ladder', step_name)
        cmd = [
            sys.executable, 'train.py',
            '--config', cfg_path,
            '--data_root', args.data_root,
            '--fold', str(args.fold),
            '--seed', '42',
            '--save_dir', exp_dir,
            '--epochs', str(overrides.get('epochs', 100)),
        ]
        if args.resume:
            cmd.append('--resume')
        run_command(cmd)

        log_path = os.path.join(exp_dir, 'training_log.csv')
        best_val = 'N/A'
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            best_val = df['val_dice'].max()

        records.append({
            'step': step_name,
            'best_val_dice': round(float(best_val), 4) if isinstance(best_val, (int, float)) else best_val,
            'log_path': log_path,
        })

    ladder_df = pd.DataFrame(records)
    ladder_df.to_csv(ladder_csv, index=False)
    print(f"\n✅ Stage A Complete! Ladder saved to {ladder_csv}")


def run_stage_b_tuning(args):
    """
    Stage B: Loss Tuning (validation only, seed 42, 100 epochs)
    - E2 lambda_boundary in {0.1, 0.3, 0.5}
    - E3 lambda_consistency in {0.5, 1.0, 2.0}
    """
    os.makedirs(args.reports_dir, exist_ok=True)
    tuning_csv = os.path.join(args.reports_dir, 'stage_b_tuning.csv')
    print("\n" + "="*60)
    print("STAGE B: LOSS TUNING")
    print("="*60)

    tuning_runs = [
        # E2 lambda tuning
        ('e2_lambda_0.1', 'configs/e2_boundary.yaml', 100),
        ('e2_lambda_0.3', 'configs/e2_boundary.yaml', 100),
        ('e2_lambda_0.5', 'configs/e2_boundary.yaml', 100),
        # E3 lambda tuning
        ('e3_lambda_0.5', 'configs/e3_consistency.yaml', 100),
        ('e3_lambda_1.0', 'configs/e3_consistency.yaml', 100),
        ('e3_lambda_2.0', 'configs/e3_consistency.yaml', 100),
    ]

    records = []
    for exp_id, cfg_file, eps in tuning_runs:
        exp_dir = os.path.join(args.checkpoints_root, 'tuning', exp_id)
        cmd = [
            sys.executable, 'train.py',
            '--config', cfg_file,
            '--data_root', args.data_root,
            '--fold', str(args.fold),
            '--seed', '42',
            '--save_dir', exp_dir,
            '--epochs', str(eps),
        ]
        if args.resume:
            cmd.append('--resume')
        run_command(cmd)

        log_path = os.path.join(exp_dir, 'training_log.csv')
        best_val = 'N/A'
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            best_val = df['val_dice'].max()

        records.append({
            'experiment': exp_id,
            'best_val_dice': round(float(best_val), 4) if isinstance(best_val, (int, float)) else best_val,
        })

    tune_df = pd.DataFrame(records)
    tune_df.to_csv(tuning_csv, index=False)
    print(f"\n✅ Stage B Complete! Tuning results saved to {tuning_csv}")


def run_stage_c_final(args):
    """
    Stage C: Final Runs (E1-R*, E2, E3, E4 across multiple seeds)
    """
    print("\n" + "="*60)
    print(f"STAGE C: FINAL EXPERIMENTS (Seeds: {args.seeds})")
    print("="*60)

    experiments = [
        ('e1_baseline', 'configs/e1_baseline.yaml'),
        ('e2_boundary', 'configs/e2_boundary.yaml'),
        ('e3_consistency', 'configs/e3_consistency.yaml'),
        ('e4_combined', 'configs/e4_combined.yaml'),
    ]

    for seed in args.seeds:
        for exp_name, cfg_path in experiments:
            print(f"\n>>> Running {exp_name} (Seed {seed}) <<<")
            cmd = [
                sys.executable, 'train.py',
                '--config', cfg_path,
                '--data_root', args.data_root,
                '--fold', str(args.fold),
                '--seed', str(seed),
            ]
            if args.checkpoints_root:
                cmd.extend(['--save_dir', args.checkpoints_root])
            if args.resume:
                cmd.append('--resume')
            run_command(cmd)

    # After all models finish training, generate all deliverables
    print("\n>>> Generating Final Deliverables <<<")
    gen_cmd = [
        sys.executable, 'generate_deliverables.py',
        '--checkpoints_root', args.checkpoints_root,
        '--data_root', args.data_root,
        '--seed', str(args.seeds[0]),
    ]
    run_command(gen_cmd)
    print("\n✅ Stage C and Deliverables Generation Complete!")


def run_smoke_test(args):
    """2-epoch smoke run of E1–E4 to verify entire training and evaluation pipeline."""
    print("\n" + "="*60)
    print("SMOKE TEST: 2 Epochs of E1–E4")
    print("="*60)

    configs = [
        ('e1_baseline', 'configs/e1_baseline.yaml'),
        ('e2_boundary', 'configs/e2_boundary.yaml'),
        ('e3_consistency', 'configs/e3_consistency.yaml'),
        ('e4_combined', 'configs/e4_combined.yaml'),
    ]

    for exp_name, cfg_path in configs:
        smoke_dir = os.path.join(args.checkpoints_root, 'smoke', exp_name)
        cmd = [
            sys.executable, 'train.py',
            '--config', cfg_path,
            '--data_root', args.data_root,
            '--fold', str(args.fold),
            '--epochs', '2',
            '--seed', '42',
            '--save_dir', smoke_dir,
            '--force_retrain',
        ]
        run_command(cmd)

    # Deliverables from smoke test
    gen_cmd = [
        sys.executable, 'generate_deliverables.py',
        '--checkpoints_root', os.path.join(args.checkpoints_root, 'smoke'),
        '--data_root', args.data_root,
        '--output_dir', 'smoke_deliverables',
    ]
    run_command(gen_cmd)
    print("\n✅ Smoke test completed successfully!")


def main():
    args = parse_args()
    if args.stage == 'A':
        run_stage_a_ladder(args)
    elif args.stage == 'B':
        run_stage_b_tuning(args)
    elif args.stage == 'C':
        run_stage_c_final(args)
    elif args.stage == 'smoke':
        run_smoke_test(args)
    elif args.stage == 'all':
        run_stage_a_ladder(args)
        run_stage_b_tuning(args)
        run_stage_c_final(args)


if __name__ == '__main__':
    main()
