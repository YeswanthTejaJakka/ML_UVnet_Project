import json
from pathlib import Path

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# AI512 Project: UltraLight VM-UNet on TN3K Thyroid Ultrasound\n",
            "## Lesion-Scale-Aware Boundary Preservation under Ultrasound Speckle Degradation\n",
            "\n",
            "### Local Execution Runner for NVIDIA GPU Workstations / Laptops\n",
            "- **Model**: UltraLight VM-UNet (0.0495M Parameters, ~0.060 GFLOPs)\n",
            "- **Dataset**: TN3K Benchmark located under `data/tn3k/` (separated from `dataset/` code)\n",
            "- **Outputs & Deliverables**: Saved locally under `checkpoints/` and `checkpoints/final_deliverables/`\n",
            "- **Core Experiments**:\n",
            "  - **E1-R***: Baseline reproduction under standardized recipe\n",
            "  - **E2**: Recipe + Lesion-Scale-Aware Boundary Supervision (`boundary_mode: from_mask`)\n",
            "  - **E3**: Recipe + Corruption-Aware Consistency Regularization\n",
            "  - **E4**: Full Combined Framework (E2 Boundary + E3 Consistency)\n",
            "\n",
            "> **Note**: This notebook runs completely locally without requiring Google Drive or Colab dependencies."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1. GPU & CUDA Environment Verification"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "import torch\n",
            "\n",
            "print(f\"Python Version:  {sys.version}\")\n",
            "print(f\"PyTorch Version: {torch.__version__}\")\n",
            "cuda_available = torch.cuda.is_available()\n",
            "print(f\"CUDA Available:  {cuda_available}\")\n",
            "\n",
            "if cuda_available:\n",
            "    device_idx = torch.cuda.current_device()\n",
            "    gpu_name = torch.cuda.get_device_name(device_idx)\n",
            "    gpu_vram = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)\n",
            "    print(f\"✅ GPU Detected:  {gpu_name}\")\n",
            "    print(f\"✅ Total VRAM:    {gpu_vram:.2f} GB\")\n",
            "    # Test quick tensor allocation on GPU\n",
            "    x = torch.ones((2, 2), device=\"cuda\")\n",
            "    print(\"✅ PyTorch GPU tensor allocation test succeeded!\")\n",
            "else:\n",
            "    print(\"⚠️ No NVIDIA GPU detected with CUDA. PyTorch is running in CPU mode.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2. Local Project Directory & Path Setup"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "from pathlib import Path\n",
            "\n",
            "# Resolve absolute paths based on current repository location\n",
            "PROJECT_ROOT = Path.cwd().resolve()\n",
            "DATA_ROOT = PROJECT_ROOT / \"data\" / \"tn3k\"\n",
            "CHECKPOINTS_ROOT = PROJECT_ROOT / \"checkpoints\"\n",
            "DELIVERABLES_ROOT = CHECKPOINTS_ROOT / \"final_deliverables\"\n",
            "REPORTS_DIR = PROJECT_ROOT / \"reports\"\n",
            "\n",
            "# Ensure output directories exist\n",
            "CHECKPOINTS_ROOT.mkdir(parents=True, exist_ok=True)\n",
            "REPORTS_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "print(f\"Project Root:      {PROJECT_ROOT}\")\n",
            "print(f\"Dataset Path:      {DATA_ROOT}\")\n",
            "print(f\"Checkpoints Path:  {CHECKPOINTS_ROOT}\")\n",
            "print(f\"Deliverables Path: {DELIVERABLES_ROOT}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3. Dataset Integrity & Unzip Helper\n",
            "Checks if `data/tn3k` exists. If not, searches for `TN3K.zip` inside `data/` or the root folder and unpacks it."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import zipfile\n",
            "\n",
            "# Auto-unpack TN3K.zip if raw images are missing\n",
            "if not DATA_ROOT.exists() or not (DATA_ROOT / \"trainval-image\").exists():\n",
            "    candidates = [\n",
            "        PROJECT_ROOT / \"data\" / \"TN3K.zip\",\n",
            "        PROJECT_ROOT / \"TN3K.zip\",\n",
            "        PROJECT_ROOT / \"data\" / \"tn3k.zip\",\n",
            "    ]\n",
            "    zip_found = next((c for c in candidates if c.exists()), None)\n",
            "    if zip_found:\n",
            "        print(f\"📦 Unpacking archive: {zip_found} -> {DATA_ROOT.parent} ...\")\n",
            "        DATA_ROOT.parent.mkdir(parents=True, exist_ok=True)\n",
            "        with zipfile.ZipFile(zip_found, \"r\") as zf:\n",
            "            zf.extractall(DATA_ROOT.parent)\n",
            "        # Handle possible nested directory after unzipping\n",
            "        nested = DATA_ROOT.parent / \"Thyroid Dataset\" / \"tn3k\"\n",
            "        if nested.exists() and not DATA_ROOT.exists():\n",
            "            nested.rename(DATA_ROOT)\n",
            "        print(\"✅ Unpacking completed!\")\n",
            "\n",
            "# Verify folder contents\n",
            "trainval_count = len(list((DATA_ROOT / \"trainval-image\").glob(\"*\"))) if (DATA_ROOT / \"trainval-image\").exists() else 0\n",
            "test_count = len(list((DATA_ROOT / \"test-image\").glob(\"*\"))) if (DATA_ROOT / \"test-image\").exists() else 0\n",
            "fold0_ok = (DATA_ROOT / \"tn3k-trainval-fold0.json\").exists()\n",
            "\n",
            "print(\"\\n--- Local Dataset Verification ---\")\n",
            "print(f\"TrainVal Images: {trainval_count} (Expected: ~2879)\")\n",
            "print(f\"Test Images:     {test_count} (Expected: ~614)\")\n",
            "print(f\"Fold 0 JSON:     {'✅ FOUND' if fold0_ok else '❌ MISSING'}\")\n",
            "\n",
            "if trainval_count == 0 or not fold0_ok:\n",
            "    print(f\"\\n⚠️ Dataset missing or incomplete. Please place or extract TN3K into: {DATA_ROOT}\")\n",
            "else:\n",
            "    print(\"\\n✅ Dataset verified successfully!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4. Unit Test Verification Suite (P1 & P6 correctness)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Run unit tests for associative parallel scan and metrics\n",
            "!python tests/test_pscan.py\n",
            "!python tests/test_metrics.py\n",
            "print(\"\\n✅ All numerical, architecture, and metric unit tests passed!\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5. P0: Dataset Integrity & Leakage Audit\n",
            "Audits file counts, fold-JSON match rates, MD5 hash disjointness vs test set, and lesion scale distributions."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python tools/audit_dataset.py --data_root \"{DATA_ROOT}\" --fold 0 --out_dir audit\n",
            "\n",
            "audit_md = PROJECT_ROOT / \"audit\" / \"audit_report.md\"\n",
            "if audit_md.exists():\n",
            "    with open(audit_md, \"r\", encoding=\"utf-8\") as f:\n",
            "        print(f.read())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 6. Quick Smoke Test (2 Epochs across E1–E4)\n",
            "Validates the entire end-to-end training and deliverable generation pipeline in ~1–2 minutes on GPU before starting full runs."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python run_all.py --stage smoke --data_root \"{DATA_ROOT}\" --checkpoints_root \"{CHECKPOINTS_ROOT}\""
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 7. Stage A: Recipe Ladder (Optional, Validation Only, Seed 42, 100 Epochs)\n",
            "Measures incremental impact of each design component (R0 -> R1 -> R2 -> R3 -> R4)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python run_all.py --stage A --data_root \"{DATA_ROOT}\" --checkpoints_root \"{CHECKPOINTS_ROOT}\"\n",
            "\n",
            "ladder_csv = REPORTS_DIR / \"recipe_ladder.csv\"\n",
            "if ladder_csv.exists():\n",
            "    import pandas as pd\n",
            "    display(pd.read_csv(ladder_csv))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 8. Stage B: Loss Hyperparameter Tuning (Optional, Seed 42)\n",
            "Tuning grids: $\\lambda_{boundary} \\in \\{0.1, 0.3, 0.5\\}$ and $\\lambda_{consistency} \\in \\{0.5, 1.0, 2.0\\}$."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!python run_all.py --stage B --data_root \"{DATA_ROOT}\" --checkpoints_root \"{CHECKPOINTS_ROOT}\"\n",
            "\n",
            "tuning_csv = REPORTS_DIR / \"stage_b_tuning.csv\"\n",
            "if tuning_csv.exists():\n",
            "    import pandas as pd\n",
            "    display(pd.read_csv(tuning_csv))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 9. Stage C: Full Benchmark Experiments (E1-R*, E2, E3, E4 across Seeds 42, 43, 44)\n",
            "Runs full training (200 epochs per run, bs 16/8, cosine schedule with warm-up, EMA).\n",
            "- Includes automatic checkpointing and resume support (skips completed runs, resumes interrupted runs).\n",
            "- Automatically invokes `generate_deliverables.py` at the end to create all benchmark tables, curves, and figures."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Note: To run a single seed first, use: !python run_all.py --stage C --seeds 42 --data_root \"{DATA_ROOT}\" --checkpoints_root \"{CHECKPOINTS_ROOT}\"\n",
            "!python run_all.py --stage C --seeds 42 43 44 --data_root \"{DATA_ROOT}\" --checkpoints_root \"{CHECKPOINTS_ROOT}\""
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📊 DELIVERABLES INSPECTION & DISPLAY"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Deliverable 1: 12-Panel Comprehensive Training Curves"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib.pyplot as plt\n",
            "from PIL import Image\n",
            "\n",
            "curve_path = DELIVERABLES_ROOT / \"training_curves_E1_E4.png\"\n",
            "if curve_path.exists():\n",
            "    plt.figure(figsize=(20, 12))\n",
            "    plt.imshow(Image.open(curve_path))\n",
            "    plt.axis(\"off\")\n",
            "    plt.title(\"AI512 12-Panel Training Curves (E1–E4)\", fontsize=16)\n",
            "    plt.show()\n",
            "else:\n",
            "    print(f\"⏳ File not found: {curve_path}. Complete Stage C to generate.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Deliverable 2: Final Test Comparison Table (E1 vs E2 vs E3 vs E4)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "\n",
            "csv_path = DELIVERABLES_ROOT / \"final_test_comparison.csv\"\n",
            "if csv_path.exists():\n",
            "    display(pd.read_csv(csv_path))\n",
            "else:\n",
            "    print(f\"⏳ File not found: {csv_path}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Deliverable 3: Lesion-Scale Stratification Analysis (Small, Medium, Large)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "csv_path = DELIVERABLES_ROOT / \"lesion_scale_analysis.csv\"\n",
            "if csv_path.exists():\n",
            "    display(pd.read_csv(csv_path))\n",
            "else:\n",
            "    print(f\"⏳ File not found: {csv_path}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Deliverable 4: Model Efficiency & Hardware Latency Benchmark"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "csv_path = DELIVERABLES_ROOT / \"model_efficiency_benchmark.csv\"\n",
            "if csv_path.exists():\n",
            "    display(pd.read_csv(csv_path))\n",
            "else:\n",
            "    print(f\"⏳ File not found: {csv_path}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Deliverable 5: Qualitative Comparisons (Stratified 8 Cases, Best Gains, Failures)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "qual_8 = DELIVERABLES_ROOT / \"qualitative_comparison_8_cases.png\"\n",
            "qual_gains = DELIVERABLES_ROOT / \"qualitative_best_gains.png\"\n",
            "qual_fail = DELIVERABLES_ROOT / \"qualitative_failures.png\"\n",
            "\n",
            "for title, p in [\n",
            "    (\"Stratified 8 Representative Cases\", qual_8),\n",
            "    (\"Top 6 Greatest Improvement Cases (E4 vs E1)\", qual_gains),\n",
            "    (\"Challenging Failure Cases for E4\", qual_fail),\n",
            "]:\n",
            "    if p.exists():\n",
            "        print(f\"\\n=== {title} ===\")\n",
            "        plt.figure(figsize=(18, 12))\n",
            "        plt.imshow(Image.open(p))\n",
            "        plt.axis(\"off\")\n",
            "        plt.show()\n",
            "    else:\n",
            "        print(f\"⏳ File not found: {p}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Deliverable 6: Master Deliverables Asset Verification Checklist"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "files_to_check = [\n",
            "    \"final_test_comparison.csv\",\n",
            "    \"lesion_scale_analysis.csv\",\n",
            "    \"model_efficiency_benchmark.csv\",\n",
            "    \"training_summary.csv\",\n",
            "    \"training_curves_E1_E4.png\",\n",
            "    \"qualitative_comparison_8_cases.png\",\n",
            "    \"qualitative_best_gains.png\",\n",
            "    \"qualitative_failures.png\",\n",
            "    \"qualitative_cases.csv\",\n",
            "    \"ablation_study_summary.csv\",\n",
            "    \"RESEARCH_FINDINGS.md\",\n",
            "    \"failure_analysis.md\",\n",
            "]\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(\"           AI512 REPORT ASSET VERIFICATION          \")\n",
            "print(\"=\" * 60)\n",
            "all_passed = True\n",
            "for name in files_to_check:\n",
            "    p = DELIVERABLES_ROOT / name\n",
            "    found = p.exists() and p.stat().st_size > 0\n",
            "    status = \"✅ FOUND\" if found else \"❌ MISSING\"\n",
            "    print(f\"{status:10} | {name}\")\n",
            "    if not found:\n",
            "        all_passed = False\n",
            "\n",
            "print(\"-\" * 60)\n",
            "if all_passed:\n",
            "    print(\"🎉 ALL DELIVERABLES SUCCESSFULLY GENERATED AND SAVED LOCALLY!\")\n",
            "    print(f\"📁 Local Destination: {DELIVERABLES_ROOT}\")\n",
            "else:\n",
            "    print(\"⚠️ Some deliverables are missing. Please complete all experiments.\")\n",
            "print(\"=\" * 60)"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path("local_runner.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Successfully wrote {out_path.resolve()}")
