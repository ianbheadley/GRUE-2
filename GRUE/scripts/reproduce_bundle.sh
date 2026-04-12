#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[1/6] Regenerating missing-training summary"
python3 missing_training_analysis.py

echo "[2/6] Regenerating layer concept report"
python3 layer_concept_report.py

echo "[3/6] Regenerating alpha sweep summary"
python3 alpha_sweep_analysis.py \
  --datasets color_blocks cifar10 \
  --max_images 300 \
  --random_trials 3 \
  --output results/alpha_sweep_summary.json

echo "[4/6] Regenerating CIFAR dissection outputs"
python3 activation_dissection.py --model_dir models_cifar10 --output_dir results/dissection

echo "[5/6] Regenerating implicit color analysis in CIFAR"
python3 cifar_implicit_color_analysis.py --per_class_limit 30 --output results/cifar_implicit_color_summary.json

echo "[6/6] Syncing fresh results into data/ and regenerating paper figures"
cp results/missing_training_summary.json data/missing_training_summary.json
cp results/layer_concept_report.json data/layer_concept_report.json
cp results/alpha_sweep_summary.json data/alpha_sweep_summary.json
cp results/dissection/summary.json data/summary.json
cp results/cifar_implicit_color_summary.json data/cifar_implicit_color_summary.json
python3 scripts/generate_paper_figures.py

echo
echo "Bundle reproduction complete."
echo "Paper figures: $ROOT/figures"
echo "Primary summaries: $ROOT/results and $ROOT/data"
