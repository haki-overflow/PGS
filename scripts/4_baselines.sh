#!/usr/bin/env bash
# Step 4: rule and classic-ML baselines on both grids, seed 0, plus the leakage check.
set -euo pipefail
python -m pytest -q tests/test_metrics_models.py
for grid in case30 case118; do
  python -m pgs.run --grid "$grid" --splits op,cont --models rules,lr,lgbm,mlp --seeds 0
done
python -m pgs.run --grid case30 --splits op --models lgbm --seeds 0 --shuffle-labels --tag shuffled
