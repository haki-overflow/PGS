#!/usr/bin/env bash
# Step 5: GCN, GraphSAGE and GATv2 on both grids, seed 0.
set -euo pipefail
python -m pytest -q tests/test_metrics_and_models.py
for grid in case30 case118; do
  python -m pgs.train_and_evaluate --grid "$grid" --splits op,cont --models gcn,sage,gat --seeds 0
done
