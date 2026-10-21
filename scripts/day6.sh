#!/usr/bin/env bash
# Day 6: every model, both grids, both splits, 5 seeds; then the mean +/- std tables.
set -euo pipefail
for grid in case30 case118; do
  python -m pgs.run --grid "$grid" --splits op,cont --models rules,lr,lgbm,mlp,gcn,sage,gat --seeds 0-4
done
python -m pgs.aggregate
