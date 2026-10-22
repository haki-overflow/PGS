#!/usr/bin/env bash
# Step 7: sanity checks and ablations on case118, both splits, 5 seeds.
# The GNN to ablate is picked by validation PR-AUC on S2, never by test results.
set -euo pipefail
BEST=$(python -m pgs.aggregate --best-gnn case118)
echo "GNN chosen on validation: $BEST"
run() { python -m pgs.run --grid case118 --splits op,cont --seeds 0-4 "$@"; }
run --models "$BEST,lgbm" --shuffle-labels --tag shuffled
run --models "$BEST" --no-topology --tag no_topology
run --models gat --no-edge-feats --tag no_edge_feats
run --models "$BEST" --layers 1 --tag layers1
run --models "$BEST" --layers 5 --tag layers5
python -m pgs.aggregate
