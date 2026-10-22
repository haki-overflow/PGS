#!/usr/bin/env bash
# Step 7: sanity checks and ablations on case118, 5 seeds.
# The GNN to ablate is picked by validation PR-AUC on S2, never by test results.
# Ablations run on S2 (unseen outages) only: that is where the GNNs led, and running
# both splits would take 8+ hours on CPU (decision D21). The shuffled-label check
# also runs once on S1.
set -euo pipefail
BEST=$(python -m pgs.summarize_results --best-gnn case118)
echo "GNN chosen on validation: $BEST"
run() { python -m pgs.train_and_evaluate --grid case118 --splits cont --seeds 0-4 "$@"; }
run --models "$BEST,lgbm" --shuffle-labels --tag shuffled
python -m pgs.train_and_evaluate --grid case118 --splits op --seeds 0 \
  --models "$BEST,lgbm" --shuffle-labels --tag shuffled
run --models "$BEST" --no-topology --tag no_topology
run --models gat --no-edge-feats --tag no_edge_feats
run --models "$BEST" --layers 1 --tag layers1
run --models "$BEST" --layers 5 --tag layers5
python -m pgs.summarize_results
echo STEP7_DONE
