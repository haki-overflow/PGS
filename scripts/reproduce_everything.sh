#!/usr/bin/env bash
# Everything from scratch: data, tests, main results, ablations, tables.
set -euo pipefail
python -m pgs.inspect_grids
python -m pgs.generate_dataset --grid case30
python -m pgs.generate_dataset --grid case118
python -m pytest -q
python -m pgs.train_test_splits
rm -rf results/runs
bash scripts/step6_train_all_models_5_seeds.sh
bash scripts/step7_run_ablations_and_sanity_checks.sh
