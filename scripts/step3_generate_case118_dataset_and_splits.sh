#!/usr/bin/env bash
# Step 3: generate the case118 dataset, test splits and graph building, report splits.
set -euo pipefail
python -m pgs.generate_dataset --grid case118
python -m pytest -q tests/test_features_and_splits.py
python -m pgs.train_test_splits
