#!/usr/bin/env bash
# Step 3: generate the case118 dataset, test splits and graph building, report splits.
set -euo pipefail
python -m pgs.generate --grid case118
python -m pytest -q tests/test_splits_dataset.py
python -m pgs.splits
