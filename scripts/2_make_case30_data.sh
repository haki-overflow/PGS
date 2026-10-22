#!/usr/bin/env bash
# Step 2: generate the case30 dataset and test the generator.
set -euo pipefail
python -m pgs.generate --grid case30
python -m pytest -q tests/test_generate.py
