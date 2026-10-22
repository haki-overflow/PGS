#!/usr/bin/env bash
# Step 1: inspect the test grids and check the label logic.
set -euo pipefail
python -m pgs.inspect_cases
python -m pytest -q tests/test_grid.py
