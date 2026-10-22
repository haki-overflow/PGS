#!/usr/bin/env bash
# Step 1: inspect the test grids and check the label logic.
set -euo pipefail
python -m pgs.inspect_grids
python -m pytest -q tests/test_power_flow.py
