#!/usr/bin/env bash
# Everything from scratch: data, tests, main results, ablations, tables.
set -euo pipefail
python -m pgs.inspect_cases
python -m pgs.generate --grid case30
python -m pgs.generate --grid case118
python -m pytest -q
python -m pgs.splits
rm -rf results/runs
bash scripts/day6.sh
bash scripts/day7.sh
