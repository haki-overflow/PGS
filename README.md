# GNN-based N-1 Contingency Screening (Static Security Assessment) on IEEE Test Grids

Predict whether a single-branch outage leaves a power grid within its steady-state
limits (bus voltages in [0.94, 1.06] p.u., no branch above its rating, power flow
converges), using graph neural networks. Labels come from AC power flow in pandapower.

**Scope, stated plainly:** this is *static security*, not dynamic or transient stability.
The grids are public IEEE test cases, not real networks.

Beginner explanation, decisions, results and limitations: [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The pinned `torch` wheel from PyPI is a CUDA build. A GPU is not required, since these graphs are small.

## Run

| Step | Command |
|---|---|
| Day 1: inspect grids, test label logic | `bash scripts/day1.sh` |
| Day 2: generate case30 data (about 2 min on 4 cores) | `bash scripts/day2.sh` |
| Day 3: generate case118 data (about 12 min on 4 cores), test splits | `bash scripts/day3.sh` |
| Day 4: rule and classic-ML baselines, seed 0 (about 5 min) | `bash scripts/day4.sh` |

## Status

Day 4 of 9 done. Results: see the guide, Section 6.

## Tools

Code, tests and documentation were written by an AI assistant (Claude Code) under my
direction. I chose the scope and label definition, ran every experiment on my machine,
checked the outputs, and reviewed the design decisions recorded in the guide.
