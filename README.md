# GNN-based N-1 Contingency Screening (Static Security Assessment) on IEEE Test Grids

Can a graph neural network predict whether losing one line or transformer leaves a power
grid within its steady-state limits? This project generates 32,640 labelled
(operating point, branch outage) samples on the IEEE 30- and 118-bus grids with AC power
flow (pandapower). It then compares GCN, GraphSAGE and GATv2 against rule-based,
DC power-flow and classic ML baselines, using leakage-safe splits and 5 seeds.

**Scope, stated plainly:** this is *static security* (limits after the outage), not
dynamic or transient stability. The grids are public IEEE test cases, not real networks.

A beginner explanation, every design decision, all results and limitations are in
[`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md).

## Label

A sample is **insecure** if, after the outage, AC power flow (generator reactive limits
enforced) gives any bus voltage outside [0.94, 1.06] p.u., any branch above 100% of its
MVA rating, or fails to converge. Only operating points secure before the outage are kept.
Outages that split the grid into islands are excluded.

| Grid | Samples | Insecure |
|---|---|---|
| case30 (real line ratings) | 11,400 (300 operating points × 38 outages) | 12.1% |
| case118 (planning-style synthetic ratings) | 21,240 (120 × 177) | 9.1% |

## Results (test PR-AUC, mean ± std over 5 seeds)

Two test types: **S1** = operating points not seen in training (all outages seen);
**S2** = outages not seen in training (5 rotating outage folds, each outage tested once).

| Model | case30 S1 | case30 S2 | case118 S1 | case118 S2 |
|---|---|---|---|---|
| Base rate (random) | 0.117 | 0.115 | 0.079 | 0.080 |
| DC power-flow rule | 0.445 ± 0.041 | **0.550 ± 0.179** | 0.291 ± 0.068 | 0.299 ± 0.101 |
| LightGBM | **0.996 ± 0.002** | 0.437 ± 0.244 | **0.964 ± 0.015** | 0.237 ± 0.041 |
| MLP | 0.973 ± 0.009 | 0.208 ± 0.043 | 0.940 ± 0.011 | 0.196 ± 0.056 |
| GCN | 0.983 ± 0.007 | 0.452 ± 0.220 | 0.955 ± 0.015 | 0.367 ± 0.167 |
| GraphSAGE | 0.981 ± 0.007 | 0.463 ± 0.260 | 0.952 ± 0.018 | 0.390 ± 0.162 |
| GATv2 | 0.985 ± 0.006 | 0.369 ± 0.184 | 0.956 ± 0.014 | **0.404 ± 0.172** |

All models and recall/precision/F1: [`results/main_table.md`](results/main_table.md).
Ablations: [`results/ablations.md`](results/ablations.md).

**Findings**
- **Known outages (S1):** LightGBM has the best average on both grids; the GNNs come close and beat it in only 3 of 30 per-fold comparisons (all on case118).
- **Unseen outages, case118:** the GNNs have the best average (0.37-0.40 vs 0.30 for the
  best baseline), but with high variance; they beat the best baseline in 3 of 5 folds.
- **Unseen outages, case30:** no reliable winner; with 7-8 outages per fold, results swing widely.
- **Ablations (GAT, case118 S2):** removing all grid connections, the branch features, or
  changing depth (1 or 5 layers) made no measurable difference. **The GNN gain is not
  driven by grid topology.** It most likely comes from the shared per-bus model plus the
  outage features.
- **Sanity checks:** shuffled labels give chance-level LightGBM (0.088 vs 0.080).

**Limitations:** static security only; synthetic IEEE grids; synthetic ratings on case118;
single-branch outages only; small unseen-outage test sets; decision thresholds tuned on
validation outages transfer poorly (low and variable recall); no speed benchmark against an
optimised power-flow solver. Full list in the guide, Section 7.

## Repository layout

```
pgs/power_flow.py          load IEEE cases, ratings, AC/DC power flow, the security label
pgs/inspect_grids.py       Step 1: facts about each test grid
pgs/generate_dataset.py    random operating points + every N-1 outage -> labelled samples
pgs/features.py            model inputs: graph view (PyG) and flat view
pgs/train_test_splits.py   S1 (unseen operating points) and S2 (rotating outage folds)
pgs/models.py              MLP, GCN, GraphSAGE, GATv2
pgs/train_and_evaluate.py  trains/evaluates any model for given grid, split, seeds
pgs/summarize_results.py   mean ± std tables over seeds
pgs/metrics.py, config.py  PR-AUC/recall/F1 + threshold tuning; all settings
scripts/stepN_*.sh         one script per step (below)
tests/                     39 tests (label logic, leakage-safe splits, features, models)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt    # Python 3.11+, versions pinned
```

A GPU is not required. The pinned `torch` wheel is a CUDA 13 build and falls back to CPU.

## Run

| Step | Command |
|---|---|
| 1. Check grids and the label logic | `bash scripts/step1_check_grids_and_label.sh` |
| 2. Generate the case30 dataset (~2 min) | `bash scripts/step2_generate_case30_dataset.sh` |
| 3. Generate the case118 dataset and splits (~12 min on 4 cores) | `bash scripts/step3_generate_case118_dataset_and_splits.sh` |
| 4. Baselines, seed 0 (~5 min) | `bash scripts/step4_train_baselines_seed0.sh` |
| 5. GNNs, seed 0 | `bash scripts/step5_train_gnns_seed0.sh` |
| 6. All models, 5 seeds (hours on CPU) | `bash scripts/step6_train_all_models_5_seeds.sh` |
| 7. Ablations and sanity checks | `bash scripts/step7_run_ablations_and_sanity_checks.sh` |
| Everything from scratch | `bash scripts/reproduce_everything.sh` |

Reference output for each step: [`docs/EXPECTED_OUTPUT.md`](docs/EXPECTED_OUTPUT.md).
Data label hashes must match: case30 `42676ea248`, case118 `97790e93b9`.

## Tools

Code, tests and documentation were written by an AI assistant (Claude Code) under my
direction, and the AI also ran the experiments, on CPU in a cloud container. I chose the
scope and the label definition, reviewed the design decisions recorded in the guide, and
checked the results.
