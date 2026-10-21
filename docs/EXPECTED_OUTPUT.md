# Expected output, day by day

Reference output from the development run (4-CPU Linux container, no GPU, Python 3.13, the
pinned versions in `requirements.txt`). Use it to check your local runs.

**What must match exactly:** grid facts (Day 1), sample counts, insecure rates and **label
hashes** (Days 2-3), split sizes (Day 3), and the rule-based rows (`majority`, `cont_prior`,
`base_loading`, `dc_rule`).
**What may differ a little:** timings (`ms_per_pf`, `sec`, `time`), `device` (yours should say
`cuda`), and the neural models (MLP, GCN, GraphSAGE, GAT): GPU arithmetic is not bit-identical
to CPU. Expect small differences on S1 and larger ones on S2, where test sets are small.
LR and LightGBM should match to about ±0.01.
**If a label hash differs, stop and paste the output:** the data would not be the same.

Run everything from the repo root, inside the virtual environment, after
`pip install -r requirements.txt`.

---

## Day 1: `bash scripts/day1.sh` (about 1 min)

```
        case        buses        lines       trafos         gens real_ratings    islanding         vmin         vmax     max_load  base_secure    ms_per_pf
      case14           14           15            5            5         0/20            1        1.010        1.090        0.720        False           58
      case30           30           41            0            6        41/41            3        0.961        1.000        1.088        False           43
      case39           39           35           11           10        46/46           11        0.982        1.064        0.764        False           60
     case118          118          173           13           54        0/186            9        0.943        1.050        0.909         True           75
.......                                                                  [100%]
7 passed
```

## Day 2: `bash scripts/day2.sh` (about 2-3 min)

```
case30: 785 candidates -> base-secure 38.2%; kept 300 OPs x 38 outages = 11400 samples (3 islanding outages excluded)
  insecure 12.1% | overload 6.4% | v_viol 8.0% | non-converged 0 | attempts {1: 11400}
  load factor of kept OPs: min 0.70 median 0.82 max 1.07
  insecure rate per OP: min 2.6% median 10.5% max 47.4%
  outages that are insecure in every OP: 1, in no OP: 12
  time 138.3 s, label hash 42676ea248
......                                                                   [100%]
6 passed
```

## Day 3: `bash scripts/day3.sh` (about 12 min on 4 cores; faster with more)

```
case118: 259 candidates -> base-secure 46.3%; kept 120 OPs x 177 outages = 21240 samples (9 islanding outages excluded)
  insecure 9.1% | overload 3.2% | v_viol 6.1% | non-converged 0 | attempts {1: 21240}
  load factor of kept OPs: min 0.70 median 0.86 max 1.12
  insecure rate per OP: min 3.4% median 6.8% max 51.4%
  outages that are insecure in every OP: 5, in no OP: 53
  time 693.2 s, label hash 97790e93b9
..................                                                       [100%]
18 passed
case30   op    seed0 train/val/test = [7980, 1710, 1710]; insecure % over 5 seeds: train 11.4-12.5, test 11.1-12.3
case30   cont  seed0 train/val/test = [4620, 360, 360]; insecure % over 5 seeds: train 8.7-14.5, test 6.7-16.7
case118  op    seed0 train/val/test = [14868, 3186, 3186]; insecure % over 5 seeds: train 8.6-9.9, test 6.8-9.3
case118  cont  seed0 train/val/test = [8820, 648, 648]; insecure % over 5 seeds: train 8.3-10.4, test 6.5-9.7
```

## Day 4: `bash scripts/day4.sh` (about 3-5 min)

Columns: test PR-AUC, recall, precision, F1, base rate, seconds.

```
......                                                                   [100%]
6 passed
case30: 11400 samples, device cuda
split model        seed  PR-AUC  recall    prec      F1   base    sec
op    majority        0   0.112   0.000   0.000   0.000  0.112    0.0
op    cont_prior      0   0.784   0.969   0.517   0.674  0.112    0.0
op    base_loading    0   0.186   0.464   0.213   0.292  0.112    0.0
op    dc_rule         0   0.452   0.578   0.521   0.548  0.112    0.0
op    lr              0   0.967   0.906   0.892   0.899  0.112    5.2
op    lgbm            0   0.997   0.964   0.974   0.969  0.112    4.6
op    mlp             0   0.983   0.964   0.869   0.914  0.112    7.9
cont  majority        0   0.167   0.000   0.000   0.000  0.167    0.0
cont  cont_prior      0   0.167   1.000   0.167   0.286  0.167    0.0
cont  base_loading    0   0.219   0.367   0.250   0.297  0.167    0.0
cont  dc_rule         0   0.738   0.600   0.735   0.661  0.167    0.0
cont  lr              0   0.842   0.783   0.547   0.644  0.167    4.1
cont  lgbm            0   0.248   0.350   0.204   0.258  0.167    2.5
cont  mlp             0   0.224   0.417   0.243   0.307  0.167    2.6
case118: 21240 samples, device cuda
split model        seed  PR-AUC  recall    prec      F1   base    sec
op    majority        0   0.086   0.000   0.000   0.000  0.086    0.0
op    cont_prior      0   0.798   0.657   0.833   0.735  0.086    0.0
op    base_loading    0   0.143   0.226   0.175   0.197  0.086    0.0
op    dc_rule         0   0.272   0.354   0.226   0.276  0.086    0.0
op    lr              0   0.883   0.759   0.788   0.773  0.086   16.0
op    lgbm            0   0.962   0.843   0.913   0.877  0.086   13.8
op    mlp             0   0.942   0.788   0.904   0.842  0.086   30.6
cont  majority        0   0.097   0.000   0.000   0.000  0.097    0.0
cont  cont_prior      0   0.097   1.000   0.097   0.177  0.097    0.0
cont  base_loading    0   0.150   0.206   0.181   0.193  0.097    0.0
cont  dc_rule         0   0.274   0.317   0.238   0.272  0.097    0.0
cont  lr              0   0.270   0.413   0.302   0.349  0.097   12.3
cont  lgbm            0   0.244   0.175   0.379   0.239  0.097   13.0
cont  mlp             0   0.232   0.349   0.275   0.308  0.097   17.8
case30: 11400 samples, device cuda
split model        seed  PR-AUC  recall    prec      F1   base    sec
op    lgbm_shuffled    0   0.106   0.995   0.112   0.202  0.112    3.1
```

The last line is the leakage check. It must stay near the base rate (0.112).

## Day 5: `bash scripts/day5.sh` (CPU here: about 25 min for case30 + 75 min for case118; a GPU should be much faster)

```
......                                                                   [100%]
6 passed
case30: 11400 samples, device cuda
split model        seed  PR-AUC  recall    prec      F1   base    sec
op    gcn             0   0.989   0.969   0.916   0.942  0.112   89.7
op    sage            0   0.989   0.943   0.958   0.950  0.112   74.7
op    gat             0   0.994   0.984   0.913   0.947  0.112  207.4
cont  gcn             0   0.246   0.267   0.246   0.256  0.167   20.6
cont  sage            0   0.179   0.100   0.081   0.090  0.167   70.0
cont  gat             0   0.160   0.000   0.000   0.000  0.167   29.4
case118: 21240 samples, device cuda
split model        seed  PR-AUC  recall    prec      F1   base    sec
op    gcn             0   0.947   0.799   0.909   0.850  0.086  856.4
op    sage            0   0.950   0.745   0.990   0.850  0.086  505.8
op    gat   (still running when this file was written; filled in when it finishes)
cont  gcn             0   0.402   0.429   0.474   0.450  0.097  104.4
cont  sage            0   0.468   0.175   0.500   0.259  0.097  175.5
cont  gat             0   0.406   0.254   0.364   0.299  0.097  716.9
```

GNN lines on a GPU will not match to the third decimal. What should hold: on case118 S2, the
GNNs beat every Day 4 baseline row (best 0.274); on S1 they land near LightGBM.

## Days 6-7

No reference output yet: these runs (5 seeds × all models, then ablations) are meant for your
GPU. Paste the final tables (`python -m pgs.aggregate` prints them) and they go into the guide.
