# Ablations and sanity checks (test set, mean ± std over seeds)

### case118, S2 unseen outages (test base rate 0.080)

| Model | Seeds | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| gat [layers1] | 5 | 0.412 ± 0.254 | 0.390 ± 0.331 | 0.346 ± 0.159 | 0.335 ± 0.228 |
| gat [layers5] | 5 | 0.402 ± 0.204 | 0.489 ± 0.158 | 0.366 ± 0.182 | 0.387 ± 0.099 |
| gat [no_edge_feats] | 5 | 0.431 ± 0.190 | 0.559 ± 0.299 | 0.325 ± 0.149 | 0.392 ± 0.191 |
| gat [no_topology] | 5 | 0.388 ± 0.120 | 0.448 ± 0.235 | 0.365 ± 0.180 | 0.334 ± 0.124 |
| lgbm [shuffled] | 5 | 0.088 ± 0.020 | 0.716 ± 0.402 | 0.076 ± 0.022 | 0.135 ± 0.046 |
| gat [shuffled] | 5 | 0.131 ± 0.017 | 0.674 ± 0.277 | 0.127 ± 0.039 | 0.198 ± 0.040 |

### case118, S1 unseen operating points (test base rate 0.086)

| Model | Seeds | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| lgbm [shuffled] | 1 | 0.071 ± 0.000 | 0.529 ± 0.000 | 0.072 ± 0.000 | 0.127 ± 0.000 |
| gat [shuffled] | 1 | 0.054 ± 0.000 | 0.051 ± 0.000 | 0.014 ± 0.000 | 0.022 ± 0.000 |

### case30, S1 unseen operating points (test base rate 0.112)

| Model | Seeds | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| lgbm [shuffled] | 1 | 0.106 ± 0.000 | 0.995 ± 0.000 | 0.112 ± 0.000 | 0.202 ± 0.000 |
