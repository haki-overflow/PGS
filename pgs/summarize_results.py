"""Collect results/runs/*.json into mean +/- std tables over seeds.

Usage: python -m pgs.summarize_results                  (write the tables)
       python -m pgs.summarize_results --best-gnn case118 (print the GNN chosen on validation)
Writes results/main_table.md (untagged runs) and results/ablations.md (tagged runs).
"""
import glob
import json
import os
import sys

import pandas as pd

from pgs import config
from pgs.train_and_evaluate import CLASSIC, GNNS, RULES

ORDER = RULES + CLASSIC + GNNS
METRICS = ["pr_auc", "recall", "precision", "f1"]


def load_runs(results_dir=config.RESULTS_DIR):
    rows = []
    for path in glob.glob(os.path.join(results_dir, "runs", "*.json")):
        with open(path) as f:
            r = json.load(f)
        rows.append(dict(grid=r["grid"], split=r["split"], model=r["model"],
                         tag=r.get("tag", ""), seed=r["seed"], seconds=r["seconds"],
                         base_rate=r["test"]["base_rate"], val_pr_auc=r["val"]["pr_auc"],
                         **{m: r["test"][m] for m in METRICS}))
    return pd.DataFrame(rows)


def best_gnn(grid, split="cont", results_dir=config.RESULTS_DIR):
    """GNN with the highest mean VALIDATION PR-AUC (the test set is never used to choose)."""
    df = load_runs(results_dir)
    df = df[(df.grid == grid) & (df.split == split) & (df.tag == "") & df.model.isin(GNNS)]
    return df.groupby("model").val_pr_auc.mean().idxmax()


def table(df):
    g = df.groupby(["grid", "split", "model", "tag"])
    out = g[METRICS].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["seeds"] = g.seed.nunique()
    out["base_rate"] = g.base_rate.mean()
    out = out.reset_index()
    out["order"] = out.model.map({m: i for i, m in enumerate(ORDER)})
    return out.sort_values(["grid", "split", "tag", "order"]).drop(columns="order")


def to_markdown(t):
    lines = []
    for (grid, split), part in t.groupby(["grid", "split"], sort=False):
        name = {"op": "S1 unseen operating points", "cont": "S2 unseen outages"}[split]
        lines += [f"### {grid}, {name} (test base rate {part.base_rate.mean():.3f})", "",
                  "| Model | Seeds | PR-AUC | Recall | Precision | F1 |",
                  "|---|---|---|---|---|---|"]
        for _, r in part.iterrows():
            label = r.model + (f" [{r.tag}]" if r.tag else "")
            cells = [f"{r[m + '_mean']:.3f} ± {0 if pd.isna(r[m + '_std']) else r[m + '_std']:.3f}"
                     for m in METRICS]
            lines.append(f"| {label} | {r.seeds} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--best-gnn":
        print(best_gnn(sys.argv[2]))
        return
    df = load_runs()
    if df.empty:
        raise SystemExit("no runs found in results/runs")
    t = table(df)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    main_t, abl_t = t[t.tag == ""], t[t.tag != ""]
    with open(os.path.join(config.RESULTS_DIR, "main_table.md"), "w") as f:
        f.write("# Main results (test set, mean ± std over seeds)\n\n" + to_markdown(main_t))
    if len(abl_t):
        with open(os.path.join(config.RESULTS_DIR, "ablations.md"), "w") as f:
            f.write("# Ablations and sanity checks (test set, mean ± std over seeds)\n\n"
                    + to_markdown(abl_t))
    cols = ["grid", "split", "model", "tag", "seeds", "pr_auc_mean", "pr_auc_std",
            "recall_mean", "f1_mean"]
    print(t[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
