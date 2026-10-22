"""Train and evaluate models on one grid, for given splits and seeds.

Usage:
  python -m pgs.train_and_evaluate --grid case30 --splits op,cont --models rules,lr,lgbm,mlp --seeds 0
  python -m pgs.train_and_evaluate --grid case118 --models gcn,sage,gat --seeds 0-4

Each (grid, split, model, seed) writes results/runs/<name>.json with validation and
test metrics. The decision threshold is tuned on validation only.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from torch_geometric.loader import DataLoader as GraphLoader

from pgs import config
from pgs.features import Scaler, flat_features, graph_dataset, load_grid
from pgs.metrics import evaluate, tune_threshold
from pgs.models import GNN, MLP
from pgs.train_test_splits import make_split

RULES = ["majority", "cont_prior", "base_loading", "dc_rule"]
CLASSIC = ["lr", "lgbm", "mlp"]
GNNS = ["gcn", "sage", "gat"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- rule baselines: a score per sample, no learning beyond a lookup table ------
def rule_scores(name, df, split):
    if name == "majority":
        return np.zeros(len(df))
    if name == "cont_prior":  # how often this outage was insecure in training
        tr = df.iloc[split["train"]]
        rate = tr.groupby("cont").insecure.mean()
        return df.cont.map(rate).fillna(tr.insecure.mean()).to_numpy()
    if name == "base_loading":  # worst pre-outage branch loading
        return df.base_max_loading.to_numpy()
    if name == "dc_rule":  # worst post-outage DC branch loading; a failed DC solve = alarm
        return df.dc_max_loading.fillna(1e6).to_numpy()
    raise ValueError(name)


# --- torch training loop shared by the MLP and the GNNs ---------------------------
def _scores(model, loader):
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, list):
                out.append(model(batch[0].to(DEVICE)).cpu())
            else:
                out.append(model(batch.to(DEVICE)).cpu())
    return torch.cat(out).numpy()


def fit_torch(model, train_loader, val_loader, y_val, pos_weight, max_epochs=60, patience=8):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE))
    best, best_state, bad, epoch = -1.0, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            if isinstance(batch, list):
                x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            else:
                x = batch.to(DEVICE)
                y = x.y
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        ap = evaluate(y_val, _scores(model, val_loader), 0.0)["pr_auc"]
        if ap > best:
            best, bad = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return epoch


def run(grid, split_kind, model_name, seed, df, arrays, graphs=None, opts=None):
    opts = opts or {}
    np.random.seed(seed)
    torch.manual_seed(seed)
    split = make_split(df, split_kind, seed)
    y = df.insecure.to_numpy().astype(np.float32)
    y_fit = y.copy()
    if opts.get("shuffle_labels"):  # sanity check: destroy the input-label link
        rng = np.random.default_rng(seed)
        for part in ("train", "val"):
            idx = split[part]
            y_fit[idx] = rng.permutation(y_fit[idx])
    tr, va, te = split["train"], split["val"], split["test"]
    pos_weight = float((1 - y_fit[tr]).sum() / max(y_fit[tr].sum(), 1))
    t0, extra = time.time(), {}

    if model_name in RULES:
        s = rule_scores(model_name, df.assign(insecure=y_fit), split)
        s_val, s_test = s[va], s[te]
    elif model_name in ("lr", "lgbm", "mlp"):
        X = flat_features(arrays, df)
        if model_name == "lgbm":
            clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                 scale_pos_weight=pos_weight, random_state=seed, verbose=-1)
            clf.fit(X[tr], y_fit[tr])
            s_val, s_test = clf.predict_proba(X[va])[:, 1], clf.predict_proba(X[te])[:, 1]
        else:
            X = Scaler().fit(X[tr])(X).astype(np.float32)
            if model_name == "lr":
                clf = LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0)
                clf.fit(X[tr], y_fit[tr])
                s_val, s_test = clf.decision_function(X[va]), clf.decision_function(X[te])
            else:
                ds = lambda i: torch.utils.data.TensorDataset(torch.from_numpy(X[i]),
                                                              torch.from_numpy(y_fit[i]))
                g = torch.Generator().manual_seed(seed)
                train_loader = torch.utils.data.DataLoader(ds(tr), batch_size=256, shuffle=True,
                                                           generator=g)
                loader = lambda i: torch.utils.data.DataLoader(ds(i), batch_size=1024)
                model = MLP(X.shape[1])
                extra["epochs"] = fit_torch(model, train_loader, loader(va), y_fit[va], pos_weight)
                s_val, s_test = _scores(model, loader(va)), _scores(model, loader(te))
    elif model_name in GNNS:
        if opts.get("shuffle_labels"):
            graphs = [gr.clone() for gr in graphs]
            for i in np.concatenate([tr, va]):
                graphs[i].y = torch.tensor([y_fit[i]])
        train_loader = GraphLoader([graphs[i] for i in tr], batch_size=128, shuffle=True,
                                   generator=torch.Generator().manual_seed(seed))
        loader = lambda idx: GraphLoader([graphs[i] for i in idx], batch_size=512)
        model = GNN(model_name, graphs[0].x.shape[1], graphs[0].edge_attr.shape[1],
                    layers=opts.get("layers", 3), use_edges=not opts.get("no_edge_feats"),
                    no_topology=opts.get("no_topology", False))
        extra["epochs"] = fit_torch(model, train_loader, loader(va), y_fit[va], pos_weight)
        extra["n_params"] = sum(p.numel() for p in model.parameters())
        s_val = _scores(model, loader(va))
        t1 = time.time()
        s_test = _scores(model, loader(te))
        extra["test_predict_seconds"] = round(time.time() - t1, 3)
    else:
        raise ValueError(model_name)

    thr = np.inf if model_name == "majority" else tune_threshold(y_fit[va], s_val)
    return dict(grid=grid, split=split_kind, model=model_name, seed=seed, threshold=thr,
                val=evaluate(y[va], s_val, thr), test=evaluate(y[te], s_test, thr),
                seconds=round(time.time() - t0, 1), device=DEVICE, **extra)


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--splits", default="op,cont")
    ap.add_argument("--models", default="rules,lr,lgbm,mlp")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--tag", default="", help="suffix for ablation runs")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--no-edge-feats", action="store_true")
    ap.add_argument("--no-topology", action="store_true")
    ap.add_argument("--shuffle-labels", action="store_true")
    a = ap.parse_args()
    models = []
    for m in a.models.split(","):
        models += RULES if m == "rules" else [m]
    opts = dict(layers=a.layers, no_edge_feats=a.no_edge_feats, no_topology=a.no_topology,
                shuffle_labels=a.shuffle_labels)

    df, arrays = load_grid(a.grid)
    out_dir = os.path.join(config.RESULTS_DIR, "runs")
    os.makedirs(out_dir, exist_ok=True)
    print(f"{a.grid}: {len(df)} samples, device {DEVICE}")
    print(f"{'split':5s} {'model':12s} {'seed':>4s} {'PR-AUC':>7s} {'recall':>7s} "
          f"{'prec':>7s} {'F1':>7s} {'base':>6s} {'sec':>6s}")
    for split_kind in a.splits.split(","):
        for seed in parse_seeds(a.seeds):
            graphs = None
            if any(m in GNNS for m in models):
                graphs = graph_dataset(arrays, df, make_split(df, split_kind, seed)["train"])
            for m in models:
                r = run(a.grid, split_kind, m, seed, df, arrays, graphs, opts)
                r["tag"] = a.tag
                name = f"{a.grid}_{split_kind}_{m}{'_' + a.tag if a.tag else ''}_s{seed}"
                with open(os.path.join(out_dir, name + ".json"), "w") as f:
                    json.dump(r, f, indent=1)
                t = r["test"]
                print(f"{split_kind:5s} {m + ('_' + a.tag if a.tag else ''):12s} {seed:4d} "
                      f"{t['pr_auc']:7.3f} {t['recall']:7.3f} {t['precision']:7.3f} "
                      f"{t['f1']:7.3f} {t['base_rate']:6.3f} {r['seconds']:6.1f}", flush=True)


if __name__ == "__main__":
    main()
