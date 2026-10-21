"""Turn the generated arrays into model inputs.

Two views of the same sample (operating point `op`, outaged branch `cont`):
- graph: PyG Data. Nodes = buses, edges = in-service branches (both directions).
  The outaged branch is removed, and its two end buses get a flag.
- flat: one fixed-length vector for the classic baselines: all node features,
  all pre-outage branch features, and a one-hot of the outaged branch.

Feature scaling is fitted on training samples only.
"""
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from pgs import config


def load_grid(grid, data_dir=config.DATA_DIR):
    d = os.path.join(data_dir, grid)
    df = pd.read_parquet(os.path.join(d, "samples.parquet"))
    arrays = dict(np.load(os.path.join(d, "arrays.npz")))
    return df, arrays


class Scaler:
    """Standardise columns; columns with zero spread are left unscaled."""

    def fit(self, x):
        x = x.reshape(-1, x.shape[-1])
        self.mean, self.std = x.mean(0), x.std(0)
        flat = self.std < 1e-8
        self.mean[flat], self.std[flat] = 0.0, 1.0
        return self

    def __call__(self, x):
        return (x - self.mean) / self.std


def node_inputs(arrays, df):
    """Per-sample node features: pre-outage bus features + outage-endpoint flag."""
    ei = arrays["edge_index"]
    x = arrays["node_x"][df.op.to_numpy()]
    flag = np.zeros(x.shape[:2] + (1,))
    rows = np.arange(len(df))
    flag[rows, ei[0, df.cont], 0] = 1
    flag[rows, ei[1, df.cont], 0] = 1
    return np.concatenate([x, flag], axis=-1)


def edge_inputs(arrays, df):
    """Per-sample branch features (all branches, outaged one still included here)."""
    n = len(df)
    static = np.broadcast_to(arrays["edge_static"], (n,) + arrays["edge_static"].shape)
    return np.concatenate([static, arrays["edge_base"][df.op.to_numpy()]], axis=-1)


def flat_features(arrays, df):
    n_br = arrays["edge_index"].shape[1]
    x = arrays["node_x"][df.op.to_numpy()].reshape(len(df), -1)
    e = arrays["edge_base"][df.op.to_numpy()].reshape(len(df), -1)
    onehot = np.eye(n_br)[df.cont.to_numpy()]
    return np.concatenate([x, e, onehot], axis=1)


def graph_dataset(arrays, df, train_idx):
    """List of PyG graphs for every row of df, scaled with train-only statistics."""
    nx_all = node_inputs(arrays, df)
    ex_all = edge_inputs(arrays, df)
    nscale = Scaler().fit(nx_all[train_idx])
    escale = Scaler().fit(ex_all[train_idx])
    nx_all = nscale(nx_all).astype(np.float32)
    ex_all = escale(ex_all).astype(np.float32)
    p_col = len(config.EDGE_STATIC_FEATURES) + config.EDGE_BASE_FEATURES.index("p_from_base")

    ei = arrays["edge_index"]
    graphs = []
    for i, (cont, y) in enumerate(zip(df.cont.to_numpy(), df.insecure.to_numpy())):
        keep = np.arange(ei.shape[1]) != cont
        fwd = ex_all[i, keep]
        rev = fwd.copy()
        rev[:, p_col] = -rev[:, p_col] - 2 * escale.mean[p_col] / escale.std[p_col]
        edge_index = np.concatenate([ei[:, keep], ei[::-1, keep]], axis=1)
        graphs.append(Data(x=torch.from_numpy(nx_all[i]),
                           edge_index=torch.from_numpy(edge_index.copy()).long(),
                           edge_attr=torch.from_numpy(np.concatenate([fwd, rev])),
                           y=torch.tensor([float(y)])))
    return graphs
