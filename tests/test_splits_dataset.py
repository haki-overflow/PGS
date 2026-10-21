import numpy as np
import pandas as pd
import pytest

from pgs import config
from pgs.dataset import Scaler, flat_features, graph_dataset, node_inputs
from pgs.splits import make_split

N_OPS, N_BUS = 20, 4
EDGES = np.array([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]])  # ring plus a chord
CONTS = [0, 1, 2, 3, 4]


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    df = pd.DataFrame([(o, c) for o in range(N_OPS) for c in CONTS], columns=["op", "cont"])
    df["insecure"] = rng.random(len(df)) < 0.2
    arrays = dict(
        node_x=rng.normal(size=(N_OPS, N_BUS, len(config.NODE_FEATURES))),
        edge_base=rng.normal(size=(N_OPS, EDGES.shape[1], len(config.EDGE_BASE_FEATURES))),
        edge_static=rng.normal(size=(EDGES.shape[1], len(config.EDGE_STATIC_FEATURES))),
        edge_index=EDGES)
    return df, arrays


@pytest.mark.parametrize("seed", range(5))
def test_op_split_shares_no_operating_point(toy, seed):
    df, _ = toy
    s = make_split(df, "op", seed)
    ops = {k: set(df.op.iloc[v]) for k, v in s.items()}
    assert not ops["train"] & ops["val"] and not ops["train"] & ops["test"]
    assert not ops["val"] & ops["test"]
    assert sum(len(v) for v in s.values()) == len(df)  # every sample used once


@pytest.mark.parametrize("seed", range(5))
def test_cont_split_shares_no_outage_and_no_operating_point(toy, seed):
    df, _ = toy
    s = make_split(df, "cont", seed)
    for col in ("op", "cont"):
        sets = {k: set(df[col].iloc[v]) for k, v in s.items()}
        assert not sets["train"] & sets["test"], col
        assert not sets["train"] & sets["val"], col
        assert not sets["val"] & sets["test"], col


def test_splits_differ_by_seed(toy):
    df, _ = toy
    assert set(make_split(df, "op", 0)["test"]) != set(make_split(df, "op", 1)["test"])


def test_outage_flag_marks_both_end_buses(toy):
    df, arrays = toy
    x = node_inputs(arrays, df)
    row = df.index[(df.cont == 4)][0]  # branch 4 joins buses 0 and 2
    assert x[row, :, -1].tolist() == [1, 0, 1, 0]


def test_graph_removes_outaged_branch_and_keeps_both_directions(toy):
    df, arrays = toy
    graphs = graph_dataset(arrays, df, np.arange(len(df)))
    g = graphs[df.index[df.cont == 4][0]]
    pairs = set(map(tuple, g.edge_index.T.tolist()))
    assert (0, 2) not in pairs and (2, 0) not in pairs
    assert g.edge_index.shape[1] == 2 * (EDGES.shape[1] - 1)
    assert g.x.shape == (N_BUS, len(config.NODE_FEATURES) + 1)


def test_reverse_edge_flips_active_power(toy):
    df, arrays = toy
    graphs = graph_dataset(arrays, df, np.arange(len(df)))
    g = graphs[0]
    p = len(config.EDGE_STATIC_FEATURES) + config.EDGE_BASE_FEATURES.index("p_from_base")
    e = g.edge_index.shape[1] // 2
    # undo scaling via a fresh scaler fitted the same way, then compare raw values
    from pgs.dataset import edge_inputs
    s = Scaler().fit(edge_inputs(arrays, df))
    raw_fwd = g.edge_attr[:e, p].numpy() * s.std[p] + s.mean[p]
    raw_rev = g.edge_attr[e:, p].numpy() * s.std[p] + s.mean[p]
    assert np.allclose(raw_fwd, -raw_rev, atol=1e-5)


def test_scaler_uses_only_the_rows_it_is_fitted_on(toy):
    df, arrays = toy
    train = make_split(df, "op", 0)["train"]
    x = node_inputs(arrays, df)
    s = Scaler().fit(x[train])
    assert np.allclose(s.mean[:-1], x[train].reshape(-1, x.shape[-1]).mean(0)[:-1])
    assert not np.allclose(s.mean[:-1], x.reshape(-1, x.shape[-1]).mean(0)[:-1])


def test_flat_features_shape(toy):
    df, arrays = toy
    f = flat_features(arrays, df)
    n_br = EDGES.shape[1]
    assert f.shape == (len(df), N_BUS * len(config.NODE_FEATURES)
                       + n_br * len(config.EDGE_BASE_FEATURES) + n_br)
