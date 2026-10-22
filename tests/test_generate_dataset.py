import numpy as np
import pytest

from pgs import config
from pgs.generate_dataset import generate, label_hash


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    out = tmp_path_factory.mktemp("case30")
    df, meta = generate("case30", n_ops=3, max_candidates=100, workers=2, out_dir=str(out))
    return df, meta, np.load(out / "arrays.npz"), out


def test_shapes_and_counts(small):
    df, meta, a, _ = small
    n_cont = len(a["contingencies"])
    assert len(df) == 3 * n_cont == meta["n_samples"]
    assert a["node_x"].shape == (3, 30, len(config.NODE_FEATURES))
    assert a["edge_base"].shape == (3, 41, len(config.EDGE_BASE_FEATURES))
    assert a["edge_static"].shape == (41, len(config.EDGE_STATIC_FEATURES))
    assert a["edge_index"].shape == (2, 41)


def test_islanding_outages_excluded(small):
    df, _, a, _ = small
    assert len(a["islanding"]) == 3
    assert not set(df.cont) & set(a["islanding"].tolist())


def test_kept_operating_points_are_secure_before_outage(small):
    df, _, a, _ = small
    assert (df.base_max_loading <= 1.0).all()
    vm = a["node_x"][:, :, config.NODE_FEATURES.index("vm_base")]
    assert vm.min() >= 0.94 and vm.max() <= 1.06


def test_features_are_finite_and_one_hot_bus_type(small):
    _, _, a, _ = small
    for k in ("node_x", "edge_base", "edge_static"):
        assert np.isfinite(a[k]).all(), k
    i = config.NODE_FEATURES.index("is_slack")
    assert (a["node_x"][:, :, i:i + 3].sum(-1) == 1).all()
    assert (a["node_x"][:, :, i].sum(-1) == 1).all()  # exactly one slack bus


def test_inputs_hold_no_post_outage_quantities(small):
    _, _, a, _ = small
    assert set(a.files) == {"node_x", "edge_base", "edge_static", "edge_index",
                            "contingencies", "islanding"}


def test_generation_is_deterministic(small, tmp_path):
    df, _, _, _ = small
    df2, _ = generate("case30", n_ops=3, max_candidates=100, workers=1, out_dir=str(tmp_path))
    assert label_hash(df) == label_hash(df2)
