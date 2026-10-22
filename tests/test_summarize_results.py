import json

import pytest

from pgs.summarize_results import best_gnn, load_runs, table


def write_run(d, model, seed, test_ap, val_ap, split="cont", tag=""):
    m = dict(pr_auc=test_ap, recall=0.5, precision=0.5, f1=0.5, base_rate=0.1)
    r = dict(grid="g", split=split, model=model, seed=seed, tag=tag, seconds=1.0,
             test=m, val=dict(m, pr_auc=val_ap))
    (d / f"g_{split}_{model}{tag}_s{seed}.json").write_text(json.dumps(r))


@pytest.fixture
def runs(tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    for seed, (a, b) in enumerate([(0.2, 0.4), (0.4, 0.6)]):
        write_run(d, "gcn", seed, test_ap=a, val_ap=0.9)   # best on validation
        write_run(d, "gat", seed, test_ap=b, val_ap=0.5)   # best on test
    write_run(d, "gat", 0, test_ap=0.1, val_ap=0.99, tag="layers1")  # ablation: ignored
    return tmp_path


def test_mean_and_std_over_seeds(runs):
    t = table(load_runs(str(runs)))
    gcn = t[(t.model == "gcn")].iloc[0]
    assert gcn.seeds == 2
    assert gcn.pr_auc_mean == pytest.approx(0.3)
    assert gcn.pr_auc_std == pytest.approx(0.1414, abs=1e-3)  # sample std of 0.2, 0.4


def test_best_gnn_is_chosen_on_validation_not_test(runs):
    assert best_gnn("g", "cont", results_dir=str(runs)) == "gcn"
