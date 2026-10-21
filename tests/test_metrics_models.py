import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

from pgs.metrics import evaluate, tune_threshold
from pgs.models import GNN, MLP
from pgs.run import rule_scores


def test_metrics_on_known_example():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.0])
    m = evaluate(y, s, threshold=0.5)  # predicts samples 0 and 2 insecure
    assert m["recall"] == 0.5 and m["precision"] == 0.5 and m["f1"] == 0.5
    assert m["base_rate"] == 0.4 and m["accuracy"] == 0.6


def test_perfect_scores_give_pr_auc_one():
    y = np.array([0, 0, 1, 1])
    assert evaluate(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)["pr_auc"] == 1.0


def test_tuned_threshold_separates_perfect_scores():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    t = tune_threshold(y, s)
    assert 0.2 < t <= 0.8
    assert evaluate(y, s, t)["f1"] == 1.0


def test_cont_prior_uses_training_rows_only_and_falls_back_for_unseen_outages():
    df = pd.DataFrame(dict(op=[0, 0, 1, 1, 2], cont=[0, 1, 0, 1, 2],
                           insecure=[True, False, True, False, True]))
    split = {"train": np.array([0, 1, 2, 3]), "val": np.array([]), "test": np.array([4])}
    s = rule_scores("cont_prior", df, split)
    assert s[0] == 1.0 and s[1] == 0.0
    assert s[4] == 0.5  # outage 2 never seen in training -> global training rate


def test_models_return_one_logit_per_sample():
    assert MLP(7)(torch.randn(5, 7)).shape == (5,)
    g = Data(x=torch.randn(4, 11), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
             edge_attr=torch.randn(3, 7))
    batch = Batch.from_data_list([g, g, g])
    for conv in ("gcn", "sage", "gat"):
        assert GNN(conv, 11, 7)(batch).shape == (3,)
