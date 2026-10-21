"""Leakage-safe train/val/test splits. Never a random split per sample.

S1 "op":   split operating points 70/15/15. Every outage appears in every split,
           but no operating point is shared between splits.
S2 "cont": split operating points 70/15/15 AND outages 60/20/20. Test samples
           combine unseen operating points with outages never seen in training.
"""
import numpy as np

SPLITS = ("op", "cont")


def _three_way(items, fracs, rng):
    items = rng.permutation(items)
    a = int(round(fracs[0] * len(items)))
    b = a + int(round(fracs[1] * len(items)))
    return items[:a], items[a:b], items[b:]


def make_split(df, kind, seed):
    """Return {"train", "val", "test"} -> row indices into df."""
    rng = np.random.default_rng(seed)
    ops = _three_way(np.unique(df.op), (0.70, 0.15), rng)
    if kind == "op":
        return {name: np.flatnonzero(df.op.isin(o)) for name, o in
                zip(("train", "val", "test"), ops)}
    if kind == "cont":
        conts = _three_way(np.unique(df.cont), (0.60, 0.20), rng)
        return {name: np.flatnonzero(df.op.isin(o) & df.cont.isin(c)) for name, o, c in
                zip(("train", "val", "test"), ops, conts)}
    raise ValueError(kind)


def report(grids=("case30", "case118"), seeds=range(5)):
    """Print split sizes and insecure rates, to check every test set has positives."""
    from pgs.dataset import load_grid
    for grid in grids:
        df, _ = load_grid(grid)
        for kind in SPLITS:
            sizes, rates = [], []
            for seed in seeds:
                s = make_split(df, kind, seed)
                sizes.append([len(s[k]) for k in ("train", "val", "test")])
                rates.append([df.insecure.iloc[s[k]].mean() for k in ("train", "val", "test")])
            sizes, rates = np.array(sizes), np.array(rates)
            print(f"{grid:8s} {kind:5s} seed0 train/val/test = {sizes[0].tolist()}; "
                  f"insecure % over 5 seeds: train {100 * rates[:, 0].min():.1f}-"
                  f"{100 * rates[:, 0].max():.1f}, test {100 * rates[:, 2].min():.1f}-"
                  f"{100 * rates[:, 2].max():.1f}")


if __name__ == "__main__":
    report()
