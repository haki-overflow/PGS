"""Generate labelled (operating point, N-1 outage) samples for one grid.

Usage: python -m pgs.generate_dataset --grid case30 [--workers 8] [--n-ops 300]

Writes data/<grid>/samples.parquet (one row per sample: label + diagnostics),
data/<grid>/arrays.npz (input features) and data/<grid>/meta.json.
Only pre-outage quantities go into arrays.npz; post-outage results stay in
samples.parquet and are used for the label, diagnostics and rule baselines.
"""
import argparse
import hashlib
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pandapower as pp

from pgs import config
from pgs.power_flow import (S_BASE, assess, assign_ratings, branch_flows_mva,
                      branch_p_from_mw, branch_table, islanding_branches,
                      load_case, run_ac, run_ac_with_retries,
                      set_branch_in_service)

_W = {}  # per-worker state: net, branch table, ratings, contingencies


def _init_worker(grid, br, contingencies):
    net = load_case(grid)
    _W.update(net=net, br=br, rating=br.rating_mva.to_numpy(),
              contingencies=contingencies, p0=net.load.p_mw.to_numpy().copy(),
              q0=net.load.q_mvar.to_numpy().copy())


def sample_candidates(net, n, rng, sc):
    """Random operating points: scaled loads and proportionally redispatched generators."""
    p0, g0 = net.load.p_mw.to_numpy(), net.gen.p_mw.to_numpy()
    out = []
    for _ in range(n):
        factor = rng.uniform(sc["load_low"], sc["load_high"])
        per_bus = np.clip(rng.normal(1.0, sc["bus_sigma"], len(p0)), 0.0, None)
        load_scale = factor * per_bus
        share = (p0 * load_scale).sum() / p0.sum()
        noise = rng.uniform(1 - sc["gen_noise"], 1 + sc["gen_noise"], len(g0))
        gen_p = np.clip(g0 * share * noise, net.gen.min_p_mw, net.gen.max_p_mw)
        out.append(dict(load_factor=factor, load_scale=load_scale, gen_p=gen_p))
    return out


def _apply_op(op):
    net = _W["net"]
    net.load["p_mw"] = _W["p0"] * op["load_scale"]
    net.load["q_mvar"] = _W["q0"] * op["load_scale"]  # keeps each load's power factor
    net.gen["p_mw"] = op["gen_p"]


def node_features(net):
    """Pre-outage features per bus, in positional bus order (see config.NODE_FEATURES)."""
    pos = pd.Series(np.arange(len(net.bus)), index=net.bus.index)
    n = len(net.bus)
    x = np.zeros((n, len(config.NODE_FEATURES)))

    def add(col, buses, values):
        np.add.at(x[:, col], pos[buses].to_numpy(), np.asarray(values) / S_BASE)

    add(0, net.load.bus, net.load.p_mw)
    add(1, net.load.bus, net.load.q_mvar)
    add(2, net.gen.bus, net.res_gen.p_mw)
    add(3, net.gen.bus, net.res_gen.q_mvar)
    add(2, net.ext_grid.bus, net.res_ext_grid.p_mw)
    add(3, net.ext_grid.bus, net.res_ext_grid.q_mvar)
    x[pos[net.gen.bus], 4] = net.gen.vm_pu
    x[pos[net.ext_grid.bus], 4] = net.ext_grid.vm_pu
    x[pos[net.ext_grid.bus], 5] = 1
    x[pos[net.gen.bus], 6] = 1
    x[:, 7] = 1 - x[:, 5] - x[:, 6]
    x[:, 8] = net.res_bus.vm_pu.to_numpy()
    x[:, 9] = np.deg2rad(net.res_bus.va_degree.to_numpy())
    return x


def edge_static_features(net, br):
    """r, x, b in p.u. (from pandapower's internal model), rating p.u., is_trafo."""
    ppc_br = net._ppc["branch"]
    sl = net._pd2ppc_lookups["branch"]
    rows = np.concatenate([np.arange(*sl["line"])] +
                          ([np.arange(*sl["trafo"])] if "trafo" in sl else []))
    rxb = np.real(ppc_br[rows][:, 2:5])
    return np.column_stack([rxb, br.rating_mva / S_BASE, br.kind == "trafo"]).astype(float)


def _base_case(args):
    """Phase A: is this candidate operating point secure before any outage?"""
    i, op = args
    _apply_op(op)
    ok = run_ac(_W["net"])
    return i, ok and not assess(_W["net"], _W["rating"], ok)["insecure"]


def _contingencies(args):
    """Phase B: features of one secure operating point, then every N-1 outage."""
    op_id, op = args
    net, br, rating = _W["net"], _W["br"], _W["rating"]
    _apply_op(op)
    assert run_ac(net)
    base_res = net.res_bus.copy()
    x = node_features(net)
    flows = branch_flows_mva(net)
    e_base = np.column_stack([flows / rating, branch_p_from_mw(net) / S_BASE])
    base_max_loading = float((flows / rating).max())
    rows = []
    for k in _W["contingencies"]:
        row = br.loc[k]
        set_branch_in_service(net, row, False)
        attempts = run_ac_with_retries(net, base_res)
        a = assess(net, rating, attempts > 0)
        try:
            pp.rundcpp(net)
            dc = float((np.abs(branch_p_from_mw(net)) / rating).max())
        except Exception:
            dc = np.nan
        set_branch_in_service(net, row, True)
        rows.append(dict(op=op_id, cont=int(k), attempts=attempts, **a,
                         dc_max_loading=dc, base_max_loading=base_max_loading,
                         load_factor=op["load_factor"]))
    return x, e_base, rows


def label_hash(df):
    return hashlib.md5(df[["op", "cont", "insecure"]].to_csv(index=False).encode()).hexdigest()[:10]


def generate(grid, n_ops, max_candidates, workers, out_dir, sc=config.SCENARIO):
    t0 = time.time()
    net = load_case(grid)
    br = branch_table(net)
    isl = islanding_branches(net, br)
    br = assign_ratings(net, br, isl)
    contingencies = [k for k in br.index if k not in isl]
    e_static = edge_static_features(net, br)
    cands = sample_candidates(net, max_candidates, np.random.default_rng(sc["seed"]), sc)

    with Pool(workers, initializer=_init_worker, initargs=(grid, br, contingencies)) as pool:
        secure, checked = [], 0
        while len(secure) < n_ops and checked < max_candidates:
            chunk = range(checked, min(checked + max(n_ops, 4 * workers), max_candidates))
            for i, ok in pool.map(_base_case, [(i, cands[i]) for i in chunk]):
                if ok:
                    secure.append(i)
            checked = chunk.stop
        if len(secure) < n_ops:
            raise RuntimeError(f"only {len(secure)} secure OPs in {max_candidates} candidates")
        kept = secure[:n_ops]
        # number of candidates actually needed to reach n_ops (order-independent of chunking)
        needed = kept[-1] + 1
        n_secure_seen = sum(1 for i in secure if i < needed)
        out = pool.map(_contingencies, [(j, cands[i]) for j, i in enumerate(kept)], chunksize=1)

    node_x = np.stack([o[0] for o in out])
    edge_base = np.stack([o[1] for o in out])
    df = pd.DataFrame([r for o in out for r in o[2]])
    edge_index = br[["f", "t"]].to_numpy().T

    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(os.path.join(out_dir, "samples.parquet"), index=False)
    np.savez_compressed(os.path.join(out_dir, "arrays.npz"), node_x=node_x,
                        edge_base=edge_base, edge_static=e_static, edge_index=edge_index,
                        contingencies=np.array(contingencies), islanding=np.array(isl, dtype=int))
    meta = dict(grid=grid, scenario=sc, n_ops=n_ops, candidates_used=needed,
                base_secure_rate=n_secure_seen / needed, n_contingencies=len(contingencies),
                n_islanding=len(isl), n_samples=len(df), insecure_rate=float(df.insecure.mean()),
                seconds=round(time.time() - t0, 1), label_hash=label_hash(df),
                branch_kinds=br.kind.tolist(), rating_mva=br.rating_mva.round(3).tolist(),
                pandapower=pp.__version__)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return df, meta


def summarize(df, meta):
    att = df.attempts.value_counts().sort_index().to_dict()
    lf = df.groupby("op").load_factor.first()
    per_op = df.groupby("op").insecure.mean()
    lines = [
        f"{meta['grid']}: {meta['candidates_used']} candidates -> base-secure "
        f"{meta['base_secure_rate']:.1%}; kept {meta['n_ops']} OPs x "
        f"{meta['n_contingencies']} outages = {meta['n_samples']} samples "
        f"({meta['n_islanding']} islanding outages excluded)",
        f"  insecure {df.insecure.mean():.1%} | overload {df.overload.mean():.1%} | "
        f"v_viol {df.v_viol.mean():.1%} | non-converged {(~df.converged).sum()} | "
        f"attempts {att}",
        f"  load factor of kept OPs: min {lf.min():.2f} median {lf.median():.2f} max {lf.max():.2f}",
        f"  insecure rate per OP: min {per_op.min():.1%} median {per_op.median():.1%} "
        f"max {per_op.max():.1%}",
        f"  outages that are insecure in every OP: "
        f"{(df.groupby('cont').insecure.mean() == 1).sum()}, in no OP: "
        f"{(df.groupby('cont').insecure.mean() == 0).sum()}",
        f"  time {meta['seconds']} s, label hash {meta['label_hash']}",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, choices=sorted(config.GRIDS))
    ap.add_argument("--n-ops", type=int)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    g = config.GRIDS[a.grid]
    df, meta = generate(a.grid, a.n_ops or g["n_ops"], g["max_candidates"], a.workers,
                        os.path.join(config.DATA_DIR, a.grid))
    print(summarize(df, meta))


if __name__ == "__main__":
    main()
