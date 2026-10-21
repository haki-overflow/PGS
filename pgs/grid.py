"""Grid loading, branch table, thermal ratings and the security label.

Everything that touches pandapower directly lives here, so the rest of the
code only sees numpy arrays and pandas tables.
"""
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Label definition (agreed with the user; do not change silently) --------
V_MIN, V_MAX = 0.94, 1.06   # p.u. voltage band, applied to every bus
LOADING_MAX = 1.0           # branch apparent power / rating
S_BASE = 100.0              # MVA, system base for per-unit features
PLACEHOLDER_MVA = 9000.0    # MATPOWER uses 9900 MVA to mean "no rating"
RATING_MARGIN = 1.1         # synthetic rating = 1.1 x worst nominal N-1 flow

CASES = {"case14": pn.case14, "case30": pn.case30, "case39": pn.case39,
         "case118": pn.case118}


def load_case(name):
    net = CASES[name]()
    if len(net.ext_grid) != 1:
        raise ValueError(f"{name}: expected exactly one slack bus")
    return net


def run_ac(net, init="auto"):
    """AC power flow with generator reactive limits. Returns True if converged."""
    try:
        pp.runpp(net, init=init, enforce_q_lims=True, max_iteration=30)
        return True
    except pp.LoadflowNotConverged:
        return False


def run_ac_with_retries(net, base_res_bus):
    """Try base-case warm start, then flat start, then DC start.

    Returns the number of attempts used (1-3), or 0 if all failed.
    """
    for attempt, init in enumerate(("results", "flat", "dc"), start=1):
        if init == "results":
            net.res_bus = base_res_bus.copy()
        if run_ac(net, init=init):
            return attempt
    return 0


def branch_table(net):
    """One row per branch (lines first, then transformers), positional bus ids."""
    pos = pd.Series(np.arange(len(net.bus)), index=net.bus.index)
    rows = []
    for i, r in net.line.iterrows():
        rating = np.sqrt(3) * r.max_i_ka * net.bus.vn_kv.at[r.from_bus]
        rows.append(("line", i, pos[r.from_bus], pos[r.to_bus], rating))
    for i, r in net.trafo.iterrows():
        rows.append(("trafo", i, pos[r.hv_bus], pos[r.lv_bus], r.sn_mva))
    br = pd.DataFrame(rows, columns=["kind", "pp_idx", "f", "t", "rating_file_mva"])
    br["has_real_rating"] = br.rating_file_mva < PLACEHOLDER_MVA
    return br


def branch_flows_mva(net):
    """Apparent power at the worse end of each branch, in branch_table order."""
    line = np.maximum(np.hypot(net.res_line.p_from_mw, net.res_line.q_from_mvar),
                      np.hypot(net.res_line.p_to_mw, net.res_line.q_to_mvar))
    trafo = np.maximum(np.hypot(net.res_trafo.p_hv_mw, net.res_trafo.q_hv_mvar),
                       np.hypot(net.res_trafo.p_lv_mw, net.res_trafo.q_lv_mvar))
    return np.concatenate([line.to_numpy(), trafo.to_numpy()])


def branch_p_from_mw(net):
    """Signed active power entering each branch at its from/hv end."""
    return np.concatenate([net.res_line.p_from_mw.to_numpy(),
                           net.res_trafo.p_hv_mw.to_numpy()])


def assign_ratings(net, br, islanding, margin=RATING_MARGIN):
    """Keep real ratings; replace placeholders with a planning-style rating.

    Synthetic rating = margin * the largest flow the branch carries over the
    nominal base case and every non-islanding N-1 outage. So the nominal case
    is thermally N-1 secure by construction, like a planned grid.
    """
    if not run_ac(net):
        raise RuntimeError("base case does not converge")
    base_res = net.res_bus.copy()
    worst = branch_flows_mva(net)
    for k, row in br.iterrows():
        if k in islanding:
            continue
        set_branch_in_service(net, row, False)
        if run_ac_with_retries(net, base_res):
            worst = np.maximum(worst, branch_flows_mva(net))
        set_branch_in_service(net, row, True)
    run_ac(net)  # leave the net holding nominal results
    br = br.copy()
    br["rating_mva"] = np.where(br.has_real_rating, br.rating_file_mva, margin * worst)
    return br


def set_branch_in_service(net, row, in_service):
    table = net.line if row.kind == "line" else net.trafo
    table.at[row.pp_idx, "in_service"] = in_service


def islanding_branches(net, br):
    """Branch ids whose single outage splits the grid into two pieces."""
    g = nx.MultiGraph()
    g.add_nodes_from(range(len(net.bus)))
    for k, r in br.iterrows():
        g.add_edge(r.f, r.t, key=k)
    out = []
    for k, r in br.iterrows():
        g.remove_edge(r.f, r.t, key=k)
        if not nx.is_connected(g):
            out.append(k)
        g.add_edge(r.f, r.t, key=k)
    return out


def assess(net, rating_mva, converged):
    """Apply the label to a solved (or failed) power flow."""
    if not converged:
        return dict(converged=False, vmin=np.nan, vmax=np.nan, max_loading=np.nan,
                    v_viol=False, overload=False, insecure=True)
    vm = net.res_bus.vm_pu.to_numpy()
    loading = branch_flows_mva(net) / rating_mva
    v_viol = bool((vm < V_MIN).any() or (vm > V_MAX).any())
    overload = bool((loading > LOADING_MAX).any())
    return dict(converged=True, vmin=float(vm.min()), vmax=float(vm.max()),
                max_loading=float(loading.max()), v_viol=v_viol,
                overload=overload, insecure=v_viol or overload)
