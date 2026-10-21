"""Print the real numbers used in the worked examples of docs/PROJECT_GUIDE.md.

Usage: python scripts/guide_examples.py
"""
import numpy as np
import pandapower as pp

from pgs.grid import assess, branch_flows_mva, branch_table, run_ac, set_branch_in_service


def four_bus_ring():
    """Slack at bus 0, ring 0-1-2-3-0, loads on buses 1, 2, 3, every line rated 50 MVA."""
    net = pp.create_empty_network()
    b = [pp.create_bus(net, vn_kv=110.0) for _ in range(4)]
    pp.create_ext_grid(net, b[0], vm_pu=1.0)
    for f, t in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        pp.create_line_from_parameters(net, b[f], b[t], length_km=20, r_ohm_per_km=0.1,
                                       x_ohm_per_km=0.4, c_nf_per_km=0,
                                       max_i_ka=50 / (np.sqrt(3) * 110))
    for bus, mw in [(1, 20), (2, 40), (3, 20)]:
        pp.create_load(net, b[bus], p_mw=mw, q_mvar=mw * 0.2)
    return net


def show(net, br, title):
    a = assess(net, br.rating_file_mva.to_numpy(), True)
    flows = branch_flows_mva(net)
    print(title)
    for k, r in br.iterrows():
        on = (net.line.in_service.iloc[k])
        print(f"  line {r.f}-{r.t}: {flows[k]:5.1f} MVA of 50 -> {flows[k] / 50:5.0%}"
              + ("" if on else "  (OUT)"))
    print(f"  voltages: {np.round(net.res_bus.vm_pu.to_numpy(), 3).tolist()}"
          f"  -> {'INSECURE' if a['insecure'] else 'secure'}")


def main():
    net = four_bus_ring()
    br = branch_table(net)
    run_ac(net)
    show(net, br, "Example 1a: 4-bus ring, all lines in service")
    set_branch_in_service(net, br.iloc[3], False)  # outage of line 3-0
    run_ac(net)
    show(net, br, "Example 1b: same grid after N-1 outage of line 3-0")

    print("\nExample 2: one mean-aggregation message-passing step")
    x = {0: np.array([1.00, 0.0]), 1: np.array([0.98, 0.2]),
         2: np.array([0.95, 0.4]), 3: np.array([0.97, 0.2])}
    nbrs_of_2 = [1, 3]
    mean = np.mean([x[j] for j in nbrs_of_2], axis=0)
    w_self, w_nbr = 1.0, 0.5
    new = np.maximum(0, w_self * x[2] + w_nbr * mean)
    print(f"  bus 2 features [voltage, load]: {x[2].tolist()}")
    print(f"  neighbours 1 and 3 mean: {mean.round(3).tolist()}")
    print(f"  new bus 2 = ReLU(1.0*self + 0.5*mean) = {new.round(3).tolist()}")


if __name__ == "__main__":
    main()
