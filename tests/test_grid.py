import numpy as np
import pandapower as pp

from pgs.grid import (assess, branch_table, islanding_branches, run_ac,
                      set_branch_in_service)


def tiny_net(load_mw=10.0, length_km=10.0):
    """Bus 0 (slack) feeds bus 1 over two parallel lines; bus 1 feeds bus 2 radially."""
    net = pp.create_empty_network()
    b = [pp.create_bus(net, vn_kv=110.0) for _ in range(3)]
    pp.create_ext_grid(net, b[0], vm_pu=1.0)
    for f, t in [(0, 1), (0, 1), (1, 2)]:
        pp.create_line_from_parameters(net, b[f], b[t], length_km=length_km, r_ohm_per_km=0.1,
                                       x_ohm_per_km=0.4, c_nf_per_km=0, max_i_ka=0.5)
    pp.create_load(net, b[2], p_mw=load_mw, q_mvar=load_mw * 0.2)
    return net


def test_branch_table_rating_from_current_limit():
    br = branch_table(tiny_net())
    assert list(br.kind) == ["line"] * 3
    assert np.allclose(br.rating_file_mva, np.sqrt(3) * 0.5 * 110.0)


def test_nominal_case_is_secure():
    net = tiny_net()
    assert run_ac(net)
    a = assess(net, branch_table(net).rating_file_mva.to_numpy(), True)
    assert a["converged"] and not a["insecure"]


def test_low_rating_gives_overload():
    net = tiny_net()
    run_ac(net)
    a = assess(net, np.full(3, 1.0), True)  # 1 MVA ratings, 10 MW load
    assert a["overload"] and a["insecure"]


def test_heavy_load_gives_voltage_violation():
    net = tiny_net(load_mw=60.0, length_km=60.0)
    assert run_ac(net)
    a = assess(net, np.full(3, 1e6), True)
    assert a["v_viol"] and not a["overload"] and a["insecure"]


def test_non_convergence_is_insecure():
    a = assess(None, None, converged=False)
    assert a["insecure"] and not a["converged"]


def test_islanding_detection_handles_parallel_lines():
    net = tiny_net()
    # parallel lines 0 and 1 back each other up; radial line 2 islands bus 2
    assert islanding_branches(net, branch_table(net)) == [2]


def test_outage_changes_flow_and_restores():
    net = tiny_net()
    br = branch_table(net)
    run_ac(net)
    before = net.res_line.p_from_mw.iloc[0]
    set_branch_in_service(net, br.iloc[1], False)
    run_ac(net)
    assert net.res_line.p_from_mw.iloc[0] > 1.9 * before  # takes the twin's flow
    set_branch_in_service(net, br.iloc[1], True)
    assert net.line.in_service.all()
