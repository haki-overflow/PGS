"""All experiment settings in one place."""

DATA_DIR = "data"
RESULTS_DIR = "results"

# Scenario generation (see guide, decisions D9-D11)
SCENARIO = dict(
    seed=0,
    load_low=0.7,     # global load factor ~ U(load_low, load_high)
    load_high=1.3,
    bus_sigma=0.07,   # per-bus factor ~ N(1, bus_sigma), clipped at 0
    gen_noise=0.10,   # generator P ~ proportional share * U(1 - g, 1 + g)
)

GRIDS = {
    "case30": dict(n_ops=300, max_candidates=3000),
    "case118": dict(n_ops=120, max_candidates=1200),
}

NODE_FEATURES = ["p_load", "q_load", "p_gen", "q_gen", "vm_set",
                 "is_slack", "is_pv", "is_pq", "vm_base", "va_base"]
EDGE_STATIC_FEATURES = ["r", "x", "b", "rating", "is_trafo"]
EDGE_BASE_FEATURES = ["loading_base", "p_from_base"]
