"""Step 1: print the facts about each test grid that our design depends on.

Usage: python -m pgs.inspect_cases
"""
import time

from pgs.grid import (assess, assign_ratings, branch_table, islanding_branches,
                      load_case, run_ac)


def main():
    cols = ["case", "buses", "lines", "trafos", "gens", "real_ratings",
            "islanding", "vmin", "vmax", "max_load", "base_secure", "ms_per_pf"]
    print(" ".join(f"{c:>12}" for c in cols))
    for name in ["case14", "case30", "case39", "case118"]:
        net = load_case(name)
        br = branch_table(net)
        isl = islanding_branches(net, br)
        br = assign_ratings(net, br, isl)
        t = time.perf_counter()
        for _ in range(10):
            run_ac(net)
        ms = (time.perf_counter() - t) * 100
        a = assess(net, br.rating_mva.to_numpy(), True)
        row = [name, len(net.bus), len(net.line), len(net.trafo),
               len(net.gen) + 1, f"{br.has_real_rating.sum()}/{len(br)}", len(isl),
               f"{a['vmin']:.3f}", f"{a['vmax']:.3f}", f"{a['max_loading']:.3f}",
               not a["insecure"], f"{ms:.0f}"]
        print(" ".join(f"{str(v):>12}" for v in row))


if __name__ == "__main__":
    main()
