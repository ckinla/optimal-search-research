"""Every model number shown in the visual essay *The Mathematics of Looking*, recomputed.

The essay (https://optimal-search-lab.vercel.app/) computes the same quantities with a separate
TypeScript implementation; the values below are the Python reference it was checked against.
Run: uv run python examples/essay_numbers.py [--json]
"""

import argparse
import json
import math

from optimal_search.cumulative_detection import cumulative_efficiency, cumulative_stationary_snrs, log_miss_rate, required_snr_for_looks
from optimal_search.exact_spatial_search import solve_exact
from optimal_search.radar_detection import detection_probability, snr_linear_to_db
from optimal_search.spatial_search import greedy_search_plan, next_look_priority, update_after_failed_search

PFA = 1e-5
BUDGET = 60
TARGET = 0.8

# 2011 presentation, Search Example 2 (as transcribed; see docs/spatial-search-verification.md).
MISS_PRIOR = [0.1, 0.2, 0.4, 0.3]
MISS_DETECTION = 0.6
MISS_CELL = 2

# Illustrative inputs for "Where should we look next?" (not from any source).
NEXT_PRIOR = [0.5, 0.3, 0.2]
NEXT_DETECTION = [0.8, 0.9, 0.5]
NEXT_COST = [2, 1, 1]
NEXT_BUDGET = 2


def energy_for_target(snr: float) -> dict:
    """Continuous looks: N = -ln(1-Pc)/q(s); energy = N s = -ln(1-Pc)/H(s)."""
    looks = -math.log1p(-TARGET) / log_miss_rate(snr, PFA)
    return {"snr": snr, "pd": detection_probability(snr, PFA), "efficiency": cumulative_efficiency(snr, PFA),
            "looks": looks, "energy": looks * snr}


def fixtures() -> dict:
    references = []
    for pfa in (1e-5, 1e-6):
        lower, upper = cumulative_stationary_snrs(pfa)
        references.append({"pfa": pfa, "local_minimum": lower, "snr": upper, "db": snr_linear_to_db(upper),
                           "pd": detection_probability(upper, pfa), "efficiency": cumulative_efficiency(upper, pfa)})
    whole = [{"looks": n, "snr": required_snr_for_looks(n, TARGET, PFA)} for n in range(1, 13)]
    for row in whole:
        row["energy"] = row["looks"] * row["snr"]
    after_miss = update_after_failed_search(MISS_PRIOR, [MISS_DETECTION] * 4, MISS_CELL)
    plan = greedy_search_plan(NEXT_PRIOR, NEXT_DETECTION, NEXT_COST, budget=NEXT_BUDGET)
    exact = solve_exact(NEXT_PRIOR, NEXT_DETECTION, NEXT_COST, NEXT_BUDGET)
    return {
        "pfa": PFA,
        "split": {"budget": BUDGET, "looks": [
            {"looks": n, "snr": BUDGET / n, "pd": detection_probability(BUDGET / n, PFA),
             "combined": -math.expm1(n * math.log1p(-detection_probability(BUDGET / n, PFA)))} for n in (1, 4, 12)]},
        "strength": {"target": TARGET, "references": references,
                     "curve": [energy_for_target(s) for s in (1, 2, 4, 8, 12, 20, 30, 40)],
                     "whole_looks": whole},
        "miss": {"prior": MISS_PRIOR, "detection": MISS_DETECTION, "cell": MISS_CELL,
                 "first_look_success": MISS_PRIOR[MISS_CELL] * MISS_DETECTION,
                 "normaliser": 1 - MISS_PRIOR[MISS_CELL] * MISS_DETECTION,
                 "posterior": [float(value) for value in after_miss]},
        "next": {"prior": NEXT_PRIOR, "detection": NEXT_DETECTION, "cost": NEXT_COST, "budget": NEXT_BUDGET,
                 "priority": [float(value) for value in next_look_priority(NEXT_PRIOR, NEXT_DETECTION, NEXT_COST)],
                 "greedy": [{"cell": step.cell, "cumulative": step.cumulative_probability, "cost": step.cumulative_cost}
                            for step in plan],
                 "exact": {"counts": list(exact.counts), "probability": exact.detection_probability,
                           "cost": exact.total_cost, "plans_examined": exact.allocations_examined}},
    }


def summary(values: dict) -> str:
    split = values["split"]
    low, high = values["strength"]["references"]
    whole = min(values["strength"]["whole_looks"], key=lambda row: row["energy"])
    miss, nxt = values["miss"], values["next"]
    lines = [f"1. Budget of {split['budget']} energy units split into equal independent looks (Pfa = {values['pfa']:g}):"]
    lines += [f"   {row['looks']:>2} looks at strength {row['snr']:5.2f}: each detects {row['pd']:.1%}, combined {row['combined']:.1%}"
              for row in split["looks"]]
    lines += [f"2. Least energy for an {values['strength']['target']:.0%} detection goal (continuous looks):",
              f"   valley at strength {low['snr']:.4f} = {low['db']:.3f} dB, Pd per look {low['pd']:.3f}, "
              f"energy {-math.log1p(-values['strength']['target']) / low['efficiency']:.3f}",
              f"   at Pfa = 1e-6: {high['db']:.3f} dB; best whole-look design: {whole['looks']} looks of "
              f"{whole['snr']:.4f}, energy {whole['energy']:.3f}",
              f"3. 2011 Search Example 2, miss in cell {miss['cell'] + 1}: first-look success {miss['first_look_success']:.2f}, "
              f"posterior {', '.join(f'{p:.5f}' for p in miss['posterior'])}",
              f"4. Illustrative three-place budget: priorities {', '.join(f'{p:.2f}' for p in nxt['priority'])}; "
              f"greedy plan {[step['cell'] for step in nxt['greedy']]} reaches {nxt['greedy'][-1]['cumulative']:.0%}, "
              f"exact best {nxt['exact']['counts']} reaches {nxt['exact']['probability']:.0%} "
              f"({nxt['exact']['plans_examined']} plans examined)"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="print every value as JSON")
    arguments = parser.parse_args()
    values = fixtures()
    print(json.dumps(values, indent=2) if arguments.json else summary(values))


if __name__ == "__main__":
    main()
