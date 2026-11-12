"""Compare the exact budgeted spatial-search optimum with the greedy plan.

Run from the repository root::

    uv run python -m scripts.spatial_search_exact_vs_greedy

Sections: representative examples; how bad the greedy gap can get; a deterministic grid of instances;
detection probability versus expected cost; and the cost of exhaustive search. Everything is
deterministic (no sampling); timings depend on the machine.
"""

import itertools
import math
import time

from optimal_search.exact_spatial_search import (
    ExactSolution,
    count_feasible_allocations,
    expected_search_cost,
    fractional_upper_bound,
    solve_exact,
)
from optimal_search.spatial_search import greedy_search_plan


def greedy_result(prior, detection, cost, budget):
    plan = greedy_search_plan(prior, detection, cost, budget=budget)
    counts = [0] * len(prior)
    for step in plan:
        counts[step.cell] += 1
    return tuple(counts), (plan[-1].cumulative_probability if plan else 0.0)


def example_rows():
    equal_prior = [0.4, 0.3, 0.2, 0.1]
    return [
        ("Greedy counterexample (d=1, costs 2/1/1)", [0.52, 0.27, 0.21], [1.0] * 3, [2.0, 1.0, 1.0], 2.0),
        ("Equal costs, unequal priors and d", equal_prior, [0.6, 0.5, 0.9, 0.7], [1.0] * 4, 5.0),
        ("Repeated looks, dominant cell", [0.9, 0.1], [0.3, 0.3], [1.0, 1.0], 4.0),
        ("Unequal costs, budget exactly used", [0.5, 0.5], [0.9, 0.9], [1.0, 2.0], 3.0),
        ("Perfect detection knapsack (costs 3/1/2)", [0.6, 0.3, 0.1], [1.0] * 3, [3.0, 1.0, 2.0], 3.0),
        ("Cheap decoy (eps = 0.01, budget 1)", [0.8, 0.00808, 0.19192], [1.0] * 3, [1.0, 0.01, 1.0], 1.0),
    ]


def print_examples():
    print("[1] Representative examples: greedy versus exact")
    header = f"{'case':44s} {'B':>4s} {'greedy counts':>14s} {'P_greedy':>9s} {'exact counts':>13s} {'P_exact':>9s} {'gap':>7s} {'allowed':>8s}"
    print(header)
    print("-" * len(header))
    for name, prior, detection, cost, budget in example_rows():
        g_counts, g_p = greedy_result(prior, detection, cost, budget)
        exact = solve_exact(prior, detection, cost, budget)
        bound = fractional_upper_bound(prior, detection, cost, budget)
        allowed = bound.value - bound.strict_prefix_value
        print(
            f"{name:44s} {budget:4.1f} {str(g_counts):>14s} {g_p:9.4f} {str(exact.counts):>13s} {exact.detection_probability:9.4f} "
            f"{exact.detection_probability - g_p:7.4f} {allowed:8.4f}"
        )
    print("   'allowed' = fractional-knapsack bound minus the guaranteed (strict-prefix) value: an upper bound on the gap for any tie-breaking.\n")


def print_gap_sweep():
    print("[2] How large can the greedy gap get? Budget 1, cell B cheap (cost eps) with a slightly better ratio than A")
    print(f"{'eps':>8s} {'P_greedy':>10s} {'P_exact':>9s} {'abs gap':>9s} {'greedy/exact':>13s}")
    for epsilon in (0.2, 0.1, 0.01, 0.001, 0.0001):
        p_a = 0.8
        p_b = p_a * epsilon * 1.01
        prior = [p_a, p_b, 1.0 - p_a - p_b]
        cost = [1.0, epsilon, 1.0]
        _, g_p = greedy_result(prior, [1.0] * 3, cost, 1.0)
        exact = solve_exact(prior, [1.0] * 3, cost, 1.0).detection_probability
        print(f"{epsilon:8.4f} {g_p:10.6f} {exact:9.4f} {exact - g_p:9.4f} {g_p / exact:13.5f}")
    print("   Greedy keeps only the cheap cell and can then afford nothing else: the ratio tends to 0 and the gap to p_A (-> 1 as p_A -> 1).\n")


def print_grid():
    priors = [(0.5, 0.3, 0.2), (0.6, 0.3, 0.1), (0.4, 0.4, 0.2), (0.7, 0.2, 0.1), (0.34, 0.33, 0.33)]
    detections = [(0.9, 0.9, 0.9), (0.9, 0.5, 0.3), (0.3, 0.5, 0.9), (0.6, 0.6, 0.6), (1.0, 1.0, 1.0)]
    cost_sets = {
        "equal costs": [(1.0, 1.0, 1.0)],
        "unequal costs": [(1, 2, 3), (2, 1, 1), (3, 1, 2), (1, 1, 2), (1, 3, 1), (2, 2, 1), (1, 2, 2), (3, 2, 1)],
    }
    print("[3] Deterministic grid: 5 priors x 5 detection vectors x cost sets x budgets 1..8 (3 cells)")
    print(f"{'cost set':16s} {'instances':>10s} {'greedy<exact':>13s} {'share':>7s} {'max abs gap':>12s} {'max rel gap':>12s} {'bound violated':>15s}")
    for name, costs in cost_sets.items():
        total = worse = violated = 0
        max_abs = max_rel = 0.0
        for prior, detection, cost, budget in itertools.product(priors, detections, costs, range(1, 9)):
            budget = float(budget)
            exact = solve_exact(prior, detection, cost, budget).detection_probability
            _, g_p = greedy_result(prior, detection, cost, budget)
            bound = fractional_upper_bound(prior, detection, cost, budget)
            total += 1
            gap = exact - g_p
            if gap > exact * 1e-12 + 1e-15:
                worse += 1
                max_abs = max(max_abs, gap)
                max_rel = max(max_rel, gap / exact)
            if gap > bound.value - bound.strict_prefix_value + 1e-12 or exact > bound.value + 1e-12:
                violated += 1
        print(f"{name:16s} {total:10d} {worse:13d} {100 * worse / total:6.1f}% {max_abs:12.4f} {max_rel:12.3f} {violated:15d}")
    print()


def print_budget_vs_expected_cost():
    prior, detection, cost = [0.52, 0.27, 0.21], [1.0] * 3, [2.0, 1.0, 1.0]
    print("[4] Maximising detection probability within a budget is not minimising expected cost")
    print(f"    prior {prior}, d = 1, costs {cost}")
    plan = solve_exact(prior, detection, cost, 2.0)
    print(
        f"    budget-2 optimum: searches {list(plan.sequence)}  P = {plan.detection_probability:.4f}  "
        f"expected cost incurred = {plan.expected_cost:.4f}"
    )
    cheaper = expected_search_cost(prior, detection, cost, [1, 2])
    print(f"    plan [1, 2]     : P = {0.27 + 0.21:.4f}  expected cost incurred = {cheaper[0]:.4f}  (cheaper, but detects less often)")
    print("    expected cost to find the target when every cell is searched (order matters):")
    for order in itertools.permutations(range(3)):
        incurred, given = expected_search_cost(prior, detection, cost, order)
        print(f"      order {list(order)}: E[cost incurred] = {incurred:.4f}   E[cost | detected] = {given:.4f}")
    print("    The cheapest order starts with cell 1 (ratio p/c = 0.27), the budget-2 optimum searches cell 0 (0.26) alone.")
    print("    Same allocation, different order: probability is unchanged, expected cost is not:")
    for order in ([0, 1], [1, 0]):
        incurred, given = expected_search_cost([0.6, 0.4], [0.5, 0.5], [1.0, 1.0], order)
        print(f"      prior (0.6, 0.4), d = 0.5, order {order}: P = 0.5000  E[cost incurred] = {incurred:.4f}  E[cost | detected] = {given:.4f}")
    print()


def print_limits():
    print("[5] Cost of exhaustive search")
    print("    Equal costs: allocations = C(k + M, M), k = floor(budget / cost)")
    print(f"    {'M':>3s} {'k':>3s} {'allocations':>20s}")
    for cells, k in [(3, 10), (5, 10), (8, 10), (10, 10), (10, 20), (15, 20), (20, 20)]:
        print(f"    {cells:3d} {k:3d} {math.comb(k + cells, cells):20,d}")
    print("    Measured (this machine; pure Python, three passes over the allocations):")
    print(f"    {'M':>3s} {'k':>3s} {'allocations':>12s} {'seconds':>9s} {'allocations/s':>15s}")
    for cells, k in [(4, 8), (5, 10), (6, 10), (7, 10), (8, 10)]:
        prior, detection, cost = [1.0 / cells] * cells, [0.5] * cells, [1.0] * cells
        start = time.perf_counter()
        solution: ExactSolution = solve_exact(prior, detection, cost, float(k), max_allocations=5_000_000)
        seconds = time.perf_counter() - start
        print(f"    {cells:3d} {k:3d} {solution.allocations_examined:12,d} {seconds:9.3f} {solution.allocations_examined / seconds:15,.0f}")
    print(f"    Unequal costs (1, 2, 3, 4, 5, 6), budget 30: {count_feasible_allocations([1, 2, 3, 4, 5, 6], 30.0):,} allocations.")
    print("    With d = 1 the problem is 0/1 knapsack (weakly NP-hard); integer costs admit a pseudo-polynomial dynamic program.")


def main() -> None:
    print("Exact vs greedy finite-budget spatial search (maximise P(detected) subject to total cost <= budget)\n")
    print_examples()
    print_gap_sweep()
    print_grid()
    print_budget_vs_expected_cost()
    print_limits()


if __name__ == "__main__":
    main()
