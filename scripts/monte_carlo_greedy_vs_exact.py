"""Simulate greedy and exact search plans and compare with their analytical predictions.

Run from the repository root::

    uv run python -m scripts.monte_carlo_greedy_vs_exact

For each example the greedy plan and the exact plan are executed for 100,000
seeded Monte Carlo trials. Analytical predictions come from the exact-solver module's formulas, which
the simulation engine does not use. The simulation is an *estimate* with sampling error: a prediction
inside the 95% interval is consistent with the simulation, not proved by it.
"""

import math
import time

from optimal_search.exact_spatial_search import detection_probability_of_counts, expected_search_cost, solve_exact
from optimal_search.monte_carlo_spatial_search import simulate_search
from optimal_search.spatial_search import greedy_search_plan

TRIALS = 100_000

EXAMPLES = [
    ("Greedy counterexample: perfect detection, costs 2/1/1, budget 2", [0.52, 0.27, 0.21], [1.0, 1.0, 1.0], [2.0, 1.0, 1.0], 2.0, 2024),
    ("Imperfect detection, unequal costs, budget 6", [0.4, 0.3, 0.2, 0.1], [0.5, 0.6, 0.7, 0.8], [1.5, 1.0, 2.0, 0.5], 6.0, 2025),
    ("Equal costs, imperfect detection, budget 5", [0.4, 0.3, 0.2, 0.1], [0.6, 0.5, 0.9, 0.7], [1.0, 1.0, 1.0, 1.0], 5.0, 2026),
]


def analytical(prior, detection, cost, plan):
    counts = [plan.count(j) for j in range(len(prior))]
    probability = detection_probability_of_counts(prior, detection, counts)
    expected_cost, _ = expected_search_cost(prior, detection, cost, plan)
    expected_looks, _ = expected_search_cost(prior, detection, [1.0] * len(prior), plan)
    return probability, expected_cost, expected_looks


def describe(name, prior, detection, cost, plan, budget, seed):
    probability, expected_cost, expected_looks = analytical(prior, detection, cost, plan)
    started = time.perf_counter()
    result = simulate_search(prior, detection, cost, plan, TRIALS, seed=seed, budget=budget)
    elapsed = time.perf_counter() - started
    low, high = result.detection_confidence_interval()
    se = math.sqrt(probability * (1.0 - probability) / TRIALS)
    z = (result.detection_probability - probability) / se if se > 0 else 0.0
    cost_low, cost_high = result.mean_cost_confidence_interval()
    looks_low, looks_high = result.mean_looks_confidence_interval()
    print(f"  {name} plan {plan}  (seed {seed}, {elapsed * 1000:.0f} ms)")
    print(f"    P(detect)    analytical {probability:.4f} | simulated {result.detection_probability:.4f}  95% Wilson [{low:.4f}, {high:.4f}]  z = {z:+.2f}  {'inside' if low <= probability <= high else 'OUTSIDE'}")
    print(f"    mean cost    analytical {expected_cost:.4f} | simulated {result.mean_cost:.4f}  95% CI [{cost_low:.4f}, {cost_high:.4f}]  {'inside' if cost_low <= expected_cost <= cost_high else 'OUTSIDE'}")
    print(f"    mean looks   analytical {expected_looks:.4f} | simulated {result.mean_looks:.4f}  95% CI [{looks_low:.4f}, {looks_high:.4f}]  {'inside' if looks_low <= expected_looks <= looks_high else 'OUTSIDE'}")
    print(f"    max simulated cost {result.cost.max():.4f} (budget {budget})")
    by_cell = "  ".join(
        f"cell {j}: {result.detections_by_cell[j]:6d}/{result.target_counts[j]:6d}" for j in range(len(prior))
    )
    print(f"    detections / target placements by cell: {by_cell}")
    return result, probability


def main() -> None:
    print(f"Monte Carlo simulation of greedy versus exact search plans ({TRIALS:,} trials per plan)\n")
    for name, prior, detection, cost, budget, seed in EXAMPLES:
        print(name)
        greedy_plan = [step.cell for step in greedy_search_plan(prior, detection, cost, budget=budget)]
        exact_plan = list(solve_exact(prior, detection, cost, budget).sequence)
        greedy, p_greedy = describe("greedy", prior, detection, cost, greedy_plan, budget, seed)
        if exact_plan == greedy_plan:
            print("  exact  plan is identical to the greedy plan\n")
            continue
        exact, p_exact = describe("exact ", prior, detection, cost, exact_plan, budget, seed + 1000)
        difference = exact.detection_probability - greedy.detection_probability
        se = math.sqrt(
            greedy.detection_probability * (1 - greedy.detection_probability) / TRIALS
            + exact.detection_probability * (1 - exact.detection_probability) / TRIALS
        )
        print(
            f"  exact - greedy: analytical {p_exact - p_greedy:+.4f} | simulated {difference:+.4f} "
            f"(standard error {se:.4f}, {difference / se:.1f} standard errors)\n"
        )
    print("Simulated values are estimates with sampling error, not proofs. The 95% intervals fail to contain the true")
    print("value about 1 time in 20 by construction, and they describe sampling error only, not model error.")


if __name__ == "__main__":
    main()
