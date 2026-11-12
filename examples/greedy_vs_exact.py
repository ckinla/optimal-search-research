"""Myopic greedy selection versus the exact best plan under a finite budget.

Greedy searches next the affordable cell with the largest posterior * detection / cost: the best immediate
payoff per unit cost. With equal costs that is optimal for a fixed number of searches. With unequal,
indivisible costs and a budget it can lose to the exact optimum, found here by exhaustive enumeration.
A seeded Monte Carlo simulation independently checks the analytical probabilities.

Run: uv run python examples/greedy_vs_exact.py
"""

from optimal_search.exact_spatial_search import solve_exact
from optimal_search.monte_carlo_spatial_search import simulate_search
from optimal_search.spatial_search import greedy_search_plan

CASES = (
    ("Perfect detection, costs (2, 1, 1), budget 2", [0.52, 0.27, 0.21], [1.0, 1.0, 1.0], [2.0, 1.0, 1.0], 2.0),
    ("Visual essay's three places, costs (2, 1, 1), budget 2", [0.5, 0.3, 0.2], [0.8, 0.9, 0.5], [2.0, 1.0, 1.0], 2.0),
    ("Equal costs, four searches (greedy is optimal here)", [0.5, 0.3, 0.2], [0.8, 0.9, 0.5], [1.0, 1.0, 1.0], 4.0),
)
TRIALS, SEED = 200_000, 2026


def main() -> None:
    for title, prior, detection, cost, budget in CASES:
        greedy = greedy_search_plan(prior, detection, cost, budget=budget)
        greedy_plan = [step.cell for step in greedy]
        greedy_p = greedy[-1].cumulative_probability if greedy else 0.0
        exact = solve_exact(prior, detection, cost, budget)
        print(title)
        print(f"  prior {prior}, detection {detection}")
        print(f"  greedy plan {greedy_plan}: P(detected) = {greedy_p:.4f}")
        print(f"  exact plan  {list(exact.sequence)}: P(detected) = {exact.detection_probability:.4f} "
              f"({exact.allocations_examined} allocations examined)")
        for name, plan, analytical in (("greedy", greedy_plan, greedy_p), ("exact", list(exact.sequence), exact.detection_probability)):
            sim = simulate_search(prior, detection, cost, plan, trials=TRIALS, seed=SEED)
            low, high = sim.detection_confidence_interval()
            print(f"  Monte Carlo, {name}: {sim.detection_probability:.4f} (95% CI {low:.4f}-{high:.4f}; analytical {analytical:.4f})")
        print()
    print("Monte Carlo estimates carry sampling error; they corroborate the exact calculations, they do not prove them.")


if __name__ == "__main__":
    main()
