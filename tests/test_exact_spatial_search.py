"""Tests for the exact (exhaustive) spatial-search optimiser and its comparison with greedy.

The exact solver is independent of ``optimal_search.spatial_search``; only the comparison tests import both.
Independent references used here: hand enumeration, enumeration of all search *sequences*, a
pseudo-polynomial dynamic program for integer costs, and full probability-space enumeration for expected
costs. Seeded random instances only generate inputs for exact identities; nothing is simulated.
"""

import ast
import itertools
import math
from pathlib import Path

import numpy as np
import pytest

import optimal_search.exact_spatial_search as exact_module
from optimal_search.exact_spatial_search import (
    ExactSolution,
    SearchSpaceTooLargeError,
    count_feasible_allocations,
    detection_probability_of_counts,
    enumerate_feasible_allocations,
    expected_search_cost,
    fractional_upper_bound,
    order_for_minimum_expected_cost,
    solve_exact,
    total_cost_of_counts,
)
from optimal_search.spatial_search import greedy_search_plan
from tests.spatial_reference import enumerate_detection_probability, enumerate_expected_cost, knapsack_dp_max_probability


def greedy_value(prior, detection, cost, budget):
    plan = greedy_search_plan(prior, detection, cost, budget=budget)
    return plan[-1].cumulative_probability if plan else 0.0


def closed_form(prior, detection, counts):
    return 1.0 - sum(p * (1.0 - d) ** n for p, d, n in zip(prior, detection, counts))


def random_instances(seed, count, cells, equal_cost=False):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        prior = rng.dirichlet(np.ones(cells)).tolist()
        detection = rng.uniform(0.05, 0.95, size=cells).tolist()
        cost = [1.0] * cells if equal_cost else rng.uniform(0.5, 3.0, size=cells).tolist()
        budget = float(rng.integers(1, 7)) if equal_cost else float(rng.uniform(1.0, 6.0))
        yield prior, detection, cost, budget


# --- independence from the greedy implementation --------------------------------------------------------


def test_exact_solver_does_not_import_the_greedy_module():
    source = Path(exact_module.__file__).read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("spatial_search" in name.replace("exact_spatial_search", "") for name in imported)
    assert imported <= {"heapq", "math", "numbers", "collections.abc", "dataclasses"}


# --- manually enumerated problems ---------------------------------------------------------------------------


def test_feasible_allocations_hand_enumeration_equal_costs():
    # costs (1, 1), budget 2: n0 + n1 <= 2
    assert list(enumerate_feasible_allocations([1.0, 1.0], 2.0)) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]


def test_feasible_allocations_hand_enumeration_unequal_costs():
    # costs (1.5, 1), budget 3: n0 = 0 -> n1 <= 3; n0 = 1 -> n1 <= 1 (1.5 + 1 <= 3); n0 = 2 -> n1 = 0
    assert list(enumerate_feasible_allocations([1.5, 1.0], 3.0)) == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (2, 0)]


def test_hand_solved_equal_cost_problem():
    # prior (0.6, 0.4), d = 0.5, costs 1, budget 2. P by allocation:
    # (0,0)=0, (0,1)=0.2, (0,2)=0.3, (1,0)=0.3, (1,1)=0.5, (2,0)=0.45  -> optimum (1, 1)
    solution = solve_exact([0.6, 0.4], [0.5, 0.5], [1.0, 1.0], 2.0)
    assert solution.counts == (1, 1)
    assert solution.detection_probability == pytest.approx(0.5, rel=1e-14)
    assert solution.total_cost == pytest.approx(2.0)
    assert solution.allocations_examined == 6


def test_hand_solved_unequal_cost_problem():
    # costs (1.5, 1), budget 3: P = (0,3): 0.35, (1,1): 0.5, (2,0): 0.45, (1,0): 0.3 ... optimum (1, 1) at cost 2.5
    solution = solve_exact([0.6, 0.4], [0.5, 0.5], [1.5, 1.0], 3.0)
    assert solution.counts == (1, 1)
    assert solution.total_cost == pytest.approx(2.5)
    assert solution.detection_probability == pytest.approx(0.5, rel=1e-14)
    assert solution.allocations_examined == 7


def test_allocation_probabilities_match_the_papers_closed_form_for_every_feasible_allocation():
    prior, detection, cost, budget = [0.5, 0.3, 0.2], [0.9, 0.6, 0.3], [2.0, 1.0, 1.0], 4.0
    for counts in enumerate_feasible_allocations(cost, budget):
        assert detection_probability_of_counts(prior, detection, counts) == pytest.approx(closed_form(prior, detection, counts), abs=1e-14)
        assert total_cost_of_counts(cost, counts) <= budget + 1e-12


@pytest.mark.parametrize("seed", range(4))
def test_exact_solution_beats_every_enumerated_allocation_and_matches_sequence_brute_force(seed):
    for prior, detection, cost, budget in random_instances(10 + seed, 6, 3):
        if budget / min(cost) > 6:
            continue  # keep every candidate sequence within the brute-forced length
        solution = solve_exact(prior, detection, cost, budget)
        best = max(closed_form(prior, detection, counts) for counts in enumerate_feasible_allocations(cost, budget))
        assert solution.detection_probability == pytest.approx(best, abs=1e-12)
        # independent: enumerate every search *sequence* within the budget (order must not matter)
        sequence_best = 0.0
        for length in range(0, 7):
            for looks in itertools.product(range(3), repeat=length):
                if sum(cost[j] for j in looks) <= budget + 1e-12:
                    counts = [looks.count(j) for j in range(3)]
                    sequence_best = max(sequence_best, closed_form(prior, detection, counts))
        assert solution.detection_probability == pytest.approx(sequence_best, abs=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_exact_matches_dynamic_programming_for_integer_costs(seed):
    rng = np.random.default_rng(900 + seed)
    for _ in range(6):
        cells = 5
        prior = rng.dirichlet(np.ones(cells)).tolist()
        detection = rng.uniform(0.1, 0.9, size=cells).tolist()
        costs = rng.integers(1, 5, size=cells).tolist()
        budget = int(rng.integers(6, 12))
        solution = solve_exact(prior, detection, [float(c) for c in costs], float(budget))
        assert solution.detection_probability == pytest.approx(knapsack_dp_max_probability(prior, detection, costs, budget), abs=1e-12)


def test_solution_is_invariant_to_relabelling_the_cells():
    prior, detection, cost, budget = [0.5, 0.3, 0.2], [0.9, 0.6, 0.3], [2.0, 1.0, 1.5], 3.5
    reference = solve_exact(prior, detection, cost, budget).detection_probability
    for permutation in itertools.permutations(range(3)):
        value = solve_exact([prior[i] for i in permutation], [detection[i] for i in permutation], [cost[i] for i in permutation], budget)
        assert value.detection_probability == pytest.approx(reference, abs=1e-13)


# --- the unequal-cost counterexample to greedy selection -------------------------------------------------------------------


def test_pr4_counterexample_exact_beats_greedy():
    prior, detection, cost, budget = [0.52, 0.27, 0.21], [1.0, 1.0, 1.0], [2.0, 1.0, 1.0], 2.0
    solution = solve_exact(prior, detection, cost, budget)
    assert solution.counts == (1, 0, 0)
    assert solution.detection_probability == pytest.approx(0.52, rel=1e-14)
    assert solution.sequence == (0,)
    assert greedy_value(prior, detection, cost, budget) == pytest.approx(0.48, rel=1e-13)
    assert solution.detection_probability > greedy_value(prior, detection, cost, budget)


def test_greedy_can_lose_almost_everything_and_the_gap_approaches_one():
    # budget 1; cell B is cheap with a slightly better ratio than A, so greedy takes it and can no
    # longer afford A or C (cost 1 each): greedy gets p_B, exact gets p_A.
    ratios = []
    for epsilon in (0.1, 0.01, 0.001):
        p_a = 0.8
        p_b = p_a * epsilon * 1.01
        prior = [p_a, p_b, 1.0 - p_a - p_b]
        cost = [1.0, epsilon, 1.0]
        solution = solve_exact(prior, [1.0] * 3, cost, 1.0)
        assert solution.detection_probability == pytest.approx(0.8, rel=1e-12)
        greedy = greedy_value(prior, [1.0] * 3, cost, 1.0)
        assert greedy == pytest.approx(p_b, rel=1e-12)
        ratios.append(greedy / solution.detection_probability)
        assert solution.detection_probability - greedy > 0.7  # the absolute gap stays large
    assert ratios == sorted(ratios, reverse=True)
    assert ratios[-1] < 0.002


# --- greedy and exact agree under assumptions ------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_equal_costs_greedy_equals_exact(seed):
    for prior, detection, cost, budget in random_instances(100 + seed, 8, 4, equal_cost=True):
        assert greedy_value(prior, detection, cost, budget) == pytest.approx(solve_exact(prior, detection, cost, budget).detection_probability, abs=1e-12)


def test_equal_non_unit_costs_greedy_equals_exact():
    prior, detection = [0.4, 0.35, 0.25], [0.5, 0.7, 0.9]
    for cost, budget in (([2.5] * 3, 10.0), ([0.4] * 3, 1.7)):
        assert greedy_value(prior, detection, cost, budget) == pytest.approx(solve_exact(prior, detection, cost, budget).detection_probability, abs=1e-12)


def test_unequal_costs_can_still_agree_when_the_budget_is_exactly_exhausted_without_blocking():
    # costs (1, 2), budget 3: ratio order takes cell 0 (ratio 0.5*... ) then cell 1 and spends exactly 3.
    prior, detection, cost, budget = [0.5, 0.5], [0.9, 0.9], [1.0, 2.0], 3.0
    assert greedy_value(prior, detection, cost, budget) == pytest.approx(solve_exact(prior, detection, cost, budget).detection_probability, abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_greedy_never_beats_exact(seed):
    for prior, detection, cost, budget in random_instances(200 + seed, 10, 4):
        assert greedy_value(prior, detection, cost, budget) <= solve_exact(prior, detection, cost, budget).detection_probability + 1e-12


# --- divisible-effort upper bound and the bound on the greedy gap ------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_fractional_bound_bounds_exact_and_the_greedy_gap(seed):
    for prior, detection, cost, budget in random_instances(300 + seed, 10, 4):
        bound = fractional_upper_bound(prior, detection, cost, budget)
        exact = solve_exact(prior, detection, cost, budget).detection_probability
        greedy = greedy_value(prior, detection, cost, budget)
        assert exact <= bound.value + 1e-12
        # greedy takes every search whose ratio is strictly above the critical ratio, in any tie order
        assert greedy >= bound.strict_prefix_value - 1e-12
        assert exact - greedy <= bound.value - bound.strict_prefix_value + 1e-12
        assert bound.value - bound.strict_prefix_value == pytest.approx(bound.critical_ratio * (budget - bound.strict_prefix_cost), abs=1e-12)
        if bound.blocked_cell is not None:
            assert bound.value - bound.prefix_value <= bound.blocked_gain + 1e-15


def test_the_greedy_gap_bound_must_use_the_strict_prefix_when_ratios_tie():
    # Regression for an incorrect first version of the bound. Cells 1 and 2 have exactly equal ratios
    # (0.27 / 3 and 0.09 / 1): a plan that breaks the tie toward cell 1 spends the whole budget on the
    # tied searches, the greedy rule picks cell 2 (by floating-point rounding) and falls below the
    # "first-come prefix" value, so only the strict prefix is a valid guarantee.
    prior, detection, cost, budget = [0.6, 0.3, 0.1], [0.9] * 3, [1.0, 3.0, 1.0], 4.0
    bound = fractional_upper_bound(prior, detection, cost, budget)
    exact = solve_exact(prior, detection, cost, budget).detection_probability
    greedy = greedy_value(prior, detection, cost, budget)
    assert exact == pytest.approx(0.81, rel=1e-12)
    assert bound.prefix_value == pytest.approx(0.81, rel=1e-12)
    assert greedy < bound.prefix_value - 0.1  # so "greedy >= first-come prefix" is false under ties
    assert bound.strict_prefix_value == pytest.approx(0.54, rel=1e-12)
    assert bound.critical_ratio == pytest.approx(0.09, rel=1e-12)
    assert greedy >= bound.strict_prefix_value
    assert exact - greedy <= bound.critical_ratio * (budget - bound.strict_prefix_cost) + 1e-12  # 0.117 <= 0.27


def test_fractional_bound_hand_example():
    # Unequal-cost counterexample: items sorted by ratio: cell 1 (0.27/1), cell 0 (0.52/2 = 0.26), cell 2 (0.21/1).
    bound = fractional_upper_bound([0.52, 0.27, 0.21], [1.0] * 3, [2.0, 1.0, 1.0], 2.0)
    assert bound.prefix_value == pytest.approx(0.27)
    assert bound.blocked_cell == 0
    assert bound.value == pytest.approx(0.27 + 0.5 * 0.52)


def test_fractional_bound_is_tight_when_the_budget_is_exhausted_exactly():
    prior, detection = [0.5, 0.3, 0.2], [0.9, 0.6, 0.3]
    bound = fractional_upper_bound(prior, detection, [1.0] * 3, 4.0)
    assert bound.value == pytest.approx(solve_exact(prior, detection, [1.0] * 3, 4.0).detection_probability, abs=1e-13)


# --- repeated looks ------------------------------------------------------------------------------------------------------------------------


def test_repeated_searches_of_a_dominant_cell():
    # gains for cell 0: 0.27, 0.189, 0.1323, 0.09261; cell 1: 0.03 ... so all four searches go to cell 0
    solution = solve_exact([0.9, 0.1], [0.3, 0.3], [1.0, 1.0], 4.0)
    assert solution.counts == (4, 0)
    assert solution.detection_probability == pytest.approx(0.9 * (1 - 0.7**4), rel=1e-13)
    assert solution.sequence == (0, 0, 0, 0)


def test_repeated_searches_single_cell_and_independent_attempts():
    solution = solve_exact([1.0], [0.4], [1.0], 5.0)
    assert solution.counts == (5,)
    assert solution.detection_probability == pytest.approx(1 - 0.6**5, rel=1e-14)


def test_repeats_split_across_cells_when_gains_cross():
    # prior (0.6, 0.4), d = 0.5, budget 3: gains 0.3, 0.15, 0.075 / 0.2, 0.1 -> counts (2, 1) with P = 0.45 + 0.2
    solution = solve_exact([0.6, 0.4], [0.5, 0.5], [1.0, 1.0], 3.0)
    assert solution.counts == (2, 1)
    assert solution.detection_probability == pytest.approx(0.65, rel=1e-14)


# --- zero budget, perfect detection ------------------------------------------------------------------------------------------------


def test_zero_budget_gives_the_empty_plan():
    solution = solve_exact([0.5, 0.5], [0.9, 0.9], [1.0, 1.0], 0.0)
    assert solution == ExactSolution((0, 0), 0.0, 0.0, (), 0.0, None, 0.0, 1)


def test_budget_below_every_cost_gives_the_empty_plan():
    solution = solve_exact([0.5, 0.5], [0.9, 0.9], [2.0, 3.0], 1.9)
    assert solution.counts == (0, 0)
    assert solution.detection_probability == 0.0
    assert solution.sequence == ()


def test_perfect_detection_is_a_knapsack_over_single_searches():
    prior, cost = [0.5, 0.3, 0.2], [1.0, 1.0, 1.0]
    assert solve_exact(prior, [1.0] * 3, cost, 2.0).counts == (1, 1, 0)
    assert solve_exact(prior, [1.0] * 3, cost, 2.0).detection_probability == pytest.approx(0.8, rel=1e-14)
    full = solve_exact(prior, [1.0] * 3, cost, 5.0)
    assert full.counts == (1, 1, 1)  # repeats are useless and cost more
    assert full.detection_probability == pytest.approx(1.0, abs=1e-15)


def test_perfect_detection_unequal_costs_reduces_to_zero_one_knapsack():
    # values 0.52, 0.27, 0.21 with costs 2, 1, 1 and capacity 2: best subset is {cell 0}
    assert solve_exact([0.52, 0.27, 0.21], [1.0] * 3, [2.0, 1.0, 1.0], 2.0).counts == (1, 0, 0)
    assert solve_exact([0.52, 0.27, 0.21], [1.0] * 3, [2.0, 1.0, 1.0], 3.0).counts == (1, 1, 0)
    assert solve_exact([0.52, 0.27, 0.21], [1.0] * 3, [2.0, 1.0, 1.0], 4.0).counts == (1, 1, 1)


def test_zero_detection_everywhere_gives_the_empty_plan():
    solution = solve_exact([0.5, 0.5], [0.0, 0.0], [1.0, 1.0], 10.0)
    assert solution.counts == (0, 0)
    assert solution.detection_probability == 0.0


# --- deterministic ties -----------------------------------------------------------------------------------------------------------------


def test_equal_cells_prefer_the_lower_index():
    assert solve_exact([0.5, 0.5], [0.5, 0.5], [1.0, 1.0], 1.0).counts == (1, 0)
    assert solve_exact([0.25] * 4, [0.5] * 4, [1.0] * 4, 2.0).counts == (1, 1, 0, 0)


def test_ties_are_broken_by_lower_cost_before_index():
    # d = (1, 0): any extra search is useless; (1, 0) beats (1, 1), (1, 2), (2, 0) ... on cost.
    solution = solve_exact([0.5, 0.5], [1.0, 0.0], [1.0, 1.0], 3.0)
    assert solution.counts == (1, 0)
    assert solution.total_cost == 1.0


def test_cheaper_equal_probability_cell_wins_over_a_lower_index():
    # cells have identical probability contribution, cell 1 is cheaper: choose (0, 1) despite the index rule
    solution = solve_exact([0.5, 0.5], [1.0, 1.0], [2.0, 1.0], 2.0)
    assert solution.counts == (0, 1)  # P = 0.5 either way ((1,0) costs 2, (0,1) costs 1)


def test_solutions_are_deterministic_and_repeatable():
    args = ([0.25] * 4, [0.5] * 4, [1.0] * 4, 3.0)
    assert solve_exact(*args) == solve_exact(*args)
    assert solve_exact(*args).counts == (1, 1, 1, 0)


def test_near_ties_from_rounding_are_treated_as_ties():
    # costs 0.1 + 0.2 and 0.3 differ by ~5e-17 in floating point: nominally symmetric cells, so the lower index wins.
    solution = solve_exact([0.5, 0.5], [0.3, 0.3], [0.1 + 0.2, 0.3], 0.3)
    assert solution.counts == (1, 0)


def test_tie_tolerance_controls_what_counts_as_a_tie():
    # P(1,0) = 0.5 * (0.6 - 1e-6) and P(0,1) = 0.5 * 0.6 differ by 5e-7
    prior, detection = [0.5, 0.5], [0.6 - 1e-6, 0.6]
    assert solve_exact(prior, detection, [1.0, 1.0], 1.0, tie_tolerance=0.0).counts == (0, 1)
    assert solve_exact(prior, detection, [1.0, 1.0], 1.0, tie_tolerance=1e-3).counts == (1, 0)
    assert solve_exact(prior, detection, [1.0, 1.0], 1.0).counts == (0, 1)  # 5e-7 exceeds the default tolerance
    # a relative difference of ~1.7e-13 is inside the default relative tolerance (1e-12)
    assert solve_exact([0.5, 0.5], [0.6 - 1e-13, 0.6], [1.0, 1.0], 1.0).counts == (1, 0)
    # the tolerance is relative: a difference of 1e-18 on probabilities of size 3e-18 is not a tie
    assert solve_exact([1.0], [1e-18], [1.0], 3.0).counts == (3,)


# --- order: same probability, different expected cost ------------------------------------------------------------------------------------


def test_orderings_of_one_allocation_share_the_detection_probability_but_not_the_expected_cost():
    prior, detection, cost = [0.6, 0.4], [0.5, 0.5], [1.0, 1.0]
    forward = expected_search_cost(prior, detection, cost, [0, 1])
    reverse = expected_search_cost(prior, detection, cost, [1, 0])
    assert forward[0] == pytest.approx(1.7, rel=1e-14)
    assert reverse[0] == pytest.approx(1.8, rel=1e-14)
    assert forward[1] == pytest.approx((0.3 * 1 + 0.2 * 2) / 0.5, rel=1e-14)
    assert reverse[1] == pytest.approx((0.2 * 1 + 0.3 * 2) / 0.5, rel=1e-14)
    assert enumerate_detection_probability(prior, detection, [0, 1]) == pytest.approx(enumerate_detection_probability(prior, detection, [1, 0]), rel=1e-14)
    assert solve_exact(prior, detection, cost, 2.0).sequence == (0, 1)


@pytest.mark.parametrize("looks", [[0], [0, 1], [1, 0], [0, 1, 0], [2, 0, 1, 0]])
def test_expected_cost_matches_full_probability_space_enumeration(looks):
    prior, detection, cost = [0.5, 0.3, 0.2], [0.9, 0.6, 0.35], [1.0, 2.0, 0.5]
    incurred, _ = expected_search_cost(prior, detection, cost, looks)
    brute, detected = enumerate_expected_cost(prior, detection, cost, looks)
    assert incurred == pytest.approx(brute, rel=1e-13)
    assert detected == pytest.approx(closed_form(prior, detection, [looks.count(j) for j in range(3)]), rel=1e-13)


def test_the_returned_sequence_minimises_the_expected_cost_over_all_orderings():
    prior, detection, cost = [0.5, 0.3, 0.2], [0.5, 0.6, 0.9], [1.0, 2.0, 0.5]
    counts = (2, 1, 1)
    sequence = order_for_minimum_expected_cost(prior, detection, cost, counts)
    assert sorted(sequence) == [0, 0, 1, 2]
    best = expected_search_cost(prior, detection, cost, sequence)
    for permutation in set(itertools.permutations([0, 0, 1, 2])):
        other = expected_search_cost(prior, detection, cost, permutation)
        assert best[0] <= other[0] + 1e-13
        assert best[1] <= other[1] + 1e-13
    # the detection probability is the same for every ordering
    assert len({round(enumerate_detection_probability(prior, detection, list(perm)), 13) for perm in set(itertools.permutations([0, 0, 1, 2]))}) == 1


@pytest.mark.parametrize("seed", range(4))
def test_the_solvers_sequence_is_expected_cost_optimal_on_random_problems(seed):
    for prior, detection, cost, budget in random_instances(400 + seed, 6, 3):
        solution = solve_exact(prior, detection, cost, budget)
        if len(solution.sequence) > 6:
            continue
        for permutation in set(itertools.permutations(solution.sequence)):
            other = expected_search_cost(prior, detection, cost, permutation)
            assert solution.expected_cost <= other[0] + 1e-12
            if solution.expected_cost_given_detection is not None:
                assert solution.expected_cost_given_detection <= other[1] + 1e-12


def test_maximum_probability_within_a_budget_is_not_the_expected_cost_minimiser():
    # perfect detection, costs (2, 1, 1), prior (0.52, 0.27, 0.21), budget 2: the probability-optimal plan is
    # [cell 0] with P = 0.52; but searching everything in ratio order p/c (cells 1, 0, 2) has lower expected cost
    # per detection and starts in a different cell.
    prior, detection, cost = [0.52, 0.27, 0.21], [1.0] * 3, [2.0, 1.0, 1.0]
    budget_plan = solve_exact(prior, detection, cost, 2.0)
    assert budget_plan.sequence == (0,)
    orders = {perm: expected_search_cost(prior, detection, cost, perm) for perm in itertools.permutations(range(3))}
    cheapest_order = min(orders, key=lambda perm: orders[perm][0])
    assert cheapest_order == (1, 0, 2)  # ratios 0.27, 0.26, 0.21
    assert orders[cheapest_order][0] == pytest.approx(1 + 2 * 0.73 + 1 * 0.21, rel=1e-13)
    assert cheapest_order[0] != budget_plan.sequence[0]
    # and within the same budget the plan with the lower expected cost is not the probability-maximising one
    cheaper_plan = expected_search_cost(prior, detection, cost, [1, 2])
    assert cheaper_plan[0] == pytest.approx(1 + 0.73, rel=1e-13)
    assert cheaper_plan[0] < budget_plan.expected_cost
    assert detection_probability_of_counts(prior, detection, [0, 1, 1]) < budget_plan.detection_probability


# --- computational limits ------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cells, k", [(1, 5), (2, 4), (3, 6), (4, 5), (5, 7)])
def test_equal_cost_allocation_count_is_a_binomial_coefficient(cells, k):
    assert count_feasible_allocations([1.0] * cells, float(k)) == math.comb(k + cells, cells)


def test_enumeration_guard_raises_instead_of_running_away():
    with pytest.raises(SearchSpaceTooLargeError):
        solve_exact([0.25] * 4, [0.5] * 4, [1.0] * 4, 30.0, max_allocations=1000)
    with pytest.raises(SearchSpaceTooLargeError):
        solve_exact([1.0], [0.5], [1.0], 1e12)
    with pytest.raises(SearchSpaceTooLargeError):
        count_feasible_allocations([1.0] * 5, 40.0, max_allocations=10_000)
    assert solve_exact([0.25] * 4, [0.5] * 4, [1.0] * 4, 5.0, max_allocations=math.comb(9, 4)).allocations_examined == math.comb(9, 4)
    assert issubclass(SearchSpaceTooLargeError, ValueError)


def test_unequal_cost_allocation_count_is_smaller_than_the_equal_cost_count():
    assert count_feasible_allocations([1.0, 2.0, 3.0], 6.0) < count_feasible_allocations([1.0, 1.0, 1.0], 6.0)


# --- invalid inputs ------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prior",
    [[], [0.5, 0.4], [0.5, 0.6], [-0.1, 1.1], [math.nan, 1.0], [math.inf, 0.0], [0.5, 0.5, 0.5], "ab", [[0.5, 0.5]], None, ["a", "b"]],
)
def test_invalid_priors_are_rejected(prior):
    with pytest.raises((ValueError, TypeError)):
        solve_exact(prior, [0.5, 0.5], [1.0, 1.0], 2.0)


@pytest.mark.parametrize("detection", [[1.1, 0.5], [-0.1, 0.5], [math.nan, 0.5], [0.5], [0.5, 0.5, 0.5], [math.inf, 0.5]])
def test_invalid_detection_probabilities_are_rejected(detection):
    with pytest.raises(ValueError):
        solve_exact([0.5, 0.5], detection, [1.0, 1.0], 2.0)


@pytest.mark.parametrize("cost", [[0.0, 1.0], [-1.0, 1.0], [math.nan, 1.0], [math.inf, 1.0], [1.0], [1.0, 1.0, 1.0]])
def test_invalid_costs_are_rejected(cost):
    with pytest.raises(ValueError):
        solve_exact([0.5, 0.5], [0.5, 0.5], cost, 2.0)


@pytest.mark.parametrize("budget", [-1.0, -1e-9, math.nan, math.inf, -math.inf, "a", None, True])
def test_invalid_budgets_are_rejected(budget):
    with pytest.raises(ValueError):
        solve_exact([0.5, 0.5], [0.5, 0.5], [1.0, 1.0], budget)


@pytest.mark.parametrize("limit", [0, -5, 2.5, True, "10", None])
def test_invalid_enumeration_limits_are_rejected(limit):
    with pytest.raises(ValueError):
        solve_exact([0.5, 0.5], [0.5, 0.5], [1.0, 1.0], 2.0, max_allocations=limit)


@pytest.mark.parametrize("tolerance", [-1e-9, math.nan, math.inf, "x", True])
def test_invalid_tie_tolerances_are_rejected(tolerance):
    with pytest.raises(ValueError):
        solve_exact([0.5, 0.5], [0.5, 0.5], [1.0, 1.0], 2.0, tie_tolerance=tolerance)


@pytest.mark.parametrize("counts", [[1], [1, 2, 3], [-1, 0], [0.5, 1], [True, 0], ["a", 0], None])
def test_invalid_counts_are_rejected(counts):
    with pytest.raises(ValueError):
        detection_probability_of_counts([0.5, 0.5], [0.5, 0.5], counts)


@pytest.mark.parametrize("sequence", [[2], [-1], [0.5], [True], ["a"]])
def test_invalid_sequences_are_rejected(sequence):
    with pytest.raises(ValueError):
        expected_search_cost([0.5, 0.5], [0.5, 0.5], [1.0, 1.0], sequence)


def test_empty_cost_list_is_rejected_by_the_enumerators():
    with pytest.raises(ValueError):
        list(enumerate_feasible_allocations([], 1.0))


# --- numerical edge cases ------------------------------------------------------------------------------------------------------------------------


def test_budget_comparison_tolerates_floating_point_sums():
    # 3 * 0.1 = 0.30000000000000004 > 0.3 in floating point, but three searches fit a budget of 0.3
    solution = solve_exact([1.0], [0.5], [0.1], 0.3)
    assert solution.counts == (3,)
    assert solution.detection_probability == pytest.approx(0.875, rel=1e-14)


def test_tiny_detection_probability_keeps_full_precision():
    solution = solve_exact([1.0], [1e-18], [1.0], 3.0)
    assert solution.counts == (3,)
    assert solution.detection_probability == pytest.approx(3e-18, rel=1e-12)


def test_very_small_prior_and_large_costs():
    solution = solve_exact([1e-300, 1.0 - 1e-300], [0.5, 0.5], [1e12, 1e12], 3e12)
    assert solution.counts == (0, 3)
    assert solution.detection_probability == pytest.approx(0.875, rel=1e-12)


def test_probability_never_exceeds_one_and_extra_searches_after_certainty_are_not_chosen():
    solution = solve_exact([0.5, 0.5], [1.0, 1.0], [1.0, 1.0], 10.0)
    assert solution.counts == (1, 1)
    assert 0.0 < solution.detection_probability <= 1.0


def test_budget_exactly_equal_to_a_combination_of_costs():
    solution = solve_exact([0.4, 0.6], [0.9, 0.9], [0.7, 0.3], 1.0)
    assert solution.total_cost <= 1.0 + 1e-12
    assert solution.detection_probability == pytest.approx(solve_exact([0.4, 0.6], [0.9, 0.9], [0.7, 0.3], 1.0 + 1e-13).detection_probability, abs=1e-12)


def test_float_and_integer_inputs_are_accepted():
    assert solve_exact([1, 0], [1, 1], [1, 1], 2).counts == (1, 0)
    assert solve_exact(np.array([0.5, 0.5]), np.array([0.5, 0.5]), np.array([1.0, 1.0]), np.float64(2.0)).counts == (1, 1)


def test_expected_cost_given_detection_is_none_when_detection_is_impossible():
    incurred, given = expected_search_cost([0.5, 0.5], [0.0, 0.0], [1.0, 1.0], [0, 1])
    assert incurred == pytest.approx(2.0)
    assert given is None


def test_solution_costs_and_probabilities_are_consistent_with_the_counts():
    prior, detection, cost = [0.5, 0.3, 0.2], [0.9, 0.6, 0.3], [2.0, 1.0, 1.5]
    solution = solve_exact(prior, detection, cost, 5.0)
    assert solution.total_cost == pytest.approx(total_cost_of_counts(cost, solution.counts))
    assert solution.detection_probability == pytest.approx(detection_probability_of_counts(prior, detection, solution.counts))
    assert sorted(solution.sequence) == sorted(j for j, n in enumerate(solution.counts) for _ in range(n))
    assert solution.total_cost <= 5.0 + 1e-12


def test_gap_can_exceed_0_99_so_the_supremum_of_the_gap_is_one():
    for p_a, epsilon in ((0.99, 0.001), (0.999, 0.0001)):
        p_b = p_a * epsilon * 1.01
        prior, cost = [p_a, p_b, 1.0 - p_a - p_b], [1.0, epsilon, 1.0]
        exact = solve_exact(prior, [1.0] * 3, cost, 1.0).detection_probability
        gap = exact - greedy_value(prior, [1.0] * 3, cost, 1.0)
        assert exact == pytest.approx(p_a, rel=1e-12)
        assert gap > p_a - 0.002
    assert gap > 0.998


# --- regressions from code review ----------------------------------------------------------------------------------------


def test_the_enumeration_guard_trips_before_any_lookup_table_is_built():
    # 100 cells, unit costs, budget 99,998: every cell alone can be searched ~1e5 times, so there are at least
    # 1e7 feasible allocations. The guard must raise from that lower bound, not after building ~1e7 table entries.
    import time
    import tracemalloc

    tracemalloc.start()
    started = time.perf_counter()
    with pytest.raises(SearchSpaceTooLargeError):
        solve_exact([0.01] * 100, [0.5] * 100, [1.0] * 100, 99_998.0)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 20_000_000
    assert elapsed < 2.0


def test_the_lower_bound_check_never_rejects_a_problem_within_the_limit():
    # allocations = 126 = C(9, 4); the lower bound 1 + sum(max_j) = 21 must not trigger
    assert solve_exact([0.25] * 4, [0.5] * 4, [1.0] * 4, 5.0, max_allocations=126).allocations_examined == 126


def test_cells_that_cannot_be_searched_do_not_consume_recursion_depth():
    solution = solve_exact([1.0 / 2000] * 2000, [0.5] * 2000, [1.0] * 2000, 0.0)
    assert solution.counts == (0,) * 2000
    assert solution.allocations_examined == 1


def test_many_searchable_cells_do_not_hit_the_recursion_limit():
    cells = 3000  # deeper than Python's default recursion limit
    solution = solve_exact([1.0 / cells] * cells, [0.5] * cells, [1.0] * cells, 1.0)
    assert solution.allocations_examined == cells + 1
    assert solution.counts[0] == 1 and sum(solution.counts) == 1


def test_cost_ties_and_the_budget_use_relative_tolerances_for_tiny_costs():
    # perfect detection: one search is optimal and repeats only add cost, even when costs are far below 1
    assert solve_exact([1.0], [1.0], [1e-13], 1e-12).counts == (1,)
    # a tiny budget is not loosened by an absolute slack: exactly ten searches of cost 1e-13 fit in 1e-12
    assert solve_exact([1.0], [0.5], [1e-13], 1e-12).counts == (10,)
    assert solve_exact([1.0], [0.5], [1e-13], 0.0).counts == (0,)


def test_fractional_bound_is_an_upper_bound_when_the_ratio_underflows():
    # gain 5e-19 divided by a cost of 1e308 rounds to zero, which used to drop the search from the bound
    prior, detection, cost, budget = [0.5, 0.5], [0.0, 1e-18], [1e308, 1e308], 1e308
    bound = fractional_upper_bound(prior, detection, cost, budget)
    exact = solve_exact(prior, detection, cost, budget).detection_probability
    assert exact == pytest.approx(5e-19, rel=1e-12)
    assert bound.value >= exact * (1 - 1e-12)


def test_expected_cost_ignores_searches_that_can_no_longer_matter():
    # cells 0 and 1 hold all the probability and detection is perfect, so after two searches the target
    # has certainly been found; the third search (cell 2, prior 0, cost 1e18) is never made.
    incurred, given = expected_search_cost([0.7, 0.3, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1e18], [0, 1, 2])
    assert incurred == pytest.approx(1.0 + 0.3, rel=1e-12)
    assert given == pytest.approx((0.7 * 1 + 0.3 * 2) / 1.0, rel=1e-12)
