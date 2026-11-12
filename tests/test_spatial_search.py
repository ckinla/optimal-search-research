"""Tests for the Bayesian spatial-search model.

Independent verification strategy: besides hand-calculated values, the closed
forms are checked against a brute-force enumeration of the full probability
space (every target location times every detection outcome of every search).
Randomised inputs below come from a fixed-seed generator and are used only to
generate test cases for exact identities; there is no simulation.
"""

import itertools
import math

import numpy as np
import pytest

from optimal_search.spatial_search import (
    SearchStep,
    conditional_next_look_success,
    counts_from_sequence,
    cumulative_detection_probability,
    cumulative_detection_probability_for_sequence,
    greedy_search_plan,
    next_look_priority,
    posterior_after_search_sequence,
    posterior_from_counts,
    probability_of_no_detection,
    select_next_cell,
    unconditional_detection_increment,
    update_after_failed_search,
)
from tests.spatial_reference import (
    enumerate_detection_probability,
    enumerate_first_detection_at,
    enumerate_posterior_after_failures,
)


# --- independent brute-force reference lives in tests/spatial_reference.py ---------------


def sum_of_conditional_probabilities(prior, detection, looks):
    """Running sum of conditional next-look success probabilities (NOT a probability)."""
    posterior = np.array(prior, dtype=float)
    total = 0.0
    for i, cell in enumerate(looks):
        total += posterior[cell] * detection[cell]
        if i + 1 < len(looks):  # no update after the last look (it could condition on an impossible failure)
            posterior = update_after_failed_search(posterior, detection, cell)
    return total


def random_instances(seed, count, cells):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        prior = rng.dirichlet(np.ones(cells))
        detection = rng.uniform(0.05, 0.95, size=cells)
        cost = rng.uniform(0.5, 3.0, size=cells)
        looks = rng.integers(0, cells, size=rng.integers(1, 6)).tolist()
        yield prior, detection, cost, looks


# --- Bayesian posterior: hand-calculated examples ------------------------------------------


def test_failed_search_hand_example_equal_cells():
    # weights (0.6 * 0.5, 0.4) = (0.3, 0.4); total 0.7
    posterior = update_after_failed_search([0.6, 0.4], [0.5, 0.5], 0)
    assert posterior == pytest.approx([3.0 / 7.0, 4.0 / 7.0], rel=1e-14)


def test_failed_search_hand_example_unequal_cells():
    # weights (0.5, 0.3 * 0.4, 0.2) = (0.5, 0.12, 0.2); total 0.82
    posterior = update_after_failed_search([0.5, 0.3, 0.2], [0.9, 0.6, 0.3], 1)
    assert posterior == pytest.approx([0.5 / 0.82, 0.12 / 0.82, 0.2 / 0.82], rel=1e-14)


def test_failed_search_with_perfect_detection_eliminates_the_cell():
    posterior = update_after_failed_search([0.5, 0.3, 0.2], [1.0, 0.5, 0.0], 0)
    assert posterior == pytest.approx([0.0, 0.6, 0.4], rel=1e-14)


def test_failed_search_with_zero_detection_changes_nothing():
    prior = [0.5, 0.3, 0.2]
    assert update_after_failed_search(prior, [0.9, 0.0, 0.5], 1) == pytest.approx(prior, rel=1e-14)


def test_repeated_unsuccessful_looks_in_one_cell():
    # (0.5, 0.5), d = 0.5, two failures in cell 0: weights (0.125, 0.5) -> (0.2, 0.8)
    posterior = posterior_after_search_sequence([0.5, 0.5], [0.5, 0.5], [0, 0])
    assert posterior == pytest.approx([0.2, 0.8], rel=1e-14)
    # three failures: weights (0.0625, 0.5)
    assert posterior_after_search_sequence([0.5, 0.5], [0.5, 0.5], [0, 0, 0]) == pytest.approx([1.0 / 9.0, 8.0 / 9.0], rel=1e-14)


def test_other_cells_keep_their_relative_weights():
    posterior = update_after_failed_search([0.5, 0.3, 0.2], [0.9, 0.6, 0.3], 0)
    assert posterior[1] / posterior[2] == pytest.approx(0.3 / 0.2, rel=1e-14)


# --- posterior: normalisation, closed form, order independence --------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_posterior_is_normalised_and_nonnegative_after_every_update(seed):
    for prior, detection, _, looks in random_instances(seed, 20, 4):
        posterior = prior
        for cell in looks:
            posterior = update_after_failed_search(posterior, detection, cell)
            assert posterior.sum() == pytest.approx(1.0, abs=1e-14)
            assert np.all(posterior >= 0.0)


@pytest.mark.parametrize("seed", range(5))
def test_sequential_update_equals_closed_form_and_brute_force(seed):
    for prior, detection, _, looks in random_instances(100 + seed, 15, 4):
        # one explicit single-search update per look, independent of the sequence/closed-form functions
        sequential = np.array(prior, dtype=float)
        for cell in looks:
            sequential = update_after_failed_search(sequential, detection, cell)
        counts = counts_from_sequence(looks, len(prior))
        assert sequential == pytest.approx(posterior_from_counts(prior, detection, counts), rel=1e-12)
        assert sequential == pytest.approx(posterior_after_search_sequence(prior, detection, looks), rel=1e-12)
        assert sequential == pytest.approx(enumerate_posterior_after_failures(prior, detection, looks), rel=1e-12)


def test_posterior_does_not_depend_on_search_order():
    prior, detection = [0.4, 0.3, 0.2, 0.1], [0.6, 0.3, 0.8, 0.5]
    looks = [0, 1, 0, 2, 3, 0]
    reference = posterior_after_search_sequence(prior, detection, looks)
    for permutation in set(itertools.permutations(looks)):
        assert posterior_after_search_sequence(prior, detection, permutation) == pytest.approx(reference, rel=1e-12)


def test_normaliser_of_a_failed_search_is_one_minus_posterior_times_detection():
    q, d = np.array([0.5, 0.3, 0.2]), np.array([0.9, 0.6, 0.3])
    weights = q.copy()
    weights[1] *= 1.0 - d[1]
    assert weights.sum() == pytest.approx(1.0 - q[1] * d[1], rel=1e-14)


# --- unconditional cumulative detection probability ------------------------------------------


def test_cumulative_detection_hand_examples():
    prior, d = [0.6, 0.4], [0.5, 0.5]
    assert cumulative_detection_probability(prior, d, [2, 0]) == pytest.approx(0.6 * 0.75, rel=1e-14)  # counts: two looks in cell 0
    assert cumulative_detection_probability(prior, d, [1, 1]) == pytest.approx(0.5 * 1.0 * 0.6 + 0.4 * 0.5 * 1.0, rel=1e-14)
    assert cumulative_detection_probability(prior, d, [2, 1]) == pytest.approx(1.0 - (0.6 * 0.25 + 0.4 * 0.5), rel=1e-14)


def test_cumulative_detection_equals_one_minus_sum_formula():
    prior, d, counts = [0.5, 0.3, 0.2], [0.9, 0.6, 0.3], [2, 1, 3]
    expected = 1.0 - sum(p * (1 - dj) ** n for p, dj, n in zip(prior, d, counts))
    assert cumulative_detection_probability(prior, d, counts) == pytest.approx(expected, rel=1e-13)
    assert probability_of_no_detection(prior, d, counts) == pytest.approx(1.0 - expected, rel=1e-13)


def test_no_searches_give_zero_detection_probability():
    assert cumulative_detection_probability([0.5, 0.5], [0.9, 0.9], [0, 0]) == 0.0
    assert probability_of_no_detection([0.5, 0.5], [0.9, 0.9], [0, 0]) == pytest.approx(1.0, rel=1e-15)


def test_perfect_detection_in_every_cell_searched_once_gives_certainty():
    assert cumulative_detection_probability([0.2, 0.3, 0.5], [1.0, 1.0, 1.0], [1, 1, 1]) == pytest.approx(1.0, abs=1e-15)


def test_perfect_detection_counts_only_searched_cells():
    assert cumulative_detection_probability([0.2, 0.3, 0.5], [1.0, 1.0, 1.0], [1, 0, 1]) == pytest.approx(0.7, rel=1e-14)


def test_imperfect_detection_never_reaches_certainty_after_finitely_many_looks():
    value = cumulative_detection_probability([0.5, 0.5], [0.5, 0.5], [5, 5])
    assert value == pytest.approx(1.0 - 0.5**5, rel=1e-14)
    assert value < 1.0


def test_zero_detection_probability_never_detects():
    assert cumulative_detection_probability([0.5, 0.5], [0.0, 0.0], [100, 100]) == 0.0


@pytest.mark.parametrize("seed", range(5))
def test_cumulative_detection_matches_brute_force_enumeration(seed):
    for prior, detection, _, looks in random_instances(200 + seed, 12, 3):
        expected = enumerate_detection_probability(prior, detection, looks)
        assert cumulative_detection_probability_for_sequence(prior, detection, looks) == pytest.approx(expected, rel=1e-12)


def test_cumulative_detection_is_independent_of_search_order():
    prior, detection = [0.4, 0.3, 0.2, 0.1], [0.6, 0.3, 0.8, 0.5]
    looks = [0, 1, 0, 2, 3, 0]
    values = {round(cumulative_detection_probability_for_sequence(prior, detection, perm), 13) for perm in itertools.permutations(looks)}
    assert len(values) == 1


def test_cumulative_detection_increases_with_every_useful_search():
    prior, detection = [0.5, 0.3, 0.2], [0.9, 0.6, 0.3]
    previous = 0.0
    for n in range(1, 8):
        current = cumulative_detection_probability_for_sequence(prior, detection, [0, 1, 2] * n)
        assert current > previous
        previous = current
    assert previous < 1.0


# --- conditional versus unconditional next-look probability ----------------------------------------


def test_hand_example_conditional_versus_unconditional():
    # prior (0.6, 0.4), d = 0.5 everywhere; look 1 in cell 0 (failed), look 2 in cell 1.
    posterior = update_after_failed_search([0.6, 0.4], [0.5, 0.5], 0)
    assert conditional_next_look_success(posterior, [0.5, 0.5], 1) == pytest.approx(2.0 / 7.0, rel=1e-14)  # (4/7) * 0.5
    assert unconditional_detection_increment([0.6, 0.4], [0.5, 0.5], [1, 0], 1) == pytest.approx(0.2, rel=1e-14)


@pytest.mark.parametrize("seed", range(5))
def test_unconditional_increment_is_conditional_times_probability_of_prior_failure(seed):
    for prior, detection, _, looks in random_instances(300 + seed, 12, 4):
        for i, cell in enumerate(looks):
            before = looks[:i]
            counts = counts_from_sequence(before, len(prior))
            posterior = posterior_after_search_sequence(prior, detection, before)
            conditional = conditional_next_look_success(posterior, detection, cell)
            unconditional = unconditional_detection_increment(prior, detection, counts, cell)
            p_failed_so_far = 1.0 - cumulative_detection_probability(prior, detection, counts)
            assert unconditional == pytest.approx(p_failed_so_far * conditional, rel=1e-12)
            assert unconditional == pytest.approx(enumerate_first_detection_at(prior, detection, looks, i), rel=1e-12)
            assert unconditional <= conditional + 1e-15


@pytest.mark.parametrize("seed", range(5))
def test_unconditional_increments_sum_to_the_cumulative_probability(seed):
    for prior, detection, _, looks in random_instances(400 + seed, 12, 4):
        increments, counts = [], np.zeros(len(prior), dtype=int)
        for cell in looks:
            increments.append(unconditional_detection_increment(prior, detection, counts, cell))
            counts[cell] += 1
        assert sum(increments) == pytest.approx(cumulative_detection_probability(prior, detection, counts), rel=1e-12)


def test_conditional_probabilities_do_not_sum_to_the_cumulative_probability():
    prior, detection, looks = [0.6, 0.4], [0.5, 0.5], [0, 1]
    exact = cumulative_detection_probability_for_sequence(prior, detection, looks)
    assert exact == pytest.approx(0.5, rel=1e-14)
    assert sum_of_conditional_probabilities(prior, detection, looks) == pytest.approx(0.3 + 2.0 / 7.0, rel=1e-13)
    assert sum_of_conditional_probabilities(prior, detection, looks) > exact


def test_sum_of_conditional_probabilities_can_exceed_one():
    # Target is in cell 0 or 1 with equal probability and detection is perfect: detection is certain
    # (probability 1) after one look in each cell, but the conditional sum is 0.5 + 1.0 = 1.5.
    prior, detection, looks = [0.5, 0.5], [1.0, 1.0], [0, 1]
    assert cumulative_detection_probability_for_sequence(prior, detection, looks) == pytest.approx(1.0, abs=1e-15)
    assert enumerate_detection_probability(prior, detection, looks) == pytest.approx(1.0, abs=1e-15)
    assert sum_of_conditional_probabilities(prior, detection, looks) == pytest.approx(1.5, rel=1e-14)


# --- order independence for a fixed set of looks -------------------------------------------------------


def test_order_changes_intermediate_conditional_quantities_but_not_the_final_probability():
    prior, detection = [0.6, 0.4], [0.5, 0.5]
    forward, reverse = [0, 1], [1, 0]
    assert cumulative_detection_probability_for_sequence(prior, detection, forward) == pytest.approx(
        cumulative_detection_probability_for_sequence(prior, detection, reverse), rel=1e-14
    )
    # ...yet the running sums of conditional probabilities differ with the order.
    assert sum_of_conditional_probabilities(prior, detection, forward) == pytest.approx(0.3 + 0.2 / 0.7, rel=1e-13)
    assert sum_of_conditional_probabilities(prior, detection, reverse) == pytest.approx(0.2 + 0.3 / 0.8, rel=1e-13)
    assert sum_of_conditional_probabilities(prior, detection, forward) != pytest.approx(
        sum_of_conditional_probabilities(prior, detection, reverse), rel=1e-6
    )
    # Partial cumulative probabilities after the first look also depend on the order.
    assert cumulative_detection_probability_for_sequence(prior, detection, [0]) != pytest.approx(
        cumulative_detection_probability_for_sequence(prior, detection, [1]), rel=1e-6
    )


@pytest.mark.parametrize("seed", range(4))
def test_sum_of_conditionals_always_at_least_the_exact_probability(seed):
    for prior, detection, _, looks in random_instances(500 + seed, 20, 4):
        exact = cumulative_detection_probability_for_sequence(prior, detection, looks)
        assert sum_of_conditional_probabilities(prior, detection, looks) >= exact - 1e-14


# --- greedy selection ------------------------------------------------------------------------------------


def test_priority_hand_calculation():
    priority = next_look_priority([0.5, 0.3, 0.2], [0.9, 0.6, 0.3], [2.0, 1.0, 0.5])
    assert priority == pytest.approx([0.225, 0.18, 0.12], rel=1e-14)  # 0.5*0.9/2, 0.3*0.6/1, 0.2*0.3/0.5
    assert select_next_cell([0.5, 0.3, 0.2], [0.9, 0.6, 0.3], [2.0, 1.0, 0.5]) == 0


def test_unit_cost_default_and_cost_changes_the_choice():
    posterior, detection = [0.5, 0.3, 0.2], [0.9, 0.6, 0.3]
    assert select_next_cell(posterior, detection) == 0
    assert select_next_cell(posterior, detection, [10.0, 1.0, 1.0]) == 1
    assert select_next_cell(posterior, detection, [10.0, 10.0, 0.1]) == 2


def test_detection_probability_changes_the_choice():
    assert select_next_cell([0.6, 0.4], [0.1, 0.9]) == 1  # 0.06 vs 0.36
    assert select_next_cell([0.6, 0.4], [0.9, 0.9]) == 0


def test_equal_cells_choose_the_first_and_ties_are_deterministic():
    assert select_next_cell([0.25] * 4, [0.5] * 4) == 0
    assert select_next_cell([0.5, 0.5], [0.4, 0.4], [1.0, 1.0]) == 0


def test_cells_with_zero_prior_or_zero_detection_are_never_selected():
    assert select_next_cell([0.0, 1.0], [0.9, 0.1]) == 1
    assert select_next_cell([0.9, 0.1], [0.0, 0.1]) == 1


@pytest.mark.parametrize("seed", range(5))
def test_ranking_is_the_same_for_posterior_prior_weights_and_unconditional_increments(seed):
    for prior, detection, cost, looks in random_instances(600 + seed, 15, 5):
        counts = counts_from_sequence(looks, len(prior))
        posterior = posterior_from_counts(prior, detection, counts)
        by_posterior = next_look_priority(posterior, detection, cost)
        by_increment = np.array(
            [unconditional_detection_increment(prior, detection, counts, j) / cost[j] for j in range(len(prior))]
        )
        # the two differ by the common factor 1 - P_before, so the orderings agree
        factor = 1.0 - cumulative_detection_probability(prior, detection, counts)
        assert by_increment == pytest.approx(factor * by_posterior, rel=1e-12)
        assert list(np.argsort(-by_posterior, kind="stable")) == list(np.argsort(-by_increment, kind="stable"))


def test_greedy_plan_hand_example():
    plan = greedy_search_plan([0.6, 0.4], [0.5, 0.5], max_looks=4)
    assert [step.cell for step in plan] == [0, 1, 0, 1]
    assert [step.unconditional_increment for step in plan] == pytest.approx([0.3, 0.2, 0.15, 0.1], rel=1e-13)
    assert [step.cumulative_probability for step in plan] == pytest.approx([0.3, 0.5, 0.65, 0.75], rel=1e-13)
    assert [step.conditional_success for step in plan][0] == pytest.approx(0.3, rel=1e-13)
    assert plan[1].conditional_success == pytest.approx(0.4 * 0.5 / 0.7, rel=1e-13)
    assert [step.cumulative_cost for step in plan] == pytest.approx([1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("seed", range(4))
def test_greedy_plan_bookkeeping_is_consistent(seed):
    for prior, detection, cost, _ in random_instances(700 + seed, 8, 4):
        plan = greedy_search_plan(prior, detection, cost, max_looks=8)
        looks = [step.cell for step in plan]
        assert plan[-1].cumulative_probability == pytest.approx(cumulative_detection_probability_for_sequence(prior, detection, looks), rel=1e-12)
        assert sum(step.unconditional_increment for step in plan) == pytest.approx(plan[-1].cumulative_probability, rel=1e-12)
        assert [step.cumulative_probability for step in plan] == sorted(step.cumulative_probability for step in plan)
        assert plan[-1].cumulative_cost == pytest.approx(sum(cost[j] for j in looks), rel=1e-13)


def test_repeated_searches_of_one_cell_are_independent_attempts():
    plan = greedy_search_plan([1.0, 0.0], [0.3, 0.9], max_looks=5)
    assert [step.cell for step in plan] == [0] * 5
    assert [step.cumulative_probability for step in plan] == pytest.approx([1 - 0.7**n for n in range(1, 6)], rel=1e-13)
    assert [step.conditional_success for step in plan] == pytest.approx([0.3] * 5, rel=1e-13)


def test_greedy_plan_stops_when_nothing_can_detect():
    plan = greedy_search_plan([0.5, 0.5], [0.0, 0.0], max_looks=10)
    assert plan == []


def test_greedy_plan_stops_when_detection_becomes_certain():
    plan = greedy_search_plan([0.5, 0.5], [1.0, 1.0], max_looks=10)
    assert [step.cell for step in plan] == [0, 1]
    assert plan[-1].cumulative_probability == pytest.approx(1.0, abs=1e-15)


def test_greedy_plan_respects_a_budget():
    plan = greedy_search_plan([0.5, 0.3, 0.2], [0.9, 0.6, 0.3], cost=[2.0, 1.0, 1.0], budget=3.5)
    assert plan[-1].cumulative_cost <= 3.5
    # nothing affordable remains: the cheapest cell costs 1.0 and the remaining budget is below that or priority is zero
    assert 3.5 - plan[-1].cumulative_cost < 1.0


def test_greedy_with_unit_costs_is_optimal_for_a_fixed_number_of_looks_on_these_instances():
    # Unit costs: the cumulative probability is a sum of concave functions of the counts, so greedy
    # is optimal; verified here by exhaustive search over all sequences (not only by the argument).
    for seed in range(5):
        for prior, detection, _, _ in random_instances(800 + seed, 6, 3):
            n_looks = 5
            best = max(
                cumulative_detection_probability_for_sequence(prior, detection, looks)
                for looks in itertools.product(range(3), repeat=n_looks)
            )
            plan = greedy_search_plan(prior, detection, max_looks=n_looks)
            assert plan[-1].cumulative_probability == pytest.approx(best, rel=1e-12)


def test_greedy_is_not_globally_optimal_for_a_finite_budget_with_unequal_costs():
    # Cell 0 costs 2, cells 1 and 2 cost 1; detection is perfect; budget 2.
    # Priorities: 0.52/2 = 0.26, 0.27, 0.21 -> greedy searches cell 1 first and can then only afford
    # cell 2 (total 0.48), while searching cell 0 alone detects with probability 0.52.
    prior, detection, cost, budget = [0.52, 0.27, 0.21], [1.0, 1.0, 1.0], [2.0, 1.0, 1.0], 2.0
    plan = greedy_search_plan(prior, detection, cost, budget=budget)
    assert [step.cell for step in plan] == [1, 2]
    greedy_value = plan[-1].cumulative_probability
    assert greedy_value == pytest.approx(0.48, rel=1e-13)

    def affordable_sequences():
        for length in range(0, 4):
            for looks in itertools.product(range(3), repeat=length):
                if sum(cost[j] for j in looks) <= budget + 1e-12:
                    yield looks

    best = max(affordable_sequences(), key=lambda looks: cumulative_detection_probability_for_sequence(prior, detection, looks) if looks else 0.0)
    assert list(best) == [0]
    assert cumulative_detection_probability_for_sequence(prior, detection, best) == pytest.approx(0.52, rel=1e-13)
    assert greedy_value < 0.52


# --- invalid inputs and numerical edge cases ---------------------------------------------------------------


@pytest.mark.parametrize(
    "prior",
    [[], [[0.5, 0.5]], [0.5, 0.4], [0.5, 0.6], [-0.1, 1.1], [1.1, -0.1], [math.nan, 1.0], [math.inf, 0.0], [0.5, 0.5, 0.5]],
)
def test_invalid_priors_are_rejected(prior):
    with pytest.raises(ValueError):
        posterior_from_counts(prior, [0.5] * len(np.atleast_1d(prior)), [0] * len(np.atleast_1d(prior)))


@pytest.mark.parametrize("detection", [[1.1, 0.5], [-0.1, 0.5], [math.nan, 0.5], [0.5], [0.5, 0.5, 0.5]])
def test_invalid_detection_probabilities_are_rejected(detection):
    with pytest.raises(ValueError):
        cumulative_detection_probability([0.5, 0.5], detection, [1, 1])


@pytest.mark.parametrize("cost", [[0.0, 1.0], [-1.0, 1.0], [math.nan, 1.0], [math.inf, 1.0], [1.0], [1.0, 1.0, 1.0]])
def test_invalid_costs_are_rejected(cost):
    with pytest.raises(ValueError):
        next_look_priority([0.5, 0.5], [0.5, 0.5], cost)


@pytest.mark.parametrize("cell", [-1, 2, 99, 0.5, "a", True, None])
def test_invalid_cell_indices_are_rejected(cell):
    with pytest.raises(ValueError):
        update_after_failed_search([0.5, 0.5], [0.5, 0.5], cell)


@pytest.mark.parametrize("counts", [[-1, 0], [0.5, 1], [math.nan, 0], [1], [1, 1, 1], [[1, 1]]])
def test_invalid_counts_are_rejected(counts):
    with pytest.raises(ValueError):
        cumulative_detection_probability([0.5, 0.5], [0.5, 0.5], counts)


def test_integer_valued_float_counts_are_accepted():
    assert cumulative_detection_probability([0.5, 0.5], [0.5, 0.5], [1.0, 2.0]) == pytest.approx(
        cumulative_detection_probability([0.5, 0.5], [0.5, 0.5], [1, 2]), rel=1e-15
    )


def test_conditioning_on_an_impossible_failure_is_rejected():
    with pytest.raises(ValueError):
        update_after_failed_search([1.0, 0.0], [1.0, 0.5], 0)
    with pytest.raises(ValueError):
        posterior_from_counts([1.0, 0.0], [1.0, 0.5], [1, 0])
    with pytest.raises(ValueError):
        posterior_after_search_sequence([0.5, 0.5], [1.0, 1.0], [0, 1])


def test_selecting_a_cell_when_no_search_can_succeed_is_rejected():
    with pytest.raises(ValueError):
        select_next_cell([0.5, 0.5], [0.0, 0.0])
    with pytest.raises(ValueError):
        select_next_cell([1.0, 0.0], [0.0, 0.9])


def test_greedy_plan_requires_a_stopping_rule_and_valid_limits():
    with pytest.raises(ValueError):
        greedy_search_plan([0.5, 0.5], [0.5, 0.5])
    with pytest.raises(ValueError):
        greedy_search_plan([0.5, 0.5], [0.5, 0.5], max_looks=-1)
    with pytest.raises(ValueError):
        greedy_search_plan([0.5, 0.5], [0.5, 0.5], max_looks=2.5)
    with pytest.raises(ValueError):
        greedy_search_plan([0.5, 0.5], [0.5, 0.5], budget=-1.0)
    with pytest.raises(ValueError):
        greedy_search_plan([0.5, 0.5], [0.5, 0.5], budget=math.nan)
    assert greedy_search_plan([0.5, 0.5], [0.5, 0.5], max_looks=0) == []
    assert greedy_search_plan([0.5, 0.5], [0.5, 0.5], budget=0.5) == []


def test_prior_with_rounding_error_is_accepted_but_a_real_deficit_is_rejected():
    assert cumulative_detection_probability([0.1] * 10, [0.5] * 10, [1] * 10) == pytest.approx(0.5, rel=1e-12)
    with pytest.raises(ValueError):
        cumulative_detection_probability([0.1] * 10 + [1e-6], [0.5] * 11, [1] * 11)


def test_single_cell_is_a_valid_degenerate_model():
    assert update_after_failed_search([1.0], [0.5], 0) == pytest.approx([1.0])
    assert cumulative_detection_probability([1.0], [0.5], [3]) == pytest.approx(0.875, rel=1e-14)


def test_tiny_detection_probability_keeps_full_precision():
    # 1 - (1 - 1e-18)**3 rounds to 0 in double precision; the stable form keeps ~3e-18.
    value = cumulative_detection_probability([1.0], [1e-18], [3])
    assert value == pytest.approx(3e-18, rel=1e-12)


def test_very_many_failures_do_not_underflow_the_posterior():
    posterior = posterior_from_counts([1.0, 0.0], [0.9, 0.9], [400, 0])  # (0.1)**400 underflows a double
    assert posterior == pytest.approx([1.0, 0.0])
    sequential = posterior_after_search_sequence([0.6, 0.4], [0.9, 0.0], [0] * 500)
    assert sequential == pytest.approx([0.0, 1.0], abs=1e-12)
    assert sequential.sum() == pytest.approx(1.0, abs=1e-15)
    assert posterior_from_counts([0.6, 0.4], [0.9, 0.0], [500, 0]) == pytest.approx(sequential, abs=1e-12)


def test_huge_look_counts_saturate_at_the_searched_prior_mass():
    assert cumulative_detection_probability([0.7, 0.3], [0.5, 0.5], [10**9, 0]) == pytest.approx(0.7, rel=1e-14)
    assert cumulative_detection_probability([0.7, 0.3], [0.5, 0.5], [10**9, 10**9]) == pytest.approx(1.0, rel=1e-14)


def test_cumulative_probability_never_exceeds_one():
    prior = [1.0 / 3.0] * 3
    value = cumulative_detection_probability(prior, [1.0, 1.0, 1.0], [5, 5, 5])
    assert 0.0 < value <= 1.0


def test_zero_prior_cells_contribute_nothing():
    assert cumulative_detection_probability([1.0, 0.0], [0.5, 1.0], [2, 10]) == pytest.approx(0.75, rel=1e-14)
    assert posterior_after_search_sequence([1.0, 0.0], [0.5, 1.0], [1, 1, 0]) == pytest.approx([1.0, 0.0])


def test_search_step_is_an_immutable_record():
    step = greedy_search_plan([0.6, 0.4], [0.5, 0.5], max_looks=1)[0]
    assert isinstance(step, SearchStep)
    with pytest.raises(AttributeError):
        step.cell = 3


# --- regressions from code review ----------------------------------------------------------------------


def test_survival_is_accurate_for_tiny_detection_probability_and_huge_counts():
    # (1 - 1e-18) rounds to 1 in double precision; exp(n log1p(-d)) = exp(-1) is the true value.
    assert probability_of_no_detection([1.0], [1e-18], [10**18]) == pytest.approx(math.exp(-1.0), rel=1e-12)
    assert cumulative_detection_probability([1.0], [1e-18], [10**18]) == pytest.approx(1.0 - math.exp(-1.0), rel=1e-12)
    assert probability_of_no_detection([1.0], [1e-18], [10**18]) + cumulative_detection_probability([1.0], [1e-18], [10**18]) == pytest.approx(1.0, rel=1e-12)
    assert unconditional_detection_increment([1.0], [1e-18], [10**18], 0) == pytest.approx(1e-18 * math.exp(-1.0), rel=1e-12)


def test_survival_handles_perfect_detection_and_zero_counts_without_warnings():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert probability_of_no_detection([0.5, 0.5], [1.0, 1.0], [0, 0]) == pytest.approx(1.0, rel=1e-15)
        assert probability_of_no_detection([0.5, 0.5], [1.0, 1.0], [1, 0]) == pytest.approx(0.5, rel=1e-15)
        assert probability_of_no_detection([0.5, 0.5], [1.0, 1.0], [1, 3]) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("looks", [[0] * 400 + [1] * 400, [1] * 400 + [0] * 400, [0, 1] * 400])
def test_long_sequences_with_equal_counts_give_the_same_posterior_in_any_order(looks):
    # Updating one search at a time in floating point used to underflow a cell to exactly 0 and lose it.
    assert posterior_after_search_sequence([0.5, 0.5], [0.9, 0.9], looks) == pytest.approx([0.5, 0.5], rel=1e-12)


def test_long_sequence_posterior_matches_the_closed_form():
    looks = [0] * 700 + [1] * 650 + [2] * 20
    counts = counts_from_sequence(looks, 3)
    expected = posterior_from_counts([0.2, 0.3, 0.5], [0.9, 0.8, 0.05], counts)
    assert posterior_after_search_sequence([0.2, 0.3, 0.5], [0.9, 0.8, 0.05], looks) == pytest.approx(expected, rel=1e-12)
    assert expected.sum() == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize(
    "counts",
    [np.array([2**63, 0], dtype=np.uint64), np.array([2**64 - 1, 0], dtype=np.uint64), [1e30, 0], [2.0**63, 0], [-1, 0]],
)
def test_counts_outside_the_signed_64_bit_range_are_rejected(counts):
    with pytest.raises(ValueError):
        cumulative_detection_probability([0.5, 0.5], [0.5, 0.5], counts)


@pytest.mark.parametrize("counts", [np.array([2**63 - 1, 0], dtype=np.uint64), [2**63 - 1, 0], np.array([2**63 - 1, 0], dtype=np.int64)])
def test_largest_supported_count_is_accepted(counts):
    assert cumulative_detection_probability([0.5, 0.5], [0.5, 0.5], counts) == pytest.approx(0.5, rel=1e-14)
