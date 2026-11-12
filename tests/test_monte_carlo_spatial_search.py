"""Tests for the Monte Carlo spatial-search engine.

Every stochastic test uses an explicit fixed seed, so each is deterministic for a pinned NumPy; no test
touches unseeded randomness. Statistical tolerances are chosen before looking at results, as multiples of
the theoretical sampling standard error (|z| <= 4.5 for a single comparison, a two-sided probability of
about 7e-6 if the engine is right), so they would also pass under any other valid random stream. A
Monte Carlo agreement is evidence, not proof: these tests compare the simulation with independently
computed analytical values and with a brute-force enumeration of the probability space.
"""

import ast
import itertools
import math
import time
from pathlib import Path

import numpy as np
import pytest

import optimal_search.monte_carlo_spatial_search as engine
from optimal_search.exact_spatial_search import expected_search_cost, solve_exact
from optimal_search.monte_carlo_spatial_search import (
    SimulationResult,
    proportion_confidence_interval,
    simulate_replications,
    simulate_search,
)
from optimal_search.spatial_search import greedy_search_plan
from tests.spatial_reference import enumerate_detection_probability, enumerate_expected_cost

Z_LIMIT = 4.5
TRIALS = 100_000

CASES = {
    "perfect detection": ([0.5, 0.3, 0.2], [1.0, 1.0, 1.0], [1.0, 2.0, 1.5], [0, 1]),
    "imperfect detection": ([0.4, 0.3, 0.2, 0.1], [0.5, 0.6, 0.7, 0.8], [1.0, 2.0, 1.0, 3.0], [0, 1, 0, 2, 3, 1]),
    "unequal everything": ([0.55, 0.25, 0.15, 0.05], [0.3, 0.9, 0.45, 0.7], [0.5, 2.5, 1.0, 4.0], [1, 0, 0, 2, 3, 0, 1]),
    "repeated looks, one cell": ([0.9, 0.1], [0.3, 0.3], [1.0, 1.0], [0, 0, 0, 0]),
    "single cell": ([1.0], [0.35], [2.0], [0, 0, 0]),
    "low probability": ([0.5, 0.5], [0.02, 0.03], [1.0, 1.0], [0, 1]),
}


def analytic_detection(prior, detection, plan):
    counts = [plan.count(j) for j in range(len(prior))]
    return 1.0 - sum(p * (1.0 - d) ** n for p, d, n in zip(prior, detection, counts))


def z_score(estimate, truth, standard_error):
    if standard_error == 0.0:
        return 0.0 if abs(estimate - truth) < 1e-9 else math.inf
    return (estimate - truth) / standard_error


def run(case, seed, trials=TRIALS, **kwargs):
    prior, detection, cost, plan = CASES[case]
    return simulate_search(prior, detection, cost, plan, trials, seed, **kwargs)


# --- agreement with analytical probabilities and costs -----------------------------------------------------


@pytest.mark.parametrize("case", sorted(CASES))
def test_empirical_detection_probability_matches_the_analytical_formula(case):
    prior, detection, _, plan = CASES[case]
    truth = analytic_detection(prior, detection, plan)
    result = run(case, seed=11)
    assert truth == pytest.approx(enumerate_detection_probability(prior, detection, plan), abs=1e-12)
    se = math.sqrt(truth * (1 - truth) / TRIALS)
    assert abs(z_score(result.detection_probability, truth, se)) < Z_LIMIT
    low, high = result.detection_confidence_interval(0.999)
    assert low <= truth <= high


@pytest.mark.parametrize("case", sorted(CASES))
def test_empirical_mean_cost_matches_the_independent_expected_cost(case):
    prior, detection, cost, plan = CASES[case]
    brute, _ = enumerate_expected_cost(prior, detection, cost, plan)
    solver = expected_search_cost(prior, detection, cost, plan)[0]
    assert brute == pytest.approx(solver, rel=1e-12)  # the two analytical references agree
    result = run(case, seed=12)
    assert abs(z_score(result.mean_cost, brute, result.mean_cost_se)) < Z_LIMIT


@pytest.mark.parametrize("case", sorted(CASES))
def test_empirical_mean_looks_matches_the_independent_expected_number_of_looks(case):
    prior, detection, _, plan = CASES[case]
    expected_looks, _ = enumerate_expected_cost(prior, detection, [1.0] * len(prior), plan)  # unit cost per look
    result = run(case, seed=13)
    assert abs(z_score(result.mean_looks, expected_looks, result.mean_looks_se)) < Z_LIMIT


@pytest.mark.parametrize("case", sorted(CASES))
def test_detection_outcomes_by_cell_match_the_analytical_per_cell_probabilities(case):
    prior, detection, _, plan = CASES[case]
    result = run(case, seed=14)
    for j, (p, d) in enumerate(zip(prior, detection)):
        truth = p * (1.0 - (1.0 - d) ** plan.count(j))
        se = math.sqrt(truth * (1 - truth) / TRIALS)
        assert abs(z_score(result.detections_by_cell[j] / TRIALS, truth, se)) < Z_LIMIT
    assert result.detections_by_cell.sum() == result.detections


@pytest.mark.parametrize("case", sorted(CASES))
def test_target_placement_follows_the_prior(case):
    prior = CASES[case][0]
    result = run(case, seed=15)
    for j, p in enumerate(prior):
        se = math.sqrt(p * (1 - p) / TRIALS)
        assert abs(z_score(result.target_counts[j] / TRIALS, p, se)) < Z_LIMIT
    assert result.target_counts.sum() == TRIALS


@pytest.mark.parametrize("case", ["imperfect detection", "unequal everything", "repeated looks, one cell"])
def test_first_detection_look_distribution_matches_the_unconditional_increments(case):
    prior, detection, _, plan = CASES[case]
    result = run(case, seed=16)
    seen = [0] * len(prior)
    for index, cell in enumerate(plan):
        truth = prior[cell] * (1.0 - detection[cell]) ** seen[cell] * detection[cell]
        seen[cell] += 1
        observed = np.count_nonzero(result.detection_look == index) / TRIALS
        assert abs(z_score(observed, truth, math.sqrt(truth * (1 - truth) / TRIALS))) < Z_LIMIT
    assert np.count_nonzero(result.detection_look >= 0) == result.detections


# --- greedy versus exact plans -----------------------------------------------------------------------------------------


def test_greedy_and_exact_plans_simulate_to_their_analytical_values_and_exact_wins():
    prior, detection, cost, budget = [0.52, 0.27, 0.21], [1.0, 1.0, 1.0], [2.0, 1.0, 1.0], 2.0
    greedy_plan = [step.cell for step in greedy_search_plan(prior, detection, cost, budget=budget)]
    exact_plan = list(solve_exact(prior, detection, cost, budget).sequence)
    assert greedy_plan == [1, 2] and exact_plan == [0]
    greedy = simulate_search(prior, detection, cost, greedy_plan, TRIALS, seed=21, budget=budget)
    exact = simulate_search(prior, detection, cost, exact_plan, TRIALS, seed=22, budget=budget)
    for result, truth in ((greedy, 0.48), (exact, 0.52)):
        assert abs(z_score(result.detection_probability, truth, math.sqrt(truth * (1 - truth) / TRIALS))) < Z_LIMIT
    # the 0.04 advantage is statistically unmistakable at this sample size (about 12 standard errors)
    difference = exact.detection_probability - greedy.detection_probability
    se_difference = math.sqrt(0.48 * 0.52 / TRIALS + 0.52 * 0.48 / TRIALS)
    assert difference / se_difference > 8.0
    # and the greedy plan has the lower expected cost (1.73 vs 2.00), also visible in simulation
    assert abs(z_score(greedy.mean_cost, 1.73, greedy.mean_cost_se)) < Z_LIMIT
    assert exact.mean_cost == pytest.approx(2.0)


def test_simulated_plans_from_both_solvers_respect_their_budget():
    prior, detection, cost = [0.4, 0.3, 0.2, 0.1], [0.5, 0.6, 0.7, 0.8], [1.5, 1.0, 2.0, 0.5]
    for budget in (0.0, 1.0, 3.5, 6.0):
        for plan in (
            [step.cell for step in greedy_search_plan(prior, detection, cost, budget=budget)] if budget > 0 else [],
            list(solve_exact(prior, detection, cost, budget).sequence),
        ):
            result = simulate_search(prior, detection, cost, plan, 5_000, seed=23, budget=budget)
            assert result.cost.max() <= budget + 1e-12


def test_the_test_has_the_power_to_detect_a_biased_analytical_value():
    # a deliberately wrong analytical probability (off by 0.01) is rejected at 100,000 trials
    prior, detection, _, plan = CASES["imperfect detection"]
    truth = analytic_detection(prior, detection, plan)
    result = run("imperfect detection", seed=31)
    se = math.sqrt(truth * (1 - truth) / TRIALS)
    assert abs(z_score(result.detection_probability, truth + 0.01, se)) > Z_LIMIT
    assert abs(z_score(result.detection_probability, truth - 0.01, se)) > Z_LIMIT


# --- perfect and zero detection: exact invariants (no statistics needed) ---------------------------------------------------


def test_perfect_detection_finds_the_target_exactly_when_its_cell_is_searched():
    result = run("perfect detection", seed=41)
    for j in range(3):
        if j in (0, 1):
            assert result.detections_by_cell[j] == result.target_counts[j]
        else:
            assert result.detections_by_cell[j] == 0
    assert result.detections == result.target_counts[0] + result.target_counts[1]


def test_zero_detection_never_detects():
    result = simulate_search([0.5, 0.5], [0.0, 0.0], [1.0, 2.0], [0, 1, 0, 1], 20_000, seed=42)
    assert result.detections == 0 and result.detection_probability == 0.0
    assert np.all(result.looks == 4)
    assert np.all(result.cost == 6.0)
    assert np.all(result.detection_cell == -1) and np.all(result.detection_look == -1)
    assert result.detection_confidence_interval()[0] == 0.0


def test_certain_detection_at_the_first_look():
    result = simulate_search([1.0, 0.0], [1.0, 1.0], [3.0, 1.0], [0, 1, 0], 10_000, seed=43)
    assert result.detections == 10_000
    assert np.all(result.looks == 1) and np.all(result.cost == 3.0) and np.all(result.detection_look == 0)
    assert result.detection_confidence_interval()[1] == 1.0


# --- early stopping and cost bookkeeping ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(CASES))
def test_search_stops_immediately_after_detection(case):
    prior, detection, cost, plan = CASES[case]
    result = run(case, seed=51, trials=20_000)
    detected = result.detected
    assert np.array_equal(result.looks[detected], result.detection_look[detected] + 1)
    assert np.all(result.looks[~detected] == len(plan))
    assert np.all(result.looks <= len(plan))
    plan_array = np.array(plan)
    assert np.array_equal(plan_array[result.detection_look[detected]], result.detection_cell[detected])
    assert np.array_equal(result.detection_cell[detected], result.target_cell[detected])  # only the right cell can succeed
    through = np.concatenate(([0.0], np.cumsum(np.array(cost)[plan_array])))
    assert np.array_equal(result.cost, through[result.looks])  # cost paid for every look performed, none after
    assert np.all(result.cost[~detected] == through[-1])


def test_costs_never_exceed_the_plan_total_or_the_budget():
    prior, detection, cost, plan = CASES["unequal everything"]
    total = math.fsum(cost[j] for j in plan)
    result = simulate_search(prior, detection, cost, plan, 20_000, seed=52, budget=total)
    assert result.cost.max() <= total + 1e-12
    assert result.cost.min() > 0.0
    with pytest.raises(ValueError):
        simulate_search(prior, detection, cost, plan, 10, seed=52, budget=total - 0.01)


def test_a_failed_look_still_costs_and_the_failure_cost_is_the_full_plan_cost():
    result = simulate_search([0.5, 0.5], [0.5, 0.5], [1.0, 2.0], [0, 1], 20_000, seed=53)
    assert set(np.unique(result.cost)) <= {1.0, 3.0}
    assert np.all(result.cost[~result.detected] == 3.0)
    assert np.all(result.cost[result.detected & (result.looks == 1)] == 1.0)


# --- zero budget and zero-look cases ----------------------------------------------------------------------------------------------


def test_empty_plan_gives_zero_looks_zero_cost_and_no_detection():
    result = simulate_search([0.6, 0.4], [0.9, 0.9], [1.0, 1.0], [], 1_000, seed=61)
    assert result.detections == 0
    assert np.all(result.looks == 0) and np.all(result.cost == 0.0)
    assert result.mean_looks == 0.0 and result.mean_cost == 0.0
    assert result.plan == ()
    assert result.target_counts.sum() == 1_000 and result.detections_by_cell.sum() == 0
    assert result.detection_confidence_interval(method="clopper-pearson")[0] == 0.0


def test_zero_budget_plans_from_both_solvers_simulate_to_nothing():
    prior, detection, cost = [0.5, 0.5], [0.9, 0.9], [1.0, 1.0]
    assert greedy_search_plan(prior, detection, cost, budget=0.0) == []
    exact_plan = list(solve_exact(prior, detection, cost, 0.0).sequence)
    assert exact_plan == []
    result = simulate_search(prior, detection, cost, exact_plan, 500, seed=62, budget=0.0)
    assert result.detections == 0 and result.cost.max() == 0.0


def test_a_single_trial_has_no_standard_error():
    result = simulate_search([0.5, 0.5], [0.5, 0.5], [1.0, 1.0], [0, 1], 1, seed=63)
    assert result.trials == 1
    assert math.isnan(result.mean_cost_se) and math.isnan(result.mean_looks_se)
    assert all(math.isnan(x) for x in result.mean_cost_confidence_interval())
    assert 0.0 <= result.detection_confidence_interval()[0] <= result.detection_confidence_interval()[1] <= 1.0


# --- reproducibility ----------------------------------------------------------------------------------------------------------------


def _same(a: SimulationResult, b: SimulationResult) -> bool:
    arrays = ("target_cell", "detected", "looks", "cost", "detection_cell", "detection_look", "target_counts", "detections_by_cell")
    return all(np.array_equal(getattr(a, name), getattr(b, name)) for name in arrays) and a.detection_probability == b.detection_probability


def test_the_same_seed_gives_identical_results_and_different_seeds_differ():
    assert _same(run("imperfect detection", seed=71, trials=5_000), run("imperfect detection", seed=71, trials=5_000))
    assert not _same(run("imperfect detection", seed=71, trials=5_000), run("imperfect detection", seed=72, trials=5_000))


def test_results_do_not_depend_on_how_trials_are_chunked_for_memory():
    reference = run("unequal everything", seed=73, trials=5_000)
    for chunk in (1, 7, 100, 10_000, 4_000_000):
        assert _same(reference, run("unequal everything", seed=73, trials=5_000, max_chunk_elements=chunk))


def test_seed_sequences_are_accepted_and_replications_are_reproducible_and_distinct():
    prior, detection, cost, plan = CASES["imperfect detection"]
    first = simulate_replications(prior, detection, cost, plan, 2_000, 5, seed=74)
    second = simulate_replications(prior, detection, cost, plan, 2_000, 5, seed=74)
    assert all(_same(a, b) for a, b in zip(first, second))
    assert len({r.detections for r in first}) > 1  # the replications use different streams
    sequence = np.random.SeedSequence(75)
    again = simulate_replications(prior, detection, cost, plan, 2_000, 3, seed=sequence)
    once_more = simulate_replications(prior, detection, cost, plan, 2_000, 3, seed=sequence)  # the parent is not advanced
    assert all(_same(a, b) for a, b in zip(again, once_more))
    direct = simulate_search(prior, detection, cost, plan, 2_000, seed=np.random.SeedSequence(75))
    assert _same(direct, simulate_search(prior, detection, cost, plan, 2_000, seed=np.random.SeedSequence(75)))


def test_unseeded_or_invalid_seeds_are_rejected():
    for bad in (None, -1, 1.5, True, "7", [1]):
        with pytest.raises(ValueError):
            simulate_search([1.0], [0.5], [1.0], [0], 10, seed=bad)


# --- calibration across seeds --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [2024, 77, 31337])
def test_confidence_intervals_are_calibrated_across_seeds(seed):
    # 300 independent replications of 1,000 trials each. Coverage of a nominal 95% interval has standard
    # deviation sqrt(0.95 * 0.05 / 300) = 0.0126, so [0.90, 0.99] is about -4 to +3 standard deviations;
    # the z-scores of the estimate should have mean ~0 (standard error 0.058) and standard deviation ~1.
    prior, detection, cost, plan = CASES["imperfect detection"]
    truth = analytic_detection(prior, detection, plan)
    true_cost = enumerate_expected_cost(prior, detection, cost, plan)[0]
    replications = simulate_replications(prior, detection, cost, plan, 1_000, 300, seed)
    wilson = np.mean([r.detection_confidence_interval()[0] <= truth <= r.detection_confidence_interval()[1] for r in replications])
    exact = np.mean(
        [r.detection_confidence_interval(method="clopper-pearson")[0] <= truth <= r.detection_confidence_interval(method="clopper-pearson")[1] for r in replications]
    )
    mean_cost = np.mean([r.mean_cost_confidence_interval()[0] <= true_cost <= r.mean_cost_confidence_interval()[1] for r in replications])
    z = np.array([(r.detection_probability - truth) / math.sqrt(truth * (1 - truth) / 1_000) for r in replications])
    assert 0.90 <= wilson <= 0.99
    assert 0.92 <= exact <= 1.0  # Clopper-Pearson is conservative
    assert 0.89 <= mean_cost <= 0.99  # the normal approximation is slightly looser for a skewed variable
    assert abs(z.mean()) < 0.35
    assert 0.85 < z.std(ddof=1) < 1.15


def test_clopper_pearson_covers_at_least_as_often_as_wilson_for_a_small_probability():
    prior, detection, cost, plan = CASES["low probability"]
    truth = analytic_detection(prior, detection, plan)
    replications = simulate_replications(prior, detection, cost, plan, 500, 300, seed=9)
    hits = lambda method: np.mean(  # noqa: E731
        [r.detection_confidence_interval(method=method)[0] <= truth <= r.detection_confidence_interval(method=method)[1] for r in replications]
    )
    assert hits("clopper-pearson") >= 0.93
    assert hits("wilson") >= 0.90


# --- confidence-interval formulas ----------------------------------------------------------------------------------------------------


def test_wilson_interval_known_values():
    z = 1.959963984540054
    low, high = proportion_confidence_interval(0, 10)
    assert low == 0.0
    assert high == pytest.approx(z * z / (10 + z * z), rel=1e-12)  # closed form for k = 0
    low, high = proportion_confidence_interval(5, 10)
    assert (low, high) == pytest.approx((0.2366, 0.7634), abs=5e-5)
    assert proportion_confidence_interval(10, 10)[1] == 1.0


def test_clopper_pearson_interval_known_values():
    assert proportion_confidence_interval(0, 10, method="clopper-pearson")[1] == pytest.approx(1 - 0.025 ** (1 / 10), rel=1e-12)  # closed form, k = 0
    assert proportion_confidence_interval(10, 10, method="clopper-pearson")[0] == pytest.approx(0.025 ** (1 / 10), rel=1e-12)
    assert proportion_confidence_interval(5, 10, method="clopper-pearson") == pytest.approx((0.1871, 0.8129), abs=5e-5)
    assert proportion_confidence_interval(0, 10, method="clopper-pearson")[0] == 0.0
    assert proportion_confidence_interval(10, 10, method="clopper-pearson")[1] == 1.0


@pytest.mark.parametrize("method", ["wilson", "clopper-pearson"])
def test_intervals_contain_the_estimate_widen_with_level_and_narrow_with_trials(method):
    for k, n in [(0, 20), (3, 20), (10, 20), (20, 20), (700, 1000)]:
        low, high = proportion_confidence_interval(k, n, 0.95, method)
        assert 0.0 <= low <= k / n <= high <= 1.0
        low90, high90 = proportion_confidence_interval(k, n, 0.90, method)
        low99, high99 = proportion_confidence_interval(k, n, 0.99, method)
        assert low99 <= low <= low90 and high90 <= high <= high99
    small = proportion_confidence_interval(70, 100, 0.95, method)
    large = proportion_confidence_interval(7000, 10000, 0.95, method)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_clopper_pearson_is_never_narrower_than_wilson_here():
    for k, n in [(0, 30), (5, 30), (15, 30), (30, 30), (200, 1000)]:
        wilson = proportion_confidence_interval(k, n)
        exact = proportion_confidence_interval(k, n, method="clopper-pearson")
        assert exact[0] <= wilson[0] + 1e-12 and exact[1] >= wilson[1] - 1e-12


@pytest.mark.parametrize(
    "args",
    [(-1, 10), (11, 10), (0, 0), (1.5, 10), (True, 10), (5, 10.5), (5, -3)],
)
def test_invalid_proportion_arguments_are_rejected(args):
    with pytest.raises(ValueError):
        proportion_confidence_interval(*args)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5, math.nan, True, "0.95", None])
def test_invalid_confidence_levels_are_rejected(level):
    with pytest.raises(ValueError):
        proportion_confidence_interval(5, 10, level)
    result = simulate_search([0.5, 0.5], [0.5, 0.5], [1.0, 1.0], [0, 1], 100, seed=1)
    with pytest.raises(ValueError):
        result.mean_cost_confidence_interval(level)


def test_unknown_interval_method_is_rejected():
    with pytest.raises(ValueError):
        proportion_confidence_interval(5, 10, method="normal")


# --- invalid inputs and numerical edge cases ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("prior", [[], [0.5, 0.4], [0.5, 0.6], [-0.1, 1.1], [math.nan, 1.0], [math.inf, 0.0], "ab", [[0.5, 0.5]], None, ["a", "b"]])
def test_invalid_priors_are_rejected(prior):
    with pytest.raises(ValueError):
        simulate_search(prior, [0.5, 0.5], [1.0, 1.0], [0], 10, seed=1)


@pytest.mark.parametrize("detection", [[1.1, 0.5], [-0.1, 0.5], [math.nan, 0.5], [0.5], [0.5, 0.5, 0.5]])
def test_invalid_detection_probabilities_are_rejected(detection):
    with pytest.raises(ValueError):
        simulate_search([0.5, 0.5], detection, [1.0, 1.0], [0], 10, seed=1)


@pytest.mark.parametrize("cost", [[0.0, 1.0], [-1.0, 1.0], [math.nan, 1.0], [math.inf, 1.0], [1.0], [1.0, 1.0, 1.0]])
def test_invalid_costs_are_rejected(cost):
    with pytest.raises(ValueError):
        simulate_search([0.5, 0.5], [0.5, 0.5], cost, [0], 10, seed=1)


@pytest.mark.parametrize("plan", [[2], [-1], [0.5], [True], ["a"], [None], 5])
def test_invalid_plans_are_rejected(plan):
    with pytest.raises(ValueError):
        simulate_search([0.5, 0.5], [0.5, 0.5], [1.0, 1.0], plan, 10, seed=1)


@pytest.mark.parametrize("trials", [0, -5, 2.5, True, None, "10"])
def test_invalid_trial_counts_are_rejected(trials):
    with pytest.raises(ValueError):
        simulate_search([0.5, 0.5], [0.5, 0.5], [1.0, 1.0], [0], trials, seed=1)


@pytest.mark.parametrize("budget", [-1.0, math.nan, math.inf, "a", True])
def test_invalid_budgets_are_rejected(budget):
    with pytest.raises(ValueError):
        simulate_search([0.5, 0.5], [0.5, 0.5], [1.0, 1.0], [0], 10, seed=1, budget=budget)


@pytest.mark.parametrize("chunk", [0, -1, 2.5, True, None])
def test_invalid_chunk_sizes_are_rejected(chunk):
    with pytest.raises(ValueError):
        simulate_search([0.5, 0.5], [0.5, 0.5], [1.0, 1.0], [0], 10, seed=1, max_chunk_elements=chunk)


def test_invalid_replication_counts_are_rejected():
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            simulate_replications([1.0], [0.5], [1.0], [0], 10, bad, seed=1)


def test_cells_with_zero_prior_never_hold_the_target():
    result = simulate_search([0.0, 1.0, 0.0], [1.0, 0.5, 1.0], [1.0, 1.0, 1.0], [0, 1, 2, 1], 20_000, seed=81)
    assert result.target_counts[0] == 0 and result.target_counts[2] == 0
    assert result.target_counts[1] == 20_000
    assert np.all(result.detection_cell[result.detected] == 1)


def test_a_cell_with_a_tiny_prior_is_handled():
    result = simulate_search([1e-12, 1.0 - 1e-12], [1.0, 0.5], [1.0, 1.0], [0, 1], 10_000, seed=82)
    assert result.target_counts.sum() == 10_000


def test_a_tiny_detection_probability_gives_essentially_no_detections():
    result = simulate_search([1.0], [1e-9], [1.0], [0] * 5, 50_000, seed=83)
    assert result.detections <= 1
    assert np.all(result.looks[~result.detected] == 5)


def test_a_long_plan_is_chunked_without_changing_the_result():
    plan = [0, 1] * 500
    one = simulate_search([0.5, 0.5], [0.001, 0.001], [1.0, 1.0], plan, 3_000, seed=84)
    chunked = simulate_search([0.5, 0.5], [0.001, 0.001], [1.0, 1.0], plan, 3_000, seed=84, max_chunk_elements=2_500)
    assert _same(one, chunked)
    truth = 1.0 - 0.999**500
    assert abs(z_score(one.detection_probability, truth, math.sqrt(truth * (1 - truth) / 3_000))) < Z_LIMIT


def test_prior_rounding_error_is_accepted():
    result = simulate_search([0.1] * 10, [0.5] * 10, [1.0] * 10, list(range(10)), 1_000, seed=85)
    assert result.target_counts.sum() == 1_000


def test_float_and_numpy_inputs_are_accepted():
    result = simulate_search(np.array([0.5, 0.5]), np.array([0.5, 0.5]), np.array([1.0, 2.0]), np.array([0, 1]), np.int64(500), seed=np.int64(86), budget=np.float64(3.0))
    assert result.trials == 500


# --- performance ----------------------------------------------------------------------------------------------------------------------


def test_one_hundred_thousand_trials_run_quickly():
    prior, detection, cost, plan = CASES["imperfect detection"]
    started = time.perf_counter()
    result = simulate_search(prior, detection, cost, plan * 4, 100_000, seed=91)
    assert result.trials == 100_000
    assert time.perf_counter() - started < 10.0  # generous: typically a few hundredths of a second


# --- separation from the mathematical solvers -----------------------------------------------------------------------------------------


def test_the_engine_does_not_import_the_solvers_or_their_formulas():
    imported = set()
    for node in ast.walk(ast.parse(Path(engine.__file__).read_text())):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("spatial_search" in name.replace("monte_carlo_spatial_search", "") for name in imported)
    assert not any(name.startswith("optimal_search") for name in imported)


def test_the_engine_contains_no_power_operator_so_it_cannot_use_the_analytical_survival_formula():
    # the analytical probability of a miss is (1 - d)**n; the engine simulates each attempt instead
    tree = ast.parse(Path(engine.__file__).read_text())
    assert not any(isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow) for node in ast.walk(tree))


# --- regressions from code review ------------------------------------------------------------------------------------------------------


def _peak_memory(limit, plan_length):
    import tracemalloc

    tracemalloc.start()
    simulate_search([0.5, 0.5], [1e-6, 1e-6], [1.0, 1.0], [0, 1] * (plan_length // 2), 2, seed=1, max_chunk_elements=limit)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def test_a_lower_chunk_limit_reduces_memory_even_when_one_plan_exceeds_it():
    # With a 1,000,000-look plan, a limit below the plan length used to change nothing: each trial still
    # drew (and compared) all 1,000,000 uniforms at once, about 8 MB plus temporaries.
    length = 1_000_000
    unrestricted = _peak_memory(length, length)
    restricted = _peak_memory(1_000, length)
    assert unrestricted - restricted > 0.5 * 8 * length  # at least half of the uniform array is no longer allocated


@pytest.mark.parametrize("limit", [1, 3, 5, 119, 120, 121, 10_000])
def test_results_are_identical_for_any_chunk_limit_including_limits_below_the_plan_length(limit):
    prior, detection, cost = [0.5, 0.3, 0.2], [0.02, 0.03, 0.05], [1.0, 2.0, 0.5]
    plan = [0, 1, 2] * 40  # 120 looks
    reference = simulate_search(prior, detection, cost, plan, 400, seed=8)
    assert reference.detections > 0 and len(set(reference.detection_look[reference.detected])) > 10  # varied indices
    assert _same(reference, simulate_search(prior, detection, cost, plan, 400, seed=8, max_chunk_elements=limit))


@pytest.mark.parametrize("limit", [1, 7, 49, 50, 51, 56])
def test_a_detection_far_into_a_long_plan_is_found_across_block_boundaries(limit):
    # the target is certainly in cell 0 and detection is perfect; the first look in cell 0 is look 50
    plan = [1] * 50 + [0] + [0] * 5
    result = simulate_search([1.0, 0.0], [1.0, 1.0], [1.0, 2.0], plan, 25, seed=3, max_chunk_elements=limit)
    assert np.all(result.detected) and np.all(result.detection_look == 50) and np.all(result.looks == 51)
    assert np.all(result.cost == 50 * 2.0 + 1.0)
