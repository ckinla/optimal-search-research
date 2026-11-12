"""Comparison of the spatial-search implementation with equations in two primary sources.

Sources (copies held by the authors; see docs/provenance.md for public availability):

* D. J. Matthiesen, "Optimal Distribution of Radar Search Effort", 12 November 1999
  (cited below as "1999").
* D. J. Matthiesen, "Optimal Search", IEEE International Symposium on Phased Array
  Systems and Technology, 14-17 October 2003, pp. 259-264 (cited below as "2003").
* D. J. Matthiesen, "Efficient Beam Scanning, Energy Allocation, and Time Allocation for
  Search and Detection", IEEE Boston Section, 22 March 2011 (cited below as "2011"):
  Search Examples 1 and 2.

The equations were transcribed by reading the vector equation objects embedded in the
documents, not from rendered pages, so each transcription should be checked against
the printed equation. Equation numbers are the documents' own. The multi-term general-N
forms (1999 Eq. 35, 2003 Eq. 15) are encoded through the papers' own worked examples
(1999 Eq. 28 and 31, 2003 Eq. 19 and 20), which are unambiguous.

What is tested is mathematics: each transcribed expression is evaluated and compared with
the implementation and with an independent brute-force enumeration of the full
probability space. Matching or missing a number is not treated as evidence about
anyone's intent.
"""

import itertools
import math

import numpy as np
import pytest

from optimal_search.spatial_search import (
    conditional_next_look_success,
    counts_from_sequence,
    cumulative_detection_probability,
    cumulative_detection_probability_for_sequence,
    next_look_priority,
    posterior_after_search_sequence,
    greedy_search_plan,
    posterior_from_counts,
    select_next_cell,
    unconditional_detection_increment,
    update_after_failed_search,
)
from tests.spatial_reference import enumerate_detection_probability

PRIOR = [0.6, 0.4]  # P_k, P_l
DETECTION = [0.5, 0.5]  # Pd_k, Pd_l


# --- equations as transcribed ------------------------------------------------------------------


def paper_naive_cumulative(prior, detection, counts):
    """2003 Eq. 10 / 22 and 1999 Eq. 34: sum_j P_j Pd_j sum_{i=1..N_j} (1 - Pd_j)^(i-1), over cells with N_j != 0."""
    total = 0.0
    for p, d, n in zip(prior, detection, counts):
        if n:
            total += p * d * sum((1.0 - d) ** (i - 1) for i in range(1, n + 1))
    return total


def paper_posterior(prior, detection, counts):
    """2003 Eq. 8 / 1999 Eq. 32: P_j (1 - Pd_j)^N_j / (1 - naive cumulative probability)."""
    denominator = 1.0 - paper_naive_cumulative(prior, detection, counts)
    return [p * (1.0 - d) ** n / denominator for p, d, n in zip(prior, detection, counts)]


def paper_cumulative_two_looks(p_k, d_k, p_l, d_l):
    """1999 Eq. 28 / 2003 Eq. 19: P_c2 = P_k Pd_k + P_l Pd_l / (1 - P_k Pd_k)  (looks in cells k then l)."""
    return p_k * d_k + p_l * d_l / (1.0 - p_k * d_k)


def paper_cumulative_three_looks_kl_k(p_k, d_k, p_l, d_l):
    """2003 Eq. 20 (1999 Eq. 31): looks in cells k, l, k.

    P_c3 = P_k Pd_k + P_l Pd_l / (1 - P_k Pd_k) + P_k (1 - Pd_k) Pd_k / (1 - P_k Pd_k - P_l Pd_l)
    """
    return p_k * d_k + p_l * d_l / (1.0 - p_k * d_k) + p_k * (1.0 - d_k) * d_k / (1.0 - p_k * d_k - p_l * d_l)


def paper_increment(prior, detection, counts_before, cell):
    """2003 Eq. 23: Delta P_c,N = P_j (1 - Pd_j)^(N_j - 1) Pd_j / (1 - naive cumulative after N-1 looks).

    Here ``counts_before[cell]`` is N_j - 1.
    """
    naive_before = paper_naive_cumulative(prior, detection, counts_before)
    return prior[cell] * (1.0 - detection[cell]) ** counts_before[cell] * detection[cell] / (1.0 - naive_before)


# --- parts that agree --------------------------------------------------------------------------------------


@pytest.mark.parametrize("looks", [[0], [1, 0], [0, 1, 0], [0, 0, 1, 1, 0]])
def test_paper_posterior_equals_implementation(looks):
    counts = counts_from_sequence(looks, 2)
    expected = paper_posterior(PRIOR, DETECTION, counts)
    assert posterior_after_search_sequence(PRIOR, DETECTION, looks) == pytest.approx(expected, rel=1e-13)
    assert posterior_from_counts(PRIOR, DETECTION, counts) == pytest.approx(expected, rel=1e-13)


def test_paper_posterior_uses_the_unconditional_probability_of_failure_as_its_normaliser():
    counts = [2, 1]
    assert 1.0 - paper_naive_cumulative(PRIOR, DETECTION, counts) == pytest.approx(
        sum(p * (1 - d) ** n for p, d, n in zip(PRIOR, DETECTION, counts)), rel=1e-14
    )


@pytest.mark.parametrize("looks", [[0], [0, 1], [0, 1, 0], [1, 0, 0, 1], [0, 0, 0]])
def test_paper_naive_cumulative_is_exactly_the_unconditional_detection_probability(looks):
    """2003 Eq. 10 / 22 ("naive") equals the true probability of detecting within the fixed searches."""
    counts = counts_from_sequence(looks, 2)
    naive = paper_naive_cumulative(PRIOR, DETECTION, counts)
    assert naive == pytest.approx(cumulative_detection_probability(PRIOR, DETECTION, counts), rel=1e-13)
    assert naive == pytest.approx(enumerate_detection_probability(PRIOR, DETECTION, looks), rel=1e-13)


def test_paper_geometric_series_simplification_is_correct():
    """2003 Eq. 21-22: Pd sum_{i=1..N} (1 - Pd)^(i-1) = 1 - (1 - Pd)^N."""
    for d in (0.1, 0.5, 0.9):
        for n in (1, 2, 7):
            assert d * sum((1 - d) ** (i - 1) for i in range(1, n + 1)) == pytest.approx(1 - (1 - d) ** n, rel=1e-13)


def test_paper_eq_23_is_the_conditional_next_look_success_probability():
    """2003 Eq. 23 describes the probability of detection on the next look given the updated cell probability."""
    looks_before, cell = [0, 1], 0
    counts_before = counts_from_sequence(looks_before, 2)
    posterior = posterior_after_search_sequence(PRIOR, DETECTION, looks_before)
    assert paper_increment(PRIOR, DETECTION, counts_before, cell) == pytest.approx(
        conditional_next_look_success(posterior, DETECTION, cell), rel=1e-13
    )


@pytest.mark.parametrize("looks_before", [[], [0], [1, 0], [0, 1, 0]])
def test_paper_greedy_criterion_ranks_cells_like_the_implemented_priority(looks_before):
    """2003 Eq. 31 / 1999 Eq. 40: P_j (1 - Pd_j)^N_j Pd_j / cost, proportional to posterior * Pd / cost."""
    cost = [2.0, 1.0]
    counts = counts_from_sequence(looks_before, 2)
    eq31 = np.array([p * (1 - d) ** n * d / c for p, d, n, c in zip(PRIOR, DETECTION, counts, cost)])
    priority = next_look_priority(posterior_from_counts(PRIOR, DETECTION, counts), DETECTION, cost)
    factor = 1.0 - paper_naive_cumulative(PRIOR, DETECTION, counts)
    assert eq31 == pytest.approx(factor * priority, rel=1e-13)
    assert int(np.argmax(eq31)) == int(np.argmax(priority))


# --- the discrepancy: "cumulative probability of detection" (1999 Eq. 28, 31, 35; 2003 Eq. 15, 19, 20) -------


def test_paper_cumulative_is_the_running_sum_of_conditional_probabilities():
    """The papers' P_c (Eq. 19/20, 28/31) is sum_n [posterior_n * Pd]: the conditional success of each look."""
    p_k, d_k = PRIOR[0], DETECTION[0]
    p_l, d_l = PRIOR[1], DETECTION[1]
    conditional = []
    for i, cell in enumerate([0, 1, 0]):
        posterior = posterior_after_search_sequence(PRIOR, DETECTION, [0, 1, 0][:i])
        conditional.append(conditional_next_look_success(posterior, DETECTION, cell))
    assert sum(conditional[:2]) == pytest.approx(paper_cumulative_two_looks(p_k, d_k, p_l, d_l), rel=1e-13)
    assert sum(conditional) == pytest.approx(paper_cumulative_three_looks_kl_k(p_k, d_k, p_l, d_l), rel=1e-13)


def test_paper_two_look_cumulative_differs_from_the_exact_probability():
    exact = enumerate_detection_probability(PRIOR, DETECTION, [0, 1])
    paper = paper_cumulative_two_looks(PRIOR[0], DETECTION[0], PRIOR[1], DETECTION[1])
    assert exact == pytest.approx(0.5, rel=1e-14)
    assert paper == pytest.approx(0.3 + 0.2 / 0.7, rel=1e-14)
    assert paper > exact


def test_paper_three_look_cumulative_differs_from_the_exact_probability():
    exact = enumerate_detection_probability(PRIOR, DETECTION, [0, 1, 0])
    paper = paper_cumulative_three_looks_kl_k(PRIOR[0], DETECTION[0], PRIOR[1], DETECTION[1])
    assert exact == pytest.approx(0.65, rel=1e-14)
    assert paper == pytest.approx(0.3 + 0.2 / 0.7 + 0.15 / 0.5, rel=1e-14)  # about 0.8857
    assert paper > exact


def test_law_of_total_probability_gives_the_naive_value_not_the_paper_value():
    """Pr(det by 2) = Pr(det 1) + Pr(miss 1) * Pr(det 2 | miss 1); the papers omit the factor Pr(miss 1)."""
    p_k, d_k, p_l, d_l = PRIOR[0], DETECTION[0], PRIOR[1], DETECTION[1]
    p_miss_1 = 1.0 - p_k * d_k
    conditional_second = p_l * d_l / p_miss_1
    assert p_k * d_k + p_miss_1 * conditional_second == pytest.approx(p_k * d_k + p_l * d_l, rel=1e-14)
    assert p_k * d_k + conditional_second != pytest.approx(p_k * d_k + p_l * d_l, rel=1e-3)


def test_paper_cumulative_depends_on_search_order_while_the_exact_probability_does_not():
    forward = paper_cumulative_two_looks(PRIOR[0], DETECTION[0], PRIOR[1], DETECTION[1])
    reverse = paper_cumulative_two_looks(PRIOR[1], DETECTION[1], PRIOR[0], DETECTION[0])
    assert forward == pytest.approx(0.3 + 0.2 / 0.7, rel=1e-14)
    assert reverse == pytest.approx(0.2 + 0.3 / 0.8, rel=1e-14)
    assert forward != pytest.approx(reverse, rel=1e-4)
    assert cumulative_detection_probability_for_sequence(PRIOR, DETECTION, [0, 1]) == pytest.approx(
        cumulative_detection_probability_for_sequence(PRIOR, DETECTION, [1, 0]), rel=1e-14
    )


def test_paper_cumulative_can_exceed_one():
    """Two equally likely cells, perfect detection: certain detection after one look in each cell (P = 1)."""
    paper = paper_cumulative_two_looks(0.5, 1.0, 0.5, 1.0)
    assert paper == pytest.approx(1.5, rel=1e-14)
    assert enumerate_detection_probability([0.5, 0.5], [1.0, 1.0], [0, 1]) == pytest.approx(1.0, abs=1e-15)


def test_exact_probability_exceeds_neither_one_nor_the_paper_value():
    for p_k, d_k, d_l in itertools.product((0.2, 0.5, 0.8), (0.3, 0.7, 1.0), (0.3, 0.7, 1.0)):
        p_l = 1.0 - p_k
        exact = enumerate_detection_probability([p_k, p_l], [d_k, d_l], [0, 1])
        assert exact <= 1.0 + 1e-15
        assert paper_cumulative_two_looks(p_k, d_k, p_l, d_l) >= exact - 1e-15


def test_claim_that_naive_is_a_strict_lower_bound_does_not_hold_for_exact_probability():
    """1999 says the naive value is "always less than" the true cumulative probability that uses updated
    probabilities. The exact probability equals the naive value: Bayesian updating conditions on failure
    and cannot change the unconditional probability of detecting within a fixed sequence of looks."""
    for looks in ([0, 1], [0, 1, 0], [1, 1, 0, 0]):
        counts = counts_from_sequence(looks, 2)
        naive = paper_naive_cumulative(PRIOR, DETECTION, counts)
        exact = enumerate_detection_probability(PRIOR, DETECTION, looks)
        assert exact == pytest.approx(naive, rel=1e-13)


def test_paper_increment_summed_over_looks_is_not_the_increase_in_detection_probability():
    """2003 Eq. 23 is conditional; the unconditional increment carries the extra factor (1 - naive_before)."""
    looks = [0, 1, 0]
    counts = np.zeros(2, dtype=int)
    for cell in looks:
        conditional = paper_increment(PRIOR, DETECTION, counts, cell)
        unconditional = unconditional_detection_increment(PRIOR, DETECTION, counts, cell)
        assert unconditional == pytest.approx((1.0 - paper_naive_cumulative(PRIOR, DETECTION, counts)) * conditional, rel=1e-13)
        counts[cell] += 1
    assert cumulative_detection_probability(PRIOR, DETECTION, counts) == pytest.approx(0.65, rel=1e-14)
    assert math.isclose(
        sum(paper_increment(PRIOR, DETECTION, counts_from_sequence(looks[:i], 2), looks[i]) for i in range(3)),
        paper_cumulative_three_looks_kl_k(PRIOR[0], DETECTION[0], PRIOR[1], DETECTION[1]),
        rel_tol=1e-13,
    )


# --- 2011 presentation: correctly weighted worked example (Search Example 1) ---------------------------


def deck_2011_weighted_cumulative(prior, detection, looks):
    """2011 Search Example 1: Pd,cum = Pd1 P1 + (1 - Pd1 P1)(Pd2 P2) + (1 - Pd1 P1)(1 - Pd2 P2)(Pd3 P3) + ...

    P_m are the *updated* probabilities at look m, so each term is the conditional success probability
    weighted by the probability that all earlier looks failed.
    """
    posterior = np.array(prior, dtype=float)
    survive, total = 1.0, 0.0
    for i, cell in enumerate(looks):
        conditional = posterior[cell] * detection[cell]
        total += survive * conditional
        survive *= 1.0 - conditional
        if i + 1 < len(looks):
            posterior = update_after_failed_search(posterior, detection, cell)
    return total


def test_2011_search_example_1_values_are_reproduced():
    """Four equally likely cells, Pd = 1: Pd,cum = 0.25, 0.50, 0.75, 1.00 and updated cell probabilities 1/3, 1/2, 1."""
    prior, detection = [0.25] * 4, [1.0] * 4
    looks = [0, 1, 2, 3]  # the slides look in cells 1, 2, 3, 4
    assert [deck_2011_weighted_cumulative(prior, detection, looks[: n + 1]) for n in range(4)] == pytest.approx([0.25, 0.5, 0.75, 1.0], rel=1e-14)
    assert [cumulative_detection_probability_for_sequence(prior, detection, looks[: n + 1]) for n in range(4)] == pytest.approx([0.25, 0.5, 0.75, 1.0], rel=1e-14)
    assert posterior_after_search_sequence(prior, detection, [0]) == pytest.approx([0.0, 1 / 3, 1 / 3, 1 / 3], rel=1e-14)
    assert posterior_after_search_sequence(prior, detection, [0, 1]) == pytest.approx([0.0, 0.0, 0.5, 0.5], rel=1e-14)
    assert posterior_after_search_sequence(prior, detection, [0, 1, 2]) == pytest.approx([0.0, 0.0, 0.0, 1.0], rel=1e-14)
    plan = greedy_search_plan(prior, detection, max_looks=4)
    assert [step.conditional_success for step in plan] == pytest.approx([0.25, 1 / 3, 0.5, 1.0], rel=1e-14)
    assert [step.cumulative_probability for step in plan] == pytest.approx([0.25, 0.5, 0.75, 1.0], rel=1e-14)


def test_2011_search_example_1_differs_from_the_1999_and_2003_expression_on_the_same_inputs():
    """After two looks the 2011 weighting gives 0.5 (the exact value); the 1999 Eq. 28 / 2003 Eq. 19 pattern gives 7/12."""
    prior, detection = [0.25] * 4, [1.0] * 4
    assert deck_2011_weighted_cumulative(prior, detection, [0, 1]) == pytest.approx(0.5, rel=1e-14)
    assert paper_cumulative_two_looks(0.25, 1.0, 0.25, 1.0) == pytest.approx(0.25 + 0.25 / 0.75, rel=1e-14)
    assert paper_cumulative_two_looks(0.25, 1.0, 0.25, 1.0) == pytest.approx(7.0 / 12.0, rel=1e-14)


def test_2011_weighting_is_the_law_of_total_probability_and_matches_exact_enumeration():
    for prior, detection, looks in [
        ([0.6, 0.4], [0.5, 0.5], [0, 1, 0]),
        ([0.5, 0.5], [1.0, 1.0], [0, 1]),
        ([0.2, 0.3, 0.5], [0.9, 0.4, 0.7], [2, 0, 1, 2, 2]),
    ]:
        assert deck_2011_weighted_cumulative(prior, detection, looks) == pytest.approx(enumerate_detection_probability(prior, detection, looks), rel=1e-13)


def test_two_cell_perfect_detection_counterexample_2011_weighting_versus_1999_2003_expression():
    """Two equally likely cells, Pd = 1, one look in each: exact 1.0; 2011 weighting 1.0; 1999 Eq. 28 / 2003 Eq. 19 give 1.5."""
    prior, detection, looks = [0.5, 0.5], [1.0, 1.0], [0, 1]
    assert enumerate_detection_probability(prior, detection, looks) == pytest.approx(1.0, abs=1e-15)
    assert deck_2011_weighted_cumulative(prior, detection, looks) == pytest.approx(1.0, abs=1e-15)  # 1/2 + (1 - 1/2)(1)
    assert paper_cumulative_two_looks(0.5, 1.0, 0.5, 1.0) == pytest.approx(1.5, rel=1e-14)  # 1/2 + (1/2)/(1 - 1/2)


def test_2011_search_example_2_first_look_and_posterior_are_reproduced():
    """Unequal cells (0.1, 0.2, 0.4, 0.3), Pd = 0.6: look in cell 3 first, Pd,cum = 0.24, normaliser 0.76."""
    prior, detection = [0.1, 0.2, 0.4, 0.3], [0.6] * 4
    assert select_next_cell(prior, detection) == 2
    plan = greedy_search_plan(prior, detection, max_looks=1)
    assert plan[0].cell == 2
    assert plan[0].cumulative_probability == pytest.approx(0.24, rel=1e-14)
    assert 1.0 - prior[2] * detection[2] == pytest.approx(0.76, rel=1e-14)
    posterior = update_after_failed_search(prior, detection, 2)
    assert posterior == pytest.approx([0.1 / 0.76, 0.2 / 0.76, 0.16 / 0.76, 0.3 / 0.76], rel=1e-14)
    assert posterior.round(5).tolist() == [0.13158, 0.26316, 0.21053, 0.39474]
