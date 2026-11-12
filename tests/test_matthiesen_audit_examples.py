"""Independent arithmetic and exhaustive sequence checks for the audit examples."""

import itertools
import math
from decimal import Decimal, localcontext

import pytest

from optimal_search.cumulative_detection import cumulative_stationary_snrs
from scripts.matthiesen_audit_examples import (
    ACTIONS, BUDGET, PFA, allocation_counts, best_toy, fixed_detection_rows,
    report, toy_greedy, toy_probability,
)


def decimal_detection(strength, pfa=PFA):
    with localcontext() as context:
        context.prec = 70
        return Decimal(str(pfa)) ** (1 / (1 + Decimal(str(strength))))


def independent_probability(prior, gain, sequence):
    with localcontext() as context:
        context.prec = 70
        survival = [Decimal(1), Decimal(1)]
        for action in sequence:
            cell, energy = ACTIONS[action]
            survival[cell] *= 1 - decimal_detection(gain[cell] * energy)
        return float(sum(Decimal(str(probability)) * (1 - miss) for probability, miss in zip(prior, survival)))


@pytest.mark.parametrize("row", fixed_detection_rows())
def test_fixed_target_inversion_against_seventy_digit_direct_formula(row):
    with localcontext() as context:
        context.prec = 70
        required_pd = 1 - Decimal("0.2") ** (1 / Decimal(row["looks"]))
        strength = Decimal("0.00001").ln() / required_pd.ln() - 1
        assert row["snr"] == pytest.approx(float(strength), rel=2e-14)
        assert row["energy"] == pytest.approx(float(strength * row["looks"]), rel=2e-14)
        assert 1 - (1 - decimal_detection(row["snr"])) ** row["looks"] == pytest.approx(Decimal("0.8"), abs=Decimal("2e-15"))
        assert 1 <= row["snr"] <= 100


@pytest.mark.parametrize("pfa", [1e-5, 1e-6])
def test_upper_stationary_root_against_independent_decimal_bisection(pfa):
    with localcontext() as context:
        context.prec = 70
        lower, upper = Decimal(10), Decimal(25)
        for _ in range(180):
            strength = (lower + upper) / 2
            pd = decimal_detection(strength, pfa)
            residual = strength * pd * -Decimal(str(pfa)).ln() / (1 + strength) ** 2 + (1 - pd) * (1 - pd).ln()
            if residual > 0:
                lower = strength
            else:
                upper = strength
        assert cumulative_stationary_snrs(pfa)[1] == pytest.approx(float((lower + upper) / 2), rel=2e-14)


@pytest.mark.parametrize("looks", range(1, 13))
def test_fixed_budget_challenge_has_four_looks_as_best_without_changing_its_model(looks):
    rows = report()["fixed_60_equal_looks"]
    with localcontext() as context:
        context.prec = 70
        pd = decimal_detection(Decimal(60) / Decimal(looks))
        expected = float(1 - (1 - pd) ** looks)
    assert rows[looks - 1]["pc"] == pytest.approx(expected, abs=2e-15)
    assert max(rows, key=lambda row: row["pc"])["looks"] == 4


@pytest.mark.parametrize("prior,gain,expected", [
    ((0.75, 0.25), (0.5, 1.0), 0.5881070911063921),
    ((0.5, 0.5), (1.0, 1.0), 0.7367976797655608),
])
def test_joint_count_enumeration_against_all_ordered_sequences_and_independent_detection(prior, gain, expected):
    best = best_toy(prior, gain)
    assert best["allocations_examined"] == 260
    assert best["probability"] == pytest.approx(expected, abs=2e-15)
    probabilities = [float(decimal_detection(gain[cell] * energy)) for cell, energy in ACTIONS]

    def all_sequences(remaining, survival):
        maximum = math.fsum(probability * (1 - miss) for probability, miss in zip(prior, survival))
        for index, (cell, energy) in enumerate(ACTIONS):
            if energy <= remaining:
                changed = list(survival)
                changed[cell] *= 1 - probabilities[index]
                maximum = max(maximum, all_sequences(remaining - energy, changed))
        return maximum

    assert all_sequences(BUDGET, [1.0, 1.0]) == pytest.approx(best["probability"], abs=2e-15)
    for counts in allocation_counts():
        sequence = tuple(index for index, count in enumerate(counts) for _ in range(count))
        assert sum(ACTIONS[index][1] for index in sequence) <= BUDGET
        assert toy_probability(prior, gain, sequence) == pytest.approx(independent_probability(prior, gain, sequence), abs=2e-15)


def test_exact_menu_example_has_no_hidden_budget_advantage_or_order_dependent_probability():
    data = report()
    for example in (data["joint_unequal"], data["joint_symmetric"]):
        for strategy in example["strategies"].values():
            assert strategy["energy"] == BUDGET
    unequal = data["joint_unequal"]["strategies"]
    assert unequal["finite_menu_best"]["actions"] == [(0, 20), (0, 30), (1, 10)]
    assert unequal["finite_menu_best"]["probability"] > unequal["joint_myopic"]["probability"]
    symmetric = data["joint_symmetric"]["strategies"]
    assert symmetric["finite_menu_best"]["probability"] == symmetric["uniform_15"]["probability"]
    prior, gain = (0.75, 0.25), (0.5, 1)
    best = best_toy(prior, gain)
    for sequence in itertools.permutations(best["sequence"]):
        assert toy_probability(prior, gain, sequence) == pytest.approx(best["probability"], abs=2e-15)
    greedy = toy_greedy(prior, gain)
    assert [ACTIONS[index] for index in greedy] == [(0, 20), (1, 10), (0, 20), (1, 10)]


def test_fixed_target_discrete_best_is_close_to_not_identical_to_continuous_local_minimum():
    data = report()
    best = min(data["fixed_pc_0_8"], key=lambda row: row["energy"])
    continuous = data["continuous_references"][0]
    assert best["looks"] == 3
    assert best["energy"] == pytest.approx(36.29310489155377, abs=2e-13)
    assert continuous["continuous_local_energy_for_pc_0_8"] < best["energy"]
    assert continuous["continuous_looks_for_pc_0_8"] == pytest.approx(2.6064105851953987)
