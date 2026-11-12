"""Checks for the visual essay's reference values, against first principles and the transcribed sources."""

import itertools
import json
import math

import pytest

from optimal_search.radar_detection import detection_probability
from examples.essay_numbers import fixtures, summary

DATA = fixtures()


def test_values_are_json_serialisable_and_summarised():
    assert json.loads(json.dumps(DATA)) == DATA
    assert "11.416 dB" in summary(DATA) and "12.356 dB" in summary(DATA)


def test_budget_split_uses_the_single_pulse_curve():
    rows = DATA["split"]["looks"]
    assert [row["looks"] * row["snr"] for row in rows] == [60, 60, 60]
    for row in rows:
        assert row["pd"] == pytest.approx(1e-5 ** (1 / (1 + row["snr"])), rel=1e-14)
    assert [round(row["pd"], 3) for row in rows] == [0.828, 0.487, 0.147]
    # Matches the fixed-60 values in scripts/matthiesen_audit_examples.py for 1, 4 and 12 looks.
    assert [row["combined"] for row in rows] == pytest.approx([0.828004671229, 0.930724538623, 0.851155506812], abs=1e-12)


def test_efficiency_references_match_the_recorded_2011_table_after_rounding():
    low, high = DATA["strength"]["references"]
    assert round(low["db"], 1) == 11.4 and round(high["db"], 1) == 12.4
    assert low["snr"] == pytest.approx(13.855428077203527, rel=1e-12)
    assert low["db"] == pytest.approx(11.416199482846404, rel=1e-12)
    assert high["db"] == pytest.approx(12.355677203158415, rel=1e-12)
    assert low["local_minimum"] < 1 < low["snr"]


def test_reference_is_a_local_valley_of_energy_for_the_80_percent_target():
    curve = DATA["strength"]["curve"]
    best = DATA["strength"]["references"][0]
    best_energy = -math.log(0.2) / best["efficiency"]
    assert best_energy == pytest.approx(36.1129344028368, rel=1e-12)
    assert all(point["energy"] > best_energy for point in curve)
    energies = [point["energy"] for point in curve]
    assert energies[:4] == sorted(energies[:4], reverse=True) and energies[5:] == sorted(energies[5:])
    for point in curve:
        assert 1 - (1 - point["pd"]) ** point["looks"] == pytest.approx(0.8, rel=1e-12)


def test_low_snr_pathology_makes_the_peak_local_not_global():
    reference = DATA["strength"]["references"][0]
    tiny = 1e-4
    efficiency = -math.log1p(-detection_probability(tiny, 1e-5)) / tiny
    assert efficiency > reference["efficiency"]


def test_best_whole_look_design_is_three_looks():
    whole = DATA["strength"]["whole_looks"]
    best = min(whole, key=lambda row: row["energy"])
    assert best["looks"] == 3
    assert best["snr"] == pytest.approx(12.097701631, rel=1e-9)
    assert best["energy"] == pytest.approx(36.293104892, rel=1e-9)


def test_miss_example_matches_the_2011_search_example_2():
    miss = DATA["miss"]
    assert miss["first_look_success"] == pytest.approx(0.24)
    assert miss["normaliser"] == pytest.approx(0.76)
    assert [round(value, 5) for value in miss["posterior"]] == [0.13158, 0.26316, 0.21053, 0.39474]
    assert math.fsum(miss["posterior"]) == pytest.approx(1, abs=1e-15)


def test_next_example_ranks_a_less_likely_cell_first_and_greedy_loses_to_the_exact_plan():
    nxt = DATA["next"]
    assert max(range(3), key=lambda j: nxt["prior"][j]) == 0
    assert max(range(3), key=lambda j: nxt["priority"][j]) == 1
    assert [step["cell"] for step in nxt["greedy"]] == [1, 2]
    assert nxt["greedy"][-1]["cumulative"] == pytest.approx(0.37, abs=1e-15)
    assert nxt["exact"] == {"counts": [1, 0, 0], "probability": pytest.approx(0.4, abs=1e-15), "cost": 2.0,
                            "plans_examined": 7}


def test_next_example_by_independent_enumeration_of_every_affordable_sequence():
    nxt = DATA["next"]
    p, d, c = nxt["prior"], nxt["detection"], nxt["cost"]
    best = 0.0
    for length in range(3):
        for sequence in itertools.product(range(3), repeat=length):
            if sum(c[j] for j in sequence) > nxt["budget"]:
                continue
            missed = sum(p[target] * math.prod(1 - d[j] for j in sequence if j == target) for target in range(3))
            best = max(best, 1 - missed)
    assert best == pytest.approx(0.4, abs=1e-15)


def test_section_two_claims_about_the_valley():
    """Plain-English claims in Section Two and the notebook, checked against the research module."""
    from scipy.optimize import brentq

    from optimal_search.cumulative_detection import cumulative_efficiency

    reference = DATA["strength"]["references"][0]
    peak = reference["efficiency"]
    # The efficient strength is the same for any goal: energy = -ln(1 - Pc) / H(s).
    for pc in (0.5, 0.8, 0.99):
        energy = lambda s, pc=pc: -math.log1p(-pc) / cumulative_efficiency(s, 1e-5)
        assert energy(reference["snr"]) < min(energy(reference["snr"] * 0.97), energy(reference["snr"] * 1.03))
    # "Anything from about 10 to 20 costs within a few percent of the minimum."
    excess = [cumulative_efficiency(reference["snr"], 1e-5) / cumulative_efficiency(s, 1e-5) - 1 for s in range(10, 21)]
    assert max(excess) < 0.04
    # Local minimum near 0.106; H beats the valley only below about 0.0002, over 4,000 times weaker than strength 1.
    assert reference["local_minimum"] == pytest.approx(0.106, abs=5e-4)
    crossing = brentq(lambda s: cumulative_efficiency(s, 1e-5) - peak, 1e-9, reference["local_minimum"])
    assert 0.0002 < crossing < 0.00025 and 1 / crossing > 4000
    # Within the charted range, 1-40, the reference is the lowest point.
    assert all(cumulative_efficiency(s / 10, 1e-5) <= peak for s in range(10, 401))
