"""Reproduce the joint where-and-how-strongly research examples; not a production solver.

The joint toy maximizes target-cell detection opportunity under an integer
energy budget, using the existing detector curve. It omits off-target false
alarms, correlation, integration, time overhead and beam footprints. This is
our finite-action extension, NOT a reproduction of Matthiesen's full method.
See docs/provenance.md for the objective and evidence limits.
"""

import json
import math
from collections.abc import Iterator, Sequence

from optimal_search.cumulative_detection import (
    continuous_optimal_looks,
    cumulative_efficiency,
    cumulative_stationary_snrs,
    log_miss_rate,
    required_snr_for_looks,
)
from optimal_search.radar_detection import detection_probability, snr_linear_to_db

PFA = 1e-5
ENERGIES = (10, 15, 20, 30, 60)
BUDGET = 60
ACTIONS = tuple((cell, energy) for cell in range(2) for energy in ENERGIES)


def fixed_detection_rows(pc: float = 0.8, pfa: float = PFA) -> list[dict]:
    """Compare only N=1..12 identical independent looks, with 1 <= s <= 100."""
    rows = []
    for looks in range(1, 13):
        strength = required_snr_for_looks(looks, pc, pfa)
        if 1 <= strength <= 100:
            rows.append({"looks": looks, "snr": strength, "energy": looks * strength,
                         "pd": detection_probability(strength, pfa),
                         "pc": -math.expm1(-looks * log_miss_rate(strength, pfa))})
    return rows


def allocation_counts(index: int = 0, remaining: int = BUDGET) -> Iterator[tuple[int, ...]]:
    """Exhaust every feasible action-count vector, including the empty plan."""
    if index == len(ACTIONS):
        yield ()
        return
    energy = ACTIONS[index][1]
    for count in range(remaining // energy + 1):
        for suffix in allocation_counts(index + 1, remaining - count * energy):
            yield (count, *suffix)


def toy_probability(prior: Sequence[float], gain: Sequence[float], sequence: Sequence[int]) -> float:
    rates = [0.0, 0.0]
    for action in sequence:
        cell, energy = ACTIONS[action]
        rates[cell] += log_miss_rate(gain[cell] * energy, PFA)
    return -math.fsum(probability * math.expm1(-rate) for probability, rate in zip(prior, rates))


def toy_greedy(prior: Sequence[float], gain: Sequence[float], fixed_energy: int | None = None) -> tuple[int, ...]:
    """Myopic target-cell detection increment per energy; not fixed-budget optimal."""
    remaining = BUDGET
    rates = [0.0, 0.0]
    sequence = []
    while True:
        affordable = [index for index, (_, energy) in enumerate(ACTIONS)
                      if energy <= remaining and (fixed_energy is None or energy == fixed_energy)]
        if not affordable:
            return tuple(sequence)
        action = max(affordable, key=lambda index:
                     prior[ACTIONS[index][0]] * math.exp(-rates[ACTIONS[index][0]])
                     * detection_probability(gain[ACTIONS[index][0]] * ACTIONS[index][1], PFA)
                     / ACTIONS[index][1])
        cell, energy = ACTIONS[action]
        sequence.append(action)
        remaining -= energy
        rates[cell] += log_miss_rate(gain[cell] * energy, PFA)


def best_toy(prior: Sequence[float], gain: Sequence[float]) -> dict:
    candidates = []
    for counts in allocation_counts():
        sequence = tuple(index for index, count in enumerate(counts) for _ in range(count))
        candidates.append((toy_probability(prior, gain, sequence), sequence))
    maximum = max(probability for probability, _ in candidates)
    tied = [sequence for probability, sequence in candidates
            if math.isclose(probability, maximum, rel_tol=1e-12, abs_tol=1e-14)]
    chosen = min(tied, key=lambda sequence: (sum(ACTIONS[index][1] for index in sequence), sequence))
    return {"probability": toy_probability(prior, gain, chosen), "sequence": chosen,
            "allocations_examined": len(candidates), "tied_allocations": len(tied)}


def toy_example(prior: Sequence[float], gain: Sequence[float]) -> dict:
    uniform = tuple(ACTIONS.index((cell, 15)) for cell in (0, 0, 1, 1))
    exact = best_toy(prior, gain)
    strategies = {"uniform_15": uniform, "posterior_greedy_15": toy_greedy(prior, gain, 15),
                  "joint_myopic": toy_greedy(prior, gain), "finite_menu_best": exact["sequence"]}
    return {"prior": prior, "gain": gain, "budget": BUDGET, "pfa": PFA, "menu": ENERGIES,
            "allocations_examined": exact["allocations_examined"], "tied_allocations": exact["tied_allocations"],
            "strategies": {name: {"actions": [ACTIONS[index] for index in sequence],
                                  "energy": sum(ACTIONS[index][1] for index in sequence),
                                  "probability": toy_probability(prior, gain, sequence)}
                           for name, sequence in strategies.items()}}


def report() -> dict:
    references = []
    for pfa in (1e-5, 1e-6):
        strength = cumulative_stationary_snrs(pfa)[1]
        references.append({"pfa": pfa, "snr": strength, "db": snr_linear_to_db(strength),
                           "pd": detection_probability(strength, pfa),
                           "continuous_looks_for_pc_0_8": continuous_optimal_looks(0.8, pfa),
                           "continuous_local_energy_for_pc_0_8": -math.log1p(-0.8) / cumulative_efficiency(strength, pfa)})
    return {"continuous_references": references, "fixed_pc_0_8": fixed_detection_rows(),
            "fixed_60_equal_looks": [{"looks": looks, "pc": -math.expm1(-looks * log_miss_rate(60 / looks, PFA))}
                                    for looks in range(1, 13)],
            "joint_unequal": toy_example((0.75, 0.25), (0.5, 1.0)),
            "joint_symmetric": toy_example((0.5, 0.5), (1.0, 1.0))}


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, allow_nan=False))
