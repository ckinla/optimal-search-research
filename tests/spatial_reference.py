"""Independent brute-force references for the spatial-search tests (no closed forms used)."""

import itertools


def enumerate_detection_probability(prior, detection, looks):
    """P(at least one detection) by enumerating target cell x outcome of every search.

    A search of cell c detects with probability d[c] if the target is in c and
    with probability 0 otherwise.
    """
    total = 0.0
    for target, p_target in enumerate(prior):
        for outcomes in itertools.product((False, True), repeat=len(looks)):
            probability = p_target
            for cell, success in zip(looks, outcomes):
                p_success = detection[cell] if cell == target else 0.0
                probability *= p_success if success else 1.0 - p_success
            if any(outcomes):
                total += probability
    return total


def enumerate_posterior_after_failures(prior, detection, looks):
    """P(target in j | every search failed), by enumerating the joint distribution."""
    joint = []
    for target, p_target in enumerate(prior):
        probability = p_target
        for cell in looks:
            p_success = detection[cell] if cell == target else 0.0
            probability *= 1.0 - p_success
        joint.append(probability)
    total = sum(joint)
    return [x / total for x in joint]


def enumerate_first_detection_at(prior, detection, looks, index):
    """P(search number ``index`` is the first detection), by enumeration."""
    total = 0.0
    for target, p_target in enumerate(prior):
        probability = p_target
        for i, cell in enumerate(looks[: index + 1]):
            p_success = detection[cell] if cell == target else 0.0
            probability *= p_success if i == index else 1.0 - p_success
        total += probability
    return total


def enumerate_expected_cost(prior, detection, cost, looks):
    """(E[cost incurred], P(detected)) for searching ``looks`` and stopping at the first detection.

    Enumerates target cell x outcome of every search. The cost of a search is incurred whether or
    not it succeeds; after a success no further searches are made.
    """
    expected, detected = 0.0, 0.0
    for target, p_target in enumerate(prior):
        for outcomes in itertools.product((False, True), repeat=len(looks)):
            probability, incurred, stopped = p_target, 0.0, False
            for cell, success in zip(looks, outcomes):
                if stopped:
                    # outcomes after the first success are not realised: count each realised history once
                    if success:
                        probability = 0.0
                        break
                    continue
                incurred += cost[cell]
                p_success = detection[cell] if cell == target else 0.0
                probability *= p_success if success else 1.0 - p_success
                if success:
                    stopped = True
            if probability:
                expected += probability * incurred
                if stopped:
                    detected += probability
    return expected, detected


def knapsack_dp_max_probability(prior, detection, integer_costs, integer_budget):
    """Maximum P(detected) for integer costs by dynamic programming over cells (pseudo-polynomial).

    best[b] = max probability using total cost at most b with the cells processed so far.
    """
    best = [0.0] * (integer_budget + 1)
    for p, d, c in zip(prior, detection, integer_costs):
        updated = list(best)
        for b in range(integer_budget + 1):
            n = 1
            while n * c <= b:
                value = best[b - n * c] + p * (1.0 - (1.0 - d) ** n)
                if value > updated[b]:
                    updated[b] = value
                n += 1
        best = updated
    return best[integer_budget]
