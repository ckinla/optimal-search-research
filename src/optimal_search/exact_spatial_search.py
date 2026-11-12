"""Exact (exhaustive) optimisation of a finite-budget spatial search.

Problem
-------
One stationary target is in exactly one of ``M`` cells. Cell ``j`` has prior
``p_j`` (summing to 1), a per-search detection probability ``d_j`` (constant,
independent between repeated searches) and a positive search cost ``c_j``.
Choose non-negative integer search counts ``n_j`` to maximise the
unconditional probability of detecting the target::

    P(n) = 1 - sum_j p_j (1 - d_j)**n_j            subject to   sum_j n_j c_j <= budget

The solver enumerates *every* feasible allocation, so the result is the exact
optimum. It is deliberately independent of the greedy implementation in
:mod:`optimal_search.spatial_search`: it shares no code with it (not even input
validation) and imports nothing from it, so the two can be compared.

Order matters for time, not for probability
-------------------------------------------
``P(n)`` depends only on the counts. Every ordering of the same searches has
the same probability of detecting the target *within the full plan*. But if the
search stops as soon as the target is detected, the *expected cost incurred*
depends on the order::

    E[cost] = sum_n c_n * S_{n-1},      S_n = P(not detected after n searches)

and so does the expected cost given that detection occurs. For a fixed
allocation both are minimised by sequencing the searches in non-increasing
order of ``a / c``, where ``a = p_j (1 - d_j)**k d_j`` is the unconditional
probability that the next search of cell ``j`` (after ``k`` earlier ones) is
the first detection. (Adjacent-exchange argument: swapping neighbours ``i, j``
changes the expected cost by ``c_j a_i - c_i a_j`` in favour of ``i`` first when
``a_i / c_i > a_j / c_j``; the tests check this against all permutations.) The
solver returns its sequence in that order. The *choice of allocation* is made
only on ``P`` (then cost, then a fixed index preference), never on time.

Deterministic tie-breaking
--------------------------
Among all feasible allocations whose probability is within a *relative*
``tie_tolerance`` of the maximum (``P >= P_max * (1 - tie_tolerance)``), choose

1. the smallest total cost (within a relative tolerance), then
2. the lexicographically greatest count vector, i.e. prefer searching
   lower-index cells.

A tolerance is used because rounding can separate nominally tied values by
~1e-16 of their size; it is relative so that problems with tiny probabilities
(for example ``d = 1e-18``) are not mistaken for ties. Budget feasibility uses the same kind of tolerance
(``sum n_j c_j <= budget * (1 + 1e-12)``, relative, so a tiny budget is not loosened) so that, for example, three
searches of cost 0.1 fit in a budget of 0.3.

Limits
------
Enumeration visits every feasible allocation. With equal costs ``c`` and
``k = floor(budget / c)`` there are ``C(k + M, M)`` of them, which grows quickly
(see :func:`count_feasible_allocations` and ``docs/exact-spatial-search.md``).
With ``d = 1`` the problem is the 0/1 knapsack problem (searching a cell more
than once is useless, so ``P = sum_j p_j x_j`` with ``x_j`` in ``{0, 1}``), hence
weakly NP-hard in general. ``max_allocations`` guards against accidental
blow-up and raises :class:`SearchSpaceTooLargeError`.
"""

import heapq
import math
import numbers
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

DEFAULT_MAX_ALLOCATIONS = 1_000_000
DEFAULT_TIE_TOLERANCE = 1e-12
_PRIOR_TOLERANCE = 1e-9
_BUDGET_RTOL = 1e-12  # relative slack on the budget
_COST_TIE_RTOL = 1e-12  # relative tolerance for treating two total costs as tied
_TIE_RTOL = 1e-12  # relative tolerance for treating two search ratios as tied


class SearchSpaceTooLargeError(ValueError):
    """Raised when exhaustive enumeration would exceed ``max_allocations`` allocations."""


@dataclass(frozen=True)
class ExactSolution:
    """An optimal allocation and a search sequence realising it."""

    counts: tuple[int, ...]
    total_cost: float
    detection_probability: float  # P(detected within the full plan); order-independent
    sequence: tuple[int, ...]  # cells in the order searched (minimises expected cost for these counts)
    expected_cost: float  # expected cost incurred along ``sequence`` if the search stops at the first detection
    expected_cost_given_detection: float | None  # expected cost at detection given detection occurs; None if P = 0
    budget: float
    allocations_examined: int


@dataclass(frozen=True)
class FractionalBound:
    """Divisible-effort relaxation of the budgeted problem (an upper bound on the exact optimum).

    Searches are taken in non-increasing ``a / c`` order while they fit; ``blocked_*`` describes the first
    one that does not, of which a fraction is taken: ``value = prefix_value + (remaining / blocked_cost) *
    blocked_gain``. ``critical_ratio`` is that search's ``a / c`` (0 if nothing blocks).

    ``strict_prefix_*`` count only the searches whose ratio is *strictly* greater than ``critical_ratio``
    (by more than a relative 1e-12). Every greedy plan, whatever its tie-breaking among equal ratios, takes
    all of them, so ``exact - greedy <= value - strict_prefix_value = critical_ratio * (budget - strict_prefix_cost)``.
    """

    value: float
    prefix_value: float
    prefix_cost: float
    blocked_cell: int | None
    blocked_gain: float
    blocked_cost: float
    critical_ratio: float
    strict_prefix_value: float
    strict_prefix_cost: float


# --- input validation (independent of optimal_search.spatial_search) -----------------------------------


def _probabilities(values: Sequence[float], name: str) -> list[float]:
    try:
        array = [float(v) for v in values]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a sequence of numbers") from error
    if not array:
        raise ValueError(f"{name} must be non-empty")
    for v in array:
        if not math.isfinite(v) or v < 0.0 or v > 1.0:
            raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return array


def _validate_problem(prior, detection, cost, budget) -> tuple[list[float], list[float], list[float], float]:
    if isinstance(prior, (str, bytes)) or isinstance(detection, (str, bytes)) or isinstance(cost, (str, bytes)):
        raise ValueError("prior, detection and cost must be sequences of numbers")
    p = _probabilities(prior, "prior")
    if not math.isclose(math.fsum(p), 1.0, rel_tol=0.0, abs_tol=_PRIOR_TOLERANCE):
        raise ValueError(f"prior must sum to 1 (got {math.fsum(p)!r})")
    d = _probabilities(detection, "detection")
    if len(d) != len(p):
        raise ValueError(f"detection has {len(d)} cells but prior has {len(p)}")
    try:
        c = [float(v) for v in cost]
    except (TypeError, ValueError) as error:
        raise ValueError("cost must be a sequence of numbers") from error
    if len(c) != len(p):
        raise ValueError(f"cost has {len(c)} cells but prior has {len(p)}")
    if any(not math.isfinite(v) or v <= 0.0 for v in c):
        raise ValueError("cost must contain finite values > 0")
    if isinstance(budget, bool):
        raise ValueError("budget must be a finite number >= 0")
    try:
        b = float(budget)
    except (TypeError, ValueError) as error:
        raise ValueError("budget must be a finite number >= 0") from error
    if not math.isfinite(b) or b < 0.0:
        raise ValueError("budget must be a finite number >= 0")
    return p, d, c, b


def _validate_limit(max_allocations: int) -> int:
    if not _is_integer(max_allocations) or max_allocations < 1:
        raise ValueError("max_allocations must be a positive integer")
    return max_allocations


def _validate_tolerance(tie_tolerance: float) -> float:
    if isinstance(tie_tolerance, bool) or not isinstance(tie_tolerance, (int, float)) or not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be a finite number >= 0")
    return float(tie_tolerance)


def _is_integer(value: object) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _validate_counts(counts: Sequence[int], size: int) -> list[int]:
    try:
        values = list(counts)
    except TypeError as error:
        raise ValueError("counts must be a sequence of non-negative integers") from error
    if len(values) != size:
        raise ValueError(f"counts has {len(values)} cells but expected {size}")
    if any(not _is_integer(v) or v < 0 for v in values):
        raise ValueError("counts must be non-negative integers")
    return [int(v) for v in values]


def _validate_sequence(sequence: Sequence[int], size: int) -> list[int]:
    cells = list(sequence)
    if any(not _is_integer(cell) or not 0 <= cell < size for cell in cells):
        raise ValueError(f"every cell index must be an integer in [0, {size})")
    return [int(cell) for cell in cells]


# --- objective ----------------------------------------------------------------------------------------


def _detected_given_cell(d: float, n: int) -> float:
    """``1 - (1 - d)**n`` without cancellation (1 if ``d = 1`` and ``n > 0``)."""
    if n == 0 or d == 0.0:
        return 0.0
    if d == 1.0:
        return 1.0
    return -math.expm1(n * math.log1p(-d))


def _increment(p: float, d: float, k: int) -> float:
    """``p (1 - d)**k d``: probability that the ``(k+1)``-th search of a cell is the first detection."""
    if d == 0.0 or p == 0.0:
        return 0.0
    if d == 1.0:
        return p if k == 0 else 0.0
    return p * math.exp(k * math.log1p(-d)) * d


def _miss_factor(d: float, n: int) -> float:
    """``(1 - d)**n`` as ``exp(n log1p(-d))`` (0 if ``d = 1`` and ``n > 0``)."""
    if n == 0:
        return 1.0
    if d == 1.0:
        return 0.0
    return math.exp(n * math.log1p(-d))


def detection_probability_of_counts(prior: Sequence[float], detection: Sequence[float], counts: Sequence[int]) -> float:
    """``P(detected) = sum_j p_j [1 - (1 - d_j)**n_j]`` for an allocation (order-independent)."""
    p = _probabilities(prior, "prior")
    d = _probabilities(detection, "detection")
    if len(d) != len(p):
        raise ValueError(f"detection has {len(d)} cells but prior has {len(p)}")
    n = _validate_counts(counts, len(p))
    return min(math.fsum(p_j * _detected_given_cell(d_j, n_j) for p_j, d_j, n_j in zip(p, d, n)), 1.0)


def total_cost_of_counts(cost: Sequence[float], counts: Sequence[int]) -> float:
    """``sum_j n_j c_j``."""
    c = [float(v) for v in cost]
    n = _validate_counts(counts, len(c))
    return math.fsum(c_j * n_j for c_j, n_j in zip(c, n))


def expected_search_cost(
    prior: Sequence[float], detection: Sequence[float], cost: Sequence[float], sequence: Sequence[int]
) -> tuple[float, float | None]:
    """``(E[cost incurred], E[cost | detected])`` for searching ``sequence`` and stopping at the first detection.

    The cost of a search is incurred whether or not it succeeds.
    ``E[cost incurred] = sum_n c_n S_{n-1}`` where ``S_n`` is the probability of no detection after
    ``n`` searches. ``E[cost | detected] = sum_n C_n a_n / P`` with ``C_n`` the cumulative cost through
    search ``n``, ``a_n`` its unconditional first-detection probability, and ``P = sum_n a_n``
    (``None`` if ``P = 0``). Unlike the detection probability, both depend on the order of ``sequence``.
    """
    p, d, c, _ = _validate_problem(prior, detection, cost, 0.0)
    cells = _validate_sequence(sequence, len(p))
    seen = [0] * len(p)
    weights = list(p)  # p_j (1 - d_j)^k_j: probability the target is in cell j and every search so far missed it
    cumulative_cost = 0.0
    incurred, weighted_cost, detected = 0.0, 0.0, 0.0
    for cell in cells:
        # Survival is the sum of the remaining weights, not 1 minus the detections so far: repeated
        # subtraction leaves a rounding remainder after detection is certain, which would charge the
        # full cost of every later (unreachable) search.
        survival = math.fsum(weights)
        incurred += c[cell] * survival
        cumulative_cost += c[cell]
        a = _increment(p[cell], d[cell], seen[cell])
        weighted_cost += cumulative_cost * a
        detected += a
        seen[cell] += 1
        weights[cell] = p[cell] * _miss_factor(d[cell], seen[cell])
    return incurred, (weighted_cost / detected if detected > 0.0 else None)


def order_for_minimum_expected_cost(
    prior: Sequence[float], detection: Sequence[float], cost: Sequence[float], counts: Sequence[int]
) -> tuple[int, ...]:
    """Order the searches of an allocation to minimise the expected cost (non-increasing ``a / c``).

    ``a`` is the unconditional first-detection probability of the next search of a cell. Ties go to the
    lower cell index. The detection probability of the full plan does not depend on this order.
    """
    p, d, c, _ = _validate_problem(prior, detection, cost, 0.0)
    remaining = _validate_counts(counts, len(p))
    placed = [0] * len(p)
    sequence: list[int] = []
    for _ in range(sum(remaining)):
        best, best_ratio = -1, -1.0
        for j in range(len(p)):
            if remaining[j] > 0:
                ratio = _increment(p[j], d[j], placed[j]) / c[j]
                if ratio > best_ratio:
                    best, best_ratio = j, ratio
        sequence.append(best)
        placed[best] += 1
        remaining[best] -= 1
    return tuple(sequence)


# --- enumeration --------------------------------------------------------------------------------------------


def _cap(budget: float) -> float:
    return budget * (1.0 + _BUDGET_RTOL)


def _max_counts(c: Sequence[float], budget: float, limit: int) -> list[int]:
    cap = _cap(budget)
    maxima = []
    for c_j in c:
        quotient = cap / c_j
        if quotient >= limit:
            raise SearchSpaceTooLargeError(
                f"a single cell can be searched up to {quotient:.3g} times, exceeding max_allocations={limit}"
            )
        maxima.append(int(math.floor(quotient)))
    # Every allocation that searches a single cell (0..max_j times) is feasible, so at least
    # 1 + sum_j max_j allocations exist. Checking this before building any table bounds memory.
    if 1 + sum(maxima) > limit:
        raise SearchSpaceTooLargeError(f"at least {1 + sum(maxima):,} feasible allocations, exceeding max_allocations={limit}")
    return maxima


def _enumerate(
    p: Sequence[float], d: Sequence[float], c: Sequence[float], budget: float, limit: int
) -> Iterator[tuple[tuple[int, ...], float, float]]:
    """Yield ``(counts, total_cost, probability)`` for every feasible allocation in lexicographic order.

    Iterative depth-first search over the cells that can be searched at all, so the depth is not limited
    by Python's recursion limit and cells with a maximum count of zero cost nothing.
    """
    cap = _cap(budget)
    maxima = _max_counts(c, budget, limit)
    active = [j for j in range(len(p)) if maxima[j] > 0]
    tables = {j: [p[j] * _detected_given_cell(d[j], n) for n in range(maxima[j] + 1)] for j in active}
    counts = [0] * len(p)
    depth_count = len(active)
    examined = 0

    def emit(spent: float, probability: float):
        nonlocal examined
        examined += 1
        if examined > limit:
            raise SearchSpaceTooLargeError(f"more than max_allocations={limit} feasible allocations")
        return tuple(counts), spent, probability

    if depth_count == 0:
        yield emit(0.0, 0.0)
        return
    n = [-1] * depth_count
    spent_before = [0.0] * (depth_count + 1)
    probability_before = [0.0] * (depth_count + 1)
    depth = 0
    while depth >= 0:
        n[depth] += 1
        j = active[depth]
        cost_here = spent_before[depth] + n[depth] * c[j]
        if n[depth] > maxima[j] or cost_here > cap:
            counts[j] = 0
            n[depth] = -1
            depth -= 1
            continue
        counts[j] = n[depth]
        spent_before[depth + 1] = cost_here
        probability_before[depth + 1] = probability_before[depth] + tables[j][n[depth]]
        if depth == depth_count - 1:
            yield emit(cost_here, probability_before[depth + 1])
        else:
            depth += 1


def enumerate_feasible_allocations(
    cost: Sequence[float], budget: float, max_allocations: int = DEFAULT_MAX_ALLOCATIONS
) -> Iterator[tuple[int, ...]]:
    """Every feasible count vector in lexicographic order, starting with all zeros."""
    p = [1.0 / len(cost)] * len(cost) if len(cost) else []
    _, _, c, b = _validate_problem(p, [0.0] * len(p), cost, budget)
    limit = _validate_limit(max_allocations)
    for counts, _, _ in _enumerate(p, [0.0] * len(p), c, b, limit):
        yield counts


def count_feasible_allocations(cost: Sequence[float], budget: float, max_allocations: int = DEFAULT_MAX_ALLOCATIONS) -> int:
    """Number of feasible allocations (including the empty one); raises if it exceeds ``max_allocations``."""
    return sum(1 for _ in enumerate_feasible_allocations(cost, budget, max_allocations))


# --- exact solver -------------------------------------------------------------------------------------------


def solve_exact(
    prior: Sequence[float],
    detection: Sequence[float],
    cost: Sequence[float],
    budget: float,
    max_allocations: int = DEFAULT_MAX_ALLOCATIONS,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> ExactSolution:
    """Allocation maximising ``P(detected)`` subject to ``sum_j n_j c_j <= budget``, by exhaustive search.

    Ties (probabilities with ``P >= P_max * (1 - tie_tolerance)``) are resolved by the smallest total
    cost, then by the lexicographically greatest count vector. Three enumeration passes keep memory
    constant: find the best probability, find the cheapest near-best allocation, then pick the
    lexicographically greatest among those.

    Raises :class:`SearchSpaceTooLargeError` if more than ``max_allocations`` allocations are feasible.
    """
    p, d, c, b = _validate_problem(prior, detection, cost, budget)
    limit = _validate_limit(max_allocations)
    tol = _validate_tolerance(tie_tolerance)

    best_probability, examined = -1.0, 0
    for _, _, probability in _enumerate(p, d, c, b, limit):
        examined += 1
        if probability > best_probability:
            best_probability = probability
    floor_probability = best_probability * (1.0 - tol)

    cheapest = math.inf
    for _, spent, probability in _enumerate(p, d, c, b, limit):
        if probability >= floor_probability and spent < cheapest:
            cheapest = spent
    cost_ceiling = cheapest * (1.0 + _COST_TIE_RTOL)

    chosen: tuple[int, ...] | None = None
    for counts, spent, probability in _enumerate(p, d, c, b, limit):
        if probability >= floor_probability and spent <= cost_ceiling and (chosen is None or counts > chosen):
            chosen = counts
    assert chosen is not None  # the empty allocation is always feasible

    sequence = order_for_minimum_expected_cost(p, d, c, chosen)
    incurred, given_detection = expected_search_cost(p, d, c, sequence)
    return ExactSolution(
        counts=chosen,
        total_cost=total_cost_of_counts(c, chosen),
        detection_probability=detection_probability_of_counts(p, d, chosen),
        sequence=sequence,
        expected_cost=incurred,
        expected_cost_given_detection=given_detection,
        budget=b,
        allocations_examined=examined,
    )


# --- divisible-effort relaxation (upper bound) --------------------------------------------------------------------


def fractional_upper_bound(
    prior: Sequence[float], detection: Sequence[float], cost: Sequence[float], budget: float, max_steps: int = DEFAULT_MAX_ALLOCATIONS
) -> FractionalBound:
    """Upper bound on the exact optimum from the divisible-effort relaxation.

    Treat each possible search as an item with value ``a = p_j (1 - d_j)**k d_j`` (the gain from the
    ``(k+1)``-th search of cell ``j``) and cost ``c_j``. Any feasible allocation picks a prefix of each cell's
    items, so its value is at most that of the best *subset* of items, which is at most the fractional
    knapsack value: take items in non-increasing ``a / c`` order while they fit, then a fraction of the
    first item that does not. The bound is an independent check on :func:`solve_exact`. It also bounds how
    much a greedy plan can lose: every greedy plan takes all searches whose ratio is strictly above the
    critical ratio (see :class:`FractionalBound`), so ``exact - greedy <= value - strict_prefix_value``.
    Searches tied at the critical ratio can be taken in different orders, which is why the guarantee is
    stated for the strict prefix and not for the first-come prefix.
    """
    p, d, c, b = _validate_problem(prior, detection, cost, budget)
    cap = _cap(b)
    seen = [0] * len(p)

    def log_ratio(j: int) -> float:
        # log(a / c) = log(a) - log(c) cannot underflow, so the order of positive gains is kept even when
        # a / c would round to zero; a gain of zero sorts last.
        gain = _increment(p[j], d[j], seen[j])
        return math.log(gain) - math.log(c[j]) if gain > 0.0 else -math.inf

    heap = [(-log_ratio(j), j) for j in range(len(p))]
    heapq.heapify(heap)
    taken: list[tuple[float, float, float]] = []  # (log ratio, gain, cost) of the searches taken in full
    spent, value = 0.0, 0.0

    def strict_prefix(critical_log_ratio: float) -> tuple[float, float]:
        threshold = critical_log_ratio + _TIE_RTOL
        strict = [(gain, cost_) for lr, gain, cost_ in taken if lr > threshold]
        return math.fsum(g for g, _ in strict), math.fsum(c_ for _, c_ in strict)

    for _ in range(max_steps):
        _, j = heapq.heappop(heap)
        gain = _increment(p[j], d[j], seen[j])
        if gain <= 0.0:
            return FractionalBound(value, value, spent, None, 0.0, 0.0, 0.0, value, spent)
        if spent + c[j] > cap:
            fraction = max(b - spent, 0.0) / c[j]
            critical_log = math.log(gain) - math.log(c[j])
            strict_value, strict_cost = strict_prefix(critical_log)
            return FractionalBound(value + fraction * gain, value, spent, j, gain, c[j], gain / c[j], strict_value, strict_cost)
        taken.append((math.log(gain) - math.log(c[j]), gain, c[j]))
        spent += c[j]
        value += gain
        seen[j] += 1
        heapq.heappush(heap, (-log_ratio(j), j))
    raise SearchSpaceTooLargeError(f"more than max_steps={max_steps} searches fit in the budget")
