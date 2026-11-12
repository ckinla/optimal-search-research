"""Bayesian spatial search for one stationary target in one of ``M`` cells.

Model
-----
A single stationary target occupies exactly one of ``M`` mutually exclusive
and exhaustive cells. Cell ``j`` has

* ``prior[j]``: probability that the target is in cell ``j`` (sums to 1),
* ``detection[j]``: probability ``d_j`` that one search of cell ``j`` detects
  the target *given that it is there*, constant and independent between
  repeated searches,
* ``cost[j]``: cost ``c_j > 0`` of one search of cell ``j``.

A search of cell ``j`` can only detect the target if it is in cell ``j``.
Nothing here depends on radar energy or SNR: ``d_j`` is simply an input.

Independently derived mathematics
---------------------------------
Let ``n_j`` be the number of searches made in cell ``j`` so far, and suppose
every search so far failed.

*Posterior.* By Bayes' theorem the unnormalised weight of cell ``j`` is
``p_j (1 - d_j)**n_j`` (a failed search of cell ``j`` multiplies only that
cell's weight by ``1 - d_j``); the posterior is that weight divided by the sum
of all weights. A failed search of cell ``k`` therefore updates the posterior
``q`` by ``q_k -> q_k (1 - d_k)`` and renormalises, and the normaliser is
``1 - q_k d_k``. Because the final weights depend only on the counts ``n_j``,
the posterior does not depend on the order of the searches.

*Unconditional cumulative detection probability.* For a fixed set of searches,
the target is *not* detected iff, for the cell ``j`` it occupies, all ``n_j``
searches of that cell fail. By the law of total probability::

    P(not detected) = sum_j p_j (1 - d_j)**n_j
    P(detected by the end) = 1 - sum_j p_j (1 - d_j)**n_j
                           = sum_j p_j [1 - (1 - d_j)**n_j]        (sum_j p_j = 1)

This depends only on the counts ``n_j``, **not on the order** of the searches.
The posterior updates are conditioning on failure; they do not change the
unconditional probability of the outcome "detected somewhere in this fixed
sequence".

*Conditional versus unconditional next-look probability.* After ``N - 1``
failed searches the posterior is ``q``. The probability that the *next* search,
in cell ``k``, succeeds *given* all previous failures is the **conditional**
quantity ``q_k d_k``. The probability that this search is the one that makes the
first detection, as seen from the start, is the **unconditional** increment
``p_k (1 - d_k)**n_k d_k = (1 - P_{N-1}) q_k d_k``, where ``P_{N-1}`` is the
unconditional probability of detection after ``N - 1`` searches. Unconditional
increments sum to ``P_N`` (they are probabilities of disjoint events);
conditional probabilities do not sum to anything meaningful and can add up to
more than 1.

*Greedy rule.* The next search goes to the cell maximising
``q_k d_k / c_k``, the conditional success probability per unit cost. The
unconditional increment is the same quantity times the factor ``1 - P_{N-1}``,
which is common to all cells, so the ranking is identical.

Limitations
-----------
* Greedy selection maximises the *immediate* payoff per unit cost. With equal
  costs, constant independent per-search detection probabilities and a fixed
  number of indivisible searches it maximises the unconditional cumulative
  detection probability. It is not automatically globally optimal for
  arbitrary costs or finite budgets: with unequal indivisible costs under a
  budget it is a knapsack-like heuristic and can be beaten by a different
  sequence (see the tests for an exhaustively verified example and
  ``docs/spatial-search-verification.md``).
* The target is stationary; the detection probability ``d_j`` is constant and
  independent between repeated searches of a cell (no look-dependent or
  energy-dependent detection, no false alarms).
* Searches are made one at a time with failure as the only observed outcome;
  the plan returned by :func:`greedy_search_plan` is the all-failures path.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_PRIOR_TOLERANCE = 1e-9


def _as_probability_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.array(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return array


def _validate_distribution(values: Sequence[float], name: str) -> np.ndarray:
    array = _as_probability_vector(values, name)
    if not math.isclose(float(array.sum()), 1.0, rel_tol=0.0, abs_tol=_PRIOR_TOLERANCE):
        raise ValueError(f"{name} must sum to 1 (got {float(array.sum())!r})")
    return array


def _validate_prior(prior: Sequence[float]) -> np.ndarray:
    return _validate_distribution(prior, "prior")


def _validate_detection(detection: Sequence[float], size: int) -> np.ndarray:
    array = _as_probability_vector(detection, "detection")
    if array.size != size:
        raise ValueError(f"detection has {array.size} cells but expected {size}")
    return array


def _validate_cost(cost: Sequence[float] | None, size: int) -> np.ndarray:
    if cost is None:
        return np.ones(size)
    array = np.array(cost, dtype=float)
    if array.ndim != 1 or array.size != size:
        raise ValueError(f"cost must be a one-dimensional sequence of {size} values")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError("cost must contain finite values > 0")
    return array


def _validate_cell(cell: int, size: int) -> int:
    if isinstance(cell, bool) or not isinstance(cell, (int, np.integer)):
        raise ValueError(f"cell must be an integer index, got {cell!r}")
    if not 0 <= cell < size:
        raise ValueError(f"cell {cell} out of range for {size} cells")
    return int(cell)


_MAX_COUNT = int(np.iinfo(np.int64).max)


def _validate_counts(counts: Sequence[int], size: int) -> np.ndarray:
    array = np.array(counts)
    if array.ndim != 1 or array.size != size:
        raise ValueError(f"counts must be a one-dimensional sequence of {size} values")
    if array.dtype.kind in "iu":
        if np.any(array < 0) or np.any(array > _MAX_COUNT):
            raise ValueError(f"counts must be integers in [0, {_MAX_COUNT}]")
    elif array.dtype.kind == "f":
        # float(_MAX_COUNT) rounds up to 2**63, which is not representable as a signed 64-bit integer
        if not np.all(np.isfinite(array)) or np.any(array != np.floor(array)) or np.any(array < 0) or np.any(array >= 2.0**63):
            raise ValueError(f"counts must be integers in [0, {_MAX_COUNT}]")
    else:
        raise ValueError("counts must be non-negative integers")
    return array.astype(np.int64)


def counts_from_sequence(looks: Sequence[int], size: int) -> np.ndarray:
    """Number of searches ``n_j`` of each cell in a sequence of searched cell indices."""
    counts = np.zeros(size, dtype=int)
    for cell in looks:
        counts[_validate_cell(cell, size)] += 1
    return counts


def _survival(detection: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """``(1 - d_j)**n_j`` per cell as ``exp(n_j * log1p(-d_j))``, with ``n_j = 0`` giving 1 and ``d_j = 1, n_j > 0`` giving 0.

    ``log1p`` keeps tiny ``d_j`` (for which ``1 - d_j`` rounds to 1) accurate.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log_survival = np.where(counts == 0, 0.0, counts * np.log1p(-detection))
    return np.exp(log_survival)


def update_after_failed_search(posterior: Sequence[float], detection: Sequence[float], cell: int) -> np.ndarray:
    """Posterior after one failed search of ``cell``.

    The weight of ``cell`` becomes ``q[cell] * (1 - d[cell])``, all other
    weights are unchanged, and the result is renormalised to sum to 1.

    Raises ``ValueError`` if the failure is impossible (the target is certainly
    in ``cell`` and ``d[cell] = 1``), because conditioning on a zero-probability
    event is undefined.
    """
    q = _validate_distribution(posterior, "posterior")
    d = _validate_detection(detection, q.size)
    k = _validate_cell(cell, q.size)
    weights = q.copy()
    weights[k] *= 1.0 - d[k]
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("a failed search of this cell has probability zero; the posterior is undefined")
    return weights / total


def posterior_after_search_sequence(prior: Sequence[float], detection: Sequence[float], looks: Sequence[int]) -> np.ndarray:
    """Posterior after failed searches of ``looks`` (cell indices).

    Mathematically identical to applying :func:`update_after_failed_search`
    once per search, and computed from the counts in log space. Updating one
    search at a time in floating point can underflow a cell's probability to
    exactly zero after a few hundred failures and then never recover it, which
    would make the result depend on the order of the searches.
    """
    p = _validate_prior(prior)
    return posterior_from_counts(p, detection, counts_from_sequence(looks, p.size))


def posterior_from_counts(prior: Sequence[float], detection: Sequence[float], counts: Sequence[int]) -> np.ndarray:
    """Closed-form posterior ``p_j (1 - d_j)**n_j / sum_i p_i (1 - d_i)**n_i`` after failed searches.

    Equal to repeated :func:`update_after_failed_search` for any ordering with
    the same counts. Computed from log-weights so it does not underflow after
    many failures. Raises ``ValueError`` if the all-failures outcome has
    probability zero.
    """
    p = _validate_prior(prior)
    d = _validate_detection(detection, p.size)
    n = _validate_counts(counts, p.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_survival = np.where(n == 0, 0.0, n * np.log1p(-d))  # d = 1 and n > 0 gives -inf
        log_weights = np.log(p) + log_survival
    peak = float(np.max(log_weights))
    if not math.isfinite(peak):
        raise ValueError("the all-failures outcome has probability zero; the posterior is undefined")
    weights = np.exp(log_weights - peak)
    return weights / float(weights.sum())


def probability_of_no_detection(prior: Sequence[float], detection: Sequence[float], counts: Sequence[int]) -> float:
    """Unconditional ``P(not detected) = sum_j p_j (1 - d_j)**n_j``."""
    p = _validate_prior(prior)
    d = _validate_detection(detection, p.size)
    n = _validate_counts(counts, p.size)
    return float(np.sum(p * _survival(d, n)))


def cumulative_detection_probability(prior: Sequence[float], detection: Sequence[float], counts: Sequence[int]) -> float:
    """Unconditional probability of detecting the target within the searches in ``counts``.

    ``P = sum_j p_j [1 - (1 - d_j)**n_j]``, which equals
    ``1 - sum_j p_j (1 - d_j)**n_j`` for a prior summing to 1. Each bracket is
    evaluated as ``-expm1(n_j * log1p(-d_j))`` so that it stays accurate for
    tiny ``d_j``, where ``1 - (1 - d_j)**n_j`` would round to zero. It depends
    only on the counts ``n_j``, never on the order of the searches.
    """
    p = _validate_prior(prior)
    d = _validate_detection(detection, p.size)
    n = _validate_counts(counts, p.size)
    total = 0.0
    for p_j, d_j, n_j in zip(p, d, n):
        if n_j == 0 or p_j == 0.0:
            continue
        miss_free = 1.0 if d_j == 1.0 else -math.expm1(n_j * math.log1p(-d_j))
        total += p_j * miss_free
    return float(min(total, 1.0))


def cumulative_detection_probability_for_sequence(prior: Sequence[float], detection: Sequence[float], looks: Sequence[int]) -> float:
    """:func:`cumulative_detection_probability` for a sequence of searched cell indices."""
    p = _validate_prior(prior)
    return cumulative_detection_probability(p, detection, counts_from_sequence(looks, p.size))


def conditional_next_look_success(posterior: Sequence[float], detection: Sequence[float], cell: int) -> float:
    """``q[cell] * d[cell]``: P(next search of ``cell`` succeeds | all previous searches failed).

    This is a **conditional** probability. It is *not* the increase in the
    unconditional cumulative detection probability; see
    :func:`unconditional_detection_increment`.
    """
    q = _validate_distribution(posterior, "posterior")
    d = _validate_detection(detection, q.size)
    k = _validate_cell(cell, q.size)
    return float(q[k] * d[k])


def unconditional_detection_increment(prior: Sequence[float], detection: Sequence[float], counts: Sequence[int], cell: int) -> float:
    """Increase in cumulative detection probability from one more search of ``cell``.

    Equals ``p_k (1 - d_k)**n_k d_k``: the (unconditional) probability that
    this search is the first to detect the target. It equals
    ``(1 - P_before) * conditional_next_look_success`` where ``P_before`` is
    the cumulative detection probability before the search.
    """
    p = _validate_prior(prior)
    d = _validate_detection(detection, p.size)
    n = _validate_counts(counts, p.size)
    k = _validate_cell(cell, p.size)
    return float(p[k] * _survival(d, n)[k] * d[k])


def next_look_priority(posterior: Sequence[float], detection: Sequence[float], cost: Sequence[float] | None = None) -> np.ndarray:
    """Greedy priority ``posterior[j] * d[j] / cost[j]`` for every cell (unit costs if ``cost`` is None)."""
    q = _validate_distribution(posterior, "posterior")
    d = _validate_detection(detection, q.size)
    c = _validate_cost(cost, q.size)
    return q * d / c


def select_next_cell(posterior: Sequence[float], detection: Sequence[float], cost: Sequence[float] | None = None) -> int:
    """Index of the cell with the largest :func:`next_look_priority`; ties go to the lowest index.

    Raises ``ValueError`` if every priority is zero (no search can detect the target).
    """
    priority = next_look_priority(posterior, detection, cost)
    best = int(np.argmax(priority))
    if priority[best] <= 0.0:
        raise ValueError("no cell has positive priority: no search can detect the target")
    return best


@dataclass(frozen=True)
class SearchStep:
    """One step of a greedy plan along the all-failures path."""

    cell: int
    conditional_success: float  # P(this search succeeds | previous searches failed)
    unconditional_increment: float  # P(this search is the first detection)
    cumulative_probability: float  # P(detected by the end of this search)
    cumulative_cost: float


def greedy_search_plan(
    prior: Sequence[float],
    detection: Sequence[float],
    cost: Sequence[float] | None = None,
    max_looks: int | None = None,
    budget: float | None = None,
) -> list[SearchStep]:
    """Greedy sequence of searches along the all-failures path.

    Each step searches the cell with the largest ``posterior * d / cost`` among
    the cells whose cost fits in the remaining ``budget`` (all cells if there is
    no budget), then updates the posterior as if that search failed. Stops after
    ``max_looks`` steps, when nothing affordable has positive priority, or when
    the all-failures outcome becomes impossible. At least one of ``max_looks``
    and ``budget`` is required.

    This is a *myopic* heuristic: it maximises the immediate payoff per unit
    cost, which is not guaranteed to maximise the cumulative detection
    probability for a finite budget with unequal costs.
    """
    if max_looks is None and budget is None:
        raise ValueError("provide max_looks and/or budget")
    if max_looks is not None and (isinstance(max_looks, bool) or not isinstance(max_looks, (int, np.integer)) or max_looks < 0):
        raise ValueError("max_looks must be a non-negative integer")
    if budget is not None and (not math.isfinite(budget) or budget < 0.0):
        raise ValueError("budget must be finite and >= 0")

    p = _validate_prior(prior)
    d = _validate_detection(detection, p.size)
    c = _validate_cost(cost, p.size)
    posterior = p
    counts = np.zeros(p.size, dtype=int)
    spent = 0.0
    cumulative = 0.0
    plan: list[SearchStep] = []
    while max_looks is None or len(plan) < max_looks:
        affordable = np.ones(p.size, dtype=bool) if budget is None else spent + c <= budget * (1.0 + 1e-12)
        priority = np.where(affordable, posterior * d / c, 0.0)
        cell = int(np.argmax(priority))
        if priority[cell] <= 0.0:
            break
        conditional = float(posterior[cell] * d[cell])
        increment = float(p[cell] * _survival(d, counts)[cell] * d[cell])
        counts[cell] += 1
        spent += float(c[cell])
        cumulative = cumulative_detection_probability(p, d, counts)
        plan.append(SearchStep(cell, conditional, increment, cumulative, spent))
        try:
            posterior = update_after_failed_search(posterior, d, cell)
        except ValueError:
            break  # the all-failures path now has probability zero: detection is certain by now
    return plan
