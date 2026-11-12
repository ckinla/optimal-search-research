"""Monte Carlo simulation of a fixed spatial-search plan.

What this is, and is not
------------------------
This module *estimates* probabilities and costs by randomised experiment. A Monte Carlo
estimate carries sampling error and is **not a proof**: agreement with an analytical formula
within the stated confidence interval is evidence that formula and simulation describe the
same model, and a persistent disagreement beyond sampling error would indicate a defect in
one of them; it can never establish that a formula is exactly right.

The engine is deliberately separate from the mathematical solvers. It imports nothing from
:mod:`optimal_search.spatial_search` or :mod:`optimal_search.exact_spatial_search`, and it does not use
their formulas: detection is simulated attempt by attempt with Bernoulli draws, so the
analytical ``(1 - d)**n`` never appears in it. A *plan* is just a sequence of cell indices,
for example ``[step.cell for step in greedy_search_plan(...)]`` or
``solve_exact(...).sequence``.

Model
-----
One stationary target is in exactly one of ``M`` cells, drawn from ``prior``. The plan is
executed look by look. A look in cell ``j`` costs ``cost[j]`` whether or not it succeeds. If
the target is in ``j`` the look detects it with probability ``detection[j]``, independently of
every other look; if the target is elsewhere the look cannot detect it. The search stops
immediately on detection or when the plan is exhausted.

Randomness and reproducibility
------------------------------
Each trial uses NumPy's :class:`numpy.random.Generator` seeded explicitly
(``numpy.random.default_rng(seed)``); an unseeded run is rejected. The target cells of all
trials are drawn first (inverse-CDF on the cumulative prior), then one uniform per
(trial, look) is drawn in row-major order and compared with the look's detection
probability. Because NumPy fills these draws sequentially, the result for a given seed does
not depend on how the trials are chunked for memory. NumPy does not promise identical
streams across versions, so exact values are reproducible for a pinned NumPy (this project
pins it in ``uv.lock``); the tests therefore check statistical properties and exact
internal invariants, not hard-coded random outputs.

Confidence intervals
--------------------
* Detection probability: the trials are independent Bernoulli observations. The default is the
  **Wilson score interval**; the **Clopper-Pearson** ("exact") interval is also available.
  Wilson is approximate, with coverage that fluctuates around the nominal level (worst for very
  small ``n`` or ``p`` near 0 or 1); Clopper-Pearson is conservative (coverage at least nominal,
  often noticeably higher).
* Mean looks and mean cost: a **normal approximation** ``mean +/- z * s / sqrt(n)`` with the
  sample standard deviation ``s`` (``ddof = 1``), justified by the central limit theorem for
  large ``n``. These variables are bounded and often skewed or nearly degenerate, so it can be
  poor for small ``n``; a zero sample variance gives a zero-width interval.
* All intervals describe *sampling* error only. They say nothing about whether the model
  (stationary target, independent attempts, constant detection probability) is right, and a
  95% interval fails to contain the true value in about 1 run out of 20 by design.
"""

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from scipy.stats import beta

DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_MAX_CHUNK_ELEMENTS = 4_000_000
_PRIOR_TOLERANCE = 1e-9
_BUDGET_RTOL = 1e-12
_METHODS = ("wilson", "clopper-pearson")


@dataclass(frozen=True, eq=False)
class SimulationResult:
    """Per-trial records and aggregate statistics of a simulated search plan.

    Per-trial arrays have length ``trials``. ``detection_cell`` and ``detection_look`` are ``-1``
    when the target was not detected; otherwise ``plan[detection_look] == detection_cell``, where
    ``detection_look`` is the 0-based index of the successful look in the plan and
    ``looks == detection_look + 1`` (the search stopped there).
    """

    trials: int
    plan: tuple[int, ...]
    cells: int
    target_cell: np.ndarray  # truth: the cell the target was placed in
    detected: np.ndarray  # bool
    looks: np.ndarray  # number of looks performed
    cost: np.ndarray  # search cost actually incurred
    detection_cell: np.ndarray
    detection_look: np.ndarray
    detections: int
    detection_probability: float
    mean_looks: float
    mean_cost: float
    mean_looks_se: float  # nan if trials == 1
    mean_cost_se: float  # nan if trials == 1
    target_counts: np.ndarray  # per cell: trials in which the target was placed there
    detections_by_cell: np.ndarray  # per cell: trials in which the target was there and detected

    def detection_confidence_interval(
        self, level: float = DEFAULT_CONFIDENCE_LEVEL, method: str = "wilson"
    ) -> tuple[float, float]:
        """Confidence interval for the probability of detection (see the module docstring)."""
        return proportion_confidence_interval(self.detections, self.trials, level, method)

    def mean_cost_confidence_interval(self, level: float = DEFAULT_CONFIDENCE_LEVEL) -> tuple[float, float]:
        """Normal-approximation confidence interval for the mean cost incurred."""
        return _mean_interval(self.mean_cost, self.mean_cost_se, level)

    def mean_looks_confidence_interval(self, level: float = DEFAULT_CONFIDENCE_LEVEL) -> tuple[float, float]:
        """Normal-approximation confidence interval for the mean number of looks."""
        return _mean_interval(self.mean_looks, self.mean_looks_se, level)


# --- validation ---------------------------------------------------------------------------------------


def _is_integer(value: object) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _probability_vector(values: Sequence[float], name: str) -> np.ndarray:
    try:
        array = np.array(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a sequence of numbers") from error
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return array


def _validate_confidence_level(level: float) -> float:
    if isinstance(level, bool) or not isinstance(level, (int, float)) or not 0.0 < level < 1.0:
        raise ValueError("confidence level must be a number strictly between 0 and 1")
    return float(level)


def _validate_problem(prior, detection, cost, plan, budget):
    if isinstance(prior, (str, bytes)) or isinstance(detection, (str, bytes)) or isinstance(cost, (str, bytes)):
        raise ValueError("prior, detection and cost must be sequences of numbers")
    p = _probability_vector(prior, "prior")
    if not math.isclose(float(p.sum()), 1.0, rel_tol=0.0, abs_tol=_PRIOR_TOLERANCE):
        raise ValueError(f"prior must sum to 1 (got {float(p.sum())!r})")
    d = _probability_vector(detection, "detection")
    if d.size != p.size:
        raise ValueError(f"detection has {d.size} cells but prior has {p.size}")
    try:
        c = np.array(cost, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("cost must be a sequence of numbers") from error
    if c.ndim != 1 or c.size != p.size:
        raise ValueError(f"cost must be a one-dimensional sequence of {p.size} values")
    if not np.all(np.isfinite(c)) or np.any(c <= 0.0):
        raise ValueError("cost must contain finite values > 0")
    try:
        looks = list(plan)
    except TypeError as error:
        raise ValueError("plan must be a sequence of cell indices") from error
    if any(not _is_integer(cell) or not 0 <= cell < p.size for cell in looks):
        raise ValueError(f"every plan entry must be an integer cell index in [0, {p.size})")
    plan_array = np.array(looks, dtype=np.int64)
    if budget is not None:
        if isinstance(budget, bool) or not isinstance(budget, (int, float, np.floating, np.integer)) or not math.isfinite(budget) or budget < 0:
            raise ValueError("budget must be a finite number >= 0")
        plan_cost = math.fsum(c[plan_array]) if plan_array.size else 0.0
        if plan_cost > float(budget) * (1.0 + _BUDGET_RTOL):
            raise ValueError(f"the plan costs {plan_cost!r}, which exceeds the budget {float(budget)!r}")
    return p, d, c, plan_array


def _validate_trials(trials: int) -> int:
    if not _is_integer(trials) or trials < 1:
        raise ValueError("trials must be a positive integer")
    return int(trials)


def _validate_seed(seed: object) -> object:
    if isinstance(seed, np.random.SeedSequence):
        return seed
    if not _is_integer(seed) or seed < 0:
        raise ValueError("seed must be an explicit non-negative integer or a numpy SeedSequence")
    return int(seed)


# --- confidence intervals -------------------------------------------------------------------------------------


def proportion_confidence_interval(
    successes: int, trials: int, level: float = DEFAULT_CONFIDENCE_LEVEL, method: str = "wilson"
) -> tuple[float, float]:
    """Confidence interval for a binomial proportion (``method``: ``"wilson"`` or ``"clopper-pearson"``).

    Wilson score: ``(p + z^2/2n +/- z sqrt(p(1-p)/n + z^2/4n^2)) / (1 + z^2/n)``.
    Clopper-Pearson: ``Beta(alpha/2; k, n-k+1)`` and ``Beta(1-alpha/2; k+1, n-k)`` quantiles, with
    the lower end 0 when ``k = 0`` and the upper end 1 when ``k = n``.
    """
    level = _validate_confidence_level(level)
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    if not _is_integer(trials) or trials < 1:
        raise ValueError("trials must be a positive integer")
    if not _is_integer(successes) or not 0 <= successes <= trials:
        raise ValueError("successes must be an integer in [0, trials]")
    k, n = int(successes), int(trials)
    alpha = 1.0 - level
    if method == "wilson":
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        p_hat = k / n
        denominator = 1.0 + z * z / n
        center = (p_hat + z * z / (2.0 * n)) / denominator
        half = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) / denominator
        return max(0.0, center - half), min(1.0, center + half)
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def _mean_interval(mean: float, standard_error: float, level: float) -> tuple[float, float]:
    level = _validate_confidence_level(level)
    if math.isnan(standard_error):
        return (math.nan, math.nan)
    z = NormalDist().inv_cdf(1.0 - (1.0 - level) / 2.0)
    return mean - z * standard_error, mean + z * standard_error


# --- simulation ------------------------------------------------------------------------------------------------------


def simulate_search(
    prior: Sequence[float],
    detection: Sequence[float],
    cost: Sequence[float],
    plan: Sequence[int],
    trials: int,
    seed: int | np.random.SeedSequence,
    budget: float | None = None,
    max_chunk_elements: int = DEFAULT_MAX_CHUNK_ELEMENTS,
) -> SimulationResult:
    """Simulate ``trials`` independent searches that execute ``plan`` until detection or exhaustion.

    Args:
        prior: probability that the target is in each cell (sums to 1).
        detection: per-look detection probability of each cell, given the target is there.
        cost: positive cost of one look in each cell.
        plan: the cells to look in, in order (for example a greedy or exact solver's plan).
        trials: number of independent trials (a positive integer).
        seed: explicit non-negative integer or ``numpy.random.SeedSequence``; ``None`` is rejected.
        budget: optional; if given, a plan whose total cost exceeds it is rejected, so every
            simulated cost is at most the budget.
        max_chunk_elements: memory control: at most this many uniforms are drawn at once, including when
            a single trial's plan is longer than the limit (it is then drawn in blocks). It does not change
            the result for a given seed. Memory is O(max_chunk_elements + trials + plan length): the
            per-trial records and the plan itself are not chunked.

    Returns:
        A :class:`SimulationResult` with per-trial records and aggregates.
    """
    p, d, c, plan_array = _validate_problem(prior, detection, cost, plan, budget)
    n = _validate_trials(trials)
    entropy = _validate_seed(seed)
    if not _is_integer(max_chunk_elements) or max_chunk_elements < 1:
        raise ValueError("max_chunk_elements must be a positive integer")

    rng = np.random.default_rng(entropy)
    cells = p.size
    length = int(plan_array.size)

    # 1. place the target: inverse-CDF sampling of the prior (zero-probability cells have zero width)
    cumulative = np.cumsum(p)
    cumulative[-1] = 1.0
    target = np.searchsorted(cumulative, rng.random(n), side="right").astype(np.int64)

    # 2. independent Bernoulli attempts, one uniform per (trial, look); only a look in the target's cell can succeed
    detected = np.zeros(n, dtype=bool)
    first_hit = np.full(n, -1, dtype=np.int64)
    if length:
        plan_detection = d[plan_array]
        if length <= max_chunk_elements:
            rows_per_chunk = max(1, max_chunk_elements // length)
            for start in range(0, n, rows_per_chunk):
                stop = min(start + rows_per_chunk, n)
                uniforms = rng.random((stop - start, length))
                hits = (plan_array[None, :] == target[start:stop, None]) & (uniforms < plan_detection[None, :])
                any_hit = hits.any(axis=1)
                detected[start:stop] = any_hit
                first_hit[start:stop] = np.where(any_hit, hits.argmax(axis=1), -1)
        else:
            # A single trial's plan is longer than the chunk limit: draw each trial's uniforms in consecutive
            # blocks of at most max_chunk_elements. Consecutive draws continue one stream, so this produces
            # exactly the values a whole-row draw would, and the result does not depend on the limit.
            for trial in range(n):
                for block_start in range(0, length, max_chunk_elements):
                    block_stop = min(block_start + max_chunk_elements, length)
                    uniforms = rng.random(block_stop - block_start)
                    block_hits = (plan_array[block_start:block_stop] == target[trial]) & (uniforms < plan_detection[block_start:block_stop])
                    if first_hit[trial] < 0 and block_hits.any():
                        first_hit[trial] = block_start + int(block_hits.argmax())
                detected[trial] = first_hit[trial] >= 0

    # 3. stop at the first success: looks performed and cost incurred (a look's cost is paid even if it fails)
    looks = np.where(detected, first_hit + 1, length).astype(np.int64)
    cost_through = np.concatenate(([0.0], np.cumsum(c[plan_array]))) if length else np.zeros(1)
    incurred = cost_through[looks]
    detection_cell = np.where(detected, target, -1).astype(np.int64)

    detections = int(detected.sum())
    se_looks = float(looks.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    se_cost = float(incurred.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    return SimulationResult(
        trials=n,
        plan=tuple(int(cell) for cell in plan_array),
        cells=cells,
        target_cell=target,
        detected=detected,
        looks=looks,
        cost=incurred,
        detection_cell=detection_cell,
        detection_look=first_hit,
        detections=detections,
        detection_probability=detections / n,
        mean_looks=float(looks.mean()),
        mean_cost=float(incurred.mean()),
        mean_looks_se=se_looks,
        mean_cost_se=se_cost,
        target_counts=np.bincount(target, minlength=cells),
        detections_by_cell=np.bincount(target[detected], minlength=cells),
    )


def simulate_replications(
    prior: Sequence[float],
    detection: Sequence[float],
    cost: Sequence[float],
    plan: Sequence[int],
    trials: int,
    replications: int,
    seed: int | np.random.SeedSequence,
    budget: float | None = None,
) -> list[SimulationResult]:
    """Independent replications with statistically independent streams spawned from one seed.

    ``numpy.random.SeedSequence(seed).spawn(replications)`` gives each replication its own
    non-overlapping stream, so the replications are independent and the whole set is reproducible
    from ``seed``. Used to check the calibration of the confidence intervals.
    """
    if not _is_integer(replications) or replications < 1:
        raise ValueError("replications must be a positive integer")
    checked = _validate_seed(seed)
    if isinstance(checked, np.random.SeedSequence):
        # spawn() advances the parent's child counter; rebuild it so repeated calls give identical children
        root = np.random.SeedSequence(entropy=checked.entropy, spawn_key=checked.spawn_key, pool_size=checked.pool_size)
    else:
        root = np.random.SeedSequence(checked)
    return [
        simulate_search(prior, detection, cost, plan, trials, child, budget=budget)
        for child in root.spawn(int(replications))
    ]
