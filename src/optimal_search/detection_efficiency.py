"""Detection-efficiency objectives for single-pulse Swerling I/II detection.

This module investigates how "detection efficiency" varies with SNR for the
model in :mod:`optimal_search.radar_detection`, ``Pd(s) = Pfa ** (1 / (1 + s))``,
with ``s`` the linear mean SNR. Write ``L = ln(1 / Pfa)`` so that
``Pd(s) = exp(-L / (1 + s))``.

Two objectives are studied. They are different objectives and must not be
conflated.

Historical objective
--------------------
``E(s) = Pd(s) / s``, detection probability per unit linear SNR. ("Historical"
is this project's label; ``docs/historical-source-verification.md`` shows that
``Pd / s`` is only the single-look special case of the source's payoff ratio,
not its optimal detection condition.)

* Boundary behaviour: ``Pd(0) = Pfa > 0``, so ``E(s) ~ Pfa / s`` and
  ``E -> +infinity`` as ``s -> 0+``. **E has no global maximum on (0, inf)**;
  its supremum is infinite and is "attained" only in a degenerate limit with
  no signal at all, where "detection" is just a false alarm. As ``s -> inf``,
  ``E -> 0``.
* Interior extrema (closed form). ``d ln E / ds = L / (1 + s)**2 - 1 / s``,
  which vanishes when ``s**2 + (2 - L) s + 1 = 0``. Real roots exist iff
  ``L >= 4`` (``Pfa <= exp(-4) ~ 0.0183``); they are local extrema only for
  ``L > 4`` (``Pfa < exp(-4)``), and then::

      s_lo, s_hi = ((L - 2) -/+ sqrt(L (L - 4))) / 2,   s_lo * s_hi = 1

  ``E`` decreases on ``(0, s_lo)``, increases on ``(s_lo, s_hi)`` and decreases
  on ``(s_hi, inf)``. Hence ``s_lo`` is a **local minimum** and ``s_hi`` is a
  **local maximum**; because ``s_lo * s_hi = 1`` they are symmetric about
  0 dB. For ``L < 4`` there is no stationary point and ``E`` is monotonically
  decreasing. At ``L = 4`` the roots merge at ``s = 1``: a stationary
  inflection, not an extremum.
* ``s_hi`` is the *global* maximum only on windows whose lower bound is above
  a crossover SNR ``s_c < s_lo`` where ``E(s_c) = E(s_hi)``. On any window
  reaching below ``s_c`` the window maximum sits on the lower boundary. This
  is a statement about the arbitrary window, not about the physics.

Baseline-corrected objective
----------------------------
``G(s) = (Pd(s) - Pfa) / s``, *excess* detection probability per unit SNR.

* Limit at zero is finite: ``G(0+) = Pd'(0) = Pfa * L``. (Computed stably as
  ``Pfa * expm1(L s / (1 + s)) / s``, algebraically identical to
  ``(Pd - Pfa) / s``.) ``G -> 0`` as ``s -> inf``.
* ``G`` is positive for all ``s > 0``. For ``L > 2`` it increases away from
  ``s = 0`` (``d ln G / ds -> L/2 - 1 > 0`` there), so the boundary is not a
  maximum and the maximum is interior. In the variable ``u = s / (1 + s)``
  the stationarity condition is ``L e**(L u) / expm1(L u) = 1/u + 1/(1-u)``,
  solved here with a root finder. This is not closed-form.
* Uniqueness of the stationary point is checked numerically (one sign change
  of the derivative on a fine grid for the Pfa values tested), not proven.
* Since ``G = E - Pfa / s``, at ``s_hi`` we have ``G' = Pfa / s_hi**2 > 0``, so
  the baseline-corrected optimum lies strictly above ``s_hi``.

Is the baseline correction physically justified?
------------------------------------------------
Partly, as a modelling choice, not as something derived from the model.
``Pd - Pfa`` is the increase in the declaration probability caused by the
target's presence relative to the no-target baseline; for a binary decision it
equals Youden's J statistic (true-positive rate minus false-positive rate).
It removes the unphysical "free detections" at zero power. Dividing by linear
SNR additionally assumes that the cost of a measurement is proportional to
SNR (transmit energy), which is an assumption of the *objective*, not of the
radar model. Other corrections, e.g. ``(Pd - Pfa) / (1 - Pfa)``, are equally
arguable. The correction is **not** the historical objective, and it is not
used here to reproduce any published value.

Assumptions and limitations
---------------------------
* Everything inherits the assumptions of :mod:`optimal_search.radar_detection`
  (single pulse, exponential target power, Gaussian noise, square-law
  detection, fixed threshold).
* The objectives take SNR in linear units; dB appears only for reporting.
* Numerical searches run on a finite dB window. Interior extrema are detected
  by a grid scan and refined with a bounded scalar minimiser; extrema closer
  together than the grid spacing could be missed. Boundary values are
  reported separately and never relabelled as optima.
* Agreement or disagreement of a numerical optimum with a published value
  does not validate or refute the published derivation; the published
  objective and SNR definition may differ from the ones used here.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from optimal_search.radar_detection import detection_probability, snr_db_to_linear

Objective = Callable[[float, float], float]

DEFAULT_LOWER_DB = -60.0
DEFAULT_UPPER_DB = 60.0
DEFAULT_GRID_POINTS = 4001


def _check_snr_positive(snr_linear: float) -> None:
    if not math.isfinite(snr_linear) or snr_linear <= 0.0:
        raise ValueError(f"snr_linear must be finite and > 0, got {snr_linear!r}")


def _log_inverse_pfa(pfa: float) -> float:
    detection_probability(0.0, pfa)  # validates pfa
    return -math.log(pfa)


def historical_efficiency(snr_linear: float, pfa: float) -> float:
    """Historical objective ``Pd(SNR) / SNR`` (SNR linear, strictly positive)."""
    _check_snr_positive(snr_linear)
    return detection_probability(snr_linear, pfa) / snr_linear


def baseline_corrected_efficiency(snr_linear: float, pfa: float) -> float:
    """Baseline-corrected objective ``(Pd(SNR) - Pfa) / SNR``.

    Evaluated as ``Pd * (1 - exp(-x)) / s`` with ``x = L s / (1 + s)`` and
    ``L = ln(1/Pfa)``. This equals ``(Pd - Pfa) / s`` exactly, because
    ``Pfa / Pd = exp(-x)``, but it avoids catastrophic cancellation for small
    ``s`` (``-expm1``) and never evaluates a large positive exponential, so it
    cannot overflow for tiny ``Pfa``.
    """
    _check_snr_positive(snr_linear)
    log_inv = _log_inverse_pfa(pfa)
    x = log_inv * snr_linear / (1.0 + snr_linear)
    return detection_probability(snr_linear, pfa) * -math.expm1(-x) / snr_linear


def baseline_corrected_limit_at_zero(pfa: float) -> float:
    """Limit of ``(Pd - Pfa) / SNR`` as SNR -> 0+, equal to ``Pfa * ln(1/Pfa)``."""
    return pfa * _log_inverse_pfa(pfa)


def historical_stationary_snrs(pfa: float) -> tuple[float, ...]:
    """Closed-form stationary points ``(s_lo, s_hi)`` of ``Pd / SNR``, linear SNR.

    For ``Pfa < exp(-4)``, ``s_lo`` is a local minimum and ``s_hi`` a local
    maximum, with ``s_lo * s_hi = 1``. At ``Pfa = exp(-4)`` the two roots merge
    at ``s = 1`` and that point is a stationary *inflection*, not an extremum.
    Returns an empty tuple when ``Pfa > exp(-4)``, where the objective has no
    stationary point.
    """
    log_inv = _log_inverse_pfa(pfa)
    if log_inv < 4.0:
        return ()
    s_hi = ((log_inv - 2.0) + math.sqrt(log_inv * (log_inv - 4.0))) / 2.0
    return (1.0 / s_hi, s_hi)  # product of the roots is exactly 1


def _inverse_survival_excess(z: float) -> float:
    """``1 / (1 - exp(-z)) - 1 / z``, which tends to 1/2 as ``z -> 0``.

    Uses a series for small ``z`` (where the difference cancels) and bounded
    exponentials elsewhere.
    """
    if z < 1e-3:
        return 0.5 + z / 12.0 - z**3 / 720.0
    return 1.0 / -math.expm1(-z) - 1.0 / z


def baseline_corrected_optimum_snr(pfa: float) -> float:
    """Linear SNR maximising ``(Pd - Pfa) / SNR``, by root-finding.

    Solves ``L e**(L u) / expm1(L u) - 1/u - 1/(1-u) = 0`` for
    ``u = s / (1 + s)``, a formulation independent of the SNR-domain
    derivative used elsewhere. Requires ``L = ln(1/Pfa) > 2``; otherwise the
    objective's maximum is not interior. The equation is evaluated as
    ``L * h(L u) - 1/(1-u)`` with ``h(z) = 1/(1 - e**-z) - 1/z`` so that it is
    accurate as ``u -> 0`` and cannot overflow.

    Raises ``ValueError`` if ``Pfa`` is so close to ``exp(-2)`` that the
    optimum is below the smallest resolvable ``u`` (1e-12).
    """
    log_inv = _log_inverse_pfa(pfa)
    if log_inv <= 2.0:
        raise ValueError("pfa must be < exp(-2) for an interior maximum")

    def stationarity(u: float) -> float:
        return log_inv * _inverse_survival_excess(log_inv * u) - 1.0 / (1.0 - u)

    u_min, u_max = 1e-12, 1.0 - 1e-9
    if stationarity(u_min) <= 0.0:
        raise ValueError("pfa is too close to exp(-2): the optimum is below the resolvable SNR")
    u = brentq(stationarity, u_min, u_max, xtol=1e-16, rtol=1e-14)
    return u / (1.0 - u)


def historical_boundary_crossover_snr(pfa: float) -> float:
    """SNR ``s_c < s_lo`` where ``E(s_c) = E(s_hi)`` for the historical objective.

    On a window whose lower bound exceeds ``s_c`` (and whose upper bound
    exceeds ``s_hi``) the local maximum at ``s_hi`` is the window's global
    maximum; on a window reaching below ``s_c`` the global maximum is the
    lower boundary. Requires ``Pfa < exp(-4)``: at equality the stationary
    point is an inflection, so there is no local maximum and no crossover.
    """
    if _log_inverse_pfa(pfa) <= 4.0:
        raise ValueError("pfa must be < exp(-4) for the objective to have a local maximum")
    s_lo, s_hi = historical_stationary_snrs(pfa)
    target = historical_efficiency(s_hi, pfa)
    # E(s) >= Pfa / s, so E exceeds the target for s < Pfa / target.
    lower = 0.5 * pfa / target
    return brentq(
        lambda s: historical_efficiency(s, pfa) - target,
        lower,
        s_lo,
        xtol=1e-300,
        rtol=1e-14,
    )


@dataclass(frozen=True)
class Point:
    """A point on an objective curve."""

    snr_db: float
    value: float

    @property
    def snr_linear(self) -> float:
        return snr_db_to_linear(self.snr_db)


@dataclass(frozen=True)
class WindowAnalysis:
    """Extrema of an objective on a finite dB window, boundaries kept separate."""

    lower_boundary: Point
    upper_boundary: Point
    local_maxima: tuple[Point, ...]
    local_minima: tuple[Point, ...]

    @property
    def window_maximum(self) -> tuple[Point, str]:
        """Largest value on the window and where it sits.

        The location is ``"lower boundary"``, ``"upper boundary"`` or
        ``"interior"``.
        """
        candidates = [(self.lower_boundary, "lower boundary"), (self.upper_boundary, "upper boundary")]
        candidates += [(p, "interior") for p in self.local_maxima]
        return max(candidates, key=lambda c: c[0].value)


def analyze_objective(
    objective: Objective,
    pfa: float,
    lower_db: float = DEFAULT_LOWER_DB,
    upper_db: float = DEFAULT_UPPER_DB,
    n_grid: int = DEFAULT_GRID_POINTS,
) -> WindowAnalysis:
    """Locate interior local extrema of a positive objective on ``[lower_db, upper_db]``.

    The objective is scanned on a uniform dB grid; every strict interior grid
    maximum (minimum) is refined with ``scipy.optimize.minimize_scalar``
    (bounded) on the neighbouring grid interval, working with ``log`` of the
    objective for conditioning. The window endpoints are reported as
    boundary points and are never counted as local extrema.
    """
    if not (math.isfinite(lower_db) and math.isfinite(upper_db) and lower_db < upper_db):
        raise ValueError("require finite lower_db < upper_db")
    if n_grid < 3:
        raise ValueError("n_grid must be at least 3")

    grid = np.linspace(lower_db, upper_db, n_grid)
    values = np.array([objective(snr_db_to_linear(x), pfa) for x in grid])

    def log_objective(x_db: float) -> float:
        return math.log(objective(snr_db_to_linear(x_db), pfa))

    maxima: list[Point] = []
    minima: list[Point] = []
    i = 1
    while i < n_grid - 1:
        # Treat a run of equal samples grid[i..j] as one candidate, so that tied
        # neighbours at a peak are still refined (exactly once).
        j = i
        while j + 1 < n_grid and values[j + 1] == values[i]:
            j += 1
        if j < n_grid - 1:
            left, level, right = values[i - 1], values[i], values[j + 1]
            if level > left and level > right:
                sign, bucket = -1.0, maxima
            elif level < left and level < right:
                sign, bucket = 1.0, minima
            else:
                sign, bucket = 0.0, None
            if bucket is not None:
                result = minimize_scalar(
                    lambda x, sign=sign: sign * log_objective(x),
                    bounds=(grid[i - 1], grid[j + 1]),
                    method="bounded",
                    options={"xatol": 1e-12},
                )
                bucket.append(Point(float(result.x), objective(snr_db_to_linear(result.x), pfa)))
        i = j + 1

    return WindowAnalysis(
        lower_boundary=Point(float(grid[0]), float(values[0])),
        upper_boundary=Point(float(grid[-1]), float(values[-1])),
        local_maxima=tuple(maxima),
        local_minima=tuple(minima),
    )
