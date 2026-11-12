"""Cumulative-detection efficiency for N independent single-pulse looks.

Everything in this module is an *independently derived* mathematical study of
the model in :mod:`optimal_search.radar_detection`; no parameter is fitted to any
published value. A comparison with a primary source's equations (stationarity
condition identical; objective ``H`` not printed there, equivalent to the
source's payoff ratio only at fixed ``Pc``) is in
``docs/historical-source-verification.md``.

Notation: ``s`` is linear SNR, ``L = ln(1 / Pfa)``, ``x = L / (1 + s)``,
``Pd(s) = exp(-x)``, and the *log-miss rate* is ``q(s) = -ln(1 - Pd(s))``.

Derivation of the cumulative objective
--------------------------------------
Assume ``N`` statistically independent, identically distributed looks, each at
SNR ``s``, and declare a detection if *any* look crosses its threshold. Then::

    Pc = 1 - (1 - Pd)**N    =>    N = -ln(1 - Pc) / q(s)

Taking total effort to be proportional to ``N * s`` (total transmitted energy)::

    effort = N * s = [-ln(1 - Pc)] / H(s),      H(s) = q(s) / s = -ln(1 - Pd) / s

For a *specified* ``Pc`` the numerator is a constant, so minimising effort over
SNR with ``N`` treated as a continuous variable is exactly maximising ``H``.
The optimal SNR does not depend on ``Pc``. (This was cross-checked by
minimising ``N * s(N)`` directly over ``N``, see the tests.)

Behaviour of H and where the stationary points come from
--------------------------------------------------------
``H'(s) = F(s) / s**2`` with ``F(s) = s q'(s) - q(s)``, ``F'(s) = s q''(s)``.
With ``P = exp(-x)`` one finds::

    q''(s) = P x (x - 2 (1 - e**-x)) / ((1 + s)**2 (1 - P)**2)

so ``sign q'' = sign(phi(x))`` with ``phi(x) = x - 2 (1 - e**-x)``. ``phi`` is
negative on ``(0, x*)`` and positive beyond, where ``x* ~ 1.5936`` is the
unique positive root of ``x = 2 (1 - e**-x)``. Hence ``q`` has exactly one
inflection, at ``s_i = L / x* - 1`` (convex below it, concave above), and
``F`` increases on ``(0, s_i)`` and decreases on ``(s_i, inf)``.

* **Boundary at s -> 0.** ``Pd(0) = Pfa > 0`` so ``q(0) = -ln(1 - Pfa) > 0``
  and ``H ~ q(0) / s -> +infinity``. As in the single-look case there is **no
  global maximum** on ``(0, inf)``. This boundary is unphysical: at zero SNR
  the "cumulative detection" is produced entirely by false alarms (total
  effort ``N s -> 0`` while ``Pc`` is reached by accumulating false alarms).
* **Boundary at s -> inf.** ``q ~ ln((1 + s) / L)`` so ``H ~ ln(s) / s -> 0``.
* **Interior.** ``F(0+) = -q(0) < 0`` and ``F -> -inf``, and ``F`` is
  unimodal, so ``F`` has 0, 1 (tangent) or 2 zeros. When there are two, the
  smaller ``s_lo`` is a **local minimum** of ``H`` and the larger ``s_hi`` a
  **local maximum**. For ``Pfa = 1e-5`` and ``1e-6`` there are two.

Baseline-corrected logarithmic objective
----------------------------------------
``G(s) = (q(s) - q0) / s`` with ``q0 = q(0) = -ln(1 - Pfa)``, i.e.
``[-ln(1 - Pd) + ln(1 - Pfa)] / s``. Since ``exp(-N q)`` is the probability of
*no* declaration in ``N`` looks with a target, and ``exp(-N q0)`` without one,
``N (q - q0) = ln[P(no declaration | noise) / P(no declaration | target)]``.
At a fixed value ``K`` of that log-ratio, ``N s = K / G(s)``: maximising ``G``
minimises effort for a fixed *excess* log-survival ratio. This is a modelling
choice (a different target than fixed ``Pc``), not a consequence of the model.

* ``G(0+) = q'(0) = Pfa L / (1 - Pfa)`` is finite.
* ``G' = F_b / s**2`` with ``F_b = F + q0``, ``F_b(0) = 0``, ``F_b' = s q''``.
  ``F_b`` therefore rises on ``(0, s_i)``, then falls to ``-inf``: it has
  **exactly one** positive zero, a unique stationary point that is the
  **global maximum** on ``(0, inf)`` (provable here, unlike the single-look
  baseline). It exists iff ``s_i > 0``, i.e. ``L > x*``.

Assumptions and limitations
---------------------------
* **Independence.** Looks are independent with identical ``Pd``. This is
  Swerling-II-like (target decorrelates between looks). For a Swerling I
  target, which is constant within a scan, looks within a scan are fully
  correlated and the product formula for ``Pc`` does not hold.
* **Fixed per-look Pfa.** ``Pfa`` is held fixed per look, so the cumulative
  false-alarm probability ``1 - (1 - Pfa)**N`` grows with ``N``. A design that
  fixes the *cumulative* false-alarm probability would change the per-look
  threshold with ``N`` and therefore the objective.
* **Effort model.** Effort is proportional to ``N * s`` (energy), with no
  per-look overhead and no coherent or non-coherent integration gain.
* **Continuous N.** ``N = -ln(1 - Pc) / q(s)`` is real. It is only physically
  meaningful for integer ``N >= 1``. At the continuous optimum ``N* < 1``
  whenever ``Pc < Pd(s_hi)`` (about 0.46), where the relaxation is
  infeasible. The effort at ``s_hi`` is a **local minimum** of
  ``effort(N) = N s(N)``, not a global lower bound: as ``N`` approaches the
  false-alarm-only value ``-ln(1 - Pc) / (-ln(1 - Pfa))`` the required SNR
  and the effort tend to 0 (the zero-SNR boundary above; for ``Pc = 0.9``,
  ``Pfa = 1e-5``, ``N = 230000`` costs about 22 versus about 52 at
  ``s_hi``). Comparisons with integer designs are therefore only meaningful
  over a finite, physically sensible range of ``N`` (see :func:`total_effort`).
* The inherited single-look assumptions of :mod:`optimal_search.radar_detection`
  (exponential target power, Gaussian noise, square-law detector).
"""

import math

from scipy.optimize import brentq

from optimal_search.radar_detection import detection_probability


def _phi(x: float) -> float:
    return x - 2.0 * (1.0 - math.exp(-x))


# Unique positive root of x = 2 (1 - exp(-x)); phi < 0 below it and > 0 above.
INFLECTION_X = brentq(_phi, 1.0, 3.0, xtol=1e-15, rtol=1e-15)


def _check_snr_positive(snr_linear: float) -> None:
    if not math.isfinite(snr_linear) or snr_linear <= 0.0:
        raise ValueError(f"snr_linear must be finite and > 0, got {snr_linear!r}")


def _log_inverse_pfa(pfa: float) -> float:
    detection_probability(0.0, pfa)  # validates pfa
    return -math.log(pfa)


def _q_from_x(x: float) -> float:
    """``-ln(1 - exp(-x))`` without cancellation for either small or large ``Pd``."""
    if x > math.log(2.0):  # Pd = exp(-x) < 1/2: log1p keeps precision for small Pd
        return -math.log1p(-math.exp(-x))
    return -math.log(-math.expm1(-x))  # Pd near 1: expm1 keeps precision in 1 - Pd


def log_miss_rate(snr_linear: float, pfa: float) -> float:
    """``q(s) = -ln(1 - Pd(s))``, the per-look log-miss rate (SNR linear, > 0).

    Evaluated from ``x = ln(1/Pfa) / (1 + s)`` with ``log1p`` when ``Pd`` is
    small and ``expm1`` when ``Pd`` is near 1, so that neither ``1 - Pd`` nor
    ``-ln(1 - Pd)`` loses precision.
    """
    _check_snr_positive(snr_linear)
    return _q_from_x(_log_inverse_pfa(pfa) / (1.0 + snr_linear))


def cumulative_efficiency(snr_linear: float, pfa: float) -> float:
    """Cumulative-detection objective ``H(s) = -ln(1 - Pd(s)) / s``."""
    return log_miss_rate(snr_linear, pfa) / snr_linear


def baseline_corrected_log_efficiency(snr_linear: float, pfa: float) -> float:
    """Baseline-corrected objective ``G(s) = [-ln(1-Pd) + ln(1-Pfa)] / s``.

    With ``excess = (Pd - Pfa) / (1 - Pfa)`` this is ``-ln(1 - excess) / s``.
    ``Pd - Pfa`` is computed as ``Pd * (1 - exp(-x))`` with
    ``x = ln(1/Pfa) s / (1 + s)`` (exact, and free of cancellation at small
    ``s`` and of overflow). While ``excess < 1/2`` the objective is
    ``-log1p(-excess) / s``; beyond that ``1 - excess`` loses precision (it
    rounds to 0 for large ``s``), so the stable ``log_miss_rate(s) - q0`` is
    used instead.
    """
    _check_snr_positive(snr_linear)
    log_inv = _log_inverse_pfa(pfa)
    x = log_inv * snr_linear / (1.0 + snr_linear)
    excess = detection_probability(snr_linear, pfa) * -math.expm1(-x) / (1.0 - pfa)
    if excess < 0.5:
        return -math.log1p(-excess) / snr_linear
    return (log_miss_rate(snr_linear, pfa) + math.log1p(-pfa)) / snr_linear


def baseline_log_limit_at_zero(pfa: float) -> float:
    """Limit of ``G(s)`` as ``s -> 0+``: ``q'(0) = Pfa * ln(1/Pfa) / (1 - Pfa)``."""
    return pfa * _log_inverse_pfa(pfa) / (1.0 - pfa)


def inflection_snr(pfa: float) -> float:
    """SNR ``s_i = ln(1/Pfa) / x* - 1`` where ``q(s)`` changes from convex to concave.

    Negative when ``ln(1/Pfa) <= x*`` (``q`` is then concave for all ``s > 0``).
    """
    return _log_inverse_pfa(pfa) / INFLECTION_X - 1.0


def _stationarity(snr_linear: float, log_inv: float) -> float:
    """``F(s) = s q'(s) - q(s)``, so that ``H'(s) = F(s) / s**2``."""
    x = log_inv / (1.0 + snr_linear)
    q_prime = math.exp(-x) * x / (1.0 + snr_linear) / -math.expm1(-x)
    return snr_linear * q_prime - _q_from_x(x)


def cumulative_stationary_snrs(pfa: float) -> tuple[float, ...]:
    """Stationary points ``(s_lo, s_hi)`` of ``H``: a local minimum, then a local maximum.

    Found by root finding on ``F(s) = s q'(s) - q(s)``, which is unimodal
    (increasing up to :func:`inflection_snr`, decreasing after). Returns an
    empty tuple if ``F`` never becomes positive, i.e. ``H`` has no interior
    stationary point.
    """
    log_inv = _log_inverse_pfa(pfa)
    s_i = inflection_snr(pfa)
    if s_i <= 0.0 or _stationarity(s_i, log_inv) <= 0.0:
        return ()
    lower = s_i
    while _stationarity(lower, log_inv) > 0.0:
        lower /= 10.0
    upper = 2.0 * s_i
    while _stationarity(upper, log_inv) > 0.0:
        upper *= 2.0
    kwargs = {"xtol": 1e-300, "rtol": 1e-14}
    s_lo = brentq(_stationarity, lower, s_i, args=(log_inv,), **kwargs)
    s_hi = brentq(_stationarity, s_i, upper, args=(log_inv,), **kwargs)
    return (s_lo, s_hi)


def baseline_log_optimum_snr(pfa: float) -> float:
    """Unique SNR maximising ``G(s)``, the root of ``F(s) + q0 = 0`` beyond the inflection.

    Requires ``ln(1/Pfa) > x*`` (``s_i > 0``); otherwise the maximum is not interior.
    """
    log_inv = _log_inverse_pfa(pfa)
    s_i = inflection_snr(pfa)
    if s_i <= 0.0:
        raise ValueError("pfa must be < exp(-x*) (about 0.2) for an interior maximum")
    q0 = -math.log1p(-pfa)

    def stationarity(s: float) -> float:
        return _stationarity(s, log_inv) + q0

    upper = 2.0 * s_i
    while stationarity(upper) > 0.0:
        upper *= 2.0
    return brentq(stationarity, s_i, upper, xtol=1e-300, rtol=1e-14)


def required_snr_for_looks(n_looks: float, pc: float, pfa: float) -> float:
    """Linear SNR per look needed for cumulative detection ``pc`` with ``n_looks`` looks.

    Inverts ``Pc = 1 - (1 - Pd)**N`` for ``Pd`` and then ``Pd = Pfa**(1/(1+s))``
    for ``s``. Raises ``ValueError`` if even ``s = 0`` (false alarms alone)
    would reach ``pc``, i.e. the required ``Pd`` does not exceed ``pfa``.
    """
    if not math.isfinite(n_looks) or n_looks <= 0.0:
        raise ValueError(f"n_looks must be finite and > 0, got {n_looks!r}")
    if not math.isfinite(pc) or not 0.0 < pc < 1.0:
        raise ValueError(f"pc must be strictly between 0 and 1, got {pc!r}")
    log_inv = _log_inverse_pfa(pfa)
    pd_required = -math.expm1(math.log1p(-pc) / n_looks)  # 1 - (1 - pc)**(1/N)
    if pd_required <= pfa:
        raise ValueError("n_looks is so large that false alarms alone reach pc")
    return log_inv / -math.log(pd_required) - 1.0


def total_effort(n_looks: float, pc: float, pfa: float) -> float:
    """Total effort ``N * s(N)`` (proportional to energy) to reach ``pc`` with ``n_looks`` looks.

    As a function of ``N`` this has a local minimum near the continuous optimum
    and tends to 0 as ``N`` approaches the false-alarm-only value, where
    ``s -> 0``; it is not bounded below by its local minimum.
    """
    return n_looks * required_snr_for_looks(n_looks, pc, pfa)


def continuous_optimal_looks(pc: float, pfa: float) -> float:
    """Continuous ``N* = -ln(1 - pc) / q(s_hi)`` at the local maximum of ``H``.

    This is the location of a local minimum of ``N * s(N)``. It may be below 1,
    in which case the continuous relaxation is infeasible.
    """
    stationary = cumulative_stationary_snrs(pfa)
    if not stationary:
        raise ValueError("H has no interior local maximum for this pfa")
    if not math.isfinite(pc) or not 0.0 < pc < 1.0:
        raise ValueError(f"pc must be strictly between 0 and 1, got {pc!r}")
    return -math.log1p(-pc) / log_miss_rate(stationary[1], pfa)
