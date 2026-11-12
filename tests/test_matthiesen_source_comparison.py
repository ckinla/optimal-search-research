"""Comparison of the implemented objectives with equations in a primary source.

Source: D. J. Matthiesen, "Efficient Beam Scanning, Energy Allocation, and Time
Allocation for Search and Detection", IEEE Boston Section technical meeting,
22 March 2011 (PowerPoint presentation).

The deck's equations below were transcribed by reading the vector equation
objects embedded in the file, not from a rendered slide, so each transcription
should be checked against the slides (see ``docs/historical-source-verification.md``).
The deck's own derivation steps and its detection curves were not recoverable.
These tests check the *mathematics* of the transcribed equations against the
implementation; matching a published number is not used as evidence that the
equations match.
"""

import math

import pytest
from scipy.optimize import brentq, minimize_scalar

from optimal_search.cumulative_detection import (
    _stationarity,
    cumulative_efficiency,
    cumulative_stationary_snrs,
    log_miss_rate,
)
from optimal_search.detection_efficiency import historical_stationary_snrs
from optimal_search.radar_detection import detection_probability

PFAS = [1e-5, 1e-6]
DECK_TABLE = {1e-5: 11.4, 1e-6: 12.4}  # Swerling I/II optimal S/N [dB], summary slide
DECK_OPTIMAL_PD = 0.47  # single value listed for both Pfa on the summary slide


def db(snr_linear):
    return 10.0 * math.log10(snr_linear)


# --- equations transcribed from the deck ------------------------------------------


def deck_numerator_function(p):
    """f(Pd) = -(1 - Pd) ln(1 - Pd), the 'numerator function' (deck figure: max 0.36788 at 0.63212)."""
    return -(1.0 - p) * math.log1p(-p)


def deck_condition_residual(snr, pfa):
    """Optimal Detection Condition: dPd/d(S/N) = f(Pd)/(S/N), written as s*Pd' - f(Pd) = 0.

    Pd' uses the closed-form derivative of Pd = Pfa**(1/(1+s)).
    """
    log_inv = -math.log(pfa)
    pd = detection_probability(snr, pfa)
    dpd_ds = pd * log_inv / (1.0 + snr) ** 2
    return snr * dpd_ds - deck_numerator_function(pd)


def deck_payoff_ratio_energy(snr, n_looks, pfa):
    """rho_D,energy = Pc / (N_L * energy per look) = [1 - (1 - Pd)^N_L] / (N_L * S/N), with S/N proportional to energy."""
    return -math.expm1(-n_looks * log_miss_rate(snr, pfa)) / (n_looks * snr)


# --- numerator function ---------------------------------------------------------------


def test_numerator_function_maximum_matches_deck_figure():
    # The deck plots f with "max = 0.36788 at Pd = 0.63212", i.e. 1/e at 1 - 1/e.
    p_star = 1.0 - math.exp(-1.0)
    assert p_star == pytest.approx(0.63212, abs=1e-5)
    assert deck_numerator_function(p_star) == pytest.approx(0.36788, abs=1e-5)
    res = minimize_scalar(lambda p: -deck_numerator_function(p), bounds=(0.01, 0.99), method="bounded", options={"xatol": 1e-12})
    assert res.x == pytest.approx(p_star, abs=1e-6)
    assert deck_numerator_function(0.0) == 0.0
    assert deck_numerator_function(1.0 - 1e-15) < 1e-13


# --- the optimal detection condition vs the implemented stationarity -------------------


@pytest.mark.parametrize("pfa", [1e-3, 1e-5, 1e-6, 1e-9])
@pytest.mark.parametrize("snr", [0.05, 0.5, 3.0, 13.0, 50.0, 500.0])
def test_deck_condition_is_algebraically_the_derivative_of_H(snr, pfa):
    # s q' - q  (our dH/ds numerator) times (1 - Pd) equals s Pd' - f(Pd):
    # since q' = Pd'/(1 - Pd) and f(Pd) = (1 - Pd) q.  Same sign, same zeros.
    ours = _stationarity(snr, -math.log(pfa))
    pd = detection_probability(snr, pfa)
    assert deck_condition_residual(snr, pfa) == pytest.approx((1.0 - pd) * ours, rel=1e-8, abs=1e-14)


@pytest.mark.parametrize("pfa", PFAS)
def test_deck_condition_roots_equal_stationary_points_of_H(pfa):
    s_lo, s_hi = cumulative_stationary_snrs(pfa)
    root_hi = brentq(deck_condition_residual, s_hi * 0.5, s_hi * 2.0, args=(pfa,), xtol=1e-14, rtol=1e-14)
    root_lo = brentq(deck_condition_residual, s_lo * 0.5, s_lo * 2.0, args=(pfa,), xtol=1e-14, rtol=1e-14)
    assert root_hi == pytest.approx(s_hi, rel=1e-9)
    assert root_lo == pytest.approx(s_lo, rel=1e-9)


@pytest.mark.parametrize("pfa", PFAS)
def test_single_look_pd_over_snr_does_not_satisfy_the_deck_condition(pfa):
    # Pd/s is the N_L = 1 payoff ratio, not the deck's optimal detection condition.
    s_single = historical_stationary_snrs(pfa)[1]
    pd = detection_probability(s_single, pfa)
    # At its own optimum s Pd' = Pd, so the deck residual is Pd - f(Pd) > 0 (f(p) < p):
    # H is still increasing there, and its maximum lies at higher SNR.
    log_inv = -math.log(pfa)
    assert s_single * pd * log_inv / (1.0 + s_single) ** 2 == pytest.approx(pd, rel=1e-9)
    assert deck_condition_residual(s_single, pfa) == pytest.approx(pd - deck_numerator_function(pd), rel=1e-9)
    assert deck_condition_residual(s_single, pfa) > 0.0
    assert db(cumulative_stationary_snrs(pfa)[1]) - db(s_single) > 1.5


# --- payoff ratio: relation to the implemented H -----------------------------------------


@pytest.mark.parametrize("pfa", PFAS)
@pytest.mark.parametrize("pc", [0.5, 0.9, 0.99])
def test_payoff_ratio_at_fixed_pc_is_constant_times_H(pfa, pc):
    # N_L = -ln(1 - Pc)/q(s) so that Pc is held fixed as s varies.
    ratios = []
    for snr in (3.0, 10.0, 30.0, 100.0):
        n_looks = -math.log1p(-pc) / log_miss_rate(snr, pfa)
        ratios.append(deck_payoff_ratio_energy(snr, n_looks, pfa) / cumulative_efficiency(snr, pfa))
    expected = pc / -math.log1p(-pc)
    assert ratios == pytest.approx([expected] * 4, rel=1e-12)


@pytest.mark.parametrize("pfa", PFAS)
def test_payoff_ratio_with_one_look_is_pd_over_snr(pfa):
    for snr in (0.5, 5.0, 50.0):
        assert deck_payoff_ratio_energy(snr, 1.0, pfa) == pytest.approx(detection_probability(snr, pfa) / snr, rel=1e-12)


def _argmax_db_at_fixed_looks(n_looks, pfa):
    res = minimize_scalar(
        lambda x: -math.log(deck_payoff_ratio_energy(10 ** (x / 10), n_looks, pfa)),
        bounds=(0.0, 40.0),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return res.x


@pytest.mark.parametrize("pfa", PFAS)
def test_optimum_at_fixed_looks_depends_on_n_and_tends_to_H_optimum_as_n_goes_to_zero(pfa):
    h_opt = db(cumulative_stationary_snrs(pfa)[1])
    single_opt = db(historical_stationary_snrs(pfa)[1])
    assert _argmax_db_at_fixed_looks(1.0, pfa) == pytest.approx(single_opt, abs=1e-6)
    # Holding N_L fixed (not Pc) gives N_L-dependent optima, all below H's.
    optima = [_argmax_db_at_fixed_looks(n, pfa) for n in (10.0, 5.0, 2.0, 1.0, 0.1, 1e-3, 1e-6)]
    assert all(a < b for a, b in zip(optima, optima[1:]))
    assert optima[-1] == pytest.approx(h_opt, abs=1e-3)
    assert h_opt - _argmax_db_at_fixed_looks(3.7, pfa) > 2.0


# --- exponential detection law -------------------------------------------------------------


@pytest.mark.parametrize("pd", [0.1, 0.47, 0.9])
@pytest.mark.parametrize("n_looks", [1, 3, 7])
def test_cumulative_probability_equals_exponential_law(pd, n_looks):
    # Deck: Pc = 1 - (1 - Pd)^N_L = 1 - exp(-alpha t), alpha = -ln(1 - Pd)/dt, t = N_L dt.
    dt = 0.37
    alpha = -math.log1p(-pd) / dt
    assert 1.0 - (1.0 - pd) ** n_looks == pytest.approx(1.0 - math.exp(-alpha * n_looks * dt), rel=1e-12)


@pytest.mark.parametrize("pfa", PFAS)
def test_H_is_the_per_energy_analogue_of_the_deck_exponential_rate(pfa):
    # alpha * dt = q.  Replacing time-per-look dt by S/N (energy per look, in units of N0) gives q/s = H.
    s = 12.0
    assert cumulative_efficiency(s, pfa) == pytest.approx(-math.log1p(-detection_probability(s, pfa)) / s, rel=1e-12)


# --- comparison with the deck's summary table (consistency record, not evidence of equality) ---


@pytest.mark.parametrize("pfa", PFAS)
def test_computed_optimum_snr_rounds_to_deck_table(pfa):
    assert round(db(cumulative_stationary_snrs(pfa)[1]), 1) == DECK_TABLE[pfa]


@pytest.mark.parametrize("pfa", PFAS)
def test_computed_optimal_pd_is_close_to_but_not_exactly_the_deck_value(pfa):
    pd = detection_probability(cumulative_stationary_snrs(pfa)[1], pfa)
    assert abs(pd - DECK_OPTIMAL_PD) < 0.01
    # The deck prints one value (0.47) for both Pfa; ours rounds to 0.46 for 1e-5 and 0.47 for 1e-6.
    assert round(pd, 2) == {1e-5: 0.46, 1e-6: 0.47}[pfa]
