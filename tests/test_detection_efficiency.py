"""Tests for the detection-efficiency objectives and their extrema."""

import math
from decimal import Decimal, getcontext

import pytest

from optimal_search.detection_efficiency import (
    analyze_objective,
    baseline_corrected_efficiency,
    baseline_corrected_limit_at_zero,
    baseline_corrected_optimum_snr,
    historical_boundary_crossover_snr,
    historical_efficiency,
    historical_stationary_snrs,
)
from optimal_search.radar_detection import detection_probability, snr_db_to_linear, snr_linear_to_db

PFAS = [1e-5, 1e-6]
WIDE_PFAS = [1e-3, 1e-5, 1e-6, 1e-9, 1e-12]


def db(snr_linear):
    return 10.0 * math.log10(snr_linear)


def d_log(f, s, pfa, h=1e-5):
    """Central finite difference of d ln f / ds, using a relative step h in s."""
    return (math.log(f(s * (1 + h), pfa)) - math.log(f(s * (1 - h), pfa))) / (2 * h * s)


# --- historical objective: definition and boundary behaviour -----------------


@pytest.mark.parametrize("pfa", PFAS)
@pytest.mark.parametrize("snr", [0.01, 1.0, 10.0, 100.0])
def test_historical_is_pd_over_snr(snr, pfa):
    assert historical_efficiency(snr, pfa) == pytest.approx(detection_probability(snr, pfa) / snr, rel=1e-14)


@pytest.mark.parametrize("pfa", PFAS)
def test_historical_diverges_as_snr_goes_to_zero(pfa):
    snrs = [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]
    values = [historical_efficiency(s, pfa) for s in snrs]
    assert all(a < b for a, b in zip(values, values[1:]))
    # E(s) * s -> Pd(0) = Pfa, so E ~ Pfa / s without bound.
    assert values[-1] * snrs[-1] == pytest.approx(pfa, rel=1e-9)
    assert values[-1] / values[0] > 1e11  # ~1/s: s fell by a factor 1e12


@pytest.mark.parametrize("pfa", PFAS)
def test_historical_tends_to_zero_at_large_snr(pfa):
    assert historical_efficiency(1e6, pfa) < 1e-5
    assert historical_efficiency(1e9, pfa) < historical_efficiency(1e6, pfa)


# --- historical objective: analytic stationary points ------------------------


@pytest.mark.parametrize("pfa", WIDE_PFAS[1:])
def test_stationary_points_satisfy_quadratic_and_reciprocity(pfa):
    log_inv = -math.log(pfa)
    s_lo, s_hi = historical_stationary_snrs(pfa)
    assert s_lo < 1.0 < s_hi
    assert s_lo * s_hi == pytest.approx(1.0, rel=1e-12)
    for s in (s_lo, s_hi):
        assert s * s + (2.0 - log_inv) * s + 1.0 == pytest.approx(0.0, abs=1e-9 * log_inv**2)
        # d ln E / ds = L/(1+s)^2 - 1/s = 0, derived by hand.
        assert log_inv / (1.0 + s) ** 2 - 1.0 / s == pytest.approx(0.0, abs=1e-9)
    # Symmetric about 0 dB.
    assert db(s_lo) == pytest.approx(-db(s_hi), abs=1e-12)


@pytest.mark.parametrize("pfa", WIDE_PFAS[1:])
def test_stationary_points_are_min_and_max_by_finite_differences(pfa):
    s_lo, s_hi = historical_stationary_snrs(pfa)
    for s in (s_lo, s_hi):
        assert d_log(historical_efficiency, s, pfa) == pytest.approx(0.0, abs=1e-6)
    e = historical_efficiency
    assert e(s_hi, pfa) > e(s_hi * 0.9, pfa) and e(s_hi, pfa) > e(s_hi * 1.1, pfa)
    assert e(s_lo, pfa) < e(s_lo * 0.9, pfa) and e(s_lo, pfa) < e(s_lo * 1.1, pfa)


@pytest.mark.parametrize(
    "pfa, expected_hi_db",
    [(1e-5, 9.734334766), (1e-6, 10.693076731)],
)
def test_historical_local_maximum_reference_values(pfa, expected_hi_db):
    # Reference values from solving s^2 + (2 - L) s + 1 = 0 in a separate script.
    _, s_hi = historical_stationary_snrs(pfa)
    assert db(s_hi) == pytest.approx(expected_hi_db, abs=1e-6)


def test_no_stationary_point_when_pfa_above_exp_minus_4():
    assert historical_stationary_snrs(0.1) == ()
    assert historical_stationary_snrs(math.exp(-3.99)) == ()
    snrs = [10 ** (k / 10) for k in range(-60, 61)]
    values = [historical_efficiency(s, 0.1) for s in snrs]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_roots_merge_at_exp_minus_4():
    s_lo, s_hi = historical_stationary_snrs(math.exp(-4.0))
    assert s_lo == pytest.approx(1.0, abs=1e-12)
    assert s_hi == pytest.approx(1.0, abs=1e-12)


# --- historical objective: numerical optimization and window dependence -------


@pytest.mark.parametrize("pfa", PFAS)
def test_numerical_extrema_match_analytic_and_boundary_wins(pfa):
    analysis = analyze_objective(historical_efficiency, pfa)
    s_lo, s_hi = historical_stationary_snrs(pfa)
    assert len(analysis.local_maxima) == 1
    assert len(analysis.local_minima) == 1
    assert analysis.local_maxima[0].snr_db == pytest.approx(db(s_hi), abs=1e-6)
    assert analysis.local_minima[0].snr_db == pytest.approx(db(s_lo), abs=1e-6)
    assert analysis.local_maxima[0].value == pytest.approx(historical_efficiency(s_hi, pfa), rel=1e-10)

    # The interior local maximum is NOT the window maximum: the lower boundary is.
    point, location = analysis.window_maximum
    assert location == "lower boundary"
    assert point.value > analysis.local_maxima[0].value
    # Boundary value is ~ Pfa / s (divergent behaviour), not an optimum.
    assert point.value == pytest.approx(pfa / snr_db_to_linear(point.snr_db), rel=1e-3)


@pytest.mark.parametrize("pfa", PFAS)
def test_crossover_snr_separates_boundary_and_interior_global_max(pfa):
    s_lo, s_hi = historical_stationary_snrs(pfa)
    s_c = historical_boundary_crossover_snr(pfa)
    assert 0.0 < s_c < s_lo
    assert historical_efficiency(s_c, pfa) == pytest.approx(historical_efficiency(s_hi, pfa), rel=1e-10)

    above = analyze_objective(historical_efficiency, pfa, lower_db=db(s_c) + 1.0, upper_db=60.0)
    assert above.window_maximum[1] == "interior"
    assert above.window_maximum[0].snr_db == pytest.approx(db(s_hi), abs=1e-6)

    below = analyze_objective(historical_efficiency, pfa, lower_db=db(s_c) - 1.0, upper_db=60.0)
    assert below.window_maximum[1] == "lower boundary"


# --- baseline-corrected objective --------------------------------------------


@pytest.mark.parametrize("pfa", PFAS)
@pytest.mark.parametrize("snr", [0.01, 0.5, 1.0, 10.0, 100.0])
def test_baseline_form_matches_direct_difference(snr, pfa):
    direct = (detection_probability(snr, pfa) - pfa) / snr
    assert baseline_corrected_efficiency(snr, pfa) == pytest.approx(direct, rel=1e-9)


@pytest.mark.parametrize("pfa", PFAS)
def test_baseline_limit_at_zero_is_finite_and_equals_pfa_log(pfa):
    limit = baseline_corrected_limit_at_zero(pfa)
    assert limit == pytest.approx(pfa * math.log(1.0 / pfa), rel=1e-14)
    for s in (1e-6, 1e-9, 1e-12):
        assert baseline_corrected_efficiency(s, pfa) == pytest.approx(limit, rel=1e-5)


@pytest.mark.parametrize("pfa", PFAS)
def test_baseline_is_positive_and_decays_at_large_snr(pfa):
    for k in range(-60, 61, 5):
        assert baseline_corrected_efficiency(10 ** (k / 10), pfa) > 0.0
    assert baseline_corrected_efficiency(1e6, pfa) < 1e-5


@pytest.mark.parametrize("pfa", PFAS)
def test_baseline_has_single_interior_global_maximum(pfa):
    analysis = analyze_objective(baseline_corrected_efficiency, pfa)
    assert len(analysis.local_maxima) == 1
    assert analysis.local_minima == ()
    point, location = analysis.window_maximum
    assert location == "interior"
    assert point.value > analysis.lower_boundary.value
    # The lower boundary value approaches the finite limit Pfa * L, with the
    # first-order correction G ~ Pfa * L * (1 + (L/2 - 1) s) derived by series.
    log_inv = -math.log(pfa)
    s_edge = snr_db_to_linear(analysis.lower_boundary.snr_db)
    expected = baseline_corrected_limit_at_zero(pfa) * (1.0 + (log_inv / 2.0 - 1.0) * s_edge)
    assert analysis.lower_boundary.value == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("pfa", PFAS)
def test_baseline_numerical_optimum_matches_independent_root(pfa):
    # Optimizer works in dB on the (Pd - Pfa)/s form; the root finder solves a
    # different equation in u = s / (1 + s).
    numeric = analyze_objective(baseline_corrected_efficiency, pfa).local_maxima[0]
    root = baseline_corrected_optimum_snr(pfa)
    assert numeric.snr_db == pytest.approx(db(root), abs=1e-6)


@pytest.mark.parametrize("pfa", WIDE_PFAS)
def test_baseline_optimum_satisfies_derivative_condition(pfa):
    # At the optimum s * Pd'(s) = Pd(s) - Pfa with Pd'(s) = Pd * L / (1 + s)^2.
    log_inv = -math.log(pfa)
    s = baseline_corrected_optimum_snr(pfa)
    pd = detection_probability(s, pfa)
    assert s * pd * log_inv / (1.0 + s) ** 2 == pytest.approx(pd - pfa, rel=1e-9)


@pytest.mark.parametrize("pfa", WIDE_PFAS)
def test_baseline_derivative_has_single_sign_change(pfa):
    # Numerical (not a proof) uniqueness check across several Pfa values.
    log_inv = -math.log(pfa)
    signs = []
    for k in range(1, 2000):
        u = k / 2000.0
        f = log_inv * math.exp(log_inv * u) / math.expm1(log_inv * u) - 1.0 / u - 1.0 / (1.0 - u)
        signs.append(f > 0)
    assert sum(a != b for a, b in zip(signs, signs[1:])) == 1


@pytest.mark.parametrize("pfa", PFAS)
def test_baseline_optimum_is_slightly_above_historical_local_maximum(pfa):
    # G = E - Pfa/s  =>  G'(s_hi) = Pfa / s_hi^2 > 0, so the optimum shifts up.
    s_hi = historical_stationary_snrs(pfa)[1]
    s_g = baseline_corrected_optimum_snr(pfa)
    assert s_g > s_hi
    assert db(s_g) - db(s_hi) < 1e-3  # the correction barely matters for tiny Pfa


def test_baseline_maximum_not_interior_for_large_pfa():
    with pytest.raises(ValueError):
        baseline_corrected_optimum_snr(0.5)


# --- comparison with a published value (documented, not asserted) -------------


@pytest.mark.parametrize("pfa, published_db", [(1e-5, 11.4), (1e-6, 12.4)])
def test_published_values_are_not_the_optimum_of_either_objective(pfa, published_db):
    # Records the finding rather than forcing agreement: at the published SNR
    # both objectives are strictly below their maxima and the gradient is
    # still negative (the optimum lies below the published SNR).
    s_pub = snr_db_to_linear(published_db)
    s_hi = historical_stationary_snrs(pfa)[1]
    s_g = baseline_corrected_optimum_snr(pfa)
    assert historical_efficiency(s_pub, pfa) < historical_efficiency(s_hi, pfa)
    assert baseline_corrected_efficiency(s_pub, pfa) < baseline_corrected_efficiency(s_g, pfa)
    assert db(s_hi) < published_db and db(s_g) < published_db


# --- validation ----------------------------------------------------------------


@pytest.mark.parametrize("func", [historical_efficiency, baseline_corrected_efficiency])
@pytest.mark.parametrize("snr", [0.0, -1.0, math.nan, math.inf])
def test_objectives_reject_invalid_snr(func, snr):
    with pytest.raises(ValueError, match="snr_linear"):
        func(snr, 1e-6)


@pytest.mark.parametrize("func", [historical_efficiency, baseline_corrected_efficiency])
@pytest.mark.parametrize("pfa", [0.0, 1.0, -0.1, math.nan])
def test_objectives_reject_invalid_pfa(func, pfa):
    with pytest.raises(ValueError, match="pfa"):
        func(1.0, pfa)


def test_analyze_objective_validates_window():
    with pytest.raises(ValueError):
        analyze_objective(historical_efficiency, 1e-6, lower_db=10.0, upper_db=0.0)
    with pytest.raises(ValueError):
        analyze_objective(historical_efficiency, 1e-6, n_grid=2)


# --- regressions from code review ------------------------------------------------------


def _reference_baseline(snr, pfa):
    """(Pd - Pfa) / SNR in 400-digit decimal arithmetic, using the exact float inputs."""
    getcontext().prec = 400
    s, p = Decimal(snr), Decimal(pfa)
    pd = (-(-p.ln()) / (1 + s)).exp()
    return float((pd - p) / s)


@pytest.mark.parametrize("pfa", [1e-3, 1e-12, 1e-100, 1e-300, 0.2, 0.5])
@pytest.mark.parametrize("snr", [1e-12, 1e-6, 0.1, 10.0, 1e3, 1e9])
def test_baseline_matches_high_precision_reference(snr, pfa):
    assert baseline_corrected_efficiency(snr, pfa) == pytest.approx(_reference_baseline(snr, pfa), rel=1e-10)


def test_baseline_efficiency_does_not_overflow_for_tiny_pfa():
    # L = ln(1/Pfa) ~ 737 so exp(L s/(1+s)) ~ exp(729) overflows a double; the answer is finite.
    value = baseline_corrected_efficiency(100.0, 1e-320)
    assert 0.0 < value < 1e-5
    assert value == pytest.approx(_reference_baseline(100.0, 1e-320), rel=1e-12)


@pytest.mark.parametrize("pfa", [1e-300, 1e-320])
def test_baseline_optimum_does_not_overflow_for_tiny_pfa(pfa):
    s = baseline_corrected_optimum_snr(pfa)
    assert math.isfinite(s) and s > 1.0
    h = 1e-5
    d = (math.log(baseline_corrected_efficiency(s * (1 + h), pfa)) - math.log(baseline_corrected_efficiency(s * (1 - h), pfa))) / (2 * h * s)
    assert d == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("excess", [1e-4, 1e-3, 1e-2])
def test_baseline_optimum_exists_just_below_exp_minus_2(excess):
    # The optimum tends to 0 as L -> 2+: u ~ 1.5 (L/2 - 1) = 0.75 (L - 2) (series of the stationarity condition).
    pfa = math.exp(-(2.0 + excess))
    s = baseline_corrected_optimum_snr(pfa)
    u = s / (1.0 + s)
    assert u == pytest.approx(0.75 * excess, rel=excess)  # first-order series; error is about 0.6 * excess
    for factor in (0.9, 1.1):
        assert baseline_corrected_efficiency(s * factor, pfa) < baseline_corrected_efficiency(s, pfa)


def test_baseline_optimum_rejects_pfa_too_close_to_exp_minus_2():
    with pytest.raises(ValueError):
        baseline_corrected_optimum_snr(math.exp(-2.0) * (1.0 - 1e-15))


def test_analyze_objective_refines_peak_between_tied_samples():
    # Samples at linear SNR 1 and 10 tie (both 1/(1+4.5^2)), straddling the true peak at 5.5.
    analysis = analyze_objective(lambda s, p: 1.0 / (1.0 + (s - 5.5) ** 2), 0.1, lower_db=-10.0, upper_db=20.0, n_grid=4)
    assert len(analysis.local_maxima) == 1
    assert analysis.local_maxima[0].snr_linear == pytest.approx(5.5, rel=1e-6)
    assert analysis.window_maximum[1] == "interior"


def test_analyze_objective_refines_each_tied_peak_once_and_ignores_boundary_plateaus():
    flat_top = analyze_objective(lambda s, p: 1.0 / (1.0 + (math.log10(s) - 0.7) ** 2), 0.1, lower_db=-10.0, upper_db=30.0, n_grid=41)
    assert len(flat_top.local_maxima) == 1
    plateau_at_edge = analyze_objective(lambda s, p: 1.0, 0.1, lower_db=-10.0, upper_db=10.0, n_grid=11)
    assert plateau_at_edge.local_maxima == () and plateau_at_edge.local_minima == ()


def test_crossover_requires_pfa_strictly_below_exp_minus_4():
    with pytest.raises(ValueError):
        historical_boundary_crossover_snr(math.exp(-4.0))
    with pytest.raises(ValueError):
        historical_boundary_crossover_snr(0.1)
    s_lo = historical_stationary_snrs(math.exp(-4.01))[0]
    assert 0.0 < historical_boundary_crossover_snr(math.exp(-4.01)) < s_lo
