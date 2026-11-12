"""Tests for the cumulative-detection (N independent looks) objectives."""

import math
from decimal import Decimal, getcontext

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from optimal_search.cumulative_detection import (
    INFLECTION_X,
    baseline_corrected_log_efficiency,
    baseline_log_limit_at_zero,
    baseline_log_optimum_snr,
    continuous_optimal_looks,
    cumulative_efficiency,
    cumulative_stationary_snrs,
    inflection_snr,
    log_miss_rate,
    required_snr_for_looks,
    total_effort,
)
from optimal_search.detection_efficiency import analyze_objective, historical_efficiency
from optimal_search.radar_detection import detection_probability, snr_db_to_linear

PFAS = [1e-5, 1e-6]
WIDE_PFAS = [1e-3, 1e-5, 1e-6, 1e-9, 1e-12]


def db(snr_linear):
    return 10.0 * math.log10(snr_linear)


# --- log-miss rate and the effort identity -----------------------------------


@pytest.mark.parametrize("pfa", PFAS)
@pytest.mark.parametrize("snr", [1e-6, 0.1, 1.0, 10.0, 100.0])
def test_log_miss_rate_matches_direct_formula(snr, pfa):
    assert log_miss_rate(snr, pfa) == pytest.approx(-math.log1p(-detection_probability(snr, pfa)), rel=1e-9)


@pytest.mark.parametrize("pfa", PFAS)
def test_log_miss_rate_large_snr_asymptotics(pfa):
    # 1 - Pd = 1 - exp(-x) = x (1 - x/2 + x^2/6 ...), x = L/(1+s)  =>  q ~ ln(1/x) + x/2.
    log_inv = -math.log(pfa)
    for s in (1e6, 1e9, 1e12):
        x = log_inv / (1.0 + s)
        assert log_miss_rate(s, pfa) == pytest.approx(math.log(1.0 / x) + x / 2.0, rel=1e-12)


@pytest.mark.parametrize("pfa", PFAS)
@pytest.mark.parametrize("pc", [0.5, 0.9, 0.99])
@pytest.mark.parametrize("snr", [3.0, 10.0, 30.0])
def test_effort_identity_N_times_s_equals_const_over_H(snr, pc, pfa):
    pd = detection_probability(snr, pfa)
    n_looks = math.log(1.0 - pc) / math.log(1.0 - pd)  # inverts Pc = 1 - (1 - Pd)^N
    assert 1.0 - (1.0 - pd) ** n_looks == pytest.approx(pc, rel=1e-12)
    assert n_looks * snr == pytest.approx(-math.log(1.0 - pc) / cumulative_efficiency(snr, pfa), rel=1e-12)


@pytest.mark.parametrize("pfa", PFAS)
def test_required_snr_inverts_the_looks_relation(pfa):
    for n in (1.0, 2.0, 3.7, 10.0):
        s = required_snr_for_looks(n, 0.9, pfa)
        pc = 1.0 - (1.0 - detection_probability(s, pfa)) ** n
        assert pc == pytest.approx(0.9, rel=1e-10)


# --- boundary behaviour --------------------------------------------------------


@pytest.mark.parametrize("pfa", PFAS)
def test_cumulative_diverges_at_zero_snr(pfa):
    q0 = -math.log1p(-pfa)
    snrs = [1e-3, 1e-6, 1e-9, 1e-12]
    values = [cumulative_efficiency(s, pfa) for s in snrs]
    assert all(a < b for a, b in zip(values, values[1:]))
    assert values[-1] * snrs[-1] == pytest.approx(q0, rel=1e-9)


@pytest.mark.parametrize("pfa", PFAS)
def test_cumulative_tends_to_zero_at_large_snr(pfa):
    # H ~ ln(s)/s
    for s in (1e6, 1e9):
        assert cumulative_efficiency(s, pfa) < 1e-4
    assert cumulative_efficiency(1e9, pfa) < cumulative_efficiency(1e6, pfa)


@pytest.mark.parametrize("pfa", PFAS)
def test_baseline_log_limit_at_zero_is_finite(pfa):
    limit = baseline_log_limit_at_zero(pfa)
    assert limit == pytest.approx(pfa * math.log(1.0 / pfa) / (1.0 - pfa), rel=1e-14)
    for s in (1e-8, 1e-10, 1e-12):
        assert baseline_corrected_log_efficiency(s, pfa) == pytest.approx(limit, rel=1e-6)


# --- analytic structure: single inflection of q -------------------------------


def test_inflection_constant_solves_defining_equation():
    assert INFLECTION_X == pytest.approx(2.0 * (1.0 - math.exp(-INFLECTION_X)), rel=1e-14)
    assert INFLECTION_X == pytest.approx(1.5936242600400401, rel=1e-12)


@pytest.mark.parametrize("pfa", WIDE_PFAS)
def test_q_is_convex_below_and_concave_above_inflection(pfa):
    s_i = inflection_snr(pfa)
    h = 1e-3

    def second_diff(s):
        return (log_miss_rate(s * (1 + h), pfa) - 2 * log_miss_rate(s, pfa) + log_miss_rate(s * (1 - h), pfa)) / (h * s) ** 2

    assert second_diff(s_i * 0.5) > 0.0
    assert second_diff(s_i * 2.0) < 0.0
    assert abs(second_diff(s_i)) < 1e-3 * abs(second_diff(s_i * 0.5))


# --- stationary points of H ----------------------------------------------------


@pytest.mark.parametrize("pfa", WIDE_PFAS)
def test_stationary_points_exist_and_are_min_then_max(pfa):
    s_lo, s_hi = cumulative_stationary_snrs(pfa)
    assert 0.0 < s_lo < inflection_snr(pfa) < s_hi
    h = 1e-5
    for s in (s_lo, s_hi):
        d = (math.log(cumulative_efficiency(s * (1 + h), pfa)) - math.log(cumulative_efficiency(s * (1 - h), pfa))) / (2 * h * s)
        assert d == pytest.approx(0.0, abs=1e-6)
    f = cumulative_efficiency
    assert f(s_hi, pfa) > f(s_hi * 0.9, pfa) and f(s_hi, pfa) > f(s_hi * 1.1, pfa)
    assert f(s_lo, pfa) < f(s_lo * 0.9, pfa) and f(s_lo, pfa) < f(s_lo * 1.1, pfa)


@pytest.mark.parametrize("pfa", WIDE_PFAS)
def test_stationarity_function_has_exactly_two_sign_changes(pfa):
    # H'(s) has the sign of F(s) = s q' - q; count sign changes on a fine log grid.
    grid = np.logspace(-8, 8, 4001)
    signs = []
    for s in grid:
        h = 1e-6
        d = (log_miss_rate(s * (1 + h), pfa) - log_miss_rate(s * (1 - h), pfa)) / (2 * h * s)
        signs.append(s * d - log_miss_rate(s, pfa) > 0)
    assert sum(a != b for a, b in zip(signs, signs[1:])) == 2


@pytest.mark.parametrize("pfa", PFAS)
def test_numerical_scan_matches_root_finder(pfa):
    analysis = analyze_objective(cumulative_efficiency, pfa)
    s_lo, s_hi = cumulative_stationary_snrs(pfa)
    assert len(analysis.local_maxima) == 1 and len(analysis.local_minima) == 1
    assert analysis.local_maxima[0].snr_db == pytest.approx(db(s_hi), abs=1e-6)
    assert analysis.local_minima[0].snr_db == pytest.approx(db(s_lo), abs=1e-6)
    # Lower boundary value is larger (divergence), so it is not an optimum.
    point, location = analysis.window_maximum
    assert location == "lower boundary"
    assert point.value > analysis.local_maxima[0].value


@pytest.mark.parametrize(
    "pfa, expected_db, expected_pd",
    [(1e-5, 11.4162, 0.46070), (1e-6, 12.3557, 0.46812)],
)
def test_local_maximum_reference_values(pfa, expected_db, expected_pd):
    # Reference values from a separate prototype script (brentq on F, 200k-point grid).
    s_hi = cumulative_stationary_snrs(pfa)[1]
    assert db(s_hi) == pytest.approx(expected_db, abs=1e-4)
    assert detection_probability(s_hi, pfa) == pytest.approx(expected_pd, abs=1e-5)


def test_no_stationary_points_for_large_pfa():
    assert cumulative_stationary_snrs(0.5) == ()


# --- independent verification of the derivation ----------------------------------


@pytest.mark.parametrize("pfa", PFAS)
@pytest.mark.parametrize("pc", [0.9, 0.99])
def test_direct_minimisation_of_effort_over_N_agrees_with_H(pfa, pc):
    # Minimise N * s(N) over real N without using H, then compare with H's maximum.
    grid = np.linspace(1.0, 60.0, 2361)
    efforts = np.array([total_effort(n, pc, pfa) for n in grid])
    interior_min = [i for i in range(1, len(grid) - 1) if efforts[i] < efforts[i - 1] and efforts[i] < efforts[i + 1]]
    assert len(interior_min) == 1
    i = interior_min[0]
    res = minimize_scalar(lambda n: total_effort(n, pc, pfa), bounds=(grid[i - 1], grid[i + 1]), method="bounded", options={"xatol": 1e-10})

    s_hi = cumulative_stationary_snrs(pfa)[1]
    assert required_snr_for_looks(res.x, pc, pfa) == pytest.approx(s_hi, rel=1e-5)
    assert res.x == pytest.approx(continuous_optimal_looks(pc, pfa), rel=1e-5)
    assert res.fun == pytest.approx(-math.log1p(-pc) / cumulative_efficiency(s_hi, pfa), rel=1e-9)


@pytest.mark.parametrize("pfa", PFAS)
def test_optimal_snr_is_independent_of_pc(pfa):
    s_hi = cumulative_stationary_snrs(pfa)[1]
    for pc in (0.6, 0.9, 0.999):
        n_star = continuous_optimal_looks(pc, pfa)
        assert required_snr_for_looks(n_star, pc, pfa) == pytest.approx(s_hi, rel=1e-9)


@pytest.mark.parametrize("pfa", PFAS)
@pytest.mark.parametrize("pc", [0.6, 0.9, 0.99, 0.999])
def test_integer_looks_in_practical_range_do_not_beat_local_optimum(pfa, pc):
    # Only a local statement: the effort at s_hi is a local minimum of N * s(N), compared here with
    # integer N = 1..40. It is NOT a global lower bound (see the next test).
    s_hi = cumulative_stationary_snrs(pfa)[1]
    local_optimum = -math.log1p(-pc) / cumulative_efficiency(s_hi, pfa)
    for n in range(1, 41):
        assert total_effort(n, pc, pfa) >= local_optimum * (1 - 1e-12)


@pytest.mark.parametrize("pfa", PFAS)
def test_effort_at_local_optimum_is_not_a_global_bound(pfa):
    # As N approaches the false-alarm-only value, s -> 0 and the effort tends to 0.
    pc = 0.9
    s_hi = cumulative_stationary_snrs(pfa)[1]
    local_optimum = -math.log1p(-pc) / cumulative_efficiency(s_hi, pfa)
    n_false_alarm_only = math.log1p(-pc) / math.log1p(-pfa)
    # Effort is large at moderate N, then falls through the local optimum only very close to N_fa.
    efforts = [total_effort(f * n_false_alarm_only, pc, pfa) for f in (0.5, 0.9, 0.99, 0.9999, 0.999999)]
    assert all(a > b for a, b in zip(efforts, efforts[1:]))
    assert efforts[0] > local_optimum
    assert efforts[-1] < 0.01 * local_optimum
    if pfa == 1e-5:
        assert total_effort(230000, pc, pfa) == pytest.approx(22.34, abs=0.05)  # about 22 vs about 52 at s_hi


@pytest.mark.parametrize("pfa", PFAS)
def test_continuous_relaxation_infeasible_for_small_pc(pfa):
    s_hi = cumulative_stationary_snrs(pfa)[1]
    pd_star = detection_probability(s_hi, pfa)
    assert continuous_optimal_looks(pd_star * 0.9, pfa) < 1.0
    assert continuous_optimal_looks(pd_star * 1.1, pfa) > 1.0
    assert continuous_optimal_looks(pd_star, pfa) == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize("pfa", PFAS)
def test_zero_snr_cumulative_detection_is_pure_false_alarm(pfa):
    # Explains the divergence at s -> 0: reaching Pc with false alarms alone.
    pc = 0.9
    n_false_alarm_only = math.log1p(-pc) / math.log1p(-pfa)
    assert 1.0 - (1.0 - pfa) ** n_false_alarm_only == pytest.approx(pc, rel=1e-10)
    with pytest.raises(ValueError):
        required_snr_for_looks(n_false_alarm_only * 1.001, pc, pfa)


# --- relation to the single-look objective -------------------------------------------


@pytest.mark.parametrize("pfa", PFAS)
def test_low_pd_region_matches_single_look_objective(pfa):
    # For small Pd, -ln(1 - Pd) ~ Pd, so H ~ Pd/s near the lower stationary point.
    s = 0.1
    assert cumulative_efficiency(s, pfa) == pytest.approx(historical_efficiency(s, pfa), rel=1e-4)


@pytest.mark.parametrize("pfa", PFAS)
def test_cumulative_optimum_differs_from_single_look_optimum(pfa):
    from optimal_search.detection_efficiency import historical_stationary_snrs

    assert db(cumulative_stationary_snrs(pfa)[1]) > db(historical_stationary_snrs(pfa)[1]) + 1.0


# --- baseline-corrected logarithmic objective ---------------------------------------


@pytest.mark.parametrize("pfa", PFAS)
@pytest.mark.parametrize("snr", [0.01, 1.0, 10.0, 100.0])
def test_baseline_log_matches_direct_formula(snr, pfa):
    direct = (-math.log(1.0 - detection_probability(snr, pfa)) + math.log(1.0 - pfa)) / snr
    assert baseline_corrected_log_efficiency(snr, pfa) == pytest.approx(direct, rel=1e-8)


@pytest.mark.parametrize("pfa", WIDE_PFAS)
def test_baseline_log_has_unique_interior_maximum(pfa):
    s_opt = baseline_log_optimum_snr(pfa)
    assert s_opt > inflection_snr(pfa)
    h = 1e-5
    f = baseline_corrected_log_efficiency
    assert (math.log(f(s_opt * (1 + h), pfa)) - math.log(f(s_opt * (1 - h), pfa))) / (2 * h * s_opt) == pytest.approx(0.0, abs=1e-6)
    grid = np.logspace(-6, 6, 3001)
    vals = [f(s, pfa) for s in grid]
    assert int(np.argmax(vals)) not in (0, len(grid) - 1)
    slope_signs = np.sign(np.diff(vals))
    assert int(np.sum(slope_signs[1:] != slope_signs[:-1])) == 1  # up, then down: single peak


@pytest.mark.parametrize("pfa", PFAS)
def test_baseline_log_numerical_optimum_matches_root(pfa):
    analysis = analyze_objective(baseline_corrected_log_efficiency, pfa)
    assert len(analysis.local_maxima) == 1 and analysis.local_minima == ()
    assert analysis.window_maximum[1] == "interior"
    assert analysis.local_maxima[0].snr_db == pytest.approx(db(baseline_log_optimum_snr(pfa)), abs=1e-6)


@pytest.mark.parametrize("pfa", PFAS)
def test_baseline_log_optimum_slightly_above_cumulative_local_max(pfa):
    # G = H - q0/s, so G'(s_hi) = q0 / s_hi^2 > 0.
    s_hi = cumulative_stationary_snrs(pfa)[1]
    s_g = baseline_log_optimum_snr(pfa)
    assert s_g > s_hi
    assert db(s_g) - db(s_hi) < 1e-3


def test_baseline_log_optimum_requires_interior_maximum():
    with pytest.raises(ValueError):
        baseline_log_optimum_snr(0.5)


# --- comparison with reported values (consistency record, not validation) ------------


@pytest.mark.parametrize("pfa, reported_db", [(1e-5, 11.4), (1e-6, 12.4)])
def test_local_maximum_rounds_to_reported_values(pfa, reported_db):
    """Records the finding that the computed maxima round to the values listed in the 2011 presentation.

    No parameter was fitted. This is a consistency record only; it does not
    validate any historical derivation.
    """
    s_hi = cumulative_stationary_snrs(pfa)[1]
    assert round(db(s_hi), 1) == reported_db
    s_g = baseline_log_optimum_snr(pfa)
    assert round(db(s_g), 1) == reported_db


@pytest.mark.parametrize("pfa", PFAS)
def test_per_look_pd_at_optimum_is_about_47_percent(pfa):
    pd = detection_probability(cumulative_stationary_snrs(pfa)[1], pfa)
    assert 0.45 < pd < 0.48


# --- validation -----------------------------------------------------------------------


@pytest.mark.parametrize("func", [log_miss_rate, cumulative_efficiency, baseline_corrected_log_efficiency])
@pytest.mark.parametrize("snr", [0.0, -1.0, math.nan, math.inf])
def test_objectives_reject_invalid_snr(func, snr):
    with pytest.raises(ValueError, match="snr_linear"):
        func(snr, 1e-6)


@pytest.mark.parametrize("func", [log_miss_rate, cumulative_efficiency, baseline_corrected_log_efficiency])
@pytest.mark.parametrize("pfa", [0.0, 1.0, math.nan])
def test_objectives_reject_invalid_pfa(func, pfa):
    with pytest.raises(ValueError, match="pfa"):
        func(1.0, pfa)


@pytest.mark.parametrize("n_looks, pc", [(0.0, 0.9), (-1.0, 0.9), (math.nan, 0.9), (2.0, 0.0), (2.0, 1.0), (2.0, math.nan)])
def test_looks_functions_reject_invalid_arguments(n_looks, pc):
    with pytest.raises(ValueError):
        required_snr_for_looks(n_looks, pc, 1e-6)


@pytest.mark.parametrize("pfa", [1e-9, 1e-12])
def test_log_miss_rate_is_accurate_when_pd_is_tiny(pfa):
    # Regression: computing 1 - Pd first cancels catastrophically for tiny Pd.
    # q = Pd + Pd^2/2 + Pd^3/3 + ... to full precision.
    s = 0.05
    pd = detection_probability(s, pfa)
    assert pd < 1e-7  # tiny-Pd regime, where the old 1 - Pd formula lost ~5 digits
    assert log_miss_rate(s, pfa) == pytest.approx(pd + pd * pd / 2.0, rel=1e-13)


# --- regressions from code review ------------------------------------------------------


def _reference_baseline_log(snr, pfa):
    """[-ln(1 - Pd) + ln(1 - Pfa)] / SNR in 400-digit decimal arithmetic, using the exact float inputs."""
    getcontext().prec = 400
    s, p = Decimal(snr), Decimal(pfa)
    pd = (-(-p.ln()) / (1 + s)).exp()
    return float((-(1 - pd).ln() + (1 - p).ln()) / s)


@pytest.mark.parametrize("pfa", [1e-5, 1e-12, 1e-100, 0.2, 0.5])
@pytest.mark.parametrize("snr", [1e-12, 1e-6, 0.1, 10.0, 1e3, 1e9, 1e12, 1e20])
def test_baseline_log_matches_high_precision_reference(snr, pfa):
    assert baseline_corrected_log_efficiency(snr, pfa) == pytest.approx(_reference_baseline_log(snr, pfa), rel=1e-10)


def test_baseline_log_is_finite_and_positive_at_very_large_snr():
    # excess = (Pd - Pfa)/(1 - Pfa) rounds to 1 here, so log1p(-excess) alone would fail.
    for snr in (1e15, 1e20, 1e100, 1e200):
        value = baseline_corrected_log_efficiency(snr, 0.5)
        assert 0.0 < value < 1e-12
    assert baseline_corrected_log_efficiency(1e20, 0.5) > baseline_corrected_log_efficiency(1e200, 0.5)
