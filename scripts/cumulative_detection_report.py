"""Report cumulative-detection efficiency optima for N independent looks.

Run from the repository root::

    uv run python -m scripts.cumulative_detection_report

The mathematics is independently derived (see ``src/optimal_search/cumulative_detection.py``);
the values attributed to Matthiesen (11.4 dB for Pfa = 1e-5, 12.4 dB for
Pfa = 1e-6, and a per-look Pd of roughly 47%) are those listed in his 2011 presentation.
docs/historical-source-verification.md compares the presentation's Optimal Detection
Condition with this objective. Nothing here is fitted to the listed values.
"""

import math

from optimal_search.cumulative_detection import (
    baseline_corrected_log_efficiency,
    baseline_log_limit_at_zero,
    baseline_log_optimum_snr,
    continuous_optimal_looks,
    cumulative_efficiency,
    cumulative_stationary_snrs,
    inflection_snr,
    required_snr_for_looks,
    total_effort,
)
from optimal_search.detection_efficiency import analyze_objective, historical_stationary_snrs
from optimal_search.radar_detection import detection_probability, snr_db_to_linear

REPORTED_DB = {1e-5: 11.4, 1e-6: 12.4}
REPORTED_PD = 0.47
PCS = (0.9, 0.99)


def db(snr_linear: float) -> float:
    return 10.0 * math.log10(snr_linear)


def report(pfa: float) -> None:
    reported_db = REPORTED_DB[pfa]
    log_inv = -math.log(pfa)
    print("=" * 78)
    print(f"Pfa = {pfa:.0e}   (L = ln(1/Pfa) = {log_inv:.4f})")
    print("=" * 78)

    print("\n[A] Cumulative objective  H = -ln(1 - Pd) / SNR   (N independent looks)")
    q0 = -math.log1p(-pfa)
    print(f"    s -> 0 boundary: H*SNR -> -ln(1-Pfa) = {q0:.6g}  => H ~ {q0:.3g}/SNR, unbounded")
    for snr_db_value in (-20.0, -40.0, -60.0):
        s = snr_db_to_linear(snr_db_value)
        print(f"      SNR = {snr_db_value:6.1f} dB   H = {cumulative_efficiency(s, pfa):.6g}")
    print("    s -> inf boundary: H ~ ln(SNR)/SNR -> 0")
    for snr_db_value in (40.0, 60.0):
        s = snr_db_to_linear(snr_db_value)
        print(f"      SNR = {snr_db_value:6.1f} dB   H = {cumulative_efficiency(s, pfa):.6g}")
    print(f"    q = -ln(1-Pd) has one inflection at SNR = {db(inflection_snr(pfa)):.3f} dB (analytic)")
    s_lo, s_hi = cumulative_stationary_snrs(pfa)
    print("    Stationary points (roots of s q' - q = 0):")
    print(f"      local minimum  {db(s_lo):9.4f} dB   H = {cumulative_efficiency(s_lo, pfa):.6g}")
    print(f"      local maximum  {db(s_hi):9.4f} dB   H = {cumulative_efficiency(s_hi, pfa):.6g}   Pd/look = {detection_probability(s_hi, pfa):.4f}")
    analysis = analyze_objective(cumulative_efficiency, pfa)
    point, where = analysis.window_maximum
    print(f"    Numerical scan on [-60, 60] dB: local max {analysis.local_maxima[0].snr_db:.4f} dB, local min {analysis.local_minima[0].snr_db:.4f} dB;")
    print(f"      window maximum at {point.snr_db:.1f} dB -> {where} (boundary divergence, not an optimum)")

    print("\n[B] Baseline-corrected log objective  G = [-ln(1-Pd) + ln(1-Pfa)] / SNR")
    s_g = baseline_log_optimum_snr(pfa)
    gan = analyze_objective(baseline_corrected_log_efficiency, pfa)
    print(f"    limit at SNR -> 0: {baseline_log_limit_at_zero(pfa):.6g} (finite)")
    print(f"    unique maximum {db(s_g):9.4f} dB   Pd/look = {detection_probability(s_g, pfa):.4f}   (shift vs H: {db(s_g) - db(s_hi):+.6f} dB)")
    print(f"    numerical scan: {gan.window_maximum[0].snr_db:.4f} dB ({gan.window_maximum[1]}), local minima: {len(gan.local_minima)}")

    print(f"\n[C] Comparison with the presentation's values ({reported_db:.1f} dB, Pd ~ {100 * REPORTED_PD:.0f}%)")
    hist = db(historical_stationary_snrs(pfa)[1])
    print(f"    single-look Pd/SNR local max : {hist:8.4f} dB   (reported - computed = {reported_db - hist:+.3f} dB)")
    print(f"    cumulative H local max       : {db(s_hi):8.4f} dB   (reported - computed = {reported_db - db(s_hi):+.3f} dB)")
    print(f"    baseline-corrected log max   : {db(s_g):8.4f} dB   (reported - computed = {reported_db - db(s_g):+.3f} dB)")
    print(f"    Pd/look: H max {detection_probability(s_hi, pfa):.4f}; at reported SNR {detection_probability(snr_db_to_linear(reported_db), pfa):.4f}")

    print("\n[D] Continuous vs integer numbers of looks (effort = N * SNR_linear)")
    for pc in PCS:
        n_star = continuous_optimal_looks(pc, pfa)
        local_opt = -math.log1p(-pc) / cumulative_efficiency(s_hi, pfa)
        print(f"    Pc = {pc}: continuous local optimum N* = {n_star:.3f} looks at {db(s_hi):.3f} dB, effort {local_opt:.3f}")
        best_n, best_effort = min(((n, total_effort(n, pc, pfa)) for n in range(1, 41)), key=lambda t: t[1])
        for n in sorted({1, 2, 3, 4, 5, 6, 8, 10, best_n}):
            s_req = required_snr_for_looks(n, pc, pfa)
            marker = "  <- best integer in N = 1..40" if n == best_n else ""
            print(f"      N = {n:2d}: SNR/look {db(s_req):7.3f} dB, Pd/look {detection_probability(s_req, pfa):.4f}, effort {n * s_req:8.3f} (x{n * s_req / local_opt:.3f} of local optimum){marker}")
    print()


def main() -> None:
    print("Cumulative-detection efficiency, single-pulse Swerling I/II looks, Pd = Pfa^(1/(1+SNR))\n")
    for pfa in REPORTED_DB:
        report(pfa)
    print("Notes: independent looks, fixed per-look Pfa, effort proportional to N*SNR, continuous N.")
    print("The continuous optimum is a LOCAL minimum of effort over N; effort -> 0 as N approaches the")
    print("false-alarm-only value (SNR -> 0), so it is not a global lower bound on integer designs.")
    print("Agreement with the presentation's values is a consistency observation, not proof of any")
    print("historical derivation; those values were known when this objective was proposed.")


if __name__ == "__main__":
    main()
