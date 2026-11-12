"""Report detection-efficiency optima for single-pulse Swerling I/II detection.

Run from the repository root::

    uv run python -m scripts.detection_efficiency_report

Compares the numerical results with the values attributed to Matthiesen
(11.4 dB for Pfa = 1e-5, 12.4 dB for Pfa = 1e-6). The comparison is reported,
not enforced; see ``src/optimal_search/detection_efficiency.py`` for the caveats.
"""

import math

from optimal_search.detection_efficiency import (
    analyze_objective,
    baseline_corrected_efficiency,
    baseline_corrected_limit_at_zero,
    baseline_corrected_optimum_snr,
    historical_boundary_crossover_snr,
    historical_efficiency,
    historical_stationary_snrs,
)
from optimal_search.radar_detection import detection_probability, snr_db_to_linear

# Swerling I/II optima listed in Matthiesen's 2011 presentation (see docs/historical-source-verification.md).
REPORTED_DB = {1e-5: 11.4, 1e-6: 12.4}
SENSITIVITY_LOWER_DB = (-20.0, -30.0, -40.0, -50.0, -60.0)


def db(snr_linear: float) -> float:
    return 10.0 * math.log10(snr_linear)


def report(pfa: float) -> None:
    log_inv = -math.log(pfa)
    reported_db = REPORTED_DB[pfa]
    s_reported = snr_db_to_linear(reported_db)

    print("=" * 78)
    print(f"Pfa = {pfa:.0e}   (L = ln(1/Pfa) = {log_inv:.4f})")
    print("=" * 78)

    print("\n[A] Historical objective  E = Pd / SNR")
    print("    Behaviour as SNR -> 0 (E ~ Pfa/SNR, unbounded):")
    for snr_db_value in (-20.0, -40.0, -60.0, -80.0):
        s = snr_db_to_linear(snr_db_value)
        print(f"      SNR = {snr_db_value:6.1f} dB   E = {historical_efficiency(s, pfa):.6g}   E*SNR = {historical_efficiency(s, pfa) * s:.6g} (-> Pfa)")
    s_lo, s_hi = historical_stationary_snrs(pfa)
    print("    Closed-form stationary points (s^2 + (2-L)s + 1 = 0):")
    print(f"      local minimum  at {db(s_lo):9.4f} dB   E = {historical_efficiency(s_lo, pfa):.6g}")
    print(f"      local maximum  at {db(s_hi):9.4f} dB   E = {historical_efficiency(s_hi, pfa):.6g}")
    s_c = historical_boundary_crossover_snr(pfa)
    print(f"    Crossover: E(SNR) > E(local max) for SNR < {db(s_c):.2f} dB (windows reaching lower have boundary 'maxima')")

    analysis = analyze_objective(historical_efficiency, pfa)
    point, where = analysis.window_maximum
    lmax = analysis.local_maxima[0]
    print(f"    Numerical scan + bounded optimisation on [{analysis.lower_boundary.snr_db:.0f}, {analysis.upper_boundary.snr_db:.0f}] dB:")
    print(f"      interior local maximum {lmax.snr_db:9.4f} dB  (analytic {db(s_hi):.4f} dB)")
    print(f"      window maximum         {point.snr_db:9.4f} dB  E = {point.value:.4g}  -> {where}")
    print("    Window dependence of the historical 'global maximum':")
    for lower in SENSITIVITY_LOWER_DB:
        a = analyze_objective(historical_efficiency, pfa, lower_db=lower, upper_db=60.0)
        p, loc = a.window_maximum
        print(f"      lower bound {lower:6.1f} dB -> max at {p.snr_db:9.4f} dB ({loc})")
    print("    => no global maximum on (0, inf); the interior local maximum is the")
    print("       only candidate 'optimum' and is global only on windows above the crossover.")

    print("\n[B] Baseline-corrected objective  G = (Pd - Pfa) / SNR   [different objective]")
    print(f"    limit as SNR -> 0: Pfa*L = {baseline_corrected_limit_at_zero(pfa):.6g}  (finite, not a maximum)")
    s_g = baseline_corrected_optimum_snr(pfa)
    gan = analyze_objective(baseline_corrected_efficiency, pfa)
    gpoint, gwhere = gan.window_maximum
    print(f"    root-finder optimum    {db(s_g):9.4f} dB   G = {baseline_corrected_efficiency(s_g, pfa):.6g}")
    print(f"    numerical optimum      {gpoint.snr_db:9.4f} dB   ({gwhere}); local minima found: {len(gan.local_minima)}")
    print(f"    shift vs historical local max: {db(s_g) - db(s_hi):+.6f} dB")

    print(f"\n[C] Comparison with reported value {reported_db:.1f} dB")
    for name, f, s_opt in (("historical", historical_efficiency, s_hi), ("baseline-corrected", baseline_corrected_efficiency, s_g)):
        ratio = f(s_reported, pfa) / f(s_opt, pfa)
        print(
            f"    {name:19s} optimum {db(s_opt):8.4f} dB, reported - optimum = {reported_db - db(s_opt):+.4f} dB, "
            f"efficiency at reported SNR = {100 * ratio:.2f}% of optimum"
        )
    print(f"    Pd at reported SNR = {detection_probability(s_reported, pfa):.4f}; Pd at historical optimum = {detection_probability(s_hi, pfa):.4f}")
    print()


def main() -> None:
    print("Detection-efficiency optimisation, single-pulse Swerling I/II, Pd = Pfa^(1/(1+SNR))\n")
    for pfa in REPORTED_DB:
        report(pfa)
    print("Note: a numerical agreement or disagreement with the reported values does not")
    print("validate or refute the historical derivation; its objective and SNR definition")
    print("may differ from those used here.")


if __name__ == "__main__":
    main()
