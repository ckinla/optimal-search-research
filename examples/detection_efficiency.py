"""Reproduce the detection-efficiency operating points (11.416 dB and 12.356 dB).

Model: single-pulse Swerling I/II detection, Pd(s) = Pfa**(1/(1+s)), with s the linear SNR of one look.
For independent identical looks, the energy needed to reach a cumulative detection goal Pc is
-ln(1-Pc)/H(s), where H(s) = -ln(1-Pd(s))/s. The efficient strength is the local maximum of H, whose
stationarity condition is the Optimal Detection Condition in Matthiesen's 2011 presentation.

Run: uv run python examples/detection_efficiency.py
"""

import math

from scipy.optimize import brentq

from optimal_search.cumulative_detection import cumulative_efficiency, cumulative_stationary_snrs, required_snr_for_looks
from optimal_search.radar_detection import detection_probability, snr_linear_to_db

PRESENTATION_DB = {1e-5: 11.4, 1e-6: 12.4}  # listed in the 2011 presentation; not used in any calculation
GOAL = 0.8


def main() -> None:
    print("Local detection-efficiency maximum of H(s) = -ln(1 - Pd(s)) / s")
    print(f"{'Pfa':>6}  {'SNR (linear)':>13}  {'SNR (dB)':>9}  {'Pd per look':>11}  {'2011 table':>10}  {'local min of H':>14}")
    for pfa, listed in PRESENTATION_DB.items():
        lower, upper = cumulative_stationary_snrs(pfa)
        print(f"{pfa:>6g}  {upper:>13.6f}  {snr_linear_to_db(upper):>9.3f}  "
              f"{detection_probability(upper, pfa):>11.4f}  {listed:>7.1f} dB  {lower:>14.4f}")

    pfa = 1e-5
    lower, upper = cumulative_stationary_snrs(pfa)
    peak = cumulative_efficiency(upper, pfa)
    print(f"\nEnergy for an {GOAL:.0%} cumulative detection goal at Pfa = {pfa:g} (energy = looks x SNR):")
    for snr in (2.0, 5.0, 10.0, upper, 20.0, 40.0):
        looks = -math.log1p(-GOAL) / (cumulative_efficiency(snr, pfa) * snr)
        marker = "  <- local optimum" if snr == upper else ""
        print(f"  SNR {snr:7.3f} ({snr_linear_to_db(snr):6.2f} dB): {looks:6.2f} looks, energy {looks * snr:7.3f}{marker}")
    whole = min(range(1, 13), key=lambda n: n * required_snr_for_looks(n, GOAL, pfa))
    snr = required_snr_for_looks(whole, GOAL, pfa)
    print(f"  Best whole number of looks: {whole} looks of SNR {snr:.4f}, energy {whole * snr:.3f}")

    print("\nThe optimum is local, not global. Pd(0) = Pfa > 0, so H(s) diverges as s -> 0, and it")
    crossing = brentq(lambda s: cumulative_efficiency(s, pfa) - peak, 1e-12, lower)
    print(f"exceeds the peak again below s = {crossing:.6f} (H has a local minimum at s = {lower:.3f}):")
    for tiny in (lower, 1e-3, 2e-4, 1e-5):
        print(f"  H({tiny:g}) = {cumulative_efficiency(tiny, pfa):.4f}  versus  H({upper:.3f}) = {peak:.4f}")
    print("  Vanishingly weak looks 'reach' the goal on false alarms alone. The stationary point is therefore a")
    print("  local optimum of a simplified model, not a universal radar operating point.")


if __name__ == "__main__":
    main()
