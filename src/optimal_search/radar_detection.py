"""Single-pulse radar detection probability for a Swerling I/II target.

Model
-----
For one pulse, a target whose radar cross-section fluctuates according to the
Swerling I/II model (exponentially distributed received power), observed in
complex Gaussian noise by a matched filter followed by a square-law detector
and a fixed threshold, has detection probability::

    Pd = Pfa ** (1 / (1 + SNR))

where ``Pfa`` is the false-alarm probability, ``SNR`` is the *linear* mean
signal-to-noise power ratio (not dB), and ``Pd`` is the detection probability.

Derivation
----------
Normalise the noise power at the detector to 1. The square-law detector output
``y`` is then exponentially distributed with mean 1 under noise only, and
exponentially distributed with mean ``1 + SNR`` under signal plus noise (a
Rayleigh-fading target makes the in-phase and quadrature components of the
signal Gaussian, so signal plus noise is again complex Gaussian with total
power ``1 + SNR``). For a threshold ``T``::

    Pfa = P(y > T | noise only)     = exp(-T)
    Pd  = P(y > T | signal + noise) = exp(-T / (1 + SNR))

Eliminating ``T = -ln(Pfa)`` gives ``Pd = Pfa ** (1 / (1 + SNR))``.

Source
------
This is the standard textbook single-pulse result for a Swerling I/II target
(see e.g. M. I. Skolnik, *Introduction to Radar Systems*, and M. A. Richards,
*Fundamentals of Radar Signal Processing*, treatments of fluctuating-target
detection). The derivation above is self-contained, so the formula can be
checked without either book. Page and equation numbers have not been
verified against a specific edition.

Why Swerling I and II coincide here
-----------------------------------
Swerling I and II share the same exponential (Rayleigh-amplitude) power
distribution. They differ only in how fast the target decorrelates: I is
constant over all pulses of a scan and independent between scans, II is
independent from pulse to pulse. With a single pulse there is only one draw
of the target power, so the two models are indistinguishable.

Assumptions and limitations
---------------------------
* Exactly one pulse; no pulse integration of any kind.
* Exponentially distributed target power (Swerling I/II). Not valid for
  non-fluctuating (Swerling 0 / Marcum) or Swerling III/IV targets.
* Complex Gaussian noise, matched filter, square-law (or equivalent
  envelope) detector, fixed threshold, i.e. no CFAR loss.
* ``SNR`` is the mean (averaged over target fluctuation) per-pulse power SNR
  at the detector input. Losses (propagation, beam shape, processing) must
  be included in ``SNR`` by the caller.
* Single resolution cell and single target; no clutter, interference or
  multi-target effects.
"""

import math


def snr_db_to_linear(snr_db: float) -> float:
    """Convert an SNR in decibels to a linear power ratio: ``10 ** (dB / 10)``."""
    if not math.isfinite(snr_db):
        raise ValueError(f"snr_db must be finite, got {snr_db!r}")
    return 10.0 ** (snr_db / 10.0)


def snr_linear_to_db(snr_linear: float) -> float:
    """Convert a linear SNR power ratio to decibels: ``10 * log10(SNR)``.

    The linear SNR must be strictly positive; 0 has no finite dB value.
    """
    if not math.isfinite(snr_linear) or snr_linear <= 0.0:
        raise ValueError(f"snr_linear must be finite and > 0, got {snr_linear!r}")
    return 10.0 * math.log10(snr_linear)


def detection_probability(snr_linear: float, pfa: float) -> float:
    """Single-pulse detection probability for a Swerling I/II target.

    Computes ``Pd = pfa ** (1 / (1 + snr_linear))``; see the module docstring
    for the derivation, assumptions and limitations.

    Args:
        snr_linear: Mean signal-to-noise power ratio on a linear scale
            (not dB). Must be finite and >= 0; 0 means no signal.
        pfa: Probability of false alarm, strictly between 0 and 1.

    Returns:
        Probability of detection, in ``[pfa, 1]``.

    Raises:
        ValueError: If ``snr_linear`` is negative or not finite, or if
            ``pfa`` is not finite and strictly between 0 and 1.
    """
    if not math.isfinite(snr_linear) or snr_linear < 0.0:
        raise ValueError(f"snr_linear must be finite and >= 0, got {snr_linear!r}")
    if not math.isfinite(pfa) or not 0.0 < pfa < 1.0:
        raise ValueError(f"pfa must be strictly between 0 and 1, got {pfa!r}")
    return pfa ** (1.0 / (1.0 + snr_linear))
