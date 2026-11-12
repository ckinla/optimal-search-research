"""Tests for the single-pulse Swerling I/II detection probability model."""

import math

import pytest

from optimal_search.radar_detection import (
    detection_probability,
    snr_db_to_linear,
    snr_linear_to_db,
)

PFAS = [1e-9, 1e-6, 1e-5, 1e-3, 0.1, 0.5]


@pytest.mark.parametrize(
    "snr_linear, pfa, expected",
    [
        # Exact by hand: Pfa ** (1 / (1 + SNR)) with an integer exponent root.
        (1.0, 1e-6, 1e-3),  # (1e-6) ** (1/2)
        (2.0, 1e-6, 1e-2),  # (1e-6) ** (1/3)
        (3.0, 1e-8, 1e-2),  # (1e-8) ** (1/4)
        (1.0, 0.25, 0.5),  # 0.25 ** (1/2)
        (3.0, 1.0 / 16.0, 0.5),  # (1/16) ** (1/4)
    ],
)
def test_exact_hand_calculated_examples(snr_linear, pfa, expected):
    assert detection_probability(snr_linear, pfa) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "snr_db, pfa, expected",
    [
        # Reference values computed independently as exp(ln(Pfa) / (1 + 10**(dB/10))).
        (0, 1e-5, 0.003162277660),
        (10, 1e-5, 0.351119173422),
        (13, 1e-5, 0.577253260775),
        (20, 1e-5, 0.892267448630),
        (25, 1e-5, 0.964358369685),
        (0, 1e-6, 0.001000000000),
        (10, 1e-6, 0.284803586844),
        (13, 1e-6, 0.517177561113),
        (20, 1e-6, 0.872155772144),
        (25, 1e-6, 0.957383955279),
    ],
)
def test_reference_values_in_db(snr_db, pfa, expected):
    pd = detection_probability(snr_db_to_linear(snr_db), pfa)
    assert pd == pytest.approx(expected, abs=1e-11)


@pytest.mark.parametrize("pfa", PFAS)
def test_zero_snr_gives_pd_equal_pfa(pfa):
    assert detection_probability(0.0, pfa) == pytest.approx(pfa, rel=1e-12)


@pytest.mark.parametrize("pfa", PFAS)
def test_pd_increases_with_snr(pfa):
    snrs = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 31.6, 100.0, 1000.0]
    pds = [detection_probability(s, pfa) for s in snrs]
    assert all(a < b for a, b in zip(pds, pds[1:]))


@pytest.mark.parametrize("pfa", PFAS)
def test_pd_decreases_as_pfa_decreases(pfa):
    assert detection_probability(10.0, pfa / 10.0) < detection_probability(10.0, pfa)


@pytest.mark.parametrize("pfa", PFAS)
@pytest.mark.parametrize("snr_linear", [0.0, 1e-12, 0.5, 1.0, 10.0, 1e3, 1e9, 1e300])
def test_pd_within_unit_interval_and_not_below_pfa(snr_linear, pfa):
    pd = detection_probability(snr_linear, pfa)
    assert 0.0 < pd <= 1.0
    assert pd >= pfa


@pytest.mark.parametrize("pfa", PFAS)
@pytest.mark.parametrize("snr_linear", [1e2, 1e4, 1e6, 1e9])
def test_pd_high_snr_miss_probability_bound(snr_linear, pfa):
    # Pd = exp(-x) with x = ln(1/Pfa) / (1 + SNR), and 1 - exp(-x) <= x, so the
    # miss probability is bounded by x and tends to 0 as SNR grows.
    miss = 1.0 - detection_probability(snr_linear, pfa)
    bound = math.log(1.0 / pfa) / (1.0 + snr_linear)
    # Allow a few ULPs: 1 - Pd loses precision when Pd is within ~1e-10 of 1.
    assert 0.0 <= miss <= bound + 4e-16


@pytest.mark.parametrize("pfa", PFAS)
def test_pd_approaches_one_at_very_high_snr(pfa):
    assert detection_probability(1e12, pfa) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("pfa", [0.0, 1.0, -0.1, 1.5, math.nan, math.inf, -math.inf])
def test_invalid_pfa_rejected(pfa):
    with pytest.raises(ValueError, match="pfa"):
        detection_probability(10.0, pfa)


@pytest.mark.parametrize("snr", [-1e-9, -1.0, math.nan, math.inf, -math.inf])
def test_invalid_snr_rejected(snr):
    with pytest.raises(ValueError, match="snr_linear"):
        detection_probability(snr, 1e-6)


@pytest.mark.parametrize(
    "snr_db, snr_linear",
    [(0.0, 1.0), (10.0, 10.0), (20.0, 100.0), (30.0, 1000.0), (-10.0, 0.1), (3.0, 1.9952623149688795)],
)
def test_known_db_linear_conversions(snr_db, snr_linear):
    assert snr_db_to_linear(snr_db) == pytest.approx(snr_linear, rel=1e-12)
    assert snr_linear_to_db(snr_linear) == pytest.approx(snr_db, abs=1e-12)


@pytest.mark.parametrize("snr_db", [-30.0, -3.0, 0.0, 0.5, 13.0, 25.0, 60.0])
def test_db_linear_round_trip(snr_db):
    assert snr_linear_to_db(snr_db_to_linear(snr_db)) == pytest.approx(snr_db, abs=1e-12)


@pytest.mark.parametrize("snr_linear", [1e-6, 0.5, 1.0, 31.6, 1e6])
def test_linear_db_round_trip(snr_linear):
    assert snr_db_to_linear(snr_linear_to_db(snr_linear)) == pytest.approx(snr_linear, rel=1e-12)


@pytest.mark.parametrize("snr_linear", [0.0, -1.0, math.nan, math.inf])
def test_linear_to_db_rejects_invalid(snr_linear):
    with pytest.raises(ValueError, match="snr_linear"):
        snr_linear_to_db(snr_linear)


@pytest.mark.parametrize("snr_db", [math.nan, math.inf, -math.inf])
def test_db_to_linear_rejects_non_finite(snr_db):
    with pytest.raises(ValueError, match="snr_db"):
        snr_db_to_linear(snr_db)
