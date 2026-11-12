"""Print single-pulse Swerling I/II detection probability versus SNR.

Run from the repository root::

    uv run python -m scripts.detection_table
"""

from optimal_search.radar_detection import detection_probability, snr_db_to_linear

PFAS = (1e-5, 1e-6)
SNR_DB_RANGE = range(0, 26)


def main() -> None:
    print("Single-pulse Swerling I/II detection probability, Pd = Pfa^(1/(1+SNR))")
    print()
    header = f"{'SNR [dB]':>8}  {'SNR [lin]':>10}" + "".join(
        f"  {f'Pd (Pfa={pfa:.0e})':>20}" for pfa in PFAS
    )
    print(header)
    print("-" * len(header))
    for snr_db in SNR_DB_RANGE:
        snr = snr_db_to_linear(snr_db)
        row = f"{snr_db:>8}  {snr:>10.3f}" + "".join(
            f"  {detection_probability(snr, pfa):>20.6f}" for pfa in PFAS
        )
        print(row)


if __name__ == "__main__":
    main()
