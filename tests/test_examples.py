"""The README's example commands run and print the documented values."""

import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("name, expected", [
    ("detection_efficiency", ["13.855428     11.416       0.4607     11.4 dB", "17.201555     12.356       0.4681     12.4 dB",
                              "3 looks of SNR 12.0977, energy 36.293"]),
    ("bayesian_update", ["Posterior after the miss: (0.13158, 0.26316, 0.21053, 0.39474)", "searches [1, 0]: P(detected) = 0.5000"]),
    ("greedy_vs_exact", ["greedy plan [1, 2]: P(detected) = 0.4800", "exact plan  [0]: P(detected) = 0.5200",
                         "greedy plan [1, 2]: P(detected) = 0.3700", "exact plan  [0]: P(detected) = 0.4000"]),
    ("essay_numbers", ["valley at strength 13.8554 = 11.416 dB", "reaches 37%", "reaches 40%"]),
])
def test_example_prints_documented_values(name, expected, capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", [name])
    runpy.run_path(str(EXAMPLES / f"{name}.py"), run_name="__main__")
    output = capsys.readouterr().out
    for text in expected:
        assert text in output
