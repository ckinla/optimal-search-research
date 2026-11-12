# Detection efficiency: model, results and limits

This note covers `src/optimal_search/radar_detection.py`, `src/optimal_search/detection_efficiency.py` and
`src/optimal_search/cumulative_detection.py`. Attribution is summarised in [provenance](provenance.md); the comparison
with the 2011 presentation's equations is in [historical-source verification](historical-source-verification.md).
Run the short version with `uv run python examples/detection_efficiency.py`.

## Radar detection probability model

`src/optimal_search/radar_detection.py` implements the standard single-pulse detection probability for a Swerling I/II fluctuating target with square-law detection:

```
Pd = Pfa ** (1 / (1 + SNR))
```

`SNR` is the *linear* mean per-pulse SNR (not dB); `snr_db_to_linear` and `snr_linear_to_db` convert between the two. The module docstring gives the derivation, source, assumptions and limitations (single pulse, exponential target power, Gaussian noise, no integration, no CFAR loss). This is a standard textbook model, not a reproduction of any specific optimal-detection result.

Print `Pd` for 0–25 dB at `Pfa = 1e-5` and `1e-6`:

```sh
uv run python -m scripts.detection_table
```

## Single-look detection efficiency

`src/optimal_search/detection_efficiency.py` studies how detection efficiency varies with SNR for the single-pulse model above. Two **different** objectives are implemented (SNR linear, `L = ln(1/Pfa)`):

| Objective | Definition | Behaviour |
|-----------|------------|-----------|
| Historical | `Pd / SNR` | Diverges as `SNR -> 0` because `Pd(0) = Pfa > 0`, so there is **no global maximum** on `(0, inf)`. Closed-form stationary points solve `s^2 + (2 - L) s + 1 = 0`: a local minimum at `s_lo` and a local maximum at `s_hi = 1/s_lo`, **only for `Pfa < e^-4`** (about 0.018); at `Pfa = e^-4` the roots merge into a stationary inflection, and above it there is no stationary point and the objective simply decreases. |
| Baseline-corrected | `(Pd - Pfa) / SNR` | Finite limit `Pfa * L` at zero; for **`Pfa < e^-2`** (about 0.135) a single interior global maximum (checked numerically, not proven). For larger `Pfa` (e.g. 0.5) it decreases from its zero-SNR limit, so its largest value is at that boundary and there is no interior optimum. This is a modelling choice (`Pd - Pfa` is Youden's J), not part of the historical objective. |

Results (`uv run python -m scripts.detection_efficiency_report`):

| Pfa | Historical local max | Baseline-corrected max | Reported (Matthiesen) | Reported − historical |
|-----|----------------------|------------------------|-----------------------|-----------------------|
| 1e-5 | 9.734 dB | 9.734 dB (+0.0002 dB) | 11.4 dB | +1.67 dB |
| 1e-6 | 10.693 dB | 10.693 dB (+0.00002 dB) | 12.4 dB | +1.71 dB |

The single-look optima do **not** reproduce the reported values, and nothing (equations, objective, bounds) was adjusted to force agreement. The baseline correction is negligible for these `Pfa`, so it does not explain the gap. On a finite dB window the historical objective's largest value sits on the lower boundary (an artifact of the window) unless the window starts above a crossover SNR (about −35 dB for 1e-5, −45 dB for 1e-6). See the module docstring for the full assumptions and limitations.

## Cumulative detection over N independent looks

`src/optimal_search/cumulative_detection.py` adds a separate, *independently derived* objective (the two single-look objectives above are unchanged). For `N` independent looks at SNR `s`, `Pc = 1 - (1 - Pd)^N`, so `N = -ln(1 - Pc) / q(s)` with `q = -ln(1 - Pd)`. With effort proportional to `N * s`:

```
effort = N * s = [-ln(1 - Pc)] / H(s),     H(s) = -ln(1 - Pd(s)) / s
```

For a fixed `Pc` and continuous `N`, minimising effort is exactly maximising `H`; the optimal SNR does not depend on `Pc` (also confirmed by minimising `N * s(N)` directly over `N`). `H` diverges as `s -> 0` just like `Pd / s`, because at zero SNR the "cumulative detection" is produced by false alarms alone, so again there is **no global maximum**. Since `q''` has exactly one sign change (analytic, see the module docstring), `H` has at most a local minimum then a local maximum; both exist for the `Pfa` studied here, but not for every `Pfa` (e.g. none at 0.5; `cumulative_stationary_snrs` returns an empty tuple then). The baseline-corrected log objective `[-ln(1 - Pd) + ln(1 - Pfa)] / s` (a different target: a fixed excess log-survival ratio, not a fixed `Pc`) has a finite limit at zero and, **for `Pfa < e^-x*` (about 0.20, `x* ~ 1.594`)**, a provably unique interior maximum; for larger `Pfa` its maximum is the zero-SNR boundary.

Results (`uv run python -m scripts.cumulative_detection_report`):

| Pfa | `H` local max | Pd per look | Baseline-corrected log max | 2011 presentation | Computed − listed |
|-----|---------------|-------------|----------------------------|----------------|---------------------|
| 1e-5 | 11.416 dB | 0.461 | 11.416 dB (+0.0001 dB) | 11.4 dB (Pd ~ 47%) | +0.016 dB |
| 1e-6 | 12.356 dB | 0.468 | 12.356 dB (+0.00001 dB) | 12.4 dB (Pd ~ 47%) | −0.044 dB |

No parameter was fitted. The computed maxima round to the presentation's values, which is a consistency observation and **not** proof of any historical derivation: those values were known when this objective was proposed, and the per-look Pd follows from the SNR and `Pfa` through the single-pulse model, so it is not an independent confirmation of the objective (the source lists one Pd, 0.47, for both `Pfa`; ours is 0.461 and 0.468). See [historical-source verification](historical-source-verification.md) for a comparison against the equations in Matthiesen's 22 March 2011 presentation: the optimal-detection *condition* is algebraically identical to the stationarity condition of `H`; `H` itself is not printed there and matches the source's payoff ratio only at fixed `Pc`; the single-look `Pd/SNR` is the source's `N_L = 1` special case, not its optimum condition.

Limitations: looks must be independent and identically distributed (Swerling-II-like; a Swerling I target is fully correlated within a scan); `Pfa` is fixed *per look*, so the cumulative false-alarm probability grows with `N`; effort is taken proportional to `N * s`; and `N` is continuous. The continuous optimum is a **local** minimum of the effort `N * s(N)`, not a global lower bound: as `N` approaches the false-alarm-only value (where `s -> 0`) the effort tends to zero (for `Pc = 0.9`, `Pfa = 1e-5`, `N = 230000` costs about 22 versus about 52 at the optimum), which is the same zero-SNR boundary divergence as above. It also requires `N* >= 1`, i.e. `Pc >= Pd(s_hi)` (about 0.46); for smaller `Pc` the relaxation is infeasible. For the example `Pc = 0.9`, among integer designs with `N = 1..40` the best (`N = 4`) is within about 0.1–0.2% of that local optimum.
