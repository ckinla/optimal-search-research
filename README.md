# Optimal Search Research

[![tests](https://github.com/ckinla/optimal-search-research/actions/workflows/tests.yml/badge.svg)](https://github.com/ckinla/optimal-search-research/actions/workflows/tests.yml)

**How should a radar spend limited observation energy and search effort to find a target?**

A radar can make one strong look or several weaker ones, and it must decide where to look next after a miss. This
repository contains reproducible Python models, tests and numerical checks for both questions:

- **How strong should each look be?** For independent looks, there is an efficient strength per look that reaches a
  detection goal on the least total energy: 11.416 dB at a false-alarm probability of 10⁻⁵.
- **Where should the next look go?** After each miss, Bayes' rule moves belief away from the place just searched.
  Searching where the immediate payoff per cost is highest is a good rule, but under a finite budget with unequal
  costs it can lose to a plan that looks further ahead.

It is the code behind the visual essay [*The Mathematics of Looking*](https://optimal-search-lab.vercel.app/), and
an independent investigation of the radar search and detection research of **Duane J. Matthiesen** (1999–2011).

## Quick start

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/); it installs Python 3.12 if needed.

```sh
git clone https://github.com/ckinla/optimal-search-research.git
cd optimal-search-research
uv sync
uv run pytest
uv run python examples/detection_efficiency.py
```

`uv sync` installs the locked versions of NumPy, SciPy and pytest from `uv.lock`.

## Reproduce the key results

### 1. The detection-efficiency operating points

```sh
uv run python examples/detection_efficiency.py
```

```text
Local detection-efficiency maximum of H(s) = -ln(1 - Pd(s)) / s
   Pfa   SNR (linear)   SNR (dB)  Pd per look  2011 table  local min of H
 1e-05      13.855428     11.416       0.4607     11.4 dB          0.1063
 1e-06      17.201555     12.356       0.4681     12.4 dB          0.0852

Energy for an 80% cumulative detection goal at Pfa = 1e-05 (energy = looks x SNR):
  SNR   2.000 (  3.01 dB):  73.90 looks, energy 147.792
  SNR   5.000 (  6.99 dB):  10.14 looks, energy  50.695
  SNR  10.000 ( 10.00 dB):   3.72 looks, energy  37.212
  SNR  13.855 ( 11.42 dB):   2.61 looks, energy  36.113  <- local optimum
  SNR  20.000 ( 13.01 dB):   1.87 looks, energy  37.313
  SNR  40.000 ( 16.02 dB):   1.14 looks, energy  45.748
  Best whole number of looks: 3 looks of SNR 12.0977, energy 36.293
...
```

The model is the single-pulse Swerling I/II detection curve, with `s` the linear signal-to-noise ratio of one look:

```text
Pd(s) = Pfa^(1/(1+s))            probability that one look detects the target
H(s)  = −ln(1 − Pd(s)) / s       detection efficiency of independent looks
E     = −ln(1 − Pc) / H(s)       energy needed for a cumulative detection goal Pc
```

Maximising `H` minimises energy for any goal `Pc`. Its stationary condition is algebraically identical to the
*Optimal Detection Condition* in Matthiesen's 2011 presentation, and its roots round to the 11.4 and 12.4 dB listed
there. No parameter was fitted.

### 2. Bayesian updating after missed searches

```sh
uv run python examples/bayesian_update.py
```

Reproduces the 2011 presentation's *Search Example 2* (prior 0.1, 0.2, 0.4, 0.3; detection 0.6; a miss in cell 3
gives the posterior 0.13158, 0.26316, 0.21053, 0.39474) and *Search Example 1*, and shows that the probability of
detection depends only on how often each cell was searched, not on the order.

### 3. Greedy versus exact finite-budget allocation

```sh
uv run python examples/greedy_vs_exact.py
```

With perfect detection, costs (2, 1, 1) and a budget of 2, the greedy rule reaches 0.48 while the exact optimum,
found by exhaustive enumeration, reaches 0.52. A seeded Monte Carlo simulation corroborates both probabilities.

### Every number in the visual essay

```sh
uv run python examples/essay_numbers.py          # summary
uv run python examples/essay_numbers.py --json   # all values
```

### Longer reports

| Command | What it prints |
| --- | --- |
| `uv run python -m scripts.detection_table` | `Pd` from 0 to 25 dB at `Pfa` = 10⁻⁵ and 10⁻⁶ |
| `uv run python -m scripts.detection_efficiency_report` | Single-look objectives and why they do not reproduce 11.4 dB |
| `uv run python -m scripts.cumulative_detection_report` | Cumulative-detection optima, boundaries and sensitivity |
| `uv run python -m scripts.spatial_search_exact_vs_greedy` | Greedy against exact plans over a grid of instances |
| `uv run python -m scripts.monte_carlo_greedy_vs_exact` | Monte Carlo checks of greedy and exact plans |
| `uv run python -m scripts.matthiesen_audit_examples` | Fixed-goal energy, whole looks and a joint where-and-how-strongly toy model |

## What the code models

| Module | Contents |
| --- | --- |
| `optimal_search.radar_detection` | Single-pulse Swerling I/II detection probability; dB conversions |
| `optimal_search.detection_efficiency` | Single-look objectives `Pd/s` and `(Pd − Pfa)/s`, with closed-form stationary points |
| `optimal_search.cumulative_detection` | Independent looks: `H(s)`, its stationary points, required SNR and energy for a goal |
| `optimal_search.spatial_search` | One stationary target in `M` cells: posterior updates, unconditional detection probability, greedy plans |
| `optimal_search.exact_spatial_search` | Exhaustive finite-budget optimum, expected cost to detection, fractional upper bound |
| `optimal_search.monte_carlo_spatial_search` | Seeded simulation of a fixed plan with confidence intervals |

Nearly 1,000 tests check these against closed forms, high-precision arithmetic, brute-force enumeration of the probability
space, transcribed equations and worked examples, and simulation.

## Connection to Matthiesen's research

Between 1999 and 2011 Matthiesen published papers and gave a talk on optimal radar search: where to point, how much
effort to spend per look, and how often to revisit, under limited energy and time. We rebuilt parts of that
mathematics independently and tested them against the equations and examples we could read:

- **Reproduced:** the 2011 Optimal Detection Condition and its 11.4 / 12.4 dB table values; the posterior update and
  payoff-per-cost ranking of the 1999 and 2003 papers; the 2011 worked search examples.
- **Found:** the 1999 and 2003 papers' "actual" cumulative probability adds conditional probabilities as if they were
  unconditional. It depends on the search order and can exceed 1; the 2011 worked example uses the correct weighting.
- **Not reproduced:** the integrated beam-scanning, energy and time allocation method described in the 2007 and 2010
  papers. Only their abstracts were available to us.

[`docs/provenance.md`](docs/provenance.md) separates foundational results, Matthiesen's work, our reproduction and our
own extensions, and lists the bibliography with DOIs.

## Relation to classical search theory

The spatial-search part is classical Bayesian search for one stationary target. Koopman (1957) and de Guenin (1961)
studied the optimum distribution of search effort, and Black (1965) gave the cost-weighted index for discrete
sequential search with imperfect detection to which the payoff-per-cost rule here is closely related. Bayes' rule and
the greedy index are not Matthiesen's inventions; his work applies these ideas to radar detection effort, repeated
looks and search payoff, including an explicit detection condition.

## Assumptions and limitations

- **Detection model.** Single-pulse Swerling I/II with square-law detection in Gaussian noise. No pulse integration,
  CFAR loss, beam shape or hardware limits.
- **Independent looks.** A Swerling I target is correlated within a scan, so independent looks are an assumption.
- **False alarms.** `Pfa` is fixed per look, so the chance of at least one false alarm grows with the number of looks.
- **Local, not global.** Since `Pd(0) = Pfa > 0`, `H(s)` diverges as `s → 0`: in the formula, vanishingly weak looks
  "detect" on false alarms alone. The 11.416 dB optimum is a local optimum of a simplified model, not a universal
  radar setting. With whole looks the best design costs slightly more (3 looks: +0.5% energy at an 80% goal).
- **Spatial search.** One stationary target; constant, independent detection probabilities per search; no false
  alarms. Greedy selection is myopic: optimal with equal costs for a fixed number of searches, not in general.
- **Monte Carlo** results are estimates with sampling error that corroborate the exact calculations; they prove
  nothing on their own.
- **No real radar data.** Every number is a model calculation, not operational performance.

## Further reading

- [`docs/provenance.md`](docs/provenance.md): attribution, what could and could not be verified, bibliography.
- [`docs/detection-efficiency.md`](docs/detection-efficiency.md): the detection models, objectives and their limits.
- [`docs/historical-source-verification.md`](docs/historical-source-verification.md): the 2011 Optimal Detection
  Condition compared with `H(s)`.
- [`docs/spatial-search-verification.md`](docs/spatial-search-verification.md): Bayesian search and the 1999/2003
  cumulative-probability discrepancy.
- [`docs/exact-spatial-search.md`](docs/exact-spatial-search.md): the exact solver and when greedy falls short.
- [`docs/monte-carlo-spatial-search.md`](docs/monte-carlo-spatial-search.md): the simulation engine and its intervals.

## Maintenance

This repository is a published release of the Python research behind Optimal Search Lab. The models in
`src/optimal_search/` are maintained in the lab's development repository, which also holds a separate TypeScript
implementation used by the visual essay and checks the two against each other. Changes to a published result are
released here with a version number, the reason for the change and the tests that establish it.

## License

The code and documentation in this repository are released under the [MIT License](LICENSE). The license does not
cover the cited papers and presentation, which belong to their authors and publishers. The repository contains no
copies of them; see [provenance](docs/provenance.md#7-source-material-and-rights).
