# Monte Carlo spatial-search engine

Implementation: `src/optimal_search/monte_carlo_spatial_search.py`. Tests: `tests/test_monte_carlo_spatial_search.py`. Demonstration: `uv run python -m scripts.monte_carlo_greedy_vs_exact`.

## Estimates, not proofs

The engine estimates probabilities and expected costs by randomised experiment. An estimate has sampling error and **proves nothing**. Agreement with an analytical formula within the confidence interval is evidence that the formula and the simulation describe the same model. A persistent disagreement well beyond sampling error would indicate a defect in one of them. A confidence interval says nothing about model error (a wrong model simulates just as precisely), and a 95% interval misses the true value in about 1 run in 20 by design. The analytical results in this project rest on the derivations and brute-force enumerations in the earlier PRs; the simulation is an independent cross-check of them.

## Separation from the mathematical solvers

The engine imports nothing from `research.spatial_search` or `research.exact_spatial_search`, and a test checks that. A plan is a plain sequence of cell indices (for example `[step.cell for step in greedy_search_plan(...)]` or `solve_exact(...).sequence`), so any plan can be simulated. The engine never uses the analytical survival formula `(1 − d)^n`: detection is simulated attempt by attempt, and a test checks that the source contains no power operator. Historical equations are not touched, and no new search strategies or radar energy models are added.

## Model and algorithm

One stationary target is in one of `M` cells drawn from the prior. The plan is executed look by look. A look in cell `j` costs `cost[j]` whether or not it succeeds; if the target is in `j` it detects with probability `detection[j]`, independently of every other look, and a look elsewhere cannot detect it. The search stops immediately on the first detection or when the plan is exhausted.

Vectorised implementation:

1. Place all targets: `numpy.random.Generator.random(trials)` compared with the cumulative prior (`searchsorted`, `side="right"`, so zero-probability cells are never chosen).
2. For each trial and each look draw one uniform and compare it with the look's detection probability; a look succeeds only if it is in the target's cell. The first success per trial is found with `argmax`. This is done in row chunks of at most `max_chunk_elements` uniforms (default 4 million, 32 MB) to bound memory; if a single trial's plan is longer than the limit its row is drawn in consecutive blocks of at most that size, which continues the same stream and so gives the same values as a whole-row draw.
3. Looks performed, cost incurred (cumulative plan cost through the last look performed) and the detection cell and look index follow from the first success.

## Reproducibility

Every run takes an explicit seed (a non-negative integer or a `numpy.random.SeedSequence`); `None` is rejected. Target placements are drawn first and the attempt uniforms are then drawn in row-major order, which NumPy fills sequentially, so **the result for a seed does not depend on the chunk size** (tested for chunks from 1 to 4 million). `simulate_replications` spawns independent, non-overlapping child streams from one seed (`SeedSequence.spawn`), so a whole set of replications is reproducible and the replications are statistically independent; a `SeedSequence` passed in is not advanced by the call. NumPy does not guarantee identical streams across versions, so exact random values are reproducible for a pinned NumPy (this project pins it in `uv.lock`). For that reason the tests assert statistical properties with wide tolerances and exact internal invariants, never hard-coded random outputs.

## What is recorded

Per trial: the true target cell, whether it was detected, looks performed, cost incurred, and (`-1` if not detected) the detection cell and the 0-based index of the successful look in the plan (`looks == detection_look + 1`). Aggregates: empirical detection probability, mean looks and mean cost with standard errors, target placements per cell and detections per cell, and the confidence intervals below.

## Confidence intervals: how they are computed and their limits

- **Detection probability** (binomial proportion, `k` detections in `n` independent trials, level `1 − α`):
  - *Wilson score* (default): `(p̂ + z²/2n ± z·sqrt(p̂(1 − p̂)/n + z²/4n²)) / (1 + z²/n)` with `z` the `1 − α/2` normal quantile. It is approximate: its actual coverage oscillates around the nominal level and is worst for small `n` and for `p` near 0 or 1.
  - *Clopper–Pearson* (`method="clopper-pearson"`): beta-quantile ("exact") interval, `Beta(α/2; k, n − k + 1)` to `Beta(1 − α/2; k + 1, n − k)`, with lower end 0 when `k = 0` and upper end 1 when `k = n`. It is conservative: coverage is at least nominal, often higher, so it is wider.
- **Mean looks and mean cost**: normal approximation `mean ± z·s/√n` with the sample standard deviation (`ddof = 1`), justified by the central limit theorem. These variables are bounded, often skewed, and sometimes nearly constant, so it can be poor for small `n`; a zero sample variance gives a zero-width interval (for example a plan that always costs the same), and `n = 1` gives no interval (`nan`).
- All intervals assume independent identically distributed trials, which the engine provides by construction, and describe **sampling error only**.
- Comparing many quantities at once makes an occasional miss expected: the tests use a single-comparison threshold of `|z| ≤ 4.5` (a two-sided probability of about 7×10⁻⁶ if the engine is right) so that the suite is not flaky.

Calibration observed in development (300 independent replications per setting, one imperfect-detection plan, true `P = 0.772`): nominal 95% Wilson coverage 0.937–0.963 at `n = 500` and `n = 1000`, Clopper–Pearson 0.943–0.963, the mean-cost interval 0.910–0.963, and the `z`-scores of the estimate had mean within ±0.16 and standard deviation 0.94–1.05. For a small probability (`P = 0.02`, `n = 500`) Wilson covered 0.957 and Clopper–Pearson 0.970. These are sample outcomes with their own sampling error (the standard deviation of a coverage estimate from 300 replications is about 0.013), and the mean-cost interval is the most variable of the three.

## Validation

Tests compare the simulation with **independent** analytical values: the closed form `1 − Σ p_j(1 − d_j)^n_j`, a brute-force enumeration of every target cell and detection outcome (for detection probability, expected cost and expected number of looks), and the exact-solver module's expected-cost function. They check, for perfect and imperfect detection and unequal priors, probabilities and costs: detection probability, per-cell detection probability, the distribution of the first-detection look, target placement against the prior, mean cost and mean looks; exact invariants (perfect detection finds the target exactly when its cell is searched; zero detection never detects; early stopping; cost equals the cumulative cost through the last look; costs never exceed the plan total or the budget); zero-budget and zero-look cases; reproducibility, chunk invariance and replication independence; calibration of the intervals across seeds; the interval formulas against closed forms; and invalid inputs and numerical edge cases.

Two checks made during development (one-off, not part of the suite):

- **Robustness of the tolerances:** 5,000 comparisons on 600 seeds that the tests do not use had a largest `|z|` of 3.2 and none reached 4.5 (about 0.03 expected by chance).
- **Sensitivity:** eight deliberately injected engine bugs (any cell able to detect, detection probability inflated by 0.01, no stop at detection, cost off by one, looks off by one, prior ignored, failed looks free, one uniform reused for all of a trial's attempts) were each caught by the suite. The suite includes a power check that a probability that is wrong by 0.01 is rejected at 100,000 trials.

## Efficiency

100,000 trials of a 20-look plan take about 20 ms and a few tens of megabytes at most (measured on the development machine; timings are machine-dependent). The memory used for random draws is bounded by `max_chunk_elements` (also for plans longer than the limit); the per-trial records (`O(trials)`) and the plan itself (`O(plan length)`) are not chunked, so total memory is `O(max_chunk_elements + trials + plan length)`.

## Limitations

- Stationary target, constant independent per-look detection probability, no false alarms, a fixed non-adaptive plan (the all-failures path, which is exactly what the greedy and exact solvers produce); no radar energy model and no new strategies.
- The simulation validates the analytical formulas only to within its sampling error; with 100,000 trials the standard error of a probability is at most 0.0016, so differences smaller than about 0.005 are not resolvable at that size.
- Confidence intervals describe sampling error only and use the approximations described above.
