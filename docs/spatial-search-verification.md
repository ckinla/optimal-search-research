# Bayesian spatial search: model and audit against Matthiesen (1999, 2003, 2011)

Implementation: `src/optimal_search/spatial_search.py`. Tests: `tests/test_spatial_search.py` (model, brute-force verification) and `tests/test_matthiesen_spatial_comparison.py` (the sources' equations and worked examples, transcribed, evaluated against the implementation). The model is independent of radar energy: detection probabilities and costs are inputs.

## Audit summary

| # | Item | Verdict | Where |
|---|------|---------|-------|
| 1 | Bayesian posterior update after a failed search | **Valid.** Agrees with the implementation. | 1999 Eq. 22, 26, 32; 2003 Eq. 8 |
| 2 | Immediate greedy selection criterion, `posterior · d / cost` | **Valid as an immediate (myopic) rule.** Unaffected by the discrepancy below. Not a global-optimality claim. | 1999 Eq. 39–40; 2003 Eq. 30–31 |
| 3 | Unconditional cumulative-probability expression ("actual" `P_c`) | **Incorrect** as the probability of detection: it adds *conditional* probabilities as if unconditional. The papers' "naive" value is the correct one. | 1999 Eq. 25 (first look only), 28, 31, 35; 2003 Eq. 15, 18–20 |
| 4 | Greedy optimality with equal costs | **Valid under stated assumptions** (A1–A5 below); argued and verified exhaustively on test instances. | this note |
| 5 | Greedy with a finite budget and unequal costs | **Can fail.** Exhaustively verified counterexample. | this note |

## Model and independently derived mathematics

One stationary target is in exactly one of `M` cells. Cell `j` has prior `p_j` (summing to 1), per-search detection probability `d_j` (constant, independent between repeated searches) and cost `c_j > 0`. A search of cell `j` can only detect the target if it is there.

| Quantity | Formula | Notes |
|----------|---------|-------|
| Posterior after a failed search of cell `k` | weight of `k` becomes `q_k (1 − d_k)`, others unchanged, then normalise | normaliser is `1 − q_k d_k` |
| Posterior after counts `n_j` (all failures) | `p_j (1 − d_j)^n_j / Σ_i p_i (1 − d_i)^n_i` | depends only on the counts, not the order |
| **Unconditional** cumulative detection | `P = 1 − Σ_j p_j (1 − d_j)^n_j` | law of total probability; order-independent |
| **Conditional** next-look success | `q_k d_k` | `P(next search of k succeeds ∣ all previous failed)` |
| Unconditional increment | `p_k (1 − d_k)^n_k d_k = (1 − P_before) · q_k d_k` | `P(this search is the first detection)`; increments sum to `P` |
| Greedy priority | `q_j d_j / c_j` | same ranking as the unconditional increment per cost (common factor `1 − P_before`) |

Why the cumulative probability cannot depend on the order: the target is missed iff all `n_j` searches of *its own* cell fail, an event whose probability involves only the counts. Bayesian updating is conditioning on failure; it does not change the unconditional probability of "detected somewhere in this fixed sequence". The tests verify every formula against an enumeration of the whole probability space (each target cell × each detection outcome of each search), not against the formulas themselves.

## Sources and how they were read

Copies held by the authors (public availability in [provenance](provenance.md)): *Optimal Distribution of Radar Search Effort* (12 November 1999); *Optimal Search* (IEEE Phased Array Systems and Technology symposium, 14–17 October 2003, pp. 259–264); and the presentation *Efficient Beam Scanning, Energy Allocation, and Time Allocation for Search and Detection* (IEEE Boston Section, 22 March 2011). The files were read passively: text from the documents, equations from the embedded equation-editor preview objects (vector geometry, laid out by position). Nothing was executed or rendered by an office application. The 2011 worked examples are slide text and are directly legible. Equation numbers are the papers' own. Some glyphs in large equations were garbled in extraction, so the conclusions about 1999 and 2003 rest on the unambiguous worked examples (1999 Eq. 28 and 31; 2003 Eq. 19 and 20) and on the surrounding text. **Readers with the printed papers can check those equations against them.**

## Comparison with the 1999 and 2003 papers

| Item | Papers | Implemented | Status |
|------|--------|-------------|--------|
| Prior: disjoint and complete | 1999 Eq. 20; 2003 Eq. 6–7 | `prior`, sum 1 | identical |
| Posterior after failed looks | 1999 Eq. 22, 26, 32; 2003 Eq. 8: `P_j (1 − Pd_j)^N_j / (1 − P̃_c,N)` | `posterior_from_counts`, `update_after_failed_search` | identical (verified numerically) |
| "Naive" cumulative probability | 1999 Eq. 23, 27, 30, 34; 2003 Eq. 10, 12–14, 22: `Σ_j P_j [1 − (1 − Pd_j)^N_j]` | `cumulative_detection_probability` | **identical to the implemented unconditional probability**, and equal to the true probability of detection within the fixed looks |
| "Actual" cumulative probability | 1999 Eq. 25, 28, 31, 35; 2003 Eq. 15, 18–20 | not implemented | **different, and not a probability of detection** (below) |
| Increment `ΔP_c` | 2003 Eq. 23: `P_j (1 − Pd_j)^(N_j−1) Pd_j / (1 − P̃_c,N−1)` | `conditional_next_look_success` | formula identical, but it is the *conditional* next-look success, not the increase in cumulative probability (the latter is `unconditional_detection_increment`, without the denominator) |
| Greedy criterion | 1999 Eq. 39–40; 2003 Eq. 30–31 | `next_look_priority`, `select_next_cell` | identical ranking |
| Optimality statements | both: "locally-optimal ... short-sighted ... myopic", "greedy" | `greedy_search_plan` | consistent on the myopic nature; stronger claims need qualification (below) |

## The cumulative-probability discrepancy

In the papers' worked examples the "actual" cumulative probability after looks in cells `k, l` is

`P_c2 = P_k Pd_k + P_l Pd_l / (1 − P_k Pd_k)`   (1999 Eq. 28; 2003 Eq. 19),

and after looks in `k, l, k`

`P_c3 = P_k Pd_k + P_l Pd_l / (1 − P_k Pd_k) + P_k (1 − Pd_k) Pd_k / (1 − P_k Pd_k − P_l Pd_l)`   (2003 Eq. 20; 1999 Eq. 31).

Each term after the first is divided by the probability that all earlier looks failed. These terms are *conditional* probabilities (`P(look n succeeds ∣ earlier failures)`), and the papers add them as if they were probabilities of disjoint events. 1999 Eq. 28 writes the sum as `Pr(detect on look 1) + Pr(detect on look 2 ∣ miss on look 1)`. The law of total probability requires `Pr(detect on look 1) + Pr(miss on look 1) · Pr(detect on look 2 ∣ miss on look 1)`: the missing factor `Pr(miss on look 1) = 1 − P_k Pd_k` cancels the denominator and gives `P_k Pd_k + P_l Pd_l`, the papers' own "naive" value.

### Counterexample: two cells, perfect detection

Two equally likely cells, `P_k = P_l = 1/2`, perfect detection `Pd_k = Pd_l = 1`, one look in cell `k` and then one in cell `l`.

- **Exact probability of detection:** the target is in `k` (probability 1/2) and is found on look 1; otherwise it is in `l` (probability 1/2), look 1 misses it with certainty, and look 2 finds it. So `P = 1/2 + 1/2 = 1` (detection is certain). An enumeration of both target locations and all outcomes gives the same.
- **"Naive" value (1999 Eq. 34; 2003 Eq. 10):** `1/2 + 1/2 = 1`. Correct.
- **"Actual" value (1999 Eq. 28; 2003 Eq. 19):** `P_c2 = 1/2 + (1/2) / (1 − 1/2) = 1/2 + 1 = 3/2`. This exceeds 1, so it cannot be a probability.
- **Correct weighting:** `1/2 + (1 − 1/2) · 1 = 1`.

Example with a nontrivial prior: `(0.6, 0.4)`, `Pd = (0.5, 0.5)`.

| Looks (cells) | Exact (enumeration) | "Naive" (Eq. 10/34) | Papers' "actual" `P_c` |
|---------------|--------------------|---------------------|------------------------|
| k | 0.3 | 0.3 | 0.3 |
| k, l | 0.5 | 0.5 | 0.5857 |
| l, k | 0.5 | 0.5 | 0.5750 |
| k, l, k | 0.65 | 0.65 | 0.8857 |
| l, k, k | 0.65 | 0.65 | 0.8750 |

### Consequences

All verified in `tests/test_matthiesen_spatial_comparison.py`.

- The papers' "actual" quantity depends on the order of the looks; the true probability of detection does not.
- It can exceed 1 (counterexample above).
- The 1999 statements that the naive value is "always less than" the true cumulative probability and is merely "a bound" (text after Eq. 28 and 31, and after Eq. 35) do not hold: the naive value *is* the true probability. The 2003 paper does not make the "bound" claim, but it does state that the naive and "actual" values differ after the first look and that only the "actual" one depends on order (text after Eq. 20); the order-dependence claim is the same error.
- Eq. 23 as the "increment in cumulative probability" holds only if the increment is the unconditional probability; as printed (with the denominator) it is the conditional probability.

One interpretation, offered as an interpretation and not as a reading of the sources: the running sum of conditional probabilities `Σ_n h_n` is the discrete *cumulative hazard* of the look-by-look detection process. The probability of detection is `1 − Π_n (1 − h_n)`, which agrees with `Σ_n h_n` only to first order when every `h_n` is small. If the sources intend a quantity other than the probability of detection, its definition needs to be clarified.

The **historical equations were not modified**: they are reproduced as transcribed in the test file and flagged here.

### What is and is not affected

- **Unaffected:** the posterior update and the greedy ranking, because the discrepancy is a common factor across cells (the papers note this themselves, 2003 Eq. 30–31). Results that use the unconditional per-cell formula `1 − (1 − Pd)^N` (single cell) are also unaffected.
- **Affected:** anything that uses the papers' `P_c` as a probability of detection for more than one look, for example a total payoff ratio `P_c / time` (2003 Eq. 24), and the "naive is a bound" statement.

## The 2011 presentation: a correctly weighted worked example

In *Search Example 1* (four equally likely cells, `Pd = 1` per look) the 2011 slides compute

`Pd,cum = (Pd,1 P1) + (1 − Pd,1 P1)(Pd,2 P2) + (1 − Pd,1 P1)(1 − Pd,2 P2)(Pd,3 P3) + ...`

with `P2 = 1/3`, `P3 = 1/2`, `P4 = 1` (the *updated* probabilities) and obtain `0.25, 0.50, 0.75, 1.00`. Each conditional term `Pd,m P_m` is multiplied by the probability that all earlier looks failed. That is the law of total probability and equals the exact value. For the same inputs the 1999 Eq. 28 / 2003 Eq. 19 pattern gives `0.25 + 0.25/0.75 = 7/12 ≈ 0.583` after two looks, where the 2011 example gives `0.5`. For the two-cell perfect-detection counterexample above, the 2011 weighting gives 1, not 3/2. *Search Example 2* (cells `0.1, 0.2, 0.4, 0.3`, `Pd = 0.6`) gives `Pd,cum = 0.24` after the first look (cell 3), the renormalisation denominator `0.76`, and the updated probabilities `0.13158, 0.26316, 0.21053, 0.39474`; all of these are reproduced by the implementation.

What this does and does not establish:

- It shows that the 2011 worked example is correctly weighted, and it is the same law-of-total-probability weighting the implementation uses.
- It does **not** show that this was an intentional correction of the 1999 and 2003 expressions. It could equally be a different convention, or a presentation-level calculation done independently. The extracted material contains no general-`N` "actual cumulative" expression of the 1999/2003 type and no statement addressing the earlier formula; absence in a passive extraction is not proof.
- The same presentation's incremental-payoff slide still labels the conditional quantity `P_k,just before look Pd_k` as the "incremental increase in cumulative probability of detection", and still uses the term "naïve cumulative probability", as the earlier papers do. The worked example and that label are not reconciled in the material read.

## Greedy selection

### Valid immediate criterion

The rule picks the cell with the largest conditional success probability per unit cost, `q_j d_j / c_j`. This is the correct ranking of immediate payoff and is the same ranking as the unconditional increment per cost. Both papers call it locally optimal, myopic and greedy.

### Equal-cost optimality: result and assumptions

**Result.** With equal costs, the greedy plan maximises the unconditional cumulative detection probability after `N` searches, for every `N` simultaneously.

**Assumptions.**

- **A1.** All searches cost the same, so a budget is just a number of searches.
- **A2.** Each cell has a constant per-search detection probability `d_j`, and repeated attempts are independent.
- **A3.** One stationary target in exactly one cell, prior summing to 1, no false alarms.
- **A4.** The objective is the *unconditional* cumulative detection probability after a fixed number of searches, not an expected time to detection and not a running sum of conditional probabilities.
- **A5.** Searches are indivisible whole numbers of looks.

**Argument.** `P(n) = Σ_j p_j [1 − (1 − d_j)^n_j]`. The gain from the `(n_j + 1)`-th search of cell `j` is `p_j d_j (1 − d_j)^n_j`, which does not increase with `n_j`. Maximising a sum of such concave terms subject to `Σ_j n_j = N` is achieved by taking the `N` largest available gains. The greedy rule takes the largest available gain at each step (the unconditional increment, ranked identically to `q_j d_j`), so after `N` steps it holds the `N` largest. This was also verified by exhaustive search over all 243 sequences of 5 searches over 3 cells on 30 randomised instances (`test_greedy_with_unit_costs_is_optimal_for_a_fixed_number_of_looks_on_these_instances`).

The papers cite the classical optimal-search literature for stronger statements (for example minimising expected time to detection). Those were not re-derived or tested here.

### Finite-budget, unequal-cost counterexample

With indivisible searches and unequal costs, choosing searches under a budget is knapsack-like and greedy can lose. Perfect detection, budget 2, costs `(2, 1, 1)`, prior `(0.52, 0.27, 0.21)`:

- Priorities: `0.52/2 = 0.26`, `0.27`, `0.21`. Greedy searches cell 1 (priority 0.27), leaving budget 1.
- Cell 0 (cost 2) is no longer affordable; greedy searches cell 2. Total detection probability `0.27 + 0.21 = 0.48`.
- One search of cell 0 alone costs 2 and detects with probability `0.52`.

An exhaustive search over all affordable sequences confirms `0.52` is optimal. `greedy_search_plan(..., budget=...)` only restricts the choice to affordable cells; it remains a heuristic.

## Limitations of the implementation

- Stationary target; constant, independent per-search detection probability; no false alarms; no look-dependent or energy-dependent detection (radar energy optimisation is the separate detection-efficiency model).
- The plan follows the all-failures path (the only branch before a detection).
- `prior` must sum to 1 within `1e-9`; it is never silently renormalised. Conditioning on an impossible failure raises `ValueError`. Search counts must be integers in `[0, 2^63 − 1]`.
- Posteriors for sequences are computed from counts in log space, so very long failure sequences (hundreds of repeated failures) neither underflow a cell nor depend on order.
- Continuous target densities, moving targets and beam-shape coverage models from the papers are out of scope.

## Not verified

- The printed form of the transcribed 1999 and 2003 equations (see above).
- The papers' general-`N` expressions (1999 Eq. 35, 2003 Eq. 15) beyond their worked examples.
- The classical optimal-search results that the papers cite.
- Whether the 2011 worked example reflects an intended change from the earlier papers.
