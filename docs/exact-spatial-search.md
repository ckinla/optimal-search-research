# Exact finite-budget spatial search: solver, comparison with greedy, and findings

Implementation: `src/optimal_search/exact_spatial_search.py`. Tests: `tests/test_exact_spatial_search.py`. Comparison script: `uv run python -m scripts.spatial_search_exact_vs_greedy`. The solver is independent of the greedy code from PR #4: it imports nothing from `research.spatial_search`, and a test checks that by parsing its imports.

## Problem and solver

One stationary target is in exactly one of `M` cells with priors `p_j`, per-search detection probabilities `d_j` (constant, independent between repeated searches) and positive costs `c_j`. Choose integer counts `n_j ≥ 0` to maximise

`P(detected) = 1 − Σ_j p_j (1 − d_j)^n_j`   subject to   `Σ_j n_j c_j ≤ budget`.

`solve_exact` enumerates **every** feasible count vector (depth-first, lexicographic) and returns the optimal allocation, its total cost, the detection probability, and a search sequence.

- **Deterministic tie-breaking.** Among allocations with `P ≥ P_max (1 − 1e-12)`: the smallest total cost, then the lexicographically greatest counts (prefer lower-index cells). Three enumeration passes keep memory constant. The tolerance is *relative*: an absolute tolerance treated every allocation as tied when probabilities were tiny (`d = 1e-18`), which a test caught during development. Budget feasibility allows a *relative* slack, `Σ n_j c_j ≤ budget · (1 + 1e-12)`, so that three searches of cost `0.1` fit in a budget of `0.3` without loosening tiny budgets; total costs are compared as ties with the same relative tolerance.
- **Guard.** `max_allocations` (default 10⁶) raises `SearchSpaceTooLargeError`, a `ValueError`, instead of running for hours. Because every allocation that searches a single cell is feasible, at least `1 + Σ_j max_j` allocations exist; this is checked *before* any table is built, so the guard also bounds memory. The enumeration is iterative (no recursion limit) and skips cells that cannot be searched at all.
- **Independent checks used in the tests.** Hand-enumerated problems; enumeration of every search *sequence* within the budget; a pseudo-polynomial dynamic program for integer costs; full probability-space enumeration for expected costs; and a divisible-effort upper bound (`fractional_upper_bound`) built from sorted per-search gains, which shares no code with the enumerator.

## Probability does not depend on order; expected time does

`P` depends only on the counts, so every ordering of the same searches detects the target with the same probability *within the full plan*. If the search stops at the first success, the expected cost depends on the order:

`E[cost incurred] = Σ_n c_n S_{n−1}`, where `S_n` is the probability of no detection after `n` searches; and `E[cost | detected] = Σ_n C_n a_n / P` with `C_n` the cumulative cost and `a_n` the unconditional first-detection probability of search `n`.

Example, prior `(0.6, 0.4)`, `d = 0.5`, unit costs, one search of each cell: both orders give `P = 0.5`; searching cell 0 first gives `E[cost incurred] = 1.7`, cell 1 first gives `1.8` (given detection: `1.4` versus `1.6`). Both were checked against a full enumeration of target cell and outcomes.

For a fixed allocation both expected costs are minimised by sequencing the searches in non-increasing `a / c`, where `a = p_j (1 − d_j)^k d_j` is the first-detection probability of the next search of cell `j` after `k` earlier ones (adjacent exchange: swapping neighbours `i, j` changes the expected cost by `c_j a_i − c_i a_j`). The solver returns its sequence in that order. The tests confirm that this order is no worse than any other permutation. The *allocation* is chosen on `P` only, never on time.

## Research findings

### 1. When does greedy match the exact solution?

- **Equal costs.** Greedy equals exact under the assumptions of `optimal_search.spatial_search`: all searches cost the same; constant, independent per-search `d_j`; one stationary target, prior summing to 1, no false alarms; objective = unconditional cumulative probability after a fixed number of whole searches. The gain from the next search of a cell, `p_j d_j (1 − d_j)^n_j`, does not increase with `n_j`, so the `N` best gains are taken. Verified here: **0 disagreements in 200 deterministic grid instances** and on seeded random instances (cells ≤ 4, budgets ≤ 6).
- **Unequal costs, special case.** Let `C_>` be the total cost of the searches whose gain-per-cost ratio is *strictly* above the critical ratio of the divisible relaxation. If `C_> = budget`, greedy attains the relaxation value, which upper-bounds the exact optimum, so greedy is optimal. Exact ties in ratio need the strict version: an early draft of this analysis used the first-come prefix and the grid exposed a counterexample (cells with ratios `0.27/3` and `0.09/1`; a tie broken the other way leaves greedy at `0.693` against an exact `0.81`).
- **Otherwise greedy can fail** (next two items). For equal-cost-like instances where the budget is not a multiple of the cost, greedy remains optimal by the integer argument, not by the relaxation.

### 2. How large can the gap become with unequal costs?

- **Bound.** `exact − greedy ≤ r* · (B − C_>)`, where `r*` is the critical ratio of the divisible relaxation and `C_>` as above. This follows because every greedy plan, under any tie-breaking, takes all searches with ratio strictly above `r*`, and the exact optimum cannot exceed the divisible relaxation. It held on all 1,800 grid instances and on seeded random instances.
- **Worst case is unbounded in ratio and approaches 1 in absolute terms.** Budget 1, cell A of cost 1 with prior `p_A`, a cheap decoy B of cost `ε` with a ratio just above `p_A`, perfect detection. Greedy takes B and then cannot afford anything else.

| `p_A` | `ε` | greedy | exact | gap |
|-------|------|--------|-------|-----|
| 0.8 | 0.1 | 0.0808 | 0.8 | 0.719 |
| 0.8 | 0.001 | 0.000808 | 0.8 | 0.799 |
| 0.99 | 0.001 | 0.001 | 0.99 | 0.989 |
| 0.999 | 0.0001 | 0.000101 | 0.999 | 0.9989 |

- **Typical gaps on the grid are small.** Grid of 5 priors × 5 detection vectors × 8 unequal cost vectors × budgets 1–8 (3 cells, 1,600 instances): greedy is below exact in **143 (8.9%)**, with maximum absolute gap 0.200 and maximum relative gap 0.45. Those are properties of that grid, not worst cases. The PR #4 greedy restricts itself to affordable cells; variants that also compare with the best single search were not studied.

### 3. Does maximising detection probability within a budget differ from minimising expected search cost?

Yes. They are different decision problems.

- *Budgeted detection* maximises `P` over finite plans with `Σ n_j c_j ≤ B`; it is a knapsack-like problem in the counts, and its answer depends on `B`.
- *Expected cost* is a statement about a search that continues until detection (or about a fixed plan's cost along its sequence), and it depends on the order.

Example (perfect detection, prior `(0.52, 0.27, 0.21)`, costs `(2, 1, 1)`): the probability-optimal plan for budget 2 is `[cell 0]` with `P = 0.52` and expected cost `2.00`. The plan `[1, 2]` has `P = 0.48` and expected cost `1.73`, so a lower expected cost buys a lower detection probability. If every cell is searched, the order that minimises the expected cost starts with cell 1 (ratio `p/c = 0.27`, expected cost `2.67`), whereas the budget-2 optimum searches cell 0 (ratio 0.26) alone. For imperfect detection the classical index rule for minimising expected cost to detection is the literature's result; it was not re-derived or tested here.

### 4. Computational limits

- **Allocation counts.** With equal costs there are `C(k + M, M)` allocations, `k = floor(budget / cost)`: `M = 5, k = 10` gives 3,003; `M = 10, k = 10` gives 184,756; `M = 10, k = 20` gives 30,045,015; `M = 15, k = 20` gives 3.2 × 10⁹; `M = 20, k = 20` gives 1.4 × 10¹¹. Unequal costs shrink the count (costs `1..6`, budget 30: 8,547).
- **Measured speed.** About 0.85–1.1 million allocations per second in pure Python on the development machine (three passes), so the default guard of 10⁶ allocations takes a few seconds. Timings are machine-dependent.
- **Hardness.** With `d = 1` extra searches are useless and the problem is 0/1 knapsack with values `p_j`; the general problem is therefore weakly NP-hard. For integer costs a pseudo-polynomial dynamic program over cells and budget solves it (used as a test cross-check). Branch-and-bound with the divisible bound would prune most of the space; neither is implemented as part of the solver.

## Limitations

- Stationary target; constant independent per-search detection probability; no false alarms; no radar-energy or look-dependent detection.
- Costs and budget are floating point; feasibility and ties use the tolerances above, so nominal ties are resolved by rule, and exactness is relative to those tolerances.
- The greedy implementation breaks nominal ties by floating-point noise (for example `0.09000000000000001` versus `0.09`) and not by index; both orders are valid greedy plans and the gap bound covers any of them. That module was not changed.
- Exact optimisation is for small problems only.
- Not verified: classical optimality results for expected-cost minimisation with imperfect detection.
