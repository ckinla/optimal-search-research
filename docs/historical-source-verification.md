# Historical-source verification note

**Source:** D. J. Matthiesen, *Efficient Beam Scanning, Energy Allocation, and Time Allocation for Search and Detection*, IEEE Boston Section technical meeting, 22 March 2011 (PowerPoint presentation; a copy is held by the authors, and no verified public copy is known). Primary source for the comparison below; no other literature was consulted.

## How the deck was read (and its limits)

The `.ppt` was read passively: slide text from the binary text records, and equations from the vector (WMF) equation objects, which were laid out from their glyph positions. Nothing in the file was executed or rendered by an office application. Consequences:

- Simple equations are legible and unambiguous. Some multi-line ones have garbled glyphs and are not relied on.
- The **numerator function** was cross-checked against the deck's own plot (maximum 0.36788 at Pd = 0.63212, which is 1/e at 1 − 1/e).
- **Not recoverable:** the Pd-versus-S/N detection charts (so which Pd(S/N) curve the deck used is unverified), the intermediate derivation steps between the payoff ratios and the optimal condition, and the exact slide on which each equation sits.
- Readers with access to the slides can check the transcriptions below against the slides *Cumulative Probability of Detection*, *Detection Payoff Ratio*, *Incremental Detection Payoff (Rate) Ratio*, and *Optimal Detection Condition*.

## What the deck states (as read)

| Item | Deck |
|------|------|
| Cumulative detection | `Pc = 1 − (1 − Pd)^N_L`; looks assumed statistically independent; other variables, including Pfa, held constant during optimization |
| Exponential law | `Pc = 1 − exp(−α t)`, `α ≡ −ln(1 − Pd)/Δt`, `t = N_L Δt` |
| Payoff ratio | `ρ_D = Pc / (detection effort)`; `ρ_D,energy = Pc / (N_L N_p P τ) = [1 − (1 − Pd)^N_L] / (N_L N_p P τ)` |
| Incremental payoff, look j | `Δ Pc_j / Δ(effort_j)`, with `ΔPc_j = (1 − Pd)^(N_j−1) Pd` |
| **Optimal Detection Condition** | `dPd/d(S/N) = −(1 − Pd) ln(1 − Pd) / (S/N) = f(Pd)/(S/N)`, `f(p) = −(1 − p) ln(1 − p) ≥ 0` |
| Independence of Pc | The condition "is independent of the specific value of cumulative probability of detection" |
| Swerling I/II optimum | 11.4 dB (Pfa 1e-5), 12.4 dB (Pfa 1e-6), optimal Pd 0.47 (one value listed for both); single pulse or coherent integration (one look) |

## Comparison with the implemented objectives

| Question | Finding |
|----------|---------|
| **Optimal condition vs. implemented `H(s) = −ln(1 − Pd)/s`** | **Identical stationarity condition.** `dH/ds = (s q′ − q)/s²` with `q = −ln(1 − Pd)`, and `q′ = Pd′/(1 − Pd)`, so `s Pd′ − f(Pd) = (1 − Pd)(s q′ − q)`: same zeros, same sign. Verified pointwise and by root-finding in `tests/test_matthiesen_source_comparison.py`. |
| **`H` as a printed objective** | **Not printed in the extracted material.** `H` is the per-energy analogue of the deck's exponential rate `α` (per unit time): `q/s` with S/N, proportional to energy per look, in place of `Δt`. That reading is mine, not a quoted equation. |
| **Deck payoff ratio `ρ_D,energy(s, N_L)` vs. `H`** | **Equivalent only under stated assumptions; different as functions.** At fixed `Pc` (so `N_L = −ln(1 − Pc)/q`), `ρ = Pc/(−ln(1 − Pc))·H(s)`: proportional, hence the same optimum, independent of `Pc`. At fixed `N_L` the optimum depends on `N_L` (Pfa 1e-5: 9.73 dB at `N_L = 1`, 7.87 dB at `N_L = 3.73`, → 11.416 dB as `N_L → 0`). |
| **Single-look `Pd/s`** | **Different.** It is the `N_L = 1` special case of `ρ_D,energy`, not the deck's optimal condition (residual `Pd − f(Pd) > 0` at its optimum). Earlier calling it "historical" was a label of ours, not supported by the deck. |
| **Baseline-corrected variants** | **Not in the deck.** Our own modelling choices; they move the optimum by < 0.001 dB and play no role in the match. |

**Equivalence assumptions** (all stated or implied by the deck): independent looks; constant Pfa per look; effort proportional to energy; and either fixed `Pc` or the continuous/exponential limit. The deck's optimum is for the continuous formulation; how it treats integer `N_L` is not recoverable here.

## Numerical comparison (not used as evidence of equality)

| Pfa | Computed `H` optimum | Deck | Computed Pd | Deck Pd |
|-----|----------------------|------|-------------|---------|
| 1e-5 | 11.416 dB | 11.4 dB | 0.461 | 0.47 |
| 1e-6 | 12.356 dB | 12.4 dB | 0.468 | 0.47 |

The SNRs round to the deck's values. **Minor discrepancy:** the deck lists one Pd (0.47) for both Pfa, while ours is 0.461 for 1e-5 (rounds to 0.46). Possible causes (unresolved): rounding or a single representative value in the deck, or a slightly different Pd(S/N) curve than the exact single-pulse formula used here. Not adjusted.

## Not verified

- The Pd(S/N) curves behind the deck's table; our model is the standard single-pulse Swerling I/II formula.
- The deck's derivation of the condition from the payoff ratios, and its integer-`N_L` treatment.
- Duty-factor and time-allocation results (not implemented here).
- Equation transcriptions beyond those listed above.
