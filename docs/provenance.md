# Provenance: what is foundational, what is Matthiesen's, what we reproduced, and what is ours

This repository studies a question that Duane J. Matthiesen addressed in a series of radar-engineering papers and a
talk between 1999 and 2011: a radar has limited energy and time, so where should it look, how much effort should it
put into each look, and how often should it look again? This note separates four kinds of material so that nothing
is credited to the wrong source. It is not a priority survey or a claim of novelty.

Evidence labels used below:

- **Transcribed:** equations or worked examples preserved in the development project's source-comparison notes.
  This publication was checked against those recorded notes, not a fresh inspection of original documents.
  Each transcription is encoded in a test and evaluated; tests check the *mathematics* of the transcription,
  not that every glyph was read correctly. Original documents are not bundled here.
- **Abstract only:** the publisher's record or an author abstract was read; no full text was available to us.
- **Reproduced:** independently calculated and tested in this repository.
- **Ours:** a model, analysis or example we built. It is not attributed to Matthiesen.

## 1. Foundational mathematics

These predate the work discussed here and are not Matthiesen's inventions.

- **Single-pulse detection of a fluctuating target.** `Pd = Pfa^(1/(1+s))` is the standard square-law result for a
  Swerling I/II target in Gaussian noise (Swerling, 1954). See `src/optimal_search/radar_detection.py`.
- **Cumulative detection over independent looks.** `Pc = 1 − (1 − Pd)^N` is elementary probability.
- **Bayesian updating after a failed search** and the law of total probability.
- **Optimal allocation of search effort.** Koopman (1957) and de Guenin (1961) studied the optimum distribution of
  searching effort. Black (1965) gave a cost-weighted index for discrete sequential search with imperfect detection,
  proportional to `p_i α_i (1 − α_i)^(k−1) / t_i`, for minimising expected cost to detection; it is a direct
  predecessor of the payoff-per-cost ranking used here. Hahn and Gross (1969) studied surveillance optimisation for
  pencil-beam arrays. We compared Koopman and de Guenin at the level of their abstracts only, and Hahn and Gross at
  the level of the bibliographic record.

## 2. Matthiesen's research

| Year | Work | Available to us | Public copy |
| --- | --- | --- | --- |
| 1999 | *Optimal Distribution of Radar Search Effort*, manuscript dated 12 November 1999 | Transcribed | None known |
| 2002 | *Optimization of Detection Performance*, International Radar Conference, pp. 263–269 | Abstract only | Publisher (DOI) |
| 2003 | *Optimal Search*, IEEE International Symposium on Phased Array Systems and Technology, pp. 259–264 | Transcribed | Publisher (DOI) |
| 2007 | *Optimal Search and Optimal Detection*, IET International Conference on Radar Systems | Abstract only | Publisher (DOI) |
| 2010 | *Efficient Beam Scanning, Energy Allocation, and Time Allocation for Search and Detection*, IEEE International Symposium on Phased Array Systems and Technology, pp. 361–368 | Abstract only | Publisher (DOI) |
| 2011 | Presentation of the same title, IEEE Boston Section technical meeting, 22 March 2011 | Transcribed | None known |

The 2010 paper and the 2011 presentation are separate sources despite their matching title. Full citations and DOIs
are in the [bibliography](#bibliography).

What the transcribed sources document, as far as it bears on this code:

- **Posterior after missed searches** (1999 Eq. 22, 26, 32; 2003 Eq. 8) and the **immediate payoff-per-cost
  selection rule**, which both papers describe as locally optimal and myopic (1999 Eq. 39–40; 2003 Eq. 30–31).
- **The Optimal Detection Condition** (2011): `dPd/d(S/N) = −(1 − Pd) ln(1 − Pd) / (S/N)`, stated to be independent of
  the cumulative detection goal, with a table listing Swerling I/II optima of 11.4 dB at `Pfa = 10⁻⁵` and 12.4 dB at
  `Pfa = 10⁻⁶`, and an optimal Pd of 0.47.
- **Payoff ratios** (2011): cumulative detection divided by detection effort, including the energy form.
- **Worked search examples** (2011 *Search Example 1* and *Search Example 2*).

The abstracts describe a broader method that integrates beam scanning, energy and time allocation, discuss integration
choices, and report savings of several decibels for most radars. **That integrated method is not reproduced here**:
the abstracts do not contain its equations, constraints or baselines.

## 3. Independent reproduction

Each item is computed by code in `src/optimal_search/` and checked by tests.

| Result | Status | Where |
| --- | --- | --- |
| The Optimal Detection Condition is algebraically identical to the stationarity condition of `H(s) = −ln(1 − Pd(s))/s`, the energy efficiency of independent looks | Reproduced: algebra checked pointwise and by root-finding | `docs/historical-source-verification.md`, `tests/test_matthiesen_source_comparison.py` |
| Local optima 11.416 dB (`Pfa = 10⁻⁵`) and 12.356 dB (`Pfa = 10⁻⁶`), which round to the presentation's 11.4 and 12.4 dB | Reproduced; nothing fitted | `src/optimal_search/cumulative_detection.py`, `tests/test_matthiesen_audit_examples.py` (70-digit check) |
| The optimum does not depend on the detection goal | Reproduced | `tests/test_cumulative_detection.py`, `tests/test_essay_examples.py` |
| Posterior update after a miss and the payoff-per-cost ranking | Reproduced; agrees with the papers | `tests/test_matthiesen_spatial_comparison.py` |
| 2011 Search Example 1 (0.25, 0.50, 0.75, 1.00) and Search Example 2 (0.24, 0.76, posterior 0.13158, 0.26316, 0.21053, 0.39474) | Reproduced number for number | `tests/test_matthiesen_spatial_comparison.py`, `examples/bayesian_update.py` |

One recorded value does not reproduce exactly: the presentation lists a single optimal Pd of 0.47 for both false-alarm
settings, while the exact single-pulse curve gives 0.461 at `10⁻⁵` (which rounds to 0.46) and 0.468 at `10⁻⁶`. The
presentation's detection curves were not recoverable, so rounding or a slightly different curve cannot be ruled out.
Nothing was adjusted to match.

## 4. Our extensions and analyses

These are ours. They are not reconstructions of Matthiesen's method.

- **`H(s)` as an explicit objective.** The presentation states the condition and the payoff ratio; `H` is our
  equivalent rate-per-energy formulation. Its payoff ratio is proportional to `H` only when the detection goal is
  fixed; at a fixed number of looks it is a different optimisation. The single-look `Pd/s` (labelled "historical" in
  the code, a name kept for compatibility) is the presentation's one-look special case, not its optimum.
- **The low-SNR pathology.** Because `Pd(0) = Pfa > 0`, `H` diverges as `s → 0`; at `Pfa = 10⁻⁵` it has a local minimum
  near `s = 0.106` and exceeds the 11.416 dB peak again only below `s ≈ 0.000225`. So 11.416 dB is a local optimum of a
  simplified model, not a global one.
- **Baseline-corrected objectives** (`(Pd − Pfa)/s` and its log form). Modelling choices of ours; they move the optimum
  by less than 0.001 dB.
- **Whole looks.** With integer looks the best design for an 80% goal at `Pfa = 10⁻⁵` is 3 looks of SNR 12.098,
  costing 0.5% more energy than the continuous optimum.
- **The exact finite-budget solver and the greedy counterexamples.** Exhaustive enumeration shows that greedy
  selection can lose with unequal, indivisible costs, for example 0.48 against 0.52 with perfect detection, costs
  (2, 1, 1) and budget 2. See `docs/exact-spatial-search.md`.
- **The Monte Carlo engine.** An independent simulation that corroborates the analytical probabilities. See
  `docs/monte-carlo-spatial-search.md`.
- **The joint where-and-how-strongly toy model** in `scripts/matthiesen_audit_examples.py`. A finite-action extension
  that ignores off-target false alarms, correlation, integration, time overhead and beam footprints.
- **The illustrative examples** in `examples/`, including the three-place budget example used in the visual essay.

## 5. The 1999/2003 cumulative-probability discrepancy

In the worked examples of the 1999 and 2003 papers, the "actual" cumulative probability after looks in cells `k` then
`l` is `P_k Pd_k + P_l Pd_l / (1 − P_k Pd_k)` (1999 Eq. 28; 2003 Eq. 19). The second term is a *conditional*
probability added as if it were unconditional. The law of total probability multiplies it by the probability that the
first look missed, `1 − P_k Pd_k`, which gives `P_k Pd_k + P_l Pd_l`: the papers' own "naive" value, which is the true
probability of detection.

Counterexample: two equally likely cells, perfect detection, one look in each. Detection is certain (probability 1),
but the expression gives 3/2. The expression also depends on the order of the looks; the true probability does not.

The 2011 presentation's *Search Example 1* weights each term by the probability that all earlier looks failed, which
is correct. Whether that was an intentional correction is **not established** by the material we have. The posterior
update and the greedy ranking are unaffected. The full analysis, with equation numbers, is in
[`spatial-search-verification.md`](spatial-search-verification.md), and every claim is tested in
`tests/test_matthiesen_spatial_comparison.py` against a brute-force enumeration of the probability space.

## 6. What cannot be verified from the available material

- The integrated beam-scanning, energy and time allocation method of the 2007 and 2010 papers, its duty-factor
  results, and the reported savings of several decibels. We have only abstracts.
- The 2002 paper's analytical method and its discussion of a nominal 13 dB design rule. We have only the abstract.
- The detection curves behind the 2011 table, the derivation between the payoff ratios and the condition, and how
  integer numbers of looks were treated.
- The printed form of every transcribed equation: transcriptions were read from embedded equation objects, and some
  glyphs in large equations were garbled. The conclusions rest on the unambiguous worked examples.
- The classical optimal-search results that the papers cite.

## 7. Source material and rights

This repository contains our own code, tests and explanations. It does **not** contain the papers, the 1999
manuscript, the 2011 slides, or full transcriptions of any of them. Individual equations and worked-example numbers
are quoted, with their equation numbers, to the extent needed to test and discuss them. Where a public copy exists we
link to the publisher's DOI; for the 1999 manuscript and the 2011 presentation no verified public copy is known, so
none is linked. The MIT license covers our code and documentation only, not the cited works.

## Bibliography

- **[M99]** D. J. Matthiesen, *Optimal Distribution of Radar Search Effort*, manuscript dated 12 November 1999.
- **[M02]** D. J. Matthiesen, *Optimization of Detection Performance*, 2002 International Radar Conference,
  pp. 263–269. <https://doi.org/10.1109/RADAR.2002.1174694>
- **[M03]** D. J. Matthiesen, *Optimal Search*, IEEE International Symposium on Phased Array Systems and Technology,
  14–17 October 2003, pp. 259–264. <https://doi.org/10.1109/PAST.2003.1256991>
- **[M07]** D. J. Matthiesen, *Optimal Search and Optimal Detection*, IET International Conference on Radar Systems,
  2007. <https://doi.org/10.1049/cp:20070593>
- **[M10]** D. J. Matthiesen, *Efficient Beam Scanning, Energy Allocation, and Time Allocation for Search and
  Detection*, 2010 IEEE International Symposium on Phased Array Systems and Technology, pp. 361–368.
  <https://doi.org/10.1109/ARRAY.2010.5613344>
- **[M11]** D. J. Matthiesen, *Efficient Beam Scanning, Energy Allocation, and Time Allocation for Search and
  Detection*, presentation, IEEE Boston Section technical meeting, 22 March 2011.
- **[S54]** P. Swerling, *Probability of Detection for Fluctuating Targets*, RAND Research Memorandum RM-1217, 1954.
  <https://www.rand.org/pubs/research_memoranda/RM1217.html>
- **[K57]** B. O. Koopman, *The Theory of Search. III. The Optimum Distribution of Searching Effort*, Operations
  Research 5(5), 613–626, 1957. <https://doi.org/10.1287/opre.5.5.613>
- **[G61]** J. de Guenin, *Optimum Distribution of Effort: An Extension of the Koopman Basic Theory*, Operations
  Research 9(1), 1–7, 1961. <https://doi.org/10.1287/opre.9.1.1>
- **[B65]** W. L. Black, *Discrete Sequential Search*, Information and Control 8(2), 159–162, 1965.
  <https://doi.org/10.1016/S0019-9958(65)90052-5>
- **[HG69]** P. Hahn and S. Gross, *Beam Shape Loss and Surveillance Optimization for Pencil Beam Arrays*, IEEE
  Transactions on Aerospace and Electronic Systems AES-5(4), 674–675, 1969. <https://doi.org/10.1109/TAES.1969.309952>
- **[V09]** M. G. de Vilmorin, E. Duflos and P. Vanheeghe, *Radar Optimal Times Detection Allocation in Multitarget
  Environment*, IEEE Systems Journal 3(2), 210–220, 2009. <https://doi.org/10.1109/JSYST.2009.2017393>. Related later
  work on detection-time allocation that cites Matthiesen's 2002 paper. In its Section III-B it approximates `1 + s`
  by `s` at high SNR and obtains a Pd of 0.5 at its optimum; that is a different model from the exact curve used here.
