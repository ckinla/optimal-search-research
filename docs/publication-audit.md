# Publication audit and release process

Reviewed 12 November 2026. This is a reviewed Python research release, not an export of the development workspace.

## Included

- Six independently implemented model modules: radar detection, cumulative detection, detection efficiency,
  Bayesian spatial search, exact finite-budget allocation, and Monte Carlo verification.
- Tests, independent enumeration helpers, four runnable examples, and six numerical verification scripts.
- Independently written technical explanations, limited attributed equations and worked-example numbers needed
  for comparison, and a bibliography distinguishing recorded transcriptions from abstracts and reproduced results.
- Python packaging, a locked uv environment, MIT license, and a public test workflow requiring no secrets.

## Excluded

- Historical papers, manuscript files, slides, complete transcriptions, and images of publications: redistribution
  permission has not been established. Publisher records are citations, not promises of open full-text access.
- Web applications, deployment settings, internal notes, account configuration, local environments, generated
  reports, and the original Git history. The public repository began with a fresh reviewed initial commit.
- Any claim to reproduce the full integrated beam/time/energy method from the abstract-only sources.

The source available for extraction was our implementation and recorded comparison notes. This audit does not
independently authenticate the original historical documents. See [provenance](provenance.md).

## Privacy and licenses

The initial three public commits (43 tracked files) were scanned for token/key patterns, private repository URLs,
local user paths, and private-key blocks; no matches were found. Tracked files were reviewed for internal notes,
personal data and external-file dependencies. Only the public GitHub handle and no-reply commit address are used.
No historical binaries or complete source transcriptions are included. Repeat the scan for each release; a pattern
scan is not a guarantee that all possible secrets can be detected.

No vendored third-party implementation was found. NumPy (BSD-3-Clause plus bundled 0BSD/MIT/Zlib/CC0 notices) and
SciPy (BSD-3-Clause, with its own bundled dependency notices) are installed as dependencies, not relicensed here;
pytest is an MIT-licensed development
dependency. Their distributions retain their own license files. Our MIT license applies only to material we own;
citations and limited historical comparisons do not grant rights to the underlying publications.

## Mathematical parity and disclosed correction

The extracted engine retains the original calculations with import/package changes. One portability correction
was necessary: the Wilson confidence interval now returns exactly zero for zero successes and exactly one for all
successes. Floating-point cancellation could previously produce a positive lower endpoint of about 10⁻¹⁷.
The interior Wilson formula, detection equations, search algorithms and reference values are unchanged.
A regression checks both endpoints for 400 trial counts at four confidence levels (1,600 pairs, two assertions
per pair). The earlier correction commit's exploratory failure-count wording was inaccurate and should not be
used as a measurement; the committed regression is the reproducible evidence.

## Authority and manual releases

Initially the lab's development repository remains authoritative for Python mathematical changes; this public
repository is a reviewed release. Do not independently evolve two mathematical implementations.

1. Propose a change with its assumptions, derivation, numerical impact and a failing regression when applicable.
2. Run the complete Python suite and independent checks in development. Cross-check essay numbers against the
   separate TypeScript engine and committed reference fixtures. Do not regenerate fixtures just to force agreement.
3. Review the publication diff, rights and privacy again; copy only reviewed modules/tests/notes with package import
   adaptation. Compare model ASTs ignoring docstrings and the package prefix; disclose every behavioral difference.
4. In a clean public clone run `uv sync --locked`, `uv run pytest`, and all four README examples. Compare
   `uv run python examples/essay_numbers.py --json` against the development essay fixture with explicit tolerances.
5. Publish a new reviewed commit and, for mathematical releases, update the package version and release notes.
   Record both repository commits in the development release record. Require green public CI before updating links.

For the boundary correction discovered during extraction, the development follow-up mirrors the fix and test,
restoring parity. Future changes to a published result require an explanation and regression, never a silent fixture
edit. No submodule, private dependency, or cross-repository publishing automation is required.
