# Release validation

The following checks passed during preparation of this release:

- Seven CPU unit tests: weak-signal abort, either-signal continuation, threshold equality, zero attention, selected layers/steps/CFG exclusion, trial cleanup, disabled/invalid configuration, and nested boxed-answer extraction.
- Five hundred randomized attention-record cases compared against the supplied original `adavig_pre` implementation. Both feature values matched to `1e-12`, and every continue/abort decision matched.
- Synthetic four-sample pipeline: rule-based MathVision scoring, all four counterfactual labels, aligned CSV construction, and two-threshold search output.
- Launcher dry runs with a stub `torchrun`: baseline, text-only, and gated module selection, including clearing an inherited gate policy for baseline runs.
- Python compilation, Ruff undefined-name/syntax checks, shell syntax check, and evaluator CLI imports.

These are software checks, not benchmark results. The lightweight validation environment used Python 3.13 with available CPU NumPy/Pandas packages. It does not verify installation of the documented Python 3.10/CUDA dependency set. Full model loading, FlashAttention execution, measured latency/memory, and the original paper's accuracy results still require validation with the real checkpoints and benchmark splits.

## Single-trajectory runner update

The current runner adds five CPU tests covering removed CLI options, the common 0.7 temperature, one generation and one save per sample, visual/text-only control flow, and text continuation after an early gate abort. Model execution is mocked in these flow tests. Together with the original gate/answer tests, the suite contains 12 tests.

The three launcher modes were also checked against the shared argument parser, including temperature overrides and policy isolation. A further runner test verifies that processing multiple samples sets the seed only once, using a supplied seed value; the suite now contains 13 tests. Full CUDA inference remains unverified.
