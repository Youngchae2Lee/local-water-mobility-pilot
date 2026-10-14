# Software validation — 2026-10-15

Public synthetic tabular package, not real-water validation.

- Python 3.12.14 / NumPy 2.3.5: 3 tests passed; the model test skipped because scikit-learn was unavailable. No packages installed.
- Python 3.10.20 / NumPy 1.26.4 / scikit-learn 1.7.2, CPU 1: all 4 tests passed in approximately 1.1 seconds. The four-model synthetic CLI also completed.
- Synthetic cohort: train 1,692; validation 396; test 396; 360 rows purged.
- Checked common time cohorts, target endpoint boundaries, inadequate gap rejection, negative/undefined R², training-only scaler statistics and common model evaluation counts.

The split, metric, fit and evaluation functions were reused from the pilot's existing synthetic harness. The generator and public tests are separate from molecular geometry. No research-vendored module is imported. The real-data adapter and its private QC are not part of this package. No trajectory was read and no real-water model was fitted during this validation.

Synthetic model scores are intentionally not presented as water results. The CLI can regenerate its own demonstration scores. No public-water benchmark or independent-run validation has been performed.
