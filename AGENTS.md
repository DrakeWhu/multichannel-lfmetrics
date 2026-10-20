# Repository guidance: multichannel-lfmetrics

## Scope and interfaces

- Maintain the supported package under `src/lfmetrics/`. `legacy/` is historical
  material, not the supported API or dependency specification.
- The package reduces existing 3D openPMD/HDF5 particle diagnostics; it does not
  run WarpX, submit jobs, edit workflow state, or delete raw data.
- Preserve `post/particle_summary.csv`, configured plot outputs, CLI arguments,
  and the `BeamMetrics` column contract consumed by campaign-workflow.
- Read the README's optimizer interface notes. Multiple requested species
  produce multiple rows, while the current optimizer consumes the first row.
  Do not change species order or selection silently.
- Consult `docs/DEVELOPMENT_BRANCH_REVIEW_2026-10-20.md` before integrating
  Soft100, final-bunch tracking, trajectory, or field-analysis work. Their
  retention needs extend beyond the baseline's final-summary products.

## Scientific invariants

- Preserve SI momentum conversion, electron mass/charge assumptions, energy
  cuts, weighting conventions, paraxial divergence/emittance definitions, and
  final-dump selection. Electron formulas are not a generic ion analyzer.
- Preserve the distinction between weighted and unweighted statistics and
  between missing weighting and zero charge.
- `beam_quality_score` is a diagnostic ranking score, not the configured
  multi-objective optimizer objective.
- Final-frame results do not establish a propagation maximum or survival-free
  comparison. Record scientific consequences before changing sampling policy.

## Verification

- Use `PYTHONPATH=src python -B -m unittest discover -s tests -p 'test_*.py' -v`
  from this repository or an isolated export with NumPy, h5py, and matplotlib.
- See `docs/BASELINE_2026-10-20.md` for the dated environment and result.
  It is historical Cloud evidence, not validation of later revisions or HPC.
  Synthetic HDF5, CLI, and plotting tests do not validate real WarpX output.
- Report errors and skips. Do not run simulations or clean campaign data
  without task authorization.
- Write code, comments, tests, documentation, commits, and PR text in English.
