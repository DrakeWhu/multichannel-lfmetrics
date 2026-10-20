# multichannel-lfmetrics

Particle/beam metrics for 3D WarpX multichannel CNT / solid-density simulations.

This repository modernizes legacy Lynx analysis scripts such as LFMetrics and
energy/phase-space particle analysis into a small, testable Python package.

Initial scope:

- read WarpX openPMD/HDF5 particle diagnostics
- support multiple species: electrons, ionized_electrons, beam
- compute relativistic particle energy metrics
- use particle weighting when available
- write contract CSVs for campaign-workflow and MORBO

This repository does not run WarpX simulations.

## Supported package and development

The maintained implementation is `src/lfmetrics`; `legacy/` contains historical
scripts and is not the package API or a dependency specification.
`pyproject.toml` declares NumPy, h5py, and matplotlib and the `lfmetrics` command.

With the scientific environment active, run from the repository root:

```bash
PYTHONPATH=src python -B -m unittest discover -s tests -p 'test_*.py' -v
```

Tests use synthetic particles/HDF5 and temporary CSV/plot products. Use an
isolated Git export if the working tree must stay untouched. See
`docs/BASELINE_2026-10-20.md` for the executed baseline and package inventory.

## Scientific and optimizer interface

`python -m lfmetrics.cli analyze-case CASE_DIR` reads the last diagnostic
HDF5 dump and writes `post/particle_summary.csv` and species-specific plots
under `post/plots/`. Each explicitly requested species produces one summary
row in request order. `BeamMetrics` in `src/lfmetrics/beam_metrics.py` defines
the column contract.

The formulas use electron mass and charge magnitude; accepting a species name
does not make them a general ion analyzer. Momentum is converted to SI by
the reader. Selection uses kinetic energy (default threshold 5 MeV).
Energy percentiles and several moments use weights when available; quantities
such as `forward_fraction` and radial-divergence RMS are currently unweighted.
Divergence/emittance use paraxial definitions. Missing weights do not establish
zero physical charge. `beam_quality_score` is a diagnostic ranking score,
not the configured MORBO-like multi-objective target.

At the audited optimizer revision, multichannel ingestion reads only the
first summary row. The workflow's SUNRISE wrapper explicitly requests
`electrons`. Multi-species CSV support therefore does not establish explicit
species selection in the optimizer. The scientific owner must decide species
identity and missing/duplicate-row policy before that interface changes.

Final-dump metrics do not represent a propagation maximum and may be affected
by particle loss. Retention of temporal information and other reduced products
must be agreed before raw deletion. Analysis never authorizes cleanup.
See campaign-workflow's `docs/INTEGRATION_REVIEW.md` and campaign-optimizer's
`docs/OBSERVATION_CONTRACT_REVIEW.md` for the pending cross-repository review.
