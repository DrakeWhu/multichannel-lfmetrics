# Development branch review — 2026-10-20

Repository: `multichannel-lfmetrics`. Audited base: `9a3d1ad898f4dc7aa1905ffc8c77aa3666abe02c`.
Remote references observed at `2026-10-20T11:41:20.402116+00:00`.

## Evidence scope

Remote heads were fetched into a temporary bare shared clone. Source repositories
were not switched, merged, rebased, or updated with development code. Inspection
used `git show`, `git diff`, `git log`, `git rev-list --left-right --count`,
and `git merge-base --is-ancestor`. Reference acquisition used `git ls-remote
--heads origin` and `git fetch --no-tags origin
'+refs/heads/*:refs/remotes/origin/*'` in the temporary clone. No branch tests,
simulations, scheduler jobs, analysis pipelines, or cleanup commands were run.
The inventory is a dated remote snapshot; findings below cover selected relevant
implementations, not certification of every changed file.

`BASELINE_2026-10-20.md` is preserved byte-for-byte as historical evidence for
the audited base and its recorded Cloud environment. It does not validate these
development tips, future versions, SUNRISE, LYNX, or WarpX execution. Its SHA-256:
`5538d9ab4526c2fd1d7712da79e971fbe5bd5cdfabde116f6a6747af2eb5ed12`.

## Remote branch inventory

Counts are commits unique to the audited base / unique to the remote branch.
A zero branch-only count denotes an ancestor or the identical base, not pending
implementation work. Hashes, rather than moving branch names, identify evidence.

| Remote branch | Commit | Base-only / branch-only |
| --- | --- | --- |
| `feat/multichannel-soft100-v2` | `9a1c7a8ff5ce86fe84e5ba5483826428fe02cf96` | 0 / 3 |
| `main` | `9a3d1ad898f4dc7aa1905ffc8c77aa3666abe02c` | 0 / 0 |
| `publication-field-visualization` | `0faa395158cebeb063fcfcb320894a63e5e2a11a` | 0 / 54 |
| `publication-final-bunch-tracking` | `7bac8bda5b9ba314b3f5cb7ce4e7ac0b202fd5d0` | 0 / 6 |
| `publication-trajectory-analysis` | `984bbdbd673a786fc040aa834b54bff1b8f7f922` | 0 / 12 |

## Verified implementation findings

The development line is cumulative: `feat/multichannel-soft100-v2` ->
`publication-final-bunch-tracking` -> `publication-trajectory-analysis` ->
`publication-field-visualization`. Each preceding tip is an ancestor of the
next. Their branch names are not independent integration units.

- Soft100 adds smooth-acceptance metrics and schema/status fields to the
  particle products (`src/lfmetrics/soft100.py`, `contracts.py`, `cli.py`).
  The schema constant is `soft100_v1` despite the branch's `v2` name. A branch
  name must not substitute for the serialized schema identifier.
- At the publication field tip, ordinary `analyze-case`/`analyze-file` still
  iterate over requested species and emit rows. The optimizer's inspected
  multichannel consumer still reads the first row. New producer metrics do
  not resolve consumer species ambiguity.
- `tracking.py` selects persistent IDs from a supplied final snapshot with
  explicit species, an energy threshold (default 20 MeV), and optional forward
  selection (enabled by default). Backtracking reads all diagnostic files for
  that selected population, retaining per-frame presence masks, times,
  positions, momenta, and weights. It checks increasing steps/times and
  requires all selected IDs in the final frame.
- `trajectory_analysis.py` consumes trajectory NPZ arrays for histories,
  threshold crossings, and cohort summaries. Its output explicitly records
  `first_seen_is_not_capture_time`. Final-bunch history conditions on final
  survival/selection; it does not describe every particle ever accelerated.
- The field-visualization tip adds field readers, publication case auditing,
  field slices, and geometry/plotting modules. `publication_case.py` discovers
  HDF5 series and audits field/particle records and time metadata. These paths
  require inputs beyond a final summary CSV.

Evidence: the named modules at the listed commits, plus the branch file
inventories and ancestry checks. No publication pipeline was executed.

## Possible conflicts and decisions required

No reviewed branch changes a documentation path modified in this stage; that
is a path comparison, not a guarantee of semantic compatibility. The baseline
README describes supported final-dump reduction, while these branches extend
its scope. A later merge must update that scope and the raw-retention contract.

Select species explicitly at the producer/consumer boundary and decide missing
or duplicate behavior before enabling multichannel ingestion of multi-row
products. Review Soft100 thresholds and schema compatibility independently of
species selection. Distinguish final-bunch histories, time-dependent population
metrics, and nozzle's terminal/windowed objectives. Specify whether future
reselection or field-based analyses are required: final summaries, selected-ID
trajectories, and rendered plots do not preserve arbitrary raw information.
Raw deletion therefore needs a reviewed product map and cannot be justified by
the existing final-dump baseline alone.

## Integration boundary

The current stage is documentation only. No branch implementation, species
selection, objective, temporal policy, scientific-validation rule, or raw
cleanup behavior is adopted. Scientific decisions above remain pending.
Any later integration requires its own version-specific tests and site evidence.
