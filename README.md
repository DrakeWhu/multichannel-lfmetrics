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