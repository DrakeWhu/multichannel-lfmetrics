import tempfile
import unittest
from pathlib import Path

import numpy as np

from lfmetrics.beam_metrics import compute_beam_metrics
from lfmetrics.constants import C_LIGHT, M_E_KG
from lfmetrics.particles import ParticleData
from lfmetrics.plots import (
    PLOT_FILENAMES,
    _resolve_spectrum_min_energy,
    write_particle_plots,
)


def make_particles() -> ParticleData:
    momentum_scale = M_E_KG * C_LIGHT
    return ParticleData(
        species="electrons",
        step=4000,
        x_m=np.array([-1.0e-6, 0.0, 1.0e-6, 2.0e-6]),
        y_m=np.array([0.5e-6, -0.5e-6, 1.0e-6, -1.0e-6]),
        z_m=np.array([0.0, 1.0e-6, 2.0e-6, 3.0e-6]),
        px_si=np.array([0.01, -0.02, 0.03, 0.04]) * momentum_scale,
        py_si=np.array([0.02, 0.01, -0.03, 0.04]) * momentum_scale,
        pz_si=np.array([2.0, 20.0, 50.0, 65.0]) * momentum_scale,
        weighting=np.array([1.0, 2.0, 3.0, 4.0]),
    )


class ParticlePlotTests(unittest.TestCase):
    def test_fixed_particle_plot_family_is_written(self) -> None:
        particles = make_particles()
        metrics = compute_beam_metrics(particles, energy_threshold_MeV=5.0)

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_particle_plots(
                particles,
                metrics,
                output_dir=Path(tmp) / "plots",
                energy_threshold_MeV=5.0,
            )

            self.assertEqual([path.name for path in paths], list(PLOT_FILENAMES))
            for path in paths:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_no_hot_particles_still_produces_the_fixed_png_contract(self) -> None:
        particles = make_particles()
        metrics = compute_beam_metrics(particles, energy_threshold_MeV=1_000.0)

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_particle_plots(
                particles,
                metrics,
                output_dir=Path(tmp) / "plots",
                energy_threshold_MeV=1_000.0,
            )

            self.assertEqual(len(paths), len(PLOT_FILENAMES))
            for path in paths:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

    def test_spectrum_cutoff_defaults_to_hot_energy_threshold(self) -> None:
        self.assertEqual(
            _resolve_spectrum_min_energy(None, energy_threshold_MeV=5.0),
            5.0,
        )

    def test_spectrum_cutoff_can_be_overridden_for_exploration(self) -> None:
        self.assertEqual(
            _resolve_spectrum_min_energy(10.0, energy_threshold_MeV=5.0),
            10.0,
        )


if __name__ == "__main__":
    unittest.main()
