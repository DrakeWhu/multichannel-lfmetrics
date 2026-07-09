import unittest

import numpy as np

from lfmetrics.beam_metrics import (
    compute_beam_metrics,
    kinetic_energy_MeV,
    weighted_percentile,
)
from lfmetrics.constants import C_LIGHT, E_CHARGE_C, M_E_KG
from lfmetrics.particles import ParticleData


class BeamMetricsTests(unittest.TestCase):
    def test_kinetic_energy_is_zero_for_zero_momentum(self):
        energy = kinetic_energy_MeV(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        self.assertAlmostEqual(float(energy[0]), 0.0)

    def test_kinetic_energy_matches_gamma_minus_one(self):
        gamma_beta = 10.0
        pz = np.array([gamma_beta * M_E_KG * C_LIGHT])
        energy = kinetic_energy_MeV(np.array([0.0]), np.array([0.0]), pz)

        gamma = np.sqrt(1.0 + gamma_beta**2)
        expected = (gamma - 1.0) * 0.51099895
        self.assertAlmostEqual(float(energy[0]), expected, places=5)

    def test_weighted_percentile_prefers_high_weight_values(self):
        values = np.array([1.0, 10.0])
        weights = np.array([1.0, 100.0])

        self.assertEqual(weighted_percentile(values, 50, weights), 10.0)

    def test_charge_uses_weighting_when_available(self):
        pz = np.array([20.0, 30.0]) * M_E_KG * C_LIGHT
        particles = ParticleData(
            species="beam",
            step=5000,
            x_m=np.array([0.0, 1e-6]),
            y_m=np.array([0.0, 1e-6]),
            z_m=np.array([0.0, 1e-6]),
            px_si=np.array([0.0, 0.0]),
            py_si=np.array([0.0, 0.0]),
            pz_si=pz,
            weighting=np.array([10.0, 20.0]),
        )

        metrics = compute_beam_metrics(particles, energy_threshold_MeV=0.0)

        self.assertEqual(metrics.n_macroparticles_total, 2)
        self.assertEqual(metrics.n_macroparticles_selected, 2)
        self.assertEqual(metrics.weight_selected, 30.0)
        self.assertAlmostEqual(metrics.charge_selected_pC, 30.0 * E_CHARGE_C * 1e12)

    def test_charge_is_none_without_weighting(self):
        pz = np.array([20.0]) * M_E_KG * C_LIGHT
        particles = ParticleData(
            species="beam",
            step=5000,
            x_m=np.array([0.0]),
            y_m=np.array([0.0]),
            z_m=np.array([0.0]),
            px_si=np.array([0.0]),
            py_si=np.array([0.0]),
            pz_si=pz,
            weighting=None,
        )

        metrics = compute_beam_metrics(particles, energy_threshold_MeV=0.0)

        self.assertIsNone(metrics.charge_selected_pC)
        self.assertIsNone(metrics.weight_selected)
        self.assertEqual(metrics.beam_quality_status, "incomplete")
        self.assertIn("missing_charge_weighting", metrics.beam_quality_flags)

    def test_energy_threshold_selects_only_hot_particles(self):
        cold_pz = 1.0 * M_E_KG * C_LIGHT
        hot_pz = 50.0 * M_E_KG * C_LIGHT

        particles = ParticleData(
            species="beam",
            step=5000,
            x_m=np.array([0.0, 1e-6]),
            y_m=np.array([0.0, 1e-6]),
            z_m=np.array([0.0, 1e-6]),
            px_si=np.array([0.0, 0.0]),
            py_si=np.array([0.0, 0.0]),
            pz_si=np.array([cold_pz, hot_pz]),
            weighting=np.array([1.0, 2.0]),
        )

        metrics = compute_beam_metrics(particles, energy_threshold_MeV=5.0)

        self.assertEqual(metrics.n_macroparticles_total, 2)
        self.assertEqual(metrics.n_macroparticles_selected, 1)
        self.assertEqual(metrics.weight_selected, 2.0)


if __name__ == "__main__":
    unittest.main()
