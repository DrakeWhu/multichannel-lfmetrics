import unittest

import numpy as np

from lfmetrics.constants import C_LIGHT, E_CHARGE_C, M_E_KG
from lfmetrics.particles import ParticleData
from lfmetrics.soft100 import (
    SOFT100_SCHEMA_VERSION,
    Soft100Config,
    compute_soft100_metrics,
    effective_sample_size,
    smooth_energy_acceptance,
    soft_charge_pC,
    soft_particle_weights,
    soft_reliability,
    weighted_relative_spread_rms,
)


REST_ENERGY_MEV = 0.51099895


def particles_from_energy(
    energy_mev,
    *,
    weights=None,
    directions=None,
    ux=None,
    uy=None,
):
    energy = np.asarray(energy_mev, dtype=float)
    count = energy.size
    gamma = energy / REST_ENERGY_MEV + 1.0
    momentum_mec = np.sqrt(np.maximum(gamma * gamma - 1.0, 0.0))
    ux_array = np.zeros(count) if ux is None else np.asarray(ux, dtype=float)
    uy_array = np.zeros(count) if uy is None else np.asarray(uy, dtype=float)
    uz_array = np.sqrt(
        np.maximum(momentum_mec**2 - ux_array**2 - uy_array**2, 0.0)
    )
    if directions is not None:
        uz_array *= np.asarray(directions, dtype=float)

    momentum_scale = M_E_KG * C_LIGHT
    return ParticleData(
        species="electrons",
        step=5000,
        x_m=np.linspace(-2.0e-6, 1.0e-6, count),
        y_m=np.linspace(1.5e-6, -1.0e-6, count),
        z_m=np.zeros(count),
        px_si=ux_array * momentum_scale,
        py_si=uy_array * momentum_scale,
        pz_si=uz_array * momentum_scale,
        weighting=(
            None if weights is None else np.asarray(weights, dtype=float)
        ),
    )


class Soft100KernelTests(unittest.TestCase):
    def test_defaults_are_the_v2_campaign_definition(self):
        config = Soft100Config()

        self.assertEqual(SOFT100_SCHEMA_VERSION, "soft100_v1")
        self.assertEqual(config.energy_low_mev, 5.0)
        self.assertEqual(config.energy_target_mev, 100.0)
        self.assertEqual(config.reliability_floor, 0.05)
        self.assertEqual(config.effective_count_reference, 100.0)

    def test_acceptance_is_smooth_and_saturates_only_per_particle(self):
        acceptance = smooth_energy_acceptance(
            np.asarray([0.0, 5.0, 52.5, 100.0, 300.0, np.nan]),
            energy_low_mev=5.0,
            energy_target_mev=100.0,
        )

        np.testing.assert_allclose(acceptance, [0.0, 0.0, 0.5, 1.0, 1.0, 0.0])

    def test_soft_weights_use_physical_weight_times_acceptance(self):
        soft_weights = soft_particle_weights(
            np.asarray([52.5, 100.0]),
            np.asarray([2.0, 3.0]),
            energy_low_mev=5.0,
            energy_target_mev=100.0,
        )

        np.testing.assert_allclose(soft_weights, [1.0, 3.0])
        self.assertAlmostEqual(effective_sample_size(soft_weights), 16.0 / 10.0)

    def test_soft_charge_scales_linearly_and_is_not_capped(self):
        soft_weights = soft_particle_weights(
            np.asarray([100.0, 300.0]),
            np.asarray([2.0, 3.0]),
            energy_low_mev=5.0,
            energy_target_mev=100.0,
        )
        scaled_weights = soft_particle_weights(
            np.asarray([100.0, 300.0]),
            np.asarray([2.0e9, 3.0e9]),
            energy_low_mev=5.0,
            energy_target_mev=100.0,
        )

        expected = 5.0 * E_CHARGE_C * 1.0e12
        self.assertAlmostEqual(soft_charge_pC(soft_weights), expected)
        self.assertAlmostEqual(
            soft_charge_pC(scaled_weights),
            1.0e9 * soft_charge_pC(soft_weights),
            places=12,
        )

    def test_energy_above_target_is_not_clipped_in_spread(self):
        energy = np.asarray([100.0, 300.0])
        soft_weights = soft_particle_weights(
            energy,
            np.ones(2),
            energy_low_mev=5.0,
            energy_target_mev=100.0,
        )

        self.assertAlmostEqual(
            weighted_relative_spread_rms(energy, soft_weights),
            0.5,
        )

    def test_monoenergetic_spread_is_zero(self):
        self.assertEqual(
            weighted_relative_spread_rms(
                np.asarray([300.0, 300.0, 300.0]),
                np.asarray([1.0, 2.0, 4.0]),
            ),
            0.0,
        )

    def test_reliability_is_bounded_confidence_not_a_physics_cap(self):
        self.assertAlmostEqual(
            soft_reliability(
                0.0,
                reliability_floor=0.05,
                effective_count_reference=100.0,
            ),
            0.05,
        )
        self.assertLess(
            soft_reliability(
                100.0,
                reliability_floor=0.05,
                effective_count_reference=100.0,
            ),
            1.0,
        )

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            Soft100Config(energy_low_mev=100.0, energy_target_mev=100.0)


class Soft100MetricsTests(unittest.TestCase):
    def test_summary_preserves_unbounded_charge_and_energy(self):
        metrics = compute_soft100_metrics(
            particles_from_energy([120.0, 300.0], weights=[2.0e9, 3.0e9])
        )

        self.assertEqual(metrics.soft100_status, "ok")
        self.assertAlmostEqual(
            metrics.charge_soft100_pC,
            5.0e9 * E_CHARGE_C * 1.0e12,
        )
        self.assertAlmostEqual(metrics.energy_p95_soft100_MeV, 300.0, places=6)
        self.assertAlmostEqual(metrics.energy_max_soft100_MeV, 300.0, places=6)
        self.assertGreater(metrics.energy_relative_spread_rms_soft100, 0.0)

    def test_summary_effective_count_uses_soft_weights(self):
        metrics = compute_soft100_metrics(
            particles_from_energy([52.5, 120.0], weights=[2.0, 3.0])
        )

        self.assertAlmostEqual(metrics.weight_soft100, 4.0, places=6)
        self.assertAlmostEqual(metrics.n_effective_soft100, 16.0 / 10.0)

    def test_summary_is_forward_only(self):
        metrics = compute_soft100_metrics(
            particles_from_energy(
                [150.0, 300.0],
                weights=[1.0, 10.0],
                directions=[1.0, -1.0],
            )
        )

        self.assertTrue(metrics.soft100_forward_only)
        self.assertEqual(metrics.n_macroparticles_soft100, 1)
        self.assertEqual(metrics.weight_soft100, 1.0)
        self.assertAlmostEqual(metrics.energy_p95_soft100_MeV, 150.0, places=6)

    def test_halo_fraction_uses_the_matching_forward_hot_population(self):
        energy = np.asarray([10.0, 52.5, 120.0])
        acceptance = smooth_energy_acceptance(
            energy,
            energy_low_mev=5.0,
            energy_target_mev=100.0,
        )
        metrics = compute_soft100_metrics(
            particles_from_energy(energy, weights=[1.0, 1.0, 1.0])
        )

        self.assertAlmostEqual(
            metrics.halo_fraction_soft100,
            1.0 - float(np.sum(acceptance)) / 3.0,
            places=6,
        )

    def test_transverse_metrics_use_raw_angles_and_normalized_momentum(self):
        metrics = compute_soft100_metrics(
            particles_from_energy(
                [150.0, 160.0, 170.0, 180.0],
                weights=[1.0, 2.0, 3.0, 4.0],
                ux=[-0.2, 0.1, 0.3, -0.1],
                uy=[0.15, -0.25, 0.05, 0.2],
            )
        )

        self.assertGreater(metrics.theta_r_p95_soft100_mrad, 0.0)
        self.assertGreater(metrics.emitn_x_soft100_um_rad, 0.0)
        self.assertGreater(metrics.emitn_y_soft100_um_rad, 0.0)

    def test_missing_weighting_is_explicit(self):
        metrics = compute_soft100_metrics(particles_from_energy([150.0]))

        self.assertEqual(metrics.soft100_status, "missing_weighting")
        self.assertIsNone(metrics.charge_soft100_pC)
        self.assertIsNone(metrics.n_effective_soft100)

    def test_no_accepted_particles_is_explicit(self):
        metrics = compute_soft100_metrics(
            particles_from_energy([1.0, 4.0], weights=[1.0, 2.0])
        )

        self.assertEqual(metrics.soft100_status, "no_accepted_particles")
        self.assertEqual(metrics.charge_soft100_pC, 0.0)
        self.assertEqual(metrics.n_effective_soft100, 0.0)
        self.assertIsNone(metrics.energy_p95_soft100_MeV)


if __name__ == "__main__":
    unittest.main()
