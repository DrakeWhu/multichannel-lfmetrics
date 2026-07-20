import unittest

import numpy as np

from lfmetrics.constants import E_CHARGE_C
from lfmetrics.soft100 import (
    SOFT100_SCHEMA_VERSION,
    Soft100Config,
    effective_sample_size,
    smooth_energy_acceptance,
    soft_charge_pC,
    soft_particle_weights,
    soft_reliability,
    weighted_relative_spread_rms,
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


if __name__ == "__main__":
    unittest.main()
