import unittest

import numpy as np

from lfmetrics.beam_metrics import compute_beam_metrics
from lfmetrics.constants import C_LIGHT, M_E_KG
from lfmetrics.contracts import PARTICLE_SUMMARY_COLUMNS, validate_particle_summary_row
from lfmetrics.particles import ParticleData


class ContractTests(unittest.TestCase):
    def test_particle_summary_columns_include_core_outputs(self):
        for name in [
            "species",
            "step",
            "n_macroparticles_total",
            "charge_selected_pC",
            "energy_p95_MeV",
            "theta_r_p95_mrad",
            "emitn_xy_um_rad",
            "beam_quality_score",
            "beam_quality_status",
            "beam_quality_flags",
        ]:
            self.assertIn(name, PARTICLE_SUMMARY_COLUMNS)

    def test_beam_metrics_row_satisfies_contract(self):
        pz = np.array([20.0, 30.0]) * M_E_KG * C_LIGHT
        particles = ParticleData(
            species="beam",
            step=5000,
            x_m=np.array([0.0, 1e-6]),
            y_m=np.array([0.0, 1e-6]),
            z_m=np.array([0.0, 1e-6]),
            px_si=np.array([0.0, 1e-3 * M_E_KG * C_LIGHT]),
            py_si=np.array([0.0, 1e-3 * M_E_KG * C_LIGHT]),
            pz_si=pz,
            weighting=np.array([10.0, 20.0]),
        )

        row = compute_beam_metrics(particles, energy_threshold_MeV=0.0).as_row()
        validate_particle_summary_row(row)


if __name__ == "__main__":
    unittest.main()
