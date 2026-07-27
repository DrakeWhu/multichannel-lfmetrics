import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lfmetrics.cli import main
from lfmetrics.trajectory_physics import (
    TrajectoryPhysicsContext,
    analyze_trajectory_physics_file,
    build_physical_cohort_rows,
    build_physical_particle_rows,
    build_physical_threshold_rows,
    weighted_correlation,
)
from lfmetrics.trajectory_analysis import load_trajectory_npz


def write_synthetic_trajectory(path: Path) -> None:
    ids = np.asarray([11, 22], dtype=np.uint64)
    steps = np.asarray([0, 100, 200], dtype=np.int64)
    times_s = np.asarray([0.0, 1.0e-15, 2.0e-15], dtype=float)
    present = np.asarray(
        [
            [True, False],
            [True, True],
            [True, True],
        ],
        dtype=bool,
    )
    shape = present.shape
    arrays = {
        name: np.full(shape, np.nan, dtype=float)
        for name in (
            "x_m",
            "y_m",
            "z_m",
            "px_si",
            "py_si",
            "pz_si",
            "weighting",
            "energy_MeV",
        )
    }

    arrays["x_m"][:, 0] = [1.0e-6, 1.1e-6, 1.2e-6]
    arrays["y_m"][:, 0] = [0.0, 0.0, 0.0]
    arrays["z_m"][:, 0] = [10.0e-6, 11.0e-6, 12.0e-6]
    arrays["pz_si"][:, 0] = [-1.0e-21, 1.0e-21, 2.0e-21]
    arrays["weighting"][:, 0] = [2.0, 2.0, 2.0]
    arrays["energy_MeV"][:, 0] = [0.0, 5.0, 25.0]

    arrays["x_m"][1:, 1] = [2.0e-6, 2.1e-6]
    arrays["y_m"][1:, 1] = [0.0, 0.0]
    arrays["z_m"][1:, 1] = [20.0e-6, 21.0e-6]
    arrays["pz_si"][1:, 1] = [1.0e-21, 2.0e-21]
    arrays["weighting"][1:, 1] = [3.0, 3.0]
    arrays["energy_MeV"][1:, 1] = [2.0, 30.0]

    np.savez_compressed(
        path,
        id=ids,
        step=steps,
        time_s=times_s,
        present=present,
        **arrays,
    )


class TrajectoryPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.context = TrajectoryPhysicsContext(
            plasma_start_m=5.0e-6,
            window_front_z0_m=9.0e-6,
            moving_window_velocity_m_s=1.0e9,
        )

    def test_weighted_correlation(self):
        value = weighted_correlation(
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([2.0, 4.0, 6.0]),
            np.asarray([1.0, 2.0, 1.0]),
        )
        self.assertAlmostEqual(value, 1.0)

    def test_physical_rows_use_plasma_and_window_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bunch_trajectories.npz"
            write_synthetic_trajectory(path)
            data = load_trajectory_npz(path)

            rows = build_physical_particle_rows(data, self.context)
            self.assertEqual(rows[0]["first_seen_step"], 0)
            self.assertAlmostEqual(rows[0]["first_z_from_plasma_start_um"], 5.0)
            self.assertAlmostEqual(rows[0]["first_xi_window_um"], 1.0)
            self.assertEqual(rows[0]["first_sustained_forward_step"], 100)
            self.assertEqual(rows[0]["first_E_ge_20_MeV_step"], 200)

            self.assertEqual(rows[1]["first_seen_step"], 100)
            self.assertAlmostEqual(rows[1]["first_z_from_plasma_start_um"], 15.0)
            self.assertAlmostEqual(rows[1]["first_xi_window_um"], 10.0)
            self.assertEqual(rows[1]["first_E_ge_1_MeV_step"], 100)

            thresholds = build_physical_threshold_rows(data, self.context)
            selected = [
                row
                for row in thresholds
                if row["id"] == 11 and row["threshold_MeV"] == 20.0
            ][0]
            self.assertEqual(selected["first_crossing_step"], 200)
            self.assertAlmostEqual(
                selected["z_from_plasma_start_at_crossing_um"], 7.0
            )

            cohorts = build_physical_cohort_rows(data, self.context)
            self.assertEqual([row["first_seen_step"] for row in cohorts], [0, 100])
            self.assertAlmostEqual(cohorts[0]["charge_pC"], 2.0 * 1.602176634e-19 * 1.0e12)

    def test_analysis_writes_physical_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory = root / "bunch_trajectories.npz"
            output = root / "physics"
            write_synthetic_trajectory(trajectory)

            summary = analyze_trajectory_physics_file(
                trajectory,
                output,
                self.context,
            )
            self.assertEqual(summary["schema"], "multichannel_trajectory_physics_v1")
            self.assertTrue(summary["first_seen_is_not_capture_time"])
            self.assertAlmostEqual(
                summary["weighted_origin"]["first_z_from_plasma_start_p50_um"],
                15.0,
            )

            for name in (
                "particle_physics.csv",
                "threshold_physics.csv",
                "cohort_physics.csv",
                "trajectory_physics_summary.json",
            ):
                self.assertTrue((output / name).is_file())

            plots = sorted((output / "plots").glob("*.png"))
            self.assertEqual(len(plots), 4)
            self.assertTrue(all(path.stat().st_size > 0 for path in plots))

            with (output / "particle_physics.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
            parsed = json.loads(
                (output / "trajectory_physics_summary.json").read_text()
            )
            self.assertEqual(
                parsed["context"]["moving_window_velocity_m_s"], 1.0e9
            )

    def test_cli_analyze_trajectory_physics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trajectory = root / "bunch_trajectories.npz"
            output = root / "physics"
            write_synthetic_trajectory(trajectory)

            rc = main(
                [
                    "analyze-trajectory-physics",
                    str(trajectory),
                    "--output-dir",
                    str(output),
                    "--plasma-start-um",
                    "5",
                    "--window-front-z0-um",
                    "9",
                    "--moving-window-velocity-m-s",
                    "1e9",
                    "--energy-thresholds-MeV",
                    "1,5,10,20",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((output / "trajectory_physics_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
