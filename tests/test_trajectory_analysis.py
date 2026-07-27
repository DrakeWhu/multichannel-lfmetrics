import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lfmetrics.cli import main
from lfmetrics.trajectory_analysis import (
    analyze_trajectory_file,
    build_cohort_summary_rows,
    build_particle_history_rows,
    build_threshold_crossing_rows,
    load_trajectory_npz,
)


def write_synthetic_trajectory(path: Path) -> None:
    ids = np.asarray([101, 102, 103], dtype=np.uint64)
    steps = np.asarray([0, 100, 200, 300], dtype=np.int64)
    times = np.asarray([0.0, 1.0e-15, 2.0e-15, 3.0e-15])
    present = np.asarray(
        [
            [True, False, False],
            [True, True, False],
            [True, True, True],
            [True, True, True],
        ],
        dtype=bool,
    )
    shape = present.shape
    x_m = np.full(shape, np.nan)
    y_m = np.full(shape, np.nan)
    z_m = np.full(shape, np.nan)
    px_si = np.full(shape, np.nan)
    py_si = np.full(shape, np.nan)
    pz_si = np.full(shape, np.nan)
    weighting = np.full(shape, np.nan)
    energy = np.full(shape, np.nan)

    for frame in range(shape[0]):
        for particle in range(shape[1]):
            if not present[frame, particle]:
                continue
            x_m[frame, particle] = (particle + 1) * 1.0e-6
            y_m[frame, particle] = -(particle + 1) * 1.0e-6
            z_m[frame, particle] = (10 * particle + frame) * 1.0e-6
            px_si[frame, particle] = 0.0
            py_si[frame, particle] = 0.0
            pz_si[frame, particle] = (frame - particle) * 1.0e-21
            weighting[frame, particle] = float(particle + 1)

    energy[:, 0] = [0.0, 2.0, 8.0, 25.0]
    energy[1:, 1] = [0.5, 6.0, 21.0]
    energy[2:, 2] = [12.0, 30.0]

    np.savez_compressed(
        path,
        id=ids,
        step=steps,
        time_s=times,
        present=present,
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        px_si=px_si,
        py_si=py_si,
        pz_si=pz_si,
        weighting=weighting,
        energy_MeV=energy,
    )


class TrajectoryAnalysisTests(unittest.TestCase):
    def test_particle_history_and_threshold_crossings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bunch_trajectories.npz"
            write_synthetic_trajectory(path)
            data = load_trajectory_npz(path)

            history = build_particle_history_rows(data)
            self.assertEqual([row["first_seen_step"] for row in history], [0, 100, 200])
            self.assertEqual(history[0]["first_E_ge_20_MeV_step"], 300)
            self.assertEqual(history[1]["first_E_ge_5_MeV_step"], 200)
            self.assertEqual(history[2]["first_E_ge_10_MeV_step"], 200)
            self.assertEqual(history[0]["first_sustained_forward_step"], 100)

            crossings = build_threshold_crossing_rows(data)
            self.assertEqual(len(crossings), 12)
            selected = [
                row
                for row in crossings
                if row["id"] == 103 and row["threshold_MeV"] == 20.0
            ]
            self.assertEqual(selected[0]["first_crossing_step"], 300)

    def test_cohort_rows_preserve_first_seen_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bunch_trajectories.npz"
            write_synthetic_trajectory(path)
            rows = build_cohort_summary_rows(load_trajectory_npz(path))
            self.assertEqual([row["first_seen_step"] for row in rows], [0, 100, 200])
            self.assertEqual([row["n_particles"] for row in rows], [1, 1, 1])
            self.assertEqual(rows[-1]["final_energy_max_MeV"], 30.0)

    def test_analysis_writes_csv_json_and_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trajectory = tmp_path / "bunch_trajectories.npz"
            output = tmp_path / "analysis"
            write_synthetic_trajectory(trajectory)

            summary = analyze_trajectory_file(trajectory, output)
            self.assertEqual(summary["schema"], "multichannel_trajectory_analysis_v1")
            self.assertEqual(summary["n_particles"], 3)
            self.assertEqual(summary["first_seen_histogram"], {"0": 1, "100": 1, "200": 1})

            for name in (
                "particle_history.csv",
                "threshold_crossings.csv",
                "cohort_summary.csv",
                "trajectory_summary.json",
            ):
                self.assertTrue((output / name).is_file())

            plots = sorted((output / "plots").glob("*.png"))
            self.assertEqual(len(plots), 5)
            self.assertTrue(all(path.stat().st_size > 0 for path in plots))

            with (output / "particle_history.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)
            parsed = json.loads((output / "trajectory_summary.json").read_text())
            self.assertTrue(parsed["first_seen_is_not_capture_time"])

    def test_cli_analyze_trajectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trajectory = tmp_path / "bunch_trajectories.npz"
            output = tmp_path / "analysis"
            write_synthetic_trajectory(trajectory)

            rc = main(
                [
                    "analyze-trajectories",
                    str(trajectory),
                    "--output-dir",
                    str(output),
                    "--energy-thresholds-MeV",
                    "1,5,10,20",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((output / "trajectory_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
