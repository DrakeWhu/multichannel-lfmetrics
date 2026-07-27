import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lfmetrics.honeycomb_geometry import HoneycombGeometry
from lfmetrics.trajectory_analysis import TrajectoryData
from lfmetrics.trajectory_honeycomb import (
    build_frame_honeycomb_rows,
    build_particle_honeycomb_rows,
    classify_trajectory_honeycomb,
    load_and_classify_trajectory_honeycomb,
)


def geometry() -> HoneycombGeometry:
    spacing = 6.0e-6
    return HoneycombGeometry(
        cnt_radius_m=2.5e-6,
        lattice_spacing_m=spacing,
        lattice_height_m=math.sqrt(3.0) * spacing / 2.0,
        honeycomb_angle_deg=0.0,
        honeycomb_cos=1.0,
        honeycomb_sin=0.0,
        plasma_halfwidth_x_m=15.0e-6,
        plasma_halfwidth_y_m=12.0e-6,
    )


def trajectory() -> TrajectoryData:
    h = geometry().lattice_height_m
    present = np.asarray(
        [
            [True, True, False],
            [True, True, True],
            [True, True, True],
        ],
        dtype=bool,
    )
    x_m = np.asarray(
        [
            [0.0, 3.0e-6, np.nan],
            [0.0, 3.0e-6, 0.0],
            [0.0, 3.0e-6, 0.0],
        ]
    )
    y_m = np.asarray(
        [
            [0.0, h / 3.0, np.nan],
            [0.0, h / 3.0, 2.0 * h / 3.0],
            [0.0, h / 3.0, 2.0 * h / 3.0],
        ]
    )
    zeros = np.where(present, 0.0, np.nan)
    weighting = np.where(present, 2.0, np.nan)
    energy_MeV = np.asarray(
        [
            [1.0, 2.0, np.nan],
            [3.0, 4.0, 5.0],
            [20.0, 30.0, 40.0],
        ]
    )
    return TrajectoryData(
        ids=np.asarray([10, 20, 30], dtype=np.uint64),
        steps=np.asarray([0, 100, 200], dtype=np.int64),
        times_s=np.asarray([0.0, 1.0e-15, 2.0e-15]),
        present=present,
        x_m=x_m,
        y_m=y_m,
        z_m=zeros,
        px_si=zeros,
        py_si=zeros,
        pz_si=zeros,
        weighting=weighting,
        energy_MeV=energy_MeV,
    )


class TrajectoryHoneycombTests(unittest.TestCase):
    def test_classifies_rod_and_two_void_sublattices(self):
        result = classify_trajectory_honeycomb(trajectory(), geometry())

        self.assertTrue(result.inside_physical_rod[0, 0])
        self.assertTrue(result.inside_interstitial_void[0, 1])
        self.assertEqual(result.void_sublattice[0, 1], "A")
        self.assertEqual(result.void_sublattice[1, 2], "B")
        self.assertEqual(result.void_sublattice[0, 2], "")
        self.assertTrue(np.isnan(result.rod_distance_m[0, 2]))
        self.assertAlmostEqual(result.rod_clearance_m[0, 0], -2.5e-6)
        self.assertAlmostEqual(result.void_distance_m[0, 1], 0.0)
        self.assertAlmostEqual(result.void_distance_m[1, 2], 0.0)

    def test_particle_rows_keep_first_seen_distinct_from_capture(self):
        result = classify_trajectory_honeycomb(trajectory(), geometry())
        rows = build_particle_honeycomb_rows(result)

        third = rows[2]
        self.assertEqual(third["first_seen_step"], 100)
        self.assertTrue(third["first_seen_is_not_capture_time"])
        self.assertEqual(third["first_seen_void_sublattice"], "B")
        self.assertTrue(third["first_seen_inside_interstitial_void"])
        self.assertEqual(third["final_energy_MeV"], 40.0)

    def test_frame_rows_partition_present_charge(self):
        result = classify_trajectory_honeycomb(trajectory(), geometry())
        rows = build_frame_honeycomb_rows(result)

        self.assertEqual(rows[0]["n_present"], 2)
        self.assertEqual(rows[0]["n_inside_physical_rod"], 1)
        self.assertEqual(rows[0]["n_inside_interstitial_void"], 1)
        self.assertEqual(rows[0]["n_outside_plasma_bounds"], 0)
        self.assertAlmostEqual(
            rows[0]["charge_present_pC"],
            4.0 * 1.602176634e-19 * 1.0e12,
        )

    def test_present_nonfinite_xy_is_rejected(self):
        data = trajectory()
        bad_x = data.x_m.copy()
        bad_x[0, 0] = np.nan
        bad = TrajectoryData(
            ids=data.ids,
            steps=data.steps,
            times_s=data.times_s,
            present=data.present,
            x_m=bad_x,
            y_m=data.y_m,
            z_m=data.z_m,
            px_si=data.px_si,
            py_si=data.py_si,
            pz_si=data.pz_si,
            weighting=data.weighting,
            energy_MeV=data.energy_MeV,
        )
        with self.assertRaisesRegex(ValueError, "finite x_m and y_m"):
            classify_trajectory_honeycomb(bad, geometry())

    def test_outside_bounds_is_not_physical_rod_or_interstitial(self):
        data = trajectory()
        x_m = data.x_m.copy()
        y_m = data.y_m.copy()
        x_m[-1, 0] = 18.0e-6
        y_m[-1, 0] = 0.0
        outside = TrajectoryData(
            ids=data.ids,
            steps=data.steps,
            times_s=data.times_s,
            present=data.present,
            x_m=x_m,
            y_m=y_m,
            z_m=data.z_m,
            px_si=data.px_si,
            py_si=data.py_si,
            pz_si=data.pz_si,
            weighting=data.weighting,
            energy_MeV=data.energy_MeV,
        )
        result = classify_trajectory_honeycomb(outside, geometry())

        self.assertFalse(result.inside_plasma_bounds[-1, 0])
        self.assertFalse(result.inside_physical_rod[-1, 0])
        self.assertFalse(result.inside_interstitial_void[-1, 0])

    def test_loads_npz_and_resolved_geometry(self):
        data = trajectory()
        honeycomb = geometry()
        values = {
            "cnt_radius_m": honeycomb.cnt_radius_m,
            "lattice_spacing_m": honeycomb.lattice_spacing_m,
            "lattice_height_m": honeycomb.lattice_height_m,
            "honeycomb_angle_deg": honeycomb.honeycomb_angle_deg,
            "honeycomb_cos": honeycomb.honeycomb_cos,
            "honeycomb_sin": honeycomb.honeycomb_sin,
            "plasma_halfwidth_x_m": honeycomb.plasma_halfwidth_x_m,
            "plasma_halfwidth_y_m": honeycomb.plasma_halfwidth_y_m,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npz = root / "bunch_trajectories.npz"
            parameters = root / "resolved_parameters.json"
            np.savez_compressed(
                npz,
                id=data.ids,
                step=data.steps,
                time_s=data.times_s,
                present=data.present,
                x_m=data.x_m,
                y_m=data.y_m,
                z_m=data.z_m,
                px_si=data.px_si,
                py_si=data.py_si,
                pz_si=data.pz_si,
                weighting=data.weighting,
                energy_MeV=data.energy_MeV,
            )
            parameters.write_text(json.dumps(values), encoding="utf-8")
            result = load_and_classify_trajectory_honeycomb(npz, parameters)

        self.assertTrue(
            result.source_trajectory_npz.endswith("bunch_trajectories.npz")
        )
        self.assertTrue(
            result.source_geometry_json.endswith("resolved_parameters.json")
        )
        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(result.void_sublattice[1, 2], "B")


if __name__ == "__main__":
    unittest.main()
