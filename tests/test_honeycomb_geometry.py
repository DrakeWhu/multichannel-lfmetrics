import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lfmetrics.honeycomb_geometry import (
    geometry_from_mapping,
    read_honeycomb_geometry,
)


def geometry_values():
    spacing = 6.0e-6
    return {
        "cnt_radius_m": 2.5e-6,
        "lattice_spacing_m": spacing,
        "lattice_height_m": math.sqrt(3.0) * spacing / 2.0,
        "honeycomb_angle_deg": 30.0,
        "honeycomb_cos": math.cos(math.radians(30.0)),
        "honeycomb_sin": math.sin(math.radians(30.0)),
        "plasma_halfwidth_x_m": 15.0e-6,
        "plasma_halfwidth_y_m": 12.0e-6,
    }


class HoneycombGeometryTests(unittest.TestCase):
    def test_reads_resolved_parameters_and_derived_scales(self):
        values = geometry_values()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resolved_parameters.json"
            path.write_text(json.dumps(values), encoding="utf-8")
            geometry = read_honeycomb_geometry(path)

        self.assertAlmostEqual(geometry.cnt_gap_m, 1.0e-6)
        self.assertAlmostEqual(
            geometry.fill_fraction,
            math.pi * (2.5e-6) ** 2
            / (math.sqrt(3.0) * (6.0e-6) ** 2 / 2.0),
        )
        self.assertAlmostEqual(
            geometry.void_center_distance_from_rods_m,
            6.0e-6 / math.sqrt(3.0),
        )
        self.assertAlmostEqual(
            geometry.void_inscribed_radius_m,
            6.0e-6 / math.sqrt(3.0) - 2.5e-6,
        )

    def test_origin_is_a_rod_center_and_rotation_round_trips(self):
        geometry = geometry_from_mapping(geometry_values())

        center_x, center_y = geometry.nearest_rod_center_lab(0.0, 0.0)
        self.assertAlmostEqual(float(center_x), 0.0)
        self.assertAlmostEqual(float(center_y), 0.0)
        self.assertTrue(bool(geometry.inside_rod(0.0, 0.0)))

        x = np.asarray([-4.0e-6, 0.0, 5.0e-6])
        y = np.asarray([2.0e-6, -3.0e-6, 1.0e-6])
        x_rot, y_rot = geometry.lab_to_lattice(x, y)
        x_round, y_round = geometry.lattice_to_lab(x_rot, y_rot)
        np.testing.assert_allclose(x_round, x, rtol=0.0, atol=1.0e-18)
        np.testing.assert_allclose(y_round, y, rtol=0.0, atol=1.0e-18)

    def test_nearest_rod_matches_staggered_triangular_rows(self):
        values = geometry_values()
        values["honeycomb_angle_deg"] = 0.0
        values["honeycomb_cos"] = 1.0
        values["honeycomb_sin"] = 0.0
        geometry = geometry_from_mapping(values)
        height = geometry.lattice_height_m

        x = np.asarray([0.1e-6, 3.2e-6, -2.8e-6])
        y = np.asarray([0.1e-6, height * 0.9, -height * 1.1])
        center_x, center_y = geometry.nearest_rod_center_lattice(x, y)

        np.testing.assert_allclose(
            center_x,
            np.asarray([0.0, 3.0e-6, -3.0e-6]),
            rtol=0.0,
            atol=1.0e-18,
        )
        np.testing.assert_allclose(
            center_y,
            np.asarray([0.0, height, -height]),
            rtol=0.0,
            atol=1.0e-18,
        )

    def test_nearest_void_centers_form_honeycomb_dual(self):
        values = geometry_values()
        values["honeycomb_angle_deg"] = 0.0
        values["honeycomb_cos"] = 1.0
        values["honeycomb_sin"] = 0.0
        geometry = geometry_from_mapping(values)

        targets = np.asarray(
            [
                [3.0e-6, geometry.lattice_height_m / 3.0],
                [0.0, 2.0 * geometry.lattice_height_m / 3.0],
                [-3.0e-6, geometry.lattice_height_m / 3.0],
            ]
        )
        center_x, center_y = geometry.nearest_void_center_lattice(
            targets[:, 0],
            targets[:, 1],
        )
        np.testing.assert_allclose(
            np.column_stack((center_x, center_y)),
            targets,
            rtol=0.0,
            atol=1.0e-18,
        )
        np.testing.assert_allclose(
            np.hypot(targets[:, 0], targets[:, 1]),
            geometry.void_center_distance_from_rods_m,
            rtol=1.0e-15,
            atol=1.0e-18,
        )

    def test_real_case_origin_and_near_axis_wake_point_are_inside_central_rod(self):
        values = {
            "cnt_radius_m": 2.661313513840329e-6,
            "lattice_spacing_m": 5.858691108915924e-6,
            "lattice_height_m": 5.073775333247213e-6,
            "honeycomb_angle_deg": 53.66169129689802,
            "honeycomb_cos": 0.5925519005589255,
            "honeycomb_sin": 0.8055322744272915,
            "plasma_halfwidth_x_m": 1.5010003522965845e-5,
            "plasma_halfwidth_y_m": 1.2004269689232998e-5,
        }
        geometry = geometry_from_mapping(values)

        self.assertTrue(bool(geometry.inside_rod(0.0, 0.0)))
        self.assertTrue(bool(geometry.inside_rod(-0.15625e-6, 0.0)))
        self.assertAlmostEqual(geometry.cnt_gap_m, 0.5360640812352654e-6)
        self.assertAlmostEqual(
            geometry.void_inscribed_radius_m,
            0.7212033749911468e-6,
        )

        centers = geometry.rod_centers_in_bounds(
            xmin_m=-10.0e-6,
            xmax_m=10.0e-6,
            ymin_m=-10.0e-6,
            ymax_m=10.0e-6,
            margin_m=geometry.cnt_radius_m,
        )
        self.assertTrue(
            np.any(np.all(np.isclose(centers, 0.0, atol=1.0e-18), axis=1))
        )

    def test_invalid_geometry_is_rejected(self):
        values = geometry_values()
        values["lattice_height_m"] *= 1.1
        with self.assertRaisesRegex(ValueError, "triangular spacing"):
            geometry_from_mapping(values)


if __name__ == "__main__":
    unittest.main()
