import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from lfmetrics.field_slices import read_mesh_plane


def write_field_file(path: Path) -> np.ndarray:
    values = np.arange(64, dtype=float).reshape(4, 4, 4)
    with h5py.File(path, "w") as h5:
        iteration = h5.create_group("data").create_group("300")
        iteration.attrs["time"] = 2.5
        iteration.attrs["timeUnitSI"] = 1.0e-15
        fields = iteration.create_group("fields")

        rho = fields.create_dataset("rho", data=values)
        rho.attrs["axisLabels"] = np.asarray([b"z", b"y", b"x"])
        rho.attrs["gridSpacing"] = np.asarray([1.0, 2.0, 4.0])
        rho.attrs["gridGlobalOffset"] = np.asarray([10.0, -4.0, -8.0])
        rho.attrs["gridUnitSI"] = 1.0e-6
        rho.attrs["geometry"] = "cartesian"
        rho.attrs["dataOrder"] = "C"
        rho.attrs["unitSI"] = 3.0
        rho.attrs["position"] = np.asarray([0.5, 0.5, 0.5])

        electric = fields.create_group("E")
        for name in (
            "axisLabels",
            "gridSpacing",
            "gridGlobalOffset",
            "gridUnitSI",
            "geometry",
            "dataOrder",
        ):
            electric.attrs[name] = rho.attrs[name]
        ex = electric.create_dataset("x", data=values)
        ex.attrs["unitSI"] = 2.0
        ex.attrs["position"] = np.asarray([0.5, 0.5, 0.5])

    return values


class FieldSlicesTests(unittest.TestCase):
    def test_linear_xz_plane_at_exact_physical_center(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_000300.h5"
            values = write_field_file(path)

            plane = read_mesh_plane(
                path,
                "rho",
                plane="xz",
                coordinate_m=0.0,
                method="linear",
            )

            self.assertEqual(plane.axis_labels, ("z", "x"))
            self.assertEqual(plane.normal_axis, "y")
            self.assertEqual(plane.source_indices, (1, 2))
            np.testing.assert_allclose(
                plane.source_coordinates_m,
                (-1.0e-6, 1.0e-6),
            )
            np.testing.assert_allclose(plane.source_weights, (0.5, 0.5))
            self.assertEqual(plane.actual_coordinate_m, 0.0)
            np.testing.assert_allclose(
                plane.values_si,
                0.5 * (values[:, 1, :] + values[:, 2, :]) * 3.0,
            )
            np.testing.assert_allclose(
                plane.coordinate("z"),
                np.asarray([10.5, 11.5, 12.5, 13.5]) * 1.0e-6,
            )
            np.testing.assert_allclose(
                plane.coordinate("x"),
                np.asarray([-6.0, -2.0, 2.0, 6.0]) * 1.0e-6,
            )

    def test_nearest_yz_plane_reads_only_selected_component_plane(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_000300.h5"
            values = write_field_file(path)

            plane = read_mesh_plane(
                path,
                "E",
                "x",
                plane="yz",
                coordinate_m=2.1e-6,
                method="nearest",
                iteration=300,
            )

            self.assertEqual(plane.axis_labels, ("z", "y"))
            self.assertEqual(plane.normal_axis, "x")
            self.assertEqual(plane.source_indices, (2,))
            self.assertAlmostEqual(plane.actual_coordinate_m, 2.0e-6)
            self.assertEqual(plane.shape, (4, 4))
            np.testing.assert_allclose(plane.values_si, values[:, :, 2] * 2.0)

    def test_linear_method_uses_one_plane_for_exact_cell_center(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_000300.h5"
            values = write_field_file(path)

            plane = read_mesh_plane(
                path,
                "rho",
                plane="xy",
                coordinate_m=11.5e-6,
                method="linear",
            )

            self.assertEqual(plane.axis_labels, ("y", "x"))
            self.assertEqual(plane.source_indices, (1,))
            self.assertEqual(plane.source_weights, (1.0,))
            np.testing.assert_allclose(plane.values_si, values[1, :, :] * 3.0)

    def test_methods_refuse_extrapolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_000300.h5"
            write_field_file(path)

            for method in ("nearest", "linear"):
                with self.subTest(method=method):
                    with self.assertRaisesRegex(
                        ValueError,
                        "extrapolation is not allowed",
                    ):
                        read_mesh_plane(
                            path,
                            "rho",
                            plane="xz",
                            coordinate_m=10.0e-6,
                            method=method,
                        )

    def test_invalid_plane_and_method_are_clear_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_000300.h5"
            write_field_file(path)

            with self.assertRaisesRegex(ValueError, "Unsupported plane"):
                read_mesh_plane(
                    path,
                    "rho",
                    plane="zx",
                    coordinate_m=0.0,
                )
            with self.assertRaisesRegex(ValueError, "Unsupported slice method"):
                read_mesh_plane(
                    path,
                    "rho",
                    plane="xz",
                    coordinate_m=0.0,
                    method="cubic",
                )


if __name__ == "__main__":
    unittest.main()
