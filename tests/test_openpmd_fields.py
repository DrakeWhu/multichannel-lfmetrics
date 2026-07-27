import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from lfmetrics.openpmd_fields import read_mesh_component


def write_field_file(path: Path) -> None:
    with h5py.File(path, "w") as h5:
        iteration = h5.create_group("data").create_group("300")
        iteration.attrs["time"] = 2.5
        iteration.attrs["timeUnitSI"] = 1.0e-15
        fields = iteration.create_group("fields")

        electric = fields.create_group("E")
        electric.attrs["axisLabels"] = np.asarray([b"z", b"y", b"x"])
        electric.attrs["gridSpacing"] = np.asarray([0.5, 2.0, 4.0])
        electric.attrs["gridGlobalOffset"] = np.asarray([10.0, -4.0, -8.0])
        electric.attrs["gridUnitSI"] = 1.0e-6
        electric.attrs["geometry"] = "cartesian"
        electric.attrs["dataOrder"] = "C"
        ex = electric.create_dataset(
            "x",
            data=np.arange(24, dtype=float).reshape(4, 3, 2),
        )
        ex.attrs["unitSI"] = 2.0
        ex.attrs["position"] = np.asarray([0.5, 0.0, 1.0])

        rho = fields.create_group("rho")
        for name, value in electric.attrs.items():
            rho.attrs[name] = value
        scalar = rho.create_dataset("SCALAR", data=np.ones((4, 3, 2)))
        scalar.attrs["unitSI"] = 3.0
        scalar.attrs["position"] = np.asarray([0.5, 0.5, 0.5])


class OpenPMDFieldsTests(unittest.TestCase):
    def test_reads_values_and_physical_coordinates_in_axis_label_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_000300.h5"
            write_field_file(path)

            component = read_mesh_component(path, "E", "x")

            self.assertEqual(component.step, 300)
            self.assertAlmostEqual(component.time_s, 2.5e-15)
            self.assertEqual(component.axis_labels, ("z", "y", "x"))
            self.assertEqual(component.shape, (4, 3, 2))
            np.testing.assert_allclose(
                component.values_si,
                np.arange(24, dtype=float).reshape(4, 3, 2) * 2.0,
            )
            np.testing.assert_allclose(
                component.coordinate("z"),
                np.asarray([10.25, 10.75, 11.25, 11.75]) * 1.0e-6,
            )
            np.testing.assert_allclose(
                component.coordinate("y"),
                np.asarray([-4.0, -2.0, 0.0]) * 1.0e-6,
            )
            np.testing.assert_allclose(
                component.coordinate("x"),
                np.asarray([-4.0, 0.0]) * 1.0e-6,
            )

    def test_reads_scalar_mesh_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_000300.h5"
            write_field_file(path)

            rho = read_mesh_component(path, "rho")

            self.assertEqual(rho.component_name, "SCALAR")
            self.assertEqual(rho.position, (0.5, 0.5, 0.5))
            np.testing.assert_allclose(rho.values_si, 3.0)

    def test_missing_component_is_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_000300.h5"
            write_field_file(path)

            with self.assertRaisesRegex(KeyError, "available components"):
                read_mesh_component(path, "E", "q")

    def test_rejects_non_cartesian_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_000300.h5"
            write_field_file(path)
            with h5py.File(path, "r+") as h5:
                h5["data"]["300"]["fields"]["E"].attrs["geometry"] = "thetaMode"

            with self.assertRaisesRegex(ValueError, "only cartesian"):
                read_mesh_component(path, "E", "x")


if __name__ == "__main__":
    unittest.main()
