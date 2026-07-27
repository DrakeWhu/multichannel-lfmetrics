import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from lfmetrics.constants import C_LIGHT
from lfmetrics.wake_fields import read_transverse_wake_plane


def write_field_file(path: Path, *, by_position_y: float = 0.5) -> None:
    shape = (4, 4, 4)

    with h5py.File(path, "w") as h5:
        iteration = h5.create_group("data").create_group("300")
        iteration.attrs["time"] = 2.5
        iteration.attrs["timeUnitSI"] = 1.0e-15
        fields = iteration.create_group("fields")

        shared_attrs = {
            "axisLabels": np.asarray([b"z", b"y", b"x"]),
            "gridSpacing": np.asarray([1.0, 2.0, 4.0]),
            "gridGlobalOffset": np.asarray([10.0, -4.0, -8.0]),
            "gridUnitSI": 1.0e-6,
            "geometry": "cartesian",
            "dataOrder": "C",
        }

        electric = fields.create_group("E")
        magnetic = fields.create_group("B")
        for record in (electric, magnetic):
            for name, value in shared_attrs.items():
                record.attrs[name] = value

        ex = electric.create_dataset("x", data=np.full(shape, 10.0))
        ey = electric.create_dataset("y", data=np.full(shape, 20.0))
        bx = magnetic.create_dataset("x", data=np.full(shape, 3.0e-8))
        by = magnetic.create_dataset("y", data=np.full(shape, 2.0e-8))

        for component in (ex, ey, bx, by):
            component.attrs["unitSI"] = 1.0
            component.attrs["position"] = np.asarray([0.5, 0.5, 0.5])

        by.attrs.modify("position", np.asarray([0.5, by_position_y, 0.5]))


class WakeFieldsTests(unittest.TestCase):
    def test_positive_charge_wake_for_plus_z_propagation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_000300.h5"
            write_field_file(path)

            wake = read_transverse_wake_plane(
                path,
                plane="xz",
                coordinate_m=0.0,
                beta=0.5,
                propagation_sign=1,
                method="linear",
                iteration=300,
            )

            self.assertEqual(wake.step, 300)
            self.assertEqual(wake.axis_labels, ("z", "x"))
            self.assertEqual(wake.shape, (4, 4))
            self.assertEqual(wake.source_indices, (1, 2))
            self.assertEqual(wake.source_weights, (0.5, 0.5))
            self.assertEqual(wake.beta, 0.5)
            self.assertEqual(wake.propagation_sign, 1)

            np.testing.assert_allclose(
                wake.wx_v_m,
                10.0 - 0.5 * C_LIGHT * 2.0e-8,
            )
            np.testing.assert_allclose(
                wake.wy_v_m,
                20.0 + 0.5 * C_LIGHT * 3.0e-8,
            )

    def test_minus_z_propagation_reverses_magnetic_terms(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_000300.h5"
            write_field_file(path)

            wake = read_transverse_wake_plane(
                path,
                plane="xz",
                coordinate_m=0.0,
                beta=0.5,
                propagation_sign=-1,
            )

            np.testing.assert_allclose(
                wake.wx_v_m,
                10.0 + 0.5 * C_LIGHT * 2.0e-8,
            )
            np.testing.assert_allclose(
                wake.wy_v_m,
                20.0 - 0.5 * C_LIGHT * 3.0e-8,
            )

    def test_zero_beta_returns_electric_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_000300.h5"
            write_field_file(path)

            wake = read_transverse_wake_plane(
                path,
                plane="xz",
                coordinate_m=0.0,
                beta=0.0,
            )

            np.testing.assert_allclose(wake.wx_v_m, 10.0)
            np.testing.assert_allclose(wake.wy_v_m, 20.0)

    def test_invalid_beta_and_propagation_sign_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_000300.h5"
            write_field_file(path)

            for beta in (-0.1, 1.1, float("nan")):
                with self.subTest(beta=beta):
                    with self.assertRaisesRegex(ValueError, "0 <= beta <= 1"):
                        read_transverse_wake_plane(
                            path,
                            plane="xz",
                            coordinate_m=0.0,
                            beta=beta,
                        )

            for sign in (0, 2, 0.5):
                with self.subTest(sign=sign):
                    with self.assertRaisesRegex(ValueError, "propagation_sign"):
                        read_transverse_wake_plane(
                            path,
                            plane="xz",
                            coordinate_m=0.0,
                            propagation_sign=sign,
                        )

    def test_non_colocated_field_components_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "openpmd_000300.h5"
            write_field_file(path, by_position_y=0.0)

            with self.assertRaisesRegex(ValueError, "not colocated"):
                read_transverse_wake_plane(
                    path,
                    plane="xz",
                    coordinate_m=0.0,
                    beta=1.0,
                )


if __name__ == "__main__":
    unittest.main()
