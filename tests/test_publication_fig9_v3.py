import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from lfmetrics.publication_fig9 import (
    PUBLICATION_FIG9_SCHEMA,
    write_publication_fig9_snapshot,
)


def write_field_file(path: Path, *, step: int = 700, time_s: float = 9.2e-14) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (8, 10, 12)  # z, y, x
    z_index, y_index, x_index = np.indices(shape)

    ez_values = (
        np.sin(0.5 * z_index)
        * np.exp(-((x_index - 5.5) ** 2 + (y_index - 4.5) ** 2) / 20.0)
        * 2.0e12
    )
    bx_values = -(y_index - 4.5) * (0.4 + 0.2 * z_index) * 2.0e3
    by_values = (x_index - 5.5) * (0.4 + 0.2 * z_index) * 2.0e3

    with h5py.File(path, "w") as h5:
        iteration = h5.create_group("data").create_group(str(step))
        iteration.attrs["time"] = time_s
        iteration.attrs["timeUnitSI"] = 1.0
        fields = iteration.create_group("fields")

        for record_name, components in (
            ("E", {"z": ez_values}),
            ("B", {"x": bx_values, "y": by_values}),
        ):
            record = fields.create_group(record_name)
            record.attrs["axisLabels"] = np.asarray([b"z", b"y", b"x"])
            record.attrs["gridSpacing"] = np.asarray([1.0, 1.0, 1.0])
            record.attrs["gridGlobalOffset"] = np.asarray([8.0, -5.0, -6.0])
            record.attrs["gridUnitSI"] = 1.0e-6
            record.attrs["geometry"] = "cartesian"
            record.attrs["dataOrder"] = "C"

            for component_name, values in components.items():
                component = record.create_dataset(component_name, data=values)
                component.attrs["position"] = np.asarray([0.5, 0.5, 0.5])
                component.attrs["unitSI"] = 1.0


def write_particle_file(
    path: Path,
    *,
    step: int = 700,
    time_s: float = 9.2e-14,
) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.arange(2000, 2080, dtype=np.uint64)
    n = ids.size

    x_um = np.linspace(-4.5, 4.5, n)
    y_um = 2.5 * np.sin(np.linspace(0.0, 3.0 * np.pi, n))
    z_um = np.linspace(10.0, 14.5, n)
    pz = np.linspace(0.2e-21, 3.0e-21, n)

    with h5py.File(path, "w") as h5:
        iteration = h5.create_group("data").create_group(str(step))
        iteration.attrs["time"] = time_s
        iteration.attrs["timeUnitSI"] = 1.0
        electrons = iteration.create_group("particles").create_group("electrons")
        electrons.create_dataset("id", data=ids)

        weighting = electrons.create_dataset(
            "weighting",
            data=np.linspace(1.0, 2.0, n),
        )
        weighting.attrs["unitSI"] = 1.0

        position = electrons.create_group("position")
        for axis, values_um in (
            ("x", x_um),
            ("y", y_um),
            ("z", z_um),
        ):
            component = position.create_dataset(axis, data=values_um)
            component.attrs["unitSI"] = 1.0e-6

        offsets = electrons.create_group("positionOffset")
        for axis in ("x", "y", "z"):
            component = offsets.create_group(axis)
            component.attrs["value"] = 0.0
            component.attrs["unitSI"] = 1.0

        momentum = electrons.create_group("momentum")
        for axis, values in (
            ("x", np.zeros(n)),
            ("y", np.zeros(n)),
            ("z", pz),
        ):
            component = momentum.create_dataset(axis, data=values)
            component.attrs["unitSI"] = 1.0

    return ids


class PublicationFig9V3Tests(unittest.TestCase):
    def test_v3_uses_inferno_quiver_equal_markers_and_ez_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            particle_h5 = root / "3D" / "openpmd_000700.h5"
            field_h5 = root / "fields3D" / "openpmd_000700.h5"
            tracked_npy = root / "selection" / "final_bunch_ids.npy"
            output = root / "post" / "fig9_v3"

            ids = write_particle_file(particle_h5)
            write_field_file(field_h5)
            tracked_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(tracked_npy, ids[::5], allow_pickle=False)

            result = write_publication_fig9_snapshot(
                particle_h5=particle_h5,
                field_h5=field_h5,
                tracked_ids_npy=tracked_npy,
                output_dir=output,
                step=700,
                case_label="synthetic Fig. 9 v3",
                background_slab_half_width_m=2.0e-6,
                transverse_half_width_m=6.0e-6,
                xi_padding_back_m=2.0e-6,
                xi_padding_front_m=2.0e-6,
                max_background_points_per_panel=17,
                ez_color_limit_floor_TV_m=5.0,
                magnetic_visible_percentile=60.0,
                magnetic_color_percentile=99.0,
                magnetic_length_gamma=1.8,
                quiver_max_arrows_per_axis=5,
                dpi=80,
            )

            self.assertEqual(
                PUBLICATION_FIG9_SCHEMA,
                "multichannel_publication_fig9_snapshot_v3",
            )
            self.assertGreaterEqual(result.ez_color_limit_TV_m, 5.0)
            self.assertGreater(result.magnetic_color_max_kT, 0.0)
            self.assertGreaterEqual(
                result.magnetic_color_max_kT,
                result.magnetic_color_min_kT,
            )
            self.assertEqual(result.xy_vector_overlay, "B/x,B/y quiver")
            self.assertIn("same marker size", result.particle_marker_size_mode)

            manifest = json.loads(
                Path(result.manifest_json).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema"], PUBLICATION_FIG9_SCHEMA)
            self.assertEqual(manifest["magnetic_quiver_colormap"], "inferno")
            self.assertEqual(manifest["magnetic_visible_percentile"], 60.0)
            self.assertEqual(manifest["magnetic_color_percentile"], 99.0)
            self.assertEqual(manifest["magnetic_length_gamma"], 1.8)
            self.assertEqual(
                manifest["particle_marker_size_mode"],
                "same size for background and tracked",
            )
            self.assertEqual(manifest["ez_color_limit_floor_TV_m"], 5.0)
            self.assertEqual(
                set(manifest["panel_marker_sizes_points_squared"]),
                {"xy", "xz", "yz"},
            )

    def test_rejects_nonpositive_ez_floor_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            particle_h5 = root / "particle.h5"
            field_h5 = root / "field.h5"
            tracked_npy = root / "ids.npy"
            output = root / "output"

            ids = write_particle_file(particle_h5)
            write_field_file(field_h5)
            np.save(tracked_npy, ids[:3], allow_pickle=False)

            with self.assertRaisesRegex(
                ValueError,
                "ez_color_limit_floor_TV_m",
            ):
                write_publication_fig9_snapshot(
                    particle_h5=particle_h5,
                    field_h5=field_h5,
                    tracked_ids_npy=tracked_npy,
                    output_dir=output,
                    step=700,
                    case_label="synthetic",
                    ez_color_limit_floor_TV_m=0.0,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
