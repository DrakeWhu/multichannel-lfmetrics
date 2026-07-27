import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lfmetrics.publication_fig9 import write_publication_fig9_snapshot
from tests.test_publication_fig9_v3 import (
    write_field_file,
    write_particle_file,
)


class PublicationFig9MultiresolutionTests(unittest.TestCase):
    def _write_snapshot(self, root: Path, **extra):
        particle_h5 = root / "3D" / "openpmd_000700.h5"
        field_h5 = root / "fields3D" / "openpmd_000700.h5"
        tracked_npy = root / "selection" / "final_bunch_ids.npy"
        output = root / "post" / "fig9_multires"

        ids = write_particle_file(particle_h5)
        write_field_file(field_h5)
        tracked_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(tracked_npy, ids[::5], allow_pickle=False)

        return write_publication_fig9_snapshot(
            particle_h5=particle_h5,
            field_h5=field_h5,
            tracked_ids_npy=tracked_npy,
            output_dir=output,
            step=700,
            case_label="synthetic multiresolution Fig. 9",
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
            **extra,
        )

    def test_default_uses_fixed_5_to_125_kT_and_native_central_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._write_snapshot(Path(tmp))

            self.assertEqual(result.magnetic_color_min_kT, 5.0)
            self.assertEqual(result.magnetic_color_max_kT, 125.0)
            self.assertIn("native field cell", result.quiver_representation)

            manifest = json.loads(
                Path(result.manifest_json).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["magnetic_visible_floor_kT"], 5.0)
            self.assertEqual(manifest["magnetic_color_limit_kT"], 125.0)
            self.assertEqual(
                manifest["magnetic_color_scale_mode"],
                "fixed floor and ceiling across frames",
            )
            self.assertEqual(
                manifest["magnetic_quiver_grid_mode"],
                "coarse exterior plus native central grid",
            )
            self.assertEqual(manifest["quiver_central_half_width_m"], 2.5e-6)
            self.assertGreater(
                manifest["central_quiver_candidate_native_points"],
                0,
            )
            self.assertGreaterEqual(
                manifest["central_quiver_candidate_native_points"],
                manifest["central_quiver_visible_native_arrows"],
            )

    def test_explicit_fixed_scale_and_central_width_are_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._write_snapshot(
                Path(tmp),
                magnetic_visible_floor_kT=3.0,
                magnetic_color_limit_kT=150.0,
                quiver_central_half_width_m=1.5e-6,
            )

            self.assertEqual(result.magnetic_color_min_kT, 3.0)
            self.assertEqual(result.magnetic_color_max_kT, 150.0)

            manifest = json.loads(
                Path(result.manifest_json).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["quiver_central_half_width_m"], 1.5e-6)

    def test_rejects_floor_not_below_ceiling_before_reading_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"

            with self.assertRaisesRegex(
                ValueError,
                "magnetic_visible_floor_kT",
            ):
                write_publication_fig9_snapshot(
                    particle_h5=root / "missing_particle.h5",
                    field_h5=root / "missing_field.h5",
                    tracked_ids_npy=root / "missing_ids.npy",
                    output_dir=output,
                    step=700,
                    case_label="synthetic",
                    magnetic_visible_floor_kT=125.0,
                    magnetic_color_limit_kT=125.0,
                )

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
