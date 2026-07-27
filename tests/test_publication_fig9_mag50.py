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


class PublicationFig9MagneticCeilingTests(unittest.TestCase):
    def _write_snapshot(
        self,
        root: Path,
        *,
        output_name: str,
        magnetic_color_limit_kT: float | None = None,
    ):
        particle_h5 = root / "3D" / "openpmd_000700.h5"
        field_h5 = root / "fields3D" / "openpmd_000700.h5"
        tracked_npy = root / "selection" / "final_bunch_ids.npy"
        output = root / "post" / output_name

        ids = write_particle_file(particle_h5)
        write_field_file(field_h5)
        tracked_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(tracked_npy, ids[::5], allow_pickle=False)

        kwargs = {}
        if magnetic_color_limit_kT is not None:
            kwargs["magnetic_color_limit_kT"] = magnetic_color_limit_kT

        return write_publication_fig9_snapshot(
            particle_h5=particle_h5,
            field_h5=field_h5,
            tracked_ids_npy=tracked_npy,
            output_dir=output,
            step=700,
            case_label="synthetic magnetic ceiling",
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
            **kwargs,
        )

    def test_default_magnetic_color_ceiling_is_50_kT(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._write_snapshot(
                Path(tmp),
                output_name="default_50kT",
            )

            self.assertEqual(result.magnetic_color_max_kT, 50.0)

            manifest = json.loads(
                Path(result.manifest_json).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["magnetic_color_limit_kT"], 50.0)
            self.assertEqual(
                manifest["magnetic_color_scale_mode"],
                "fixed upper limit; lower limit from magnetic visibility percentile",
            )
            self.assertEqual(
                manifest["result"]["magnetic_color_max_kT"],
                50.0,
            )

    def test_explicit_magnetic_color_ceiling_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._write_snapshot(
                Path(tmp),
                output_name="explicit_40kT",
                magnetic_color_limit_kT=40.0,
            )

            self.assertEqual(result.magnetic_color_max_kT, 40.0)

    def test_rejects_nonpositive_magnetic_color_ceiling_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "post" / "invalid"

            with self.assertRaisesRegex(
                ValueError,
                "magnetic_color_limit_kT",
            ):
                write_publication_fig9_snapshot(
                    particle_h5=root / "missing_particle.h5",
                    field_h5=root / "missing_field.h5",
                    tracked_ids_npy=root / "missing_ids.npy",
                    output_dir=output,
                    step=700,
                    case_label="synthetic",
                    magnetic_color_limit_kT=0.0,
                )

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
