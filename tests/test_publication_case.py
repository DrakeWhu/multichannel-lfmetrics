import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from lfmetrics.publication_case import (
    audit_publication_case,
    find_openpmd_h5_series,
    read_openpmd_file_audits,
    write_publication_case_audit,
)


def write_particle_frame(path: Path, step: int) -> None:
    with h5py.File(path, "w") as h5:
        iteration = h5.create_group("data").create_group(str(step))
        iteration.attrs["time"] = 2.5
        iteration.attrs["timeUnitSI"] = 1.0e-15
        electrons = iteration.create_group("particles").create_group("electrons")
        electrons.create_dataset("id", data=np.asarray([10, 20], dtype=np.uint64))
        position = electrons.create_group("position")
        for axis in "xyz":
            component = position.create_dataset(axis, data=np.asarray([0.0, 1.0]))
            component.attrs["unitSI"] = 1.0e-6
        weighting = electrons.create_dataset(
            "weighting", data=np.asarray([1.0, 2.0])
        )
        weighting.attrs["unitSI"] = 1.0


def write_field_frame(path: Path, step: int) -> None:
    with h5py.File(path, "w") as h5:
        iteration = h5.create_group("data").create_group(str(step))
        iteration.attrs["time"] = 2.5
        iteration.attrs["timeUnitSI"] = 1.0e-15
        fields = iteration.create_group("fields")

        electric = fields.create_group("E")
        electric.attrs["axisLabels"] = np.asarray([b"z", b"y", b"x"])
        electric.attrs["gridSpacing"] = np.asarray([0.5, 0.25, 0.25])
        electric.attrs["gridGlobalOffset"] = np.asarray([4.0, -1.0, -1.0])
        electric.attrs["gridUnitSI"] = 1.0e-6
        electric.attrs["geometry"] = "cartesian"
        electric.attrs["dataOrder"] = "C"
        ex = electric.create_dataset("x", data=np.zeros((4, 3, 2)))
        ex.attrs["unitSI"] = 2.0
        ex.attrs["position"] = np.asarray([0.5, 0.0, 0.0])

        rho = fields.create_group("rho")
        rho.attrs["axisLabels"] = np.asarray([b"z", b"y", b"x"])
        rho.attrs["gridSpacing"] = np.asarray([0.5, 0.25, 0.25])
        rho.attrs["gridGlobalOffset"] = np.asarray([4.0, -1.0, -1.0])
        rho.attrs["gridUnitSI"] = 1.0e-6
        scalar = rho.create_dataset("SCALAR", data=np.ones((4, 3, 2)))
        scalar.attrs["unitSI"] = 3.0
        scalar.attrs["position"] = np.asarray([0.5, 0.5, 0.5])


class PublicationCaseTests(unittest.TestCase):
    def test_audit_separate_particle_and_field_series_with_same_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case"
            particle_dir = case / "particles"
            field_dir = case / "fields"
            particle_dir.mkdir(parents=True)
            field_dir.mkdir(parents=True)
            write_particle_frame(particle_dir / "openpmd_000300.h5", 300)
            write_field_frame(field_dir / "openpmd_000300.h5", 300)

            (case / "input.py").write_text("# synthetic\n", encoding="utf-8")
            (case / "resolved_parameters.json").write_text(
                "{}\n", encoding="utf-8"
            )
            selection = (
                case / "post" / "publication_tracking_deadbeef" / "selection"
            )
            backtracking = selection.parent / "backtracking"
            selection.mkdir(parents=True)
            backtracking.mkdir()
            np.save(
                selection / "final_bunch_ids.npy",
                np.asarray([10], dtype=np.uint64),
            )
            np.savez_compressed(
                backtracking / "bunch_trajectories.npz",
                id=np.asarray([10]),
            )

            audit = audit_publication_case(case)

            self.assertEqual(
                [item.series_kind for item in audit.series],
                ["fields", "particles"],
            )
            self.assertEqual([item.steps for item in audit.series], [(300,), (300,)])
            self.assertEqual(len(audit.context_files.input_py_candidates), 1)
            self.assertEqual(
                len(audit.context_files.final_bunch_ids_candidates), 1
            )

            field_series = next(
                item for item in audit.series if item.series_kind == "fields"
            )
            frame = field_series.iterations[0]
            self.assertAlmostEqual(frame.time_s, 2.5e-15)
            ex = frame.field_records["E"].components["x"]
            self.assertEqual(
                frame.field_records["E"].axis_labels,
                ("z", "y", "x"),
            )
            self.assertEqual(ex.shape, (4, 3, 2))
            self.assertEqual(ex.position, (0.5, 0.0, 0.0))
            self.assertEqual(ex.unit_si, 2.0)
            self.assertEqual(field_series.file_pattern, "openpmd_%06T.h5")

    def test_write_json_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "case"
            fields = case / "fields"
            fields.mkdir(parents=True)
            write_field_frame(fields / "openpmd_000300.h5", 300)
            audit = audit_publication_case(case)
            output = case / "post" / "audit.json"

            self.assertEqual(write_publication_case_audit(audit, output), output)
            stored = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                stored["schema"],
                "multichannel_publication_case_audit_v1",
            )
            with self.assertRaises(FileExistsError):
                write_publication_case_audit(audit, output)

    def test_missing_hdf5_is_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "No HDF5 files"):
                find_openpmd_h5_series(Path(tmp))

    def test_missing_data_group_is_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.h5"
            with h5py.File(path, "w") as h5:
                h5.create_group("not_data")
            with self.assertRaisesRegex(
                KeyError,
                "Missing openPMD 'data' group",
            ):
                read_openpmd_file_audits(path)


if __name__ == "__main__":
    unittest.main()
