from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from .beam_metrics import compute_beam_metrics
from .csv_io import write_particle_summary_csv
from .openpmd_h5 import read_particles_from_case, read_particles_from_h5
from .plots import write_particle_plots
from .soft100 import compute_soft100_metrics
from .tracking import (
    backtrack_particle_ids,
    select_final_bunch,
    write_final_bunch_selection,
    write_tracking_outputs,
)
from .trajectory_analysis import (
    DEFAULT_ENERGY_THRESHOLDS_MEV,
    analyze_trajectory_file,
)


class LFMetricsCLIError(RuntimeError):
    pass


def parse_species_arg(value: str | None) -> list[str | None]:
    """Parse comma-separated species.

    Returns [None] when species is omitted, meaning the reader chooses the
    default beam-oriented species priority.
    """
    if value is None or value.strip() == "":
        return [None]

    species = [item.strip() for item in value.split(",") if item.strip()]
    if not species:
        return [None]
    return species


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one numeric value is required")
    if any(not np.isfinite(item) or item < 0.0 for item in values):
        raise argparse.ArgumentTypeError("values must be finite and non-negative")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lfmetrics",
        description="Particle and beam metrics for WarpX multichannel openPMD/HDF5 outputs.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_case = subparsers.add_parser(
        "analyze-case",
        help="Analyze a WarpX case directory containing a 3D/openPMD HDF5 diagnostic.",
    )
    analyze_case.add_argument("case_dir", type=Path)
    analyze_case.add_argument("--diagnostics-dir", default="3D")
    analyze_case.add_argument("--species", default=None)
    analyze_case.add_argument("--energy-threshold-MeV", type=float, default=5.0)
    analyze_case.add_argument("--output", type=Path, default=None)
    analyze_case.add_argument("--plots-dir", type=Path, default=None)
    analyze_case.add_argument("--spectrum-min-energy-MeV", type=float, default=None)

    analyze_file = subparsers.add_parser(
        "analyze-h5",
        help="Analyze one openPMD HDF5 file directly.",
    )
    analyze_file.add_argument("h5_file", type=Path)
    analyze_file.add_argument("--species", default=None)
    analyze_file.add_argument("--energy-threshold-MeV", type=float, default=5.0)
    analyze_file.add_argument("--output", type=Path, required=True)

    select_bunch = subparsers.add_parser(
        "select-final-bunch",
        help="Select a final-frame bunch and persist its WarpX particle IDs.",
    )
    select_bunch.add_argument("h5_file", type=Path)
    select_bunch.add_argument("--species", default="electrons")
    select_bunch.add_argument("--energy-threshold-MeV", type=float, default=20.0)
    select_bunch.add_argument(
        "--include-backward",
        action="store_true",
        help="Do not require pz > 0 when selecting the final bunch.",
    )
    select_bunch.add_argument("--output-dir", type=Path, required=True)

    backtrack = subparsers.add_parser(
        "backtrack-bunch",
        help="Backtrack a fixed NPY list of persistent particle IDs through all frames.",
    )
    backtrack.add_argument("case_dir", type=Path)
    backtrack.add_argument("--ids", type=Path, required=True)
    backtrack.add_argument("--species", default="electrons")
    backtrack.add_argument("--diagnostics-dir", default="3D")
    backtrack.add_argument("--output-dir", type=Path, required=True)

    trajectory = subparsers.add_parser(
        "analyze-trajectories",
        help="Analyze a bunch_trajectories.npz file by origin cohort and energy crossings.",
    )
    trajectory.add_argument("trajectory_npz", type=Path)
    trajectory.add_argument("--output-dir", type=Path, required=True)
    trajectory.add_argument(
        "--energy-thresholds-MeV",
        type=parse_float_list,
        default=DEFAULT_ENERGY_THRESHOLDS_MEV,
        help="Comma-separated kinetic-energy thresholds. Default: 1,5,10,20.",
    )

    return parser


def analyze_case(args: argparse.Namespace) -> Path:
    case_dir = args.case_dir.resolve(strict=False)
    if not case_dir.is_dir():
        raise LFMetricsCLIError(f"Case directory does not exist: {case_dir}")

    output = args.output or case_dir / "post" / "particle_summary.csv"
    plots_root = args.plots_dir or case_dir / "post" / "plots"
    if not plots_root.is_absolute():
        plots_root = case_dir / plots_root

    rows: list[dict[str, object]] = []
    for species in parse_species_arg(args.species):
        particles = read_particles_from_case(
            case_dir=case_dir,
            species=species,
            diagnostics_dir=args.diagnostics_dir,
        )
        metrics = compute_beam_metrics(
            particles,
            energy_threshold_MeV=args.energy_threshold_MeV,
        )
        soft100 = compute_soft100_metrics(particles)
        rows.append({**metrics.as_row(), **soft100.as_row()})
        write_particle_plots(
            particles,
            metrics,
            output_dir=plots_root / particles.species,
            energy_threshold_MeV=args.energy_threshold_MeV,
            spectrum_min_energy_MeV=args.spectrum_min_energy_MeV,
        )

    output_path = write_particle_summary_csv(rows, output)
    print(f"[lfmetrics] wrote plots under {plots_root}")
    return output_path


def analyze_h5(args: argparse.Namespace) -> Path:
    h5_file = args.h5_file.resolve(strict=False)
    if not h5_file.is_file():
        raise LFMetricsCLIError(f"HDF5 file does not exist: {h5_file}")

    rows: list[dict[str, object]] = []
    for species in parse_species_arg(args.species):
        particles = read_particles_from_h5(h5_file, species=species)
        metrics = compute_beam_metrics(
            particles,
            energy_threshold_MeV=args.energy_threshold_MeV,
        )
        soft100 = compute_soft100_metrics(particles)
        rows.append({**metrics.as_row(), **soft100.as_row()})

    return write_particle_summary_csv(rows, args.output)


def select_final_bunch_command(args: argparse.Namespace) -> Path:
    h5_file = args.h5_file.resolve(strict=False)
    if not h5_file.is_file():
        raise LFMetricsCLIError(f"HDF5 file does not exist: {h5_file}")

    selection = select_final_bunch(
        h5_file,
        species=args.species,
        energy_threshold_MeV=args.energy_threshold_MeV,
        forward_only=not args.include_backward,
    )
    summary = write_final_bunch_selection(selection, args.output_dir)
    print(json.dumps(summary, indent=2))
    return args.output_dir


def backtrack_bunch_command(args: argparse.Namespace) -> Path:
    case_dir = args.case_dir.resolve(strict=False)
    if not case_dir.is_dir():
        raise LFMetricsCLIError(f"Case directory does not exist: {case_dir}")
    if not args.ids.is_file():
        raise LFMetricsCLIError(f"Particle ID file does not exist: {args.ids}")

    target_ids = np.load(args.ids, allow_pickle=False)
    result = backtrack_particle_ids(
        case_dir,
        target_ids,
        species=args.species,
        diagnostics_dir=args.diagnostics_dir,
    )
    summary = write_tracking_outputs(result, args.output_dir)
    print(json.dumps(summary, indent=2))
    return args.output_dir


def analyze_trajectories_command(args: argparse.Namespace) -> Path:
    trajectory_npz = args.trajectory_npz.resolve(strict=False)
    if not trajectory_npz.is_file():
        raise LFMetricsCLIError(f"Trajectory NPZ does not exist: {trajectory_npz}")
    summary = analyze_trajectory_file(
        trajectory_npz,
        args.output_dir,
        energy_thresholds_MeV=args.energy_thresholds_MeV,
    )
    print(json.dumps(summary, indent=2))
    return args.output_dir


def _main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze-case":
        output = analyze_case(args)
    elif args.command == "analyze-h5":
        output = analyze_h5(args)
    elif args.command == "select-final-bunch":
        output = select_final_bunch_command(args)
    elif args.command == "backtrack-bunch":
        output = backtrack_bunch_command(args)
    elif args.command == "analyze-trajectories":
        output = analyze_trajectories_command(args)
    else:
        raise LFMetricsCLIError(f"Unknown command: {args.command}")

    print(f"[lfmetrics] wrote {output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return exc.code
        return 2
    except Exception as exc:
        print(f"[lfmetrics] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
