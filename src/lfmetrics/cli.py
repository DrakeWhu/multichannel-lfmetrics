from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .beam_metrics import compute_beam_metrics
from .csv_io import write_particle_summary_csv
from .openpmd_h5 import read_particles_from_case, read_particles_from_h5
from .plots import write_particle_plots


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
    analyze_case.add_argument(
        "case_dir",
        type=Path,
        help="Case directory, for example combination_1 or a campaign-workflow case dir.",
    )
    analyze_case.add_argument(
        "--diagnostics-dir",
        default="3D",
        help="Diagnostics subdirectory relative to case_dir. Default: 3D.",
    )
    analyze_case.add_argument(
        "--species",
        default=None,
        help=(
            "Comma-separated species to analyze. "
            "If omitted, LFMetrics analyzes electrons."
        ),
    )
    analyze_case.add_argument(
        "--energy-threshold-MeV",
        type=float,
        default=5.0,
        help="Energy threshold for selected/hot particles. Default: 5.0 MeV.",
    )
    analyze_case.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: CASE_DIR/post/particle_summary.csv.",
    )
    analyze_case.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help=(
            "Case-local directory for fixed PNG diagnostics. "
            "Default: CASE_DIR/post/plots. Each species uses one child directory."
        ),
    )

    analyze_file = subparsers.add_parser(
        "analyze-h5",
        help="Analyze one openPMD HDF5 file directly.",
    )
    analyze_file.add_argument("h5_file", type=Path)
    analyze_file.add_argument(
        "--species",
        default=None,
        help="Comma-separated species to analyze. Current default: electrons.",
    )
    analyze_file.add_argument(
        "--energy-threshold-MeV",
        type=float,
        default=5.0,
        help="Energy threshold for selected/hot particles. Default: 5.0 MeV.",
    )
    analyze_case.add_argument(
        "--spectrum-min-energy-MeV",
        type=float,
        default=None,
        help=(
            "Lower kinetic-energy cutoff shown in energy_spectrum.png. "
            "Default: same as --energy-threshold-MeV."
        ),
    )
    analyze_file.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )

    return parser


def analyze_case(args: argparse.Namespace) -> Path:
    case_dir = args.case_dir.resolve(strict=False)
    if not case_dir.is_dir():
        raise LFMetricsCLIError(f"Case directory does not exist: {case_dir}")

    output = args.output
    if output is None:
        output = case_dir / "post" / "particle_summary.csv"

    plots_root = args.plots_dir
    if plots_root is None:
        plots_root = case_dir / "post" / "plots"
    elif not plots_root.is_absolute():
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
        rows.append(metrics.as_row())
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
        rows.append(metrics.as_row())

    return write_particle_summary_csv(rows, args.output)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze-case":
        output = analyze_case(args)
    elif args.command == "analyze-h5":
        output = analyze_h5(args)
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
