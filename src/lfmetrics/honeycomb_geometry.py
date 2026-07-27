from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class HoneycombGeometry:
    """Exact transverse geometry used by the multichannel PICMI input.

    Plasma cylinders are centered on a triangular lattice in the rotated
    coordinates used by the WarpX analytic density expression. The interstitial
    void centers form the two-sublattice honeycomb dual.
    """

    cnt_radius_m: float
    lattice_spacing_m: float
    lattice_height_m: float
    honeycomb_angle_deg: float
    honeycomb_cos: float
    honeycomb_sin: float
    plasma_halfwidth_x_m: float | None = None
    plasma_halfwidth_y_m: float | None = None

    def validate(self) -> None:
        positive = {
            "cnt_radius_m": self.cnt_radius_m,
            "lattice_spacing_m": self.lattice_spacing_m,
            "lattice_height_m": self.lattice_height_m,
        }
        invalid = [
            name
            for name, value in positive.items()
            if not np.isfinite(value) or value <= 0.0
        ]
        if invalid:
            raise ValueError(f"Positive finite geometry values required: {invalid}")

        if 2.0 * self.cnt_radius_m > self.lattice_spacing_m:
            raise ValueError(
                "CNT diameter exceeds lattice spacing; cylinders would overlap"
            )

        expected_height = math.sqrt(3.0) * self.lattice_spacing_m / 2.0
        if not np.isclose(
            self.lattice_height_m,
            expected_height,
            rtol=1.0e-12,
            atol=1.0e-18,
        ):
            raise ValueError(
                "lattice_height_m is inconsistent with triangular spacing: "
                f"{self.lattice_height_m} != {expected_height}"
            )

        norm = self.honeycomb_cos**2 + self.honeycomb_sin**2
        if not np.isclose(norm, 1.0, rtol=1.0e-12, atol=1.0e-14):
            raise ValueError(
                "honeycomb_cos and honeycomb_sin do not define a unit rotation"
            )

        expected_cos = math.cos(math.radians(self.honeycomb_angle_deg))
        expected_sin = math.sin(math.radians(self.honeycomb_angle_deg))
        if not (
            np.isclose(
                self.honeycomb_cos,
                expected_cos,
                rtol=1.0e-12,
                atol=1.0e-14,
            )
            and np.isclose(
                self.honeycomb_sin,
                expected_sin,
                rtol=1.0e-12,
                atol=1.0e-14,
            )
        ):
            raise ValueError(
                "Stored honeycomb cosine/sine are inconsistent with angle"
            )

        for name, value in (
            ("plasma_halfwidth_x_m", self.plasma_halfwidth_x_m),
            ("plasma_halfwidth_y_m", self.plasma_halfwidth_y_m),
        ):
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be positive and finite when provided")

    @property
    def cnt_gap_m(self) -> float:
        return self.lattice_spacing_m - 2.0 * self.cnt_radius_m

    @property
    def fill_fraction(self) -> float:
        cell_area = math.sqrt(3.0) * self.lattice_spacing_m**2 / 2.0
        return math.pi * self.cnt_radius_m**2 / cell_area

    @property
    def void_center_distance_from_rods_m(self) -> float:
        return self.lattice_spacing_m / math.sqrt(3.0)

    @property
    def void_inscribed_radius_m(self) -> float:
        return self.void_center_distance_from_rods_m - self.cnt_radius_m

    def lab_to_lattice(
        self,
        x_m: np.ndarray | float,
        y_m: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x_m, dtype=float)
        y = np.asarray(y_m, dtype=float)
        x, y = np.broadcast_arrays(x, y)
        x_rot = self.honeycomb_cos * x + self.honeycomb_sin * y
        y_rot = -self.honeycomb_sin * x + self.honeycomb_cos * y
        return x_rot, y_rot

    def lattice_to_lab(
        self,
        x_rot_m: np.ndarray | float,
        y_rot_m: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_rot = np.asarray(x_rot_m, dtype=float)
        y_rot = np.asarray(y_rot_m, dtype=float)
        x_rot, y_rot = np.broadcast_arrays(x_rot, y_rot)
        x = self.honeycomb_cos * x_rot - self.honeycomb_sin * y_rot
        y = self.honeycomb_sin * x_rot + self.honeycomb_cos * y_rot
        return x, y

    def _nearest_lattice_site(
        self,
        x_rot_m: np.ndarray | float,
        y_rot_m: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_rot = np.asarray(x_rot_m, dtype=float)
        y_rot = np.asarray(y_rot_m, dtype=float)
        x_rot, y_rot = np.broadcast_arrays(x_rot, y_rot)

        row = np.floor(y_rot / self.lattice_height_m + 0.5)
        parity = row - 2.0 * np.floor(row / 2.0)
        stagger = 0.5 * self.lattice_spacing_m * parity
        center_x = (
            self.lattice_spacing_m
            * np.floor((x_rot - stagger) / self.lattice_spacing_m + 0.5)
            + stagger
        )
        center_y = self.lattice_height_m * row
        return center_x, center_y

    def nearest_rod_center_lattice(
        self,
        x_rot_m: np.ndarray | float,
        y_rot_m: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._nearest_lattice_site(x_rot_m, y_rot_m)

    def nearest_rod_center_lab(
        self,
        x_m: np.ndarray | float,
        y_m: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_rot, y_rot = self.lab_to_lattice(x_m, y_m)
        center_x_rot, center_y_rot = self._nearest_lattice_site(x_rot, y_rot)
        return self.lattice_to_lab(center_x_rot, center_y_rot)

    def nearest_void_center_lattice(
        self,
        x_rot_m: np.ndarray | float,
        y_rot_m: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the nearest interstitial center of the honeycomb dual."""

        x_rot = np.asarray(x_rot_m, dtype=float)
        y_rot = np.asarray(y_rot_m, dtype=float)
        x_rot, y_rot = np.broadcast_arrays(x_rot, y_rot)

        a_x, a_y = self._nearest_lattice_site(
            x_rot - 0.5 * self.lattice_spacing_m,
            y_rot - self.lattice_height_m / 3.0,
        )
        a_x = a_x + 0.5 * self.lattice_spacing_m
        a_y = a_y + self.lattice_height_m / 3.0

        b_x, b_y = self._nearest_lattice_site(
            x_rot,
            y_rot - 2.0 * self.lattice_height_m / 3.0,
        )
        b_y = b_y + 2.0 * self.lattice_height_m / 3.0

        distance_a_sq = (x_rot - a_x) ** 2 + (y_rot - a_y) ** 2
        distance_b_sq = (x_rot - b_x) ** 2 + (y_rot - b_y) ** 2
        choose_a = distance_a_sq <= distance_b_sq

        center_x = np.where(choose_a, a_x, b_x)
        center_y = np.where(choose_a, a_y, b_y)
        return center_x, center_y

    def nearest_void_center_lab(
        self,
        x_m: np.ndarray | float,
        y_m: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_rot, y_rot = self.lab_to_lattice(x_m, y_m)
        center_x_rot, center_y_rot = self.nearest_void_center_lattice(
            x_rot,
            y_rot,
        )
        return self.lattice_to_lab(center_x_rot, center_y_rot)

    def rod_distance_m(
        self,
        x_m: np.ndarray | float,
        y_m: np.ndarray | float,
    ) -> np.ndarray:
        center_x, center_y = self.nearest_rod_center_lab(x_m, y_m)
        x = np.asarray(x_m, dtype=float)
        y = np.asarray(y_m, dtype=float)
        x, y = np.broadcast_arrays(x, y)
        return np.hypot(x - center_x, y - center_y)

    def inside_rod(
        self,
        x_m: np.ndarray | float,
        y_m: np.ndarray | float,
        *,
        include_boundary: bool = True,
    ) -> np.ndarray:
        distance = self.rod_distance_m(x_m, y_m)
        if include_boundary:
            return distance <= self.cnt_radius_m
        return distance < self.cnt_radius_m

    def inside_plasma_bounds(
        self,
        x_m: np.ndarray | float,
        y_m: np.ndarray | float,
    ) -> np.ndarray:
        if self.plasma_halfwidth_x_m is None or self.plasma_halfwidth_y_m is None:
            raise ValueError(
                "Plasma half-widths are required to evaluate transverse bounds"
            )
        x = np.asarray(x_m, dtype=float)
        y = np.asarray(y_m, dtype=float)
        x, y = np.broadcast_arrays(x, y)
        return (
            (x >= -self.plasma_halfwidth_x_m)
            & (x <= self.plasma_halfwidth_x_m)
            & (y >= -self.plasma_halfwidth_y_m)
            & (y <= self.plasma_halfwidth_y_m)
        )

    def rod_centers_in_bounds(
        self,
        *,
        xmin_m: float,
        xmax_m: float,
        ymin_m: float,
        ymax_m: float,
        margin_m: float = 0.0,
    ) -> np.ndarray:
        """Enumerate rod centers whose lab coordinates lie in padded bounds."""

        if not (xmin_m < xmax_m and ymin_m < ymax_m):
            raise ValueError("Bounds must satisfy xmin < xmax and ymin < ymax")
        if margin_m < 0.0 or not np.isfinite(margin_m):
            raise ValueError("margin_m must be finite and non-negative")

        corners_x = np.asarray([xmin_m, xmin_m, xmax_m, xmax_m])
        corners_y = np.asarray([ymin_m, ymax_m, ymin_m, ymax_m])
        corners_x_rot, corners_y_rot = self.lab_to_lattice(corners_x, corners_y)
        pad = margin_m + self.lattice_spacing_m
        x_rot_min = float(np.min(corners_x_rot) - pad)
        x_rot_max = float(np.max(corners_x_rot) + pad)
        y_rot_min = float(np.min(corners_y_rot) - pad)
        y_rot_max = float(np.max(corners_y_rot) + pad)

        n_min = math.floor(y_rot_min / self.lattice_height_m) - 1
        n_max = math.ceil(y_rot_max / self.lattice_height_m) + 1
        centers_rot: list[tuple[float, float]] = []

        for row in range(n_min, n_max + 1):
            parity = row - 2 * math.floor(row / 2)
            stagger = 0.5 * self.lattice_spacing_m * parity
            m_min = math.floor(
                (x_rot_min - stagger) / self.lattice_spacing_m
            ) - 1
            m_max = math.ceil(
                (x_rot_max - stagger) / self.lattice_spacing_m
            ) + 1
            for column in range(m_min, m_max + 1):
                centers_rot.append(
                    (
                        column * self.lattice_spacing_m + stagger,
                        row * self.lattice_height_m,
                    )
                )

        centers_array = np.asarray(centers_rot, dtype=float)
        x_lab, y_lab = self.lattice_to_lab(
            centers_array[:, 0],
            centers_array[:, 1],
        )
        centers_lab = np.column_stack((x_lab, y_lab))
        keep = (
            (centers_lab[:, 0] >= xmin_m - margin_m)
            & (centers_lab[:, 0] <= xmax_m + margin_m)
            & (centers_lab[:, 1] >= ymin_m - margin_m)
            & (centers_lab[:, 1] <= ymax_m + margin_m)
        )
        selected = centers_lab[keep]
        order = np.lexsort((selected[:, 0], selected[:, 1]))
        return selected[order]


def geometry_from_mapping(values: Mapping[str, object]) -> HoneycombGeometry:
    required = (
        "cnt_radius_m",
        "lattice_spacing_m",
        "lattice_height_m",
        "honeycomb_angle_deg",
        "honeycomb_cos",
        "honeycomb_sin",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise KeyError(f"Missing honeycomb geometry parameters: {missing}")

    geometry = HoneycombGeometry(
        cnt_radius_m=float(values["cnt_radius_m"]),
        lattice_spacing_m=float(values["lattice_spacing_m"]),
        lattice_height_m=float(values["lattice_height_m"]),
        honeycomb_angle_deg=float(values["honeycomb_angle_deg"]),
        honeycomb_cos=float(values["honeycomb_cos"]),
        honeycomb_sin=float(values["honeycomb_sin"]),
        plasma_halfwidth_x_m=(
            None
            if values.get("plasma_halfwidth_x_m") is None
            else float(values["plasma_halfwidth_x_m"])
        ),
        plasma_halfwidth_y_m=(
            None
            if values.get("plasma_halfwidth_y_m") is None
            else float(values["plasma_halfwidth_y_m"])
        ),
    )
    geometry.validate()
    return geometry


def read_honeycomb_geometry(path: str | Path) -> HoneycombGeometry:
    source = Path(path).resolve(strict=False)
    if not source.is_file():
        raise FileNotFoundError(
            f"Resolved-parameters JSON does not exist: {source}"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Resolved-parameters JSON must contain an object: {source}"
        )
    return geometry_from_mapping(payload)
