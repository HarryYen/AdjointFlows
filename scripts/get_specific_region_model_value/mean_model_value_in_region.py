"""Compute average model values inside a shapefile region.

Edit the parameters in the "User parameters" section, then run:

    python scripts/get_specific_region_model_value/mean_model_value_in_region.py

The script is intentionally CLI-free so it can be used as a small, repeatable
analysis note. It reads a SPECFEM/AdjointFlows regular-grid ``model.xyz`` file
with lon/lat/depth columns and selects points whose horizontal coordinates fall
inside a polygon shapefile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import struct

import matplotlib.path as mpath
import numpy as np


# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_XYZ = BASE_DIR / "TOMO" / "m026" / "OUTPUT" / "model.xyz"
SHAPEFILE = Path(__file__).resolve().parent / "Geological_Province" / "CR_high_vel.shp"

# model.xyz uses positive depth in km in this project, e.g. 0, 2, ..., 100.
DEPTH_MIN_KM = 8
DEPTH_MAX_KM = 9
# Columns in the current model.xyz:
# lon lat depth vp_abs vs_abs rho_abs vp_pert vs_pert rho_pert
COLUMN_NAMES = (
    "lon",
    "lat",
    "depth_km",
    "vp_abs",
    "vs_abs",
    "rho_abs",
    "vp_pert",
    "vs_pert",
    "rho_pert",
)
VALUE_COLUMNS = ("vp_abs", "vs_abs", "rho_abs", "vp_pert", "vs_pert", "rho_pert")

HEADER_LINES = 5
INCLUDE_BOUNDARY = True
OUTPUT_DIR = Path(__file__).resolve().parent / "OUTPUT"
OUTPUT_PREFIX = "COR_m030"

# Set this to True if you also want a potentially large CSV of all selected
# grid points. The summary file is always written.
WRITE_SELECTED_POINTS = False


@dataclass(frozen=True)
class PolygonPart:
    """One shapefile polygon ring."""

    points: np.ndarray
    signed_area: float

    @property
    def is_hole(self) -> bool:
        # ESRI polygon convention: outer rings are clockwise, holes are
        # counter-clockwise. With the usual signed-area formula, holes are > 0.
        return self.signed_area > 0.0


def ring_signed_area(points: np.ndarray) -> float:
    """Return signed area of a closed or open lon/lat polygon ring."""
    if points.shape[0] < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def read_polygon_shapefile(shp_path: Path) -> list[PolygonPart]:
    """Read Polygon/PolygonZ rings from an ESRI .shp file.

    This minimal reader avoids a GeoPandas/PyShp dependency. It supports the
    common polygon shape types needed for these geological province files.
    """
    if not shp_path.is_file():
        raise FileNotFoundError(f"Missing shapefile: {shp_path}")

    parts: list[PolygonPart] = []
    with shp_path.open("rb") as f:
        header = f.read(100)
        if len(header) != 100:
            raise ValueError(f"Invalid shapefile header: {shp_path}")

        file_code = struct.unpack(">i", header[:4])[0]
        if file_code != 9994:
            raise ValueError(f"Not an ESRI shapefile: {shp_path}")

        while True:
            record_header = f.read(8)
            if not record_header:
                break
            if len(record_header) != 8:
                raise ValueError(f"Truncated record header in {shp_path}")

            _record_number, content_length_words = struct.unpack(">2i", record_header)
            content = f.read(content_length_words * 2)
            if len(content) != content_length_words * 2:
                raise ValueError(f"Truncated record content in {shp_path}")

            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type == 0:
                continue
            if shape_type not in (5, 15, 25):
                raise ValueError(
                    f"Unsupported shapefile shape type {shape_type}. "
                    "Expected Polygon, PolygonZ, or PolygonM."
                )

            num_parts, num_points = struct.unpack("<2i", content[36:44])
            parts_start = 44
            points_start = parts_start + 4 * num_parts
            part_indices = list(struct.unpack(f"<{num_parts}i", content[parts_start:points_start]))
            point_values = struct.unpack(
                f"<{num_points * 2}d",
                content[points_start : points_start + num_points * 16],
            )
            all_points = np.asarray(point_values, dtype=float).reshape(num_points, 2)

            for i, start in enumerate(part_indices):
                end = part_indices[i + 1] if i + 1 < len(part_indices) else num_points
                ring = all_points[start:end]
                if ring.shape[0] >= 3:
                    parts.append(PolygonPart(points=ring, signed_area=ring_signed_area(ring)))

    if not parts:
        raise ValueError(f"No polygon rings found in {shp_path}")
    return parts


def points_in_polygon_parts(lon: np.ndarray, lat: np.ndarray, parts: list[PolygonPart]) -> np.ndarray:
    """Return mask for points inside shapefile polygon rings.

    Holes are subtracted when ring orientation follows the ESRI convention. If a
    file has ambiguous orientation, the practical fallback is union of all rings.
    """
    points = np.column_stack((lon, lat))
    radius = 1.0e-12 if INCLUDE_BOUNDARY else 0.0

    has_outer = any(not part.is_hole for part in parts)
    mask = np.zeros(points.shape[0], dtype=bool)
    hole_mask = np.zeros(points.shape[0], dtype=bool)

    for part in parts:
        path = mpath.Path(part.points)
        in_ring = path.contains_points(points, radius=radius)
        if has_outer and part.is_hole:
            hole_mask |= in_ring
        else:
            mask |= in_ring

    return mask & ~hole_mask


def load_model_xyz(model_xyz: Path) -> dict[str, np.ndarray]:
    """Load model.xyz into named columns."""
    if not model_xyz.is_file():
        raise FileNotFoundError(f"Missing model file: {model_xyz}")
    data = np.loadtxt(model_xyz, skiprows=HEADER_LINES)
    if data.ndim != 2 or data.shape[1] < len(COLUMN_NAMES):
        raise ValueError(
            f"Expected at least {len(COLUMN_NAMES)} columns in {model_xyz}, "
            f"got shape {data.shape}"
        )
    return {name: data[:, i] for i, name in enumerate(COLUMN_NAMES)}


def summarize(values: np.ndarray) -> dict[str, float | int]:
    """Return NaN-aware summary statistics."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def write_summary(summary_path: Path, stats: dict[str, dict[str, float | int]], selected_count: int) -> None:
    """Write a human-readable summary text file."""
    lines = [
        "Model region value summary",
        "",
        f"model_xyz: {MODEL_XYZ}",
        f"shapefile: {SHAPEFILE}",
        f"depth_range_km: {DEPTH_MIN_KM:.3f} {DEPTH_MAX_KM:.3f}",
        f"selected_grid_points_before_nan_filter: {selected_count}",
        "",
        "column count mean median std min max",
    ]
    for column in VALUE_COLUMNS:
        item = stats[column]
        lines.append(
            f"{column} "
            f"{item['count']} "
            f"{item['mean']:.6f} "
            f"{item['median']:.6f} "
            f"{item['std']:.6f} "
            f"{item['min']:.6f} "
            f"{item['max']:.6f}"
        )
    summary_path.write_text("\n".join(lines) + "\n")


def write_summary_csv(csv_path: Path, stats: dict[str, dict[str, float | int]]) -> None:
    """Write summary statistics as CSV for spreadsheet use."""
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["column", "count", "mean", "median", "std", "min", "max"])
        for column in VALUE_COLUMNS:
            item = stats[column]
            writer.writerow(
                [
                    column,
                    item["count"],
                    item["mean"],
                    item["median"],
                    item["std"],
                    item["min"],
                    item["max"],
                ]
            )


def write_selected_points(output_path: Path, model: dict[str, np.ndarray], mask: np.ndarray) -> None:
    """Write selected model grid points as CSV."""
    selected = np.column_stack([model[name][mask] for name in COLUMN_NAMES])
    header = ",".join(COLUMN_NAMES)
    np.savetxt(output_path, selected, delimiter=",", header=header, comments="", fmt="%.6f")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading model: {MODEL_XYZ}")
    model = load_model_xyz(MODEL_XYZ)

    print(f"Reading region shapefile: {SHAPEFILE}")
    polygon_parts = read_polygon_shapefile(SHAPEFILE)

    depth_mask = (
        np.isfinite(model["depth_km"])
        & (model["depth_km"] >= DEPTH_MIN_KM)
        & (model["depth_km"] <= DEPTH_MAX_KM)
    )
    lon_min = min(float(np.min(part.points[:, 0])) for part in polygon_parts)
    lon_max = max(float(np.max(part.points[:, 0])) for part in polygon_parts)
    lat_min = min(float(np.min(part.points[:, 1])) for part in polygon_parts)
    lat_max = max(float(np.max(part.points[:, 1])) for part in polygon_parts)
    bbox_mask = (
        (model["lon"] >= lon_min)
        & (model["lon"] <= lon_max)
        & (model["lat"] >= lat_min)
        & (model["lat"] <= lat_max)
    )
    candidate_mask = depth_mask & bbox_mask

    horizontal_mask = np.zeros_like(candidate_mask, dtype=bool)
    candidate_index = np.flatnonzero(candidate_mask)
    horizontal_mask[candidate_index] = points_in_polygon_parts(
        model["lon"][candidate_index],
        model["lat"][candidate_index],
        polygon_parts,
    )

    selected_mask = candidate_mask & horizontal_mask
    selected_count = int(np.count_nonzero(selected_mask))
    if selected_count == 0:
        raise ValueError(
            "No model grid points selected. Check depth range, shapefile CRS, "
            "and whether model lon/lat overlaps the region."
        )

    stats = {column: summarize(model[column][selected_mask]) for column in VALUE_COLUMNS}

    tag = f"{OUTPUT_PREFIX}_depth_{DEPTH_MIN_KM:g}_{DEPTH_MAX_KM:g}km"
    summary_txt = OUTPUT_DIR / f"{tag}_summary.txt"
    summary_csv = OUTPUT_DIR / f"{tag}_summary.csv"
    write_summary(summary_txt, stats, selected_count)
    write_summary_csv(summary_csv, stats)

    if WRITE_SELECTED_POINTS:
        selected_csv = OUTPUT_DIR / f"{tag}_selected_points.csv"
        write_selected_points(selected_csv, model, selected_mask)
        print(f"Wrote selected points: {selected_csv}")

    print("")
    print(f"Selected grid points before NaN filtering: {selected_count}")
    for column in VALUE_COLUMNS:
        item = stats[column]
        print(
            f"{column:8s} count={item['count']:7d} "
            f"mean={item['mean']:10.6f} "
            f"min={item['min']:10.6f} "
            f"max={item['max']:10.6f}"
        )
    print("")
    print(f"Wrote summary: {summary_txt}")
    print(f"Wrote CSV:     {summary_csv}")


if __name__ == "__main__":
    main()
