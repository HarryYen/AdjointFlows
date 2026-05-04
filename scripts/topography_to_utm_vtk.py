"""
Convert SPECFEM topography to a UTM-coordinate VTK surface.

Edit the USER PARAMETERS section, then run:

    python utils/topography_to_utm_vtk.py

The output is a legacy ASCII .vtk file that ParaView can open directly.
"""

from pathlib import Path

import numpy as np


# =============================================================================
# USER PARAMETERS
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

TOPO_FILE = ROOT_DIR / "specfem3d" / "DATA" / "meshfem3D_files" / "topo.xyz"
OUTPUT_FILE = ROOT_DIR / "topography_utm.vtk"

# Values from specfem3d/DATA/meshfem3D_files/interfaces.dat.
NXI = 1311
NETA = 1049
LON_MIN = 119.0
LAT_MIN = 26.1
SPACING_LON = 0.0046
SPACING_LAT = -0.0046

# Use the same UTM convention as the rest of this project if needed.
# Taiwan is commonly in UTM zone 51N, but some project files here use zone 50N.
UTM_ZONE = 50
IS_NORTH_HEMISPHERE = True

# Use this only for visual inspection in ParaView. The scalar elevation remains
# unchanged; only the point z coordinate is scaled.
VERTICAL_EXAGGERATION = 1.0

# SPECFEM topography files are usually written row-by-row:
# first ETA row, all XI values; then next ETA row.
DATA_ORDER = "eta_xi"


# =============================================================================
# IMPLEMENTATION
# =============================================================================


def lonlat_to_utm(lon, lat, utm_zone, is_north_hemisphere):
    """Convert longitude/latitude in degrees to WGS84 UTM metres."""
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)

    semi_major_axis = 6378137.0
    flattening = 1.0 / 298.257223563
    scale = 0.9996

    eccentricity_sq = flattening * (2.0 - flattening)
    eccentricity_prime_sq = eccentricity_sq / (1.0 - eccentricity_sq)

    lon_origin = (utm_zone - 1) * 6 - 180 + 3
    lon_origin_rad = np.radians(lon_origin)

    n = semi_major_axis / np.sqrt(1.0 - eccentricity_sq * np.sin(lat_rad) ** 2)
    t = np.tan(lat_rad) ** 2
    c = eccentricity_prime_sq * np.cos(lat_rad) ** 2
    a = np.cos(lat_rad) * (lon_rad - lon_origin_rad)

    m = semi_major_axis * (
        (1 - eccentricity_sq / 4 - 3 * eccentricity_sq**2 / 64 - 5 * eccentricity_sq**3 / 256) * lat_rad
        - (3 * eccentricity_sq / 8 + 3 * eccentricity_sq**2 / 32 + 45 * eccentricity_sq**3 / 1024) * np.sin(2 * lat_rad)
        + (15 * eccentricity_sq**2 / 256 + 45 * eccentricity_sq**3 / 1024) * np.sin(4 * lat_rad)
        - (35 * eccentricity_sq**3 / 3072) * np.sin(6 * lat_rad)
    )

    x = scale * n * (
        a
        + (1 - t + c) * a**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * eccentricity_prime_sq) * a**5 / 120
    ) + 500000.0

    y = scale * (
        m
        + n
        * np.tan(lat_rad)
        * (
            a**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * a**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * eccentricity_prime_sq) * a**6 / 720
        )
    )

    if not is_north_hemisphere:
        y = y + 10000000.0

    return x, y


def read_topography(topo_file, nxi, neta, data_order):
    topo = np.loadtxt(topo_file, dtype=float)
    expected_size = nxi * neta
    if topo.size != expected_size:
        raise ValueError(
            f"{topo_file} contains {topo.size} values, expected {expected_size}."
        )

    if data_order == "eta_xi":
        return topo.reshape(neta, nxi)
    if data_order == "xi_eta":
        return topo.reshape(nxi, neta).T

    raise ValueError("DATA_ORDER must be either 'eta_xi' or 'xi_eta'.")


def build_surface_points():
    lon = LON_MIN + np.arange(NXI) * SPACING_LON
    lat = LAT_MIN + np.arange(NETA) * SPACING_LAT
    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing="xy")

    x_grid, y_grid = lonlat_to_utm(
        lon_grid,
        lat_grid,
        utm_zone=UTM_ZONE,
        is_north_hemisphere=IS_NORTH_HEMISPHERE,
    )

    topo_grid = read_topography(TOPO_FILE, NXI, NETA, DATA_ORDER)
    z_grid = topo_grid * VERTICAL_EXAGGERATION

    return x_grid, y_grid, z_grid, topo_grid


def write_structured_grid_vtk(output_file, x_grid, y_grid, z_grid, topo_grid):
    neta, nxi = topo_grid.shape
    npoints = nxi * neta

    with open(output_file, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Topography in UTM coordinates\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_GRID\n")
        f.write(f"DIMENSIONS {nxi} {neta} 1\n")
        f.write(f"POINTS {npoints} float\n")

        for x, y, z in zip(x_grid.ravel(), y_grid.ravel(), z_grid.ravel()):
            f.write(f"{x:.3f} {y:.3f} {z:.3f}\n")

        f.write(f"\nPOINT_DATA {npoints}\n")
        f.write("SCALARS elevation_m float 1\n")
        f.write("LOOKUP_TABLE default\n")

        for elevation in topo_grid.ravel():
            f.write(f"{elevation:.3f}\n")


def main():
    print(f"Reading {TOPO_FILE}")
    x_grid, y_grid, z_grid, topo_grid = build_surface_points()

    print(f"Writing {OUTPUT_FILE}")
    write_structured_grid_vtk(OUTPUT_FILE, x_grid, y_grid, z_grid, topo_grid)

    print("Done.")


if __name__ == "__main__":
    main()
