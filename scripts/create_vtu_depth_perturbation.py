"""
Create depth-relative velocity perturbation VTU files for ParaView.

This script reads SPECFEM-style model VTU files, estimates the horizontal mean
value at each depth, and writes a new VTU containing dVp/dVs-like perturbations:

    dmodel = (model - mean_at_same_depth) / mean_at_same_depth * 100

Edit the USER PARAMETERS section below, then run:

    python utils/create_vtu_depth_perturbation.py
"""

from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


# =============================================================================
# USER PARAMETERS
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_NAME = "m030"

DATABASES_DIR = ROOT_DIR / "TOMO" / MODEL_NAME / "DATABASES_MPI"
OUTPUT_FILE = DATABASES_DIR / "velocity_depth_perturbation.vtu"

# VTU files and point-data arrays to process.
# Each file is expected to contain a point-data array with the same name.
MODEL_FIELDS = ["vp", "vs"]

# Horizontal sampling used to estimate the mean value at each depth.
# Smaller values are more accurate but slower.
HORIZONTAL_SPACING_M = 5000.0

# Depth sampling used to build the 1-D depth-average profile.
# Smaller values follow vertical variations more closely but are slower.
DEPTH_SPACING_M = 2000.0

# If True, invalid sampled points outside the mesh are ignored when averaging.
IGNORE_INVALID_PROBE_POINTS = True


# =============================================================================
# IMPLEMENTATION
# =============================================================================


def read_vtu(vtu_file):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(vtu_file))
    reader.Update()
    return reader.GetOutput()


def write_vtu(ugrid, output_file):
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(output_file))
    writer.SetInputData(ugrid)
    writer.Write()


def get_point_array(ugrid, array_name):
    data = ugrid.GetPointData().GetArray(array_name)
    if data is None:
        raise ValueError(f"Point-data array '{array_name}' was not found.")
    return vtk_to_numpy(data).astype(float)


def make_regular_xy_points(bounds, horizontal_spacing_m):
    xmin, xmax, ymin, ymax, _, _ = bounds
    x = np.arange(xmin, xmax + horizontal_spacing_m, horizontal_spacing_m)
    y = np.arange(ymin, ymax + horizontal_spacing_m, horizontal_spacing_m)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return xx.ravel(), yy.ravel()


def probe_values_at_depth(ugrid, array_name, query_x, query_y, z):
    coords = np.column_stack([query_x, query_y, np.full(query_x.size, z)])

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(coords, deep=True, array_type=vtk.VTK_DOUBLE))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    probe = vtk.vtkProbeFilter()
    probe.SetSourceData(ugrid)
    probe.SetInputData(polydata)
    probe.Update()

    sampled = probe.GetOutput()
    values_vtk = sampled.GetPointData().GetArray(array_name)
    if values_vtk is None:
        return np.full(query_x.size, np.nan)

    values = vtk_to_numpy(values_vtk).astype(float)
    if IGNORE_INVALID_PROBE_POINTS:
        mask_vtk = sampled.GetPointData().GetArray("vtkValidPointMask")
        if mask_vtk is not None:
            valid = vtk_to_numpy(mask_vtk).astype(bool)
            values[~valid] = np.nan

    return values


def build_depth_mean_profile(ugrid, array_name):
    bounds = ugrid.GetBounds()
    _, _, _, _, zmin, zmax = bounds

    query_x, query_y = make_regular_xy_points(bounds, HORIZONTAL_SPACING_M)
    depth_z = np.arange(zmin, zmax + DEPTH_SPACING_M, DEPTH_SPACING_M)
    mean_values = np.full(depth_z.size, np.nan)

    for i, z in enumerate(depth_z):
        values = probe_values_at_depth(ugrid, array_name, query_x, query_y, z)
        if not np.all(np.isnan(values)):
            mean_values[i] = np.nanmean(values)

    valid = np.isfinite(mean_values)
    if valid.sum() < 2:
        raise ValueError(
            f"Could not build a depth-average profile for '{array_name}'. "
            "Try increasing HORIZONTAL_SPACING_M or checking the VTU bounds."
        )

    mean_values = np.interp(depth_z, depth_z[valid], mean_values[valid])
    return depth_z, mean_values


def add_numpy_array(ugrid, values, array_name):
    vtk_array = numpy_to_vtk(values.astype(float), deep=True)
    vtk_array.SetName(array_name)
    ugrid.GetPointData().AddArray(vtk_array)


def add_depth_perturbation(base_grid, source_grid, array_name):
    values = get_point_array(source_grid, array_name)
    point_z = vtk_to_numpy(source_grid.GetPoints().GetData())[:, 2]

    depth_z, mean_profile = build_depth_mean_profile(source_grid, array_name)
    mean_at_points = np.interp(point_z, depth_z, mean_profile)

    perturbation = (values - mean_at_points) / mean_at_points * 100.0

    add_numpy_array(base_grid, values, array_name)
    add_numpy_array(base_grid, mean_at_points, f"{array_name}_depth_mean")
    add_numpy_array(base_grid, perturbation, f"d{array_name}")


def main():
    base_grid = None

    for array_name in MODEL_FIELDS:
        vtu_file = DATABASES_DIR / f"{array_name}.vtu"
        if not vtu_file.exists():
            raise FileNotFoundError(f"{vtu_file} does not exist.")

        print(f"Reading {vtu_file}")
        source_grid = read_vtu(vtu_file)

        if base_grid is None:
            base_grid = vtk.vtkUnstructuredGrid()
            base_grid.DeepCopy(source_grid)
            base_grid.GetPointData().Initialize()
        elif source_grid.GetNumberOfPoints() != base_grid.GetNumberOfPoints():
            raise ValueError(
                f"{vtu_file} has a different number of points from the first VTU file."
            )

        print(f"Calculating depth-relative perturbation for {array_name}")
        add_depth_perturbation(base_grid, source_grid, array_name)

    print(f"Writing {OUTPUT_FILE}")
    write_vtu(base_grid, OUTPUT_FILE)
    print("Done.")


if __name__ == "__main__":
    main()
