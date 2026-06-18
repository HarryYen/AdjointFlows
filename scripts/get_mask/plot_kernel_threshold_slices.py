#%%
"""Plot summed-kernel horizontal slices to choose a mask threshold.

Edit the config block directly, then run this file. This script expects a
regular-grid xyz file with columns like ``gradient.xyz``:

lon lat dep alpha beta rho ...
"""

from pathlib import Path
import sys

import numpy as np
import pygmt


BASE_DIR = Path("/home/harry/Work/adjflows_for_ambient_noise/AdjointFlows")
VISUALIZER_DIR = BASE_DIR / "adjointflows" / "visualizer"
if str(VISUALIZER_DIR) not in sys.path:
    sys.path.append(str(VISUALIZER_DIR))

from plotting_modules import (  # noqa: E402
    find_minmax_from_xyz_file,
    find_nxnynz_from_xyz_file,
    interp_2d_in_specific_dep,
)


# =============================================================================
# User settings: edit this block directly, then run this file.
# =============================================================================
INPUT_XYZ = (
    BASE_DIR
    / "TOMO"
    / "MASK_KERNEL_SUM"
    / "gradient_summed.xyz"
)
OUTPUT_DIR = INPUT_XYZ.parent / "fig_threshold_slices"

MAP_REGION = [119.0, 123.0, 21.0, 26.0]
DEPTH_LIST = [6, 10, 15, 20, 30, 50, 80, 120, 150]

# Columns are zero-based in the loaded numpy array.
SCALAR_COLUMNS = {
    "alpha": 3,
    "beta": 4,
    "rho": 5,
}

# "log10" is useful when summed absolute kernels span many orders of magnitude.
# Use "linear" to plot raw values.
PLOT_SCALE = "log10"
LOG_EPS = 1e-30

# Raw threshold values for mask selection. These are in the original kernel
# units, not log10 units. Set a scalar to None to skip its contour.
THRESHOLDS = {
    "alpha": 2.e-10,
    "beta": 2.e-10,
    "rho": 2.e-10,
}
CONTOUR_PEN = "0.8p,cyan"
CONTOUR_ANNOTATE = False

WRITE_MASK_XYZ = True
OUTPUT_MASK_XYZ = INPUT_XYZ.parent / "mask.xyz"
MASK_VALUE_INSIDE = 1
MASK_VALUE_OUTSIDE = 0
MASK_OUTPUT_NAMES = ["vp_mask", "vs_mask", "rho_mask"]
MASK_XYZ_FMT = ["%.3f", "%.3f", "%.3f", "%d", "%d", "%d"]

# Set to None for automatic global range per scalar.
# Otherwise use [min, max, interval], for example [-16, -8, 0.25] for log10.
CPT_RANGE = None
CMAP = "hot"
REVERSE_CMAP = False

FINE_GRID_SPACING = 0.02
FIGSIZE = ("28c", "25c")
DPI = 300

PERCENTILES_FOR_SUMMARY = [50, 70, 80, 85, 90, 95, 97, 99]


def load_scalar_array(input_xyz, scalar_column):
    """Load one scalar from an xyz file and reshape it to (nz, nx, ny)."""
    nx, ny, nz = find_nxnynz_from_xyz_file(input_xyz)
    all_arr_flat = np.atleast_2d(np.loadtxt(input_xyz, skiprows=5))

    lon_arr = all_arr_flat[:, 0]
    lat_arr = all_arr_flat[:, 1]
    dep_arr = all_arr_flat[:, 2]
    lon_uniq = np.unique(lon_arr)
    lat_uniq = np.unique(lat_arr)
    dep_uniq = np.unique(dep_arr)

    if scalar_column >= all_arr_flat.shape[1]:
        raise ValueError(
            f"{input_xyz} has {all_arr_flat.shape[1]} columns, "
            f"so column {scalar_column} is not available."
        )

    scalar_flat = all_arr_flat[:, scalar_column]
    scalar_arr = scalar_flat.reshape(nz, nx, ny)
    return nx, ny, lon_uniq, lat_uniq, dep_uniq, scalar_flat, scalar_arr


def transform_for_plot(values):
    """Transform values for plotting while preserving NaNs."""
    if PLOT_SCALE == "linear":
        return values
    if PLOT_SCALE == "log10":
        return np.log10(np.maximum(values, 0.0) + LOG_EPS)
    raise ValueError(f"Unsupported PLOT_SCALE: {PLOT_SCALE}")


def transform_threshold_for_plot(threshold):
    """Transform a raw threshold into the plotting scale."""
    if threshold is None:
        return None
    if PLOT_SCALE == "linear":
        return float(threshold)
    if PLOT_SCALE == "log10":
        if threshold < 0:
            raise ValueError("A log10 threshold must be >= 0 in raw kernel units.")
        return float(np.log10(threshold + LOG_EPS))
    raise ValueError(f"Unsupported PLOT_SCALE: {PLOT_SCALE}")


def make_auto_cpt_range(values, n_interval=100):
    """Make a robust CPT range from finite plotting values."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("No finite values available for CPT range.")

    vmin = float(np.nanmin(finite_values))
    vmax = float(np.nanmax(finite_values))
    if np.isclose(vmin, vmax):
        padding = abs(vmin) * 0.05 if vmin else 1.0
        vmin -= padding
        vmax += padding

    interval = (vmax - vmin) / n_interval
    return [vmin, vmax, interval]


def write_percentile_summary(output_file, scalar_name, scalar_values):
    """Write percentile values from the untransformed scalar field."""
    finite_values = scalar_values[np.isfinite(scalar_values)]
    if finite_values.size == 0:
        raise ValueError(f"No finite values found for {scalar_name}.")

    with open(output_file, "w") as f:
        f.write(f"scalar {scalar_name}\n")
        f.write(f"input {INPUT_XYZ}\n")
        f.write(f"plot_scale {PLOT_SCALE}\n")
        f.write(f"count {finite_values.size}\n")
        f.write(f"min {np.nanmin(finite_values):.6e}\n")
        f.write(f"max {np.nanmax(finite_values):.6e}\n")
        for percentile in PERCENTILES_FOR_SUMMARY:
            value = np.nanpercentile(finite_values, percentile)
            f.write(f"p{percentile:g} {value:.6e}\n")


def read_xyz_header(input_xyz, header_lines=5):
    """Read header lines from a gradient.xyz-like file."""
    with open(input_xyz) as f:
        return [next(f).rstrip("\n") for _ in range(header_lines)]


def update_mask_header(header_lines):
    """Update gradient.xyz-style header lines for 0/1 masks."""
    header_lines = list(header_lines)
    if len(header_lines) >= 4:
        header_lines[3] = " 0 1 0 1 0 1"
    if len(header_lines) >= 5:
        header_lines[4] = " lon lat dep " + " ".join(MASK_OUTPUT_NAMES)
    return header_lines


def write_mask_xyz():
    """Write a gradient.xyz-like file containing only vp/vs/rho 0/1 masks."""
    input_xyz = Path(INPUT_XYZ)
    if not input_xyz.is_file():
        raise FileNotFoundError(f"Input xyz file not found: {input_xyz}")

    all_arr_flat = np.atleast_2d(np.loadtxt(input_xyz, skiprows=5))
    mask_data = np.zeros((all_arr_flat.shape[0], 3 + len(SCALAR_COLUMNS)), dtype=float)
    mask_data[:, 0:3] = all_arr_flat[:, 0:3]

    for output_index, (scalar_name, scalar_column) in enumerate(SCALAR_COLUMNS.items()):
        threshold = THRESHOLDS.get(scalar_name)
        if threshold is None:
            print(f"Skipping mask for {scalar_name}: threshold is None")
            continue
        if scalar_column >= all_arr_flat.shape[1]:
            raise ValueError(
                f"{input_xyz} has {all_arr_flat.shape[1]} columns, "
                f"so column {scalar_column} is not available."
            )

        values = all_arr_flat[:, scalar_column]
        finite = np.isfinite(values)
        mask = np.full(values.shape, MASK_VALUE_OUTSIDE, dtype=int)
        mask[finite & (values >= threshold)] = MASK_VALUE_INSIDE
        mask_data[:, 3 + output_index] = mask

    output_mask_xyz = Path(OUTPUT_MASK_XYZ)
    output_mask_xyz.parent.mkdir(parents=True, exist_ok=True)
    header_lines = update_mask_header(read_xyz_header(input_xyz))
    np.savetxt(
        output_mask_xyz,
        mask_data,
        fmt=MASK_XYZ_FMT[: mask_data.shape[1]],
        header="\n".join(header_lines),
        comments="",
    )
    print(f"Wrote {output_mask_xyz}")


def plot_scalar_depth_slices(scalar_name, scalar_column):
    """Plot a 3x3 depth-slice panel for one scalar."""
    input_xyz = Path(INPUT_XYZ)
    if not input_xyz.is_file():
        raise FileNotFoundError(f"Input xyz file not found: {input_xyz}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lon_min, lon_max, lat_min, lat_max, _dep_min, _dep_max = find_minmax_from_xyz_file(
        input_xyz
    )
    grd_range = [lon_min, lon_max, lat_min, lat_max]

    nx, ny, lon_uniq, lat_uniq, dep_uniq, scalar_flat, scalar_arr = load_scalar_array(
        input_xyz, scalar_column
    )
    plot_arr = transform_for_plot(scalar_arr)
    cpt_range = CPT_RANGE if CPT_RANGE is not None else make_auto_cpt_range(plot_arr)
    contour_value = transform_threshold_for_plot(THRESHOLDS.get(scalar_name))

    summary_file = OUTPUT_DIR / f"{scalar_name}_threshold_percentiles.txt"
    write_percentile_summary(summary_file, scalar_name, scalar_flat)

    fig = pygmt.Figure()
    pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")

    with fig.subplot(
        nrows=3,
        ncols=3,
        figsize=FIGSIZE,
        margins="0.01c",
        frame=["a", "WSne"],
    ):
        for index, depth_km in enumerate(DEPTH_LIST):
            selected_df = interp_2d_in_specific_dep(
                lon_uniq,
                lat_uniq,
                dep_uniq,
                plot_arr,
                depth_km,
            )
            pygmt.xyz2grd(
                data=selected_df,
                outgrid="tmp.grd",
                region=grd_range,
                spacing=f"{nx}+n/{ny}+n",
                verbose="q",
            )
            pygmt.grdsample(
                grid="tmp.grd",
                spacing=FINE_GRID_SPACING,
                region=grd_range,
                outgrid="tmp_fine.grd",
                verbose="q",
            )

            with fig.set_panel(panel=index):
                pygmt.makecpt(
                    cmap=CMAP,
                    series=cpt_range,
                    continuous=True,
                    reverse=REVERSE_CMAP,
                )
                fig.grdimage(
                    grid="tmp_fine.grd",
                    cmap=True,
                    region=MAP_REGION,
                    projection="M?",
                    frame=True,
                )
                if contour_value is not None:
                    fig.grdcontour(
                        grid="tmp_fine.grd",
                        levels=[contour_value],
                        pen=CONTOUR_PEN,
                        annotation=CONTOUR_ANNOTATE,
                    )
                fig.coast(shorelines=True)
                fig.text(
                    text=f"dep: {depth_km:g}km",
                    font="12p,Helvetica-Bold",
                    position="BR",
                    frame=True,
                )
                pygmt.config(FONT_ANNOT_PRIMARY="20p,Helvetica")
                pygmt.config(FONT_LABEL="20p,Helvetica")
                fig.colorbar(
                    frame=f'af+l{PLOT_SCALE}({scalar_name})',
                    position="JBC+w3.5c/0.3c+v+o3.5c/-3.5c",
                )

    output_png = OUTPUT_DIR / f"{scalar_name}_{PLOT_SCALE}_depth_slices.png"
    fig.savefig(output_png, dpi=DPI, transparent=True)
    # fig.show()
    print(f"Wrote {output_png}")
    print(f"Wrote {summary_file}")


def main():
    """Plot all configured scalar columns."""
    if WRITE_MASK_XYZ:
        write_mask_xyz()

    for scalar_name, scalar_column in SCALAR_COLUMNS.items():
        print(f"Plotting {scalar_name} from column {scalar_column}")
        plot_scalar_depth_slices(scalar_name, scalar_column)


if __name__ == "__main__":
    main()
