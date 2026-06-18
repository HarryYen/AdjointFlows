"""Sum multiple regular-grid gradient.xyz files into one gradient.xyz.

Use this after kernels from different meshes have each been projected onto the
same lon/lat/depth regular grid. Edit the config block directly, then run.
"""

from pathlib import Path

import numpy as np


# =============================================================================
# User settings: edit this block directly, then run this file.
# =============================================================================
BASE_DIR = Path("/home/harry/Work/adjflows_for_ambient_noise/AdjointFlows")

INPUT_XYZ_FILES = [
    '/home/harry/Work/Other_AdjointFlows/AdjointFlows_no_smoothed/TOMO/MASK_KERNEL_SUM/KERNEL/PRECOND/gradient.xyz',
    '/home/harry/Work/AdjointFlows_mesh_2_no_smoothed/TOMO/MASK_KERNEL_SUM/KERNEL_22_25/PRECOND/gradient.xyz',
    '/home/harry/Work/AdjointFlows_mesh_2_no_smoothed/TOMO/MASK_KERNEL_SUM/KERNEL_14_21/PRECOND/gradient.xyz',
    '/home/harry/Work/adjflows_for_ambient_noise/AdjointFlows/TOMO/MASK_KERNEL_SUM/KERNEL_EQ_5_12s/PRECOND/gradient.xyz'   
]

OUTPUT_XYZ = BASE_DIR / "TOMO" / "MASK_KERNEL_SUM" / "gradient_summed.xyz"

# Columns are zero-based. For gradient.xyz: lon lat dep alpha beta rho ...
SUM_COLUMNS = [3, 4, 5]


def read_xyz_header(input_xyz, header_lines=5):
    """Read the first header lines from a gradient.xyz file."""
    with open(input_xyz) as f:
        return [next(f).rstrip("\n") for _ in range(header_lines)]


def check_same_grid(reference_data, data, input_xyz):
    """Check that lon/lat/dep columns match exactly enough for xyz summation."""
    if data.shape[0] != reference_data.shape[0]:
        raise ValueError(
            f"Row count mismatch for {input_xyz}: "
            f"{data.shape[0]} != {reference_data.shape[0]}"
        )

    if not np.allclose(data[:, 0:3], reference_data[:, 0:3], equal_nan=True):
        raise ValueError(f"Grid lon/lat/dep columns do not match: {input_xyz}")


def update_header_minmax(header_lines, output_data, sum_columns):
    """Update value min/max lines in a gradient.xyz-style header."""
    header_lines = list(header_lines)
    if len(header_lines) >= 4 and len(sum_columns) >= 3:
        minmax_values = []
        for column in sum_columns[:3]:
            minmax_values.append(np.nanmin(output_data[:, column]))
            minmax_values.append(np.nanmax(output_data[:, column]))
        header_lines[3] = " " + " ".join(f"{value:.4e}" for value in minmax_values)
    return header_lines


def sum_gradient_xyz(input_xyz_files, output_xyz, sum_columns):
    """Sum selected scalar columns from multiple gradient.xyz files."""
    if not input_xyz_files:
        raise ValueError("INPUT_XYZ_FILES is empty. Edit the config block before running.")

    input_xyz_files = [Path(input_xyz) for input_xyz in input_xyz_files]
    for input_xyz in input_xyz_files:
        if not input_xyz.is_file():
            raise FileNotFoundError(f"Input xyz file not found: {input_xyz}")

    reference_header = read_xyz_header(input_xyz_files[0])
    output_data = None
    reference_data = None

    for input_xyz in input_xyz_files:
        data = np.loadtxt(input_xyz, skiprows=5)
        if reference_data is None:
            reference_data = data
            output_data = data.copy()
            output_data[:, sum_columns] = 0.0
        else:
            check_same_grid(reference_data, data, input_xyz)

        output_data[:, sum_columns] += data[:, sum_columns]

    output_xyz = Path(output_xyz)
    output_xyz.parent.mkdir(parents=True, exist_ok=True)
    output_header = update_header_minmax(reference_header, output_data, sum_columns)
    np.savetxt(
        output_xyz,
        output_data,
        fmt="%.6e",
        header="\n".join(output_header),
        comments="",
    )
    return output_xyz


def main():
    """Sum configured gradient.xyz files."""
    output_xyz = sum_gradient_xyz(INPUT_XYZ_FILES, OUTPUT_XYZ, SUM_COLUMNS)
    print(f"Wrote {output_xyz}")


if __name__ == "__main__":
    main()
